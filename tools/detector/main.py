#!/usr/bin/env python3
"""
tiling_llm_agent_custom_api_v3.py

Adds:
 - detection of most important tiling functions (normalization process)
 - function-based extraction if file > 200 lines
 - JSON metadata describing mode (function-based vs file-based)

For each file labeled "real":
 - Output exactly ONE .cpp/.cc/.c/.hpp/.h file with the SAME NAME as original,
   but possibly normalized (subset of functions if long).
 - Also output meta.json describing:
     {
       "mode": "file-based" | "function-based",
       "functions": [ ... ]   # only for function-based
     }
"""

from __future__ import annotations
import argparse
import logging
import os
import re
import json
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import subprocess
# --- OpenAI imports with fallback ---
try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None

try:
    import requests
except Exception:
    requests = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger("tiling_llm_agent_v3")

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="tiling extractor with function-based normalization")
    p.add_argument("--op", default=None)
    p.add_argument("--base", default=None)
    p.add_argument("--repo-root", default="/home/l00936201/AscendC-copliot/ops-x/ops-nn")
    p.add_argument("--result-root", default="/home/l00936201/AscendC-copliot/tools/detector/test/result")
    p.add_argument("--max-chars", type=int, default=10000000000)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key", default="sk-d1e612e48d654731bffff6cbf35fdfd8")
    p.add_argument("--api-host", default="https://api.deepseek.com/v1")
    p.add_argument("--write-report", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()

# ============================================================
# Resolve operator base
# ============================================================

def resolve_base(base_arg, op_arg, repo_root):
    if base_arg:
        base = Path(base_arg)
        if not base.exists():
            raise FileNotFoundError(f"Provided base path does not exist: {base}")
        return base.resolve(), base.parent.name, base.name

    if not op_arg:
        raise ValueError("Either --base or --op must be provided")

    if "/" in op_arg:
        category, opname = op_arg.split("/", 1)
        base = Path(repo_root) / category / opname
        if not base.exists():
            raise FileNotFoundError(f"Operator base not found at {base}")
        return base.resolve(), category, opname

    root = Path(repo_root)
    for p in root.rglob(op_arg):
        if p.is_dir() and (p / "op_host").exists():
            return p.resolve(), p.parent.name, p.name

    raise FileNotFoundError(f"Could not find operator {op_arg} under {repo_root}")

# ============================================================
# I/O Helpers
# ============================================================

def read_text(p: Path, max_chars: int) -> str:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = p.read_text(encoding="latin-1", errors="ignore")

    if len(txt) > max_chars:
        half = max_chars // 2
        txt = txt[:half] + "\n\n/* --TRUNCATED-- (middle omitted) -- */\n\n" + txt[-half:]

    return txt

def count_lines(p: Path) -> int:
    try:
        return len(p.read_text(errors="ignore").splitlines())
    except Exception:
        return 999_999

# ============================================================
# Prompt for detecting tiling (file-level classifier)
# ============================================================

SYSTEM_PROMPT = """
You are an expert C/C++ code reviewer specializing in **tiling implementations** on Ascend / HPC-style kernels.

Your job: given the *entire contents of a single C/C++ source or header file*, decide
whether this file actually contains **REAL tiling computations**, or is only doing
registration / declarations / trivial boilerplate.

Definitions (very important):

1. REAL tiling computation  ("real")
   - There exist **function definitions with bodies** (not just declarations)
     that perform or directly drive numeric / tiling logic, such as:
       - explicit loops (`for`, `while`, `do`), especially over dimensions / blocks / tiles
       - arithmetic or bitwise math that computes tile sizes, UB usage, strides, factors, etc.
       - ceil / floor / align operations for tiles, such as:
           - `CeilAlign`, `FloorDiv`, `CeilDiv`, `FloorAlign`, `AlignUp`, etc.
           - `Ops::Base::*` tiling helpers
       - computing or updating tiling-related data structures, examples:
           - `td_.set_*` / `tilingData.*` fields for shapes, UB/GM sizes, block factors
       - logic that decides how to split work across cores, blocks, tiles, or loops
       - UB memory budgeting (e.g. compare against `ubSizePlatForm`, `coreNum`, etc.)

2. DECL / registration-only ("decl")
   - The file contains only:
       - macro calls such as `IMPL_OP_OPTILING(...)`, registration helpers, or
         thin glue code around the real tiling function in another file
       - struct / class / typedef definitions of tiling-related data without math
       - declarations of functions (without bodies) or pure header APIs
       - very small "prepare" functions that:
           - just fetch platform info or shapes
           - assign them into a struct or `compileInfo`
           - DO NOT loop, DO NOT do non-trivial arithmetic, and DO NOT choose tile sizes
       - plain data tables, enums, constants, or logging code
   - These should be labeled "decl", even if they mention "tiling" in names or comments.

Edge cases and tie-breaking:
 - If there is **any** function body that clearly performs tiling or UB/GM sizing math,
   label the file `"real"` (even if there is also registration boilerplate).
 - If you are uncertain, prefer label `"decl"` with low confidence rather than guessing.
 - Ignore comments when deciding; only behavior of actual compiled code matters.

Output format (strict):
 - Return **ONLY** a single JSON object, no extra text, no markdown.
 - The JSON MUST have exactly these keys: "label", "confidence", "explanation", "evidence".

   - "label": string, either "real" or "decl".
   - "confidence": integer 0–100 (your subjective confidence in the label).
   - "explanation": a short 1–3 sentence human-readable explanation of why
        you chose this label (mention function names or obvious patterns).
   - "evidence": list of 1–6 short strings, each describing a code region with
        approximate line numbers, for example:
        - "L120-L145: DoUbTiling uses for-loop + CeilAlign to compute UB factors"
        - "L10-L30: IMPL_OP_OPTILING(...) registration only, no loops or math"

Additional constraints:
 - If the file text contains a special marker like "--TRUNCATED--", still classify based
   on what you see; do NOT mention truncation in the JSON.
 - Do NOT invent functions that are not present in the file.
 - Do NOT output anything except the JSON object.
"""

SHOT_USER_1 = """// Example: registration-only file (short)
IMPL_OP_OPTILING(BatchNormV3).Tiling(Tiling4BatchNormV3).TilingParse<BatchNormV3CompileInfo>(TilingPrepare4BatchNormV3);

static ge::graphStatus TilingPrepare4BatchNormV3(gert::TilingParseContext* context)
{
    // only gets platform info and writes to compileInfo; no math/loops
    auto compileInfo = context->GetCompiledInfo<BatchNormV3CompileInfo>();
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}
"""

SHOT_ASSISTANT_1 = json.dumps({
    "label": "decl",
    "confidence": 95,
    "explanation": "File only registers the op and defines a small prepare function that fetches platform info and assigns fields, without any loops or tiling math.",
    "evidence": [
        "L1: IMPL_OP_OPTILING(...) registration macro, no function body here",
        "L3-L9: TilingPrepare4BatchNormV3 reads platform info and writes to compileInfo, no loops or tile size calculations"
    ]
})

SHOT_USER_2 = """// Example: real tiling implementation (short)
static uint32_t FindDichotomizeAddDiffSize(uint32_t parallelN)
{
    if ((parallelN & (parallelN - 1)) != 0) {
        uint32_t temp = parallelN - 1;
        temp |= temp >> 1;
        temp |= temp >> 2;
        return (parallelN - ((temp + 1) / 2));
    } else {
        return 0;
    }
}

int64_t BatchNormV3FullReduceTiling::DoUbTiling(const int64_t blockFactor, int64_t& aUbSize, int64_t& rUbSize)
{
    int64_t eleNum = Ops::Base::FloorDiv(commonParams.ubSizePlatForm, 2);
    int64_t aUbFactor = std::min(blockFactor, eleNum / rUbNum / (commonParams.patternR1 * commonParams.patternR0));
    while (aUbFactor > 0) {
        aUbSize = Ops::Base::CeilAlign(aUbFactor, 16);
        if (aUbSize * 20 + rUbSize * rUbNum > eleNum) {
            aUbFactor = aUbFactor - 1;
        } else {
            break;
        }
    }
    return aUbFactor;
}
"""

SHOT_ASSISTANT_2 = json.dumps({
    "label": "real",
    "confidence": 98,
    "explanation": "Contains concrete tiling logic: bitwise math to adjust a parallel factor and a tiling function that computes UB sizes using FloorDiv and CeilAlign inside a loop.",
    "evidence": [
        "L1-L10: FindDichotomizeAddDiffSize uses bitwise arithmetic to compute a parallel factor",
        "L12-L26: DoUbTiling computes UB tile sizes using Ops::Base::FloorDiv/CeilAlign and a while-loop that adjusts aUbFactor based on UB capacity"
    ]
})

def build_messages(file_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SHOT_USER_1},
        {"role": "assistant", "content": SHOT_ASSISTANT_1},
        {"role": "user", "content": SHOT_USER_2},
        {"role": "assistant", "content": SHOT_ASSISTANT_2},
        {
            "role": "user",
            "content": (
                "Now classify the following file.\n"
                "Return the JSON only, with keys: label, confidence, explanation, evidence.\n\n"
                f"File to judge:\n{file_content}\n"
            ),
        },
    ]

# ============================================================
# NEW: prompt to extract *important tiling functions*
# ============================================================

FUNC_PROMPT = """
You are an expert in C/C++ tiling kernels.

Goal:
From the given C/C++ file, identify the **most important functions that contain REAL tiling logic**.
You are NOT classifying the file; you are selecting specific function names.

What counts as REAL tiling computations or strategy:
 - loops (`for`, `while`, `do`) that iterate over dimensions / tiles / blocks
 - arithmetic or bitwise math that computes:
     - tile sizes, UB/GM sizes, strides, block factors, partition sizes
 - ceil/floor/align operations used for tiling, e.g.:
     - CeilAlign, FloorDiv, CeilDiv, AlignUp, AlignDown, etc.
     - Ops::Base::* helpers for tile size / UB computation
 - code that fills or updates tiling data structures:
     - `td_.set_*`, `tilingData.*`, or similar
 - logic that decides how to split work across cores, blocks, or UB/GM

What does NOT count as REAL tiling logic:
 - pure registration macros or glue such as IMPL_OP_OPTILING(...), etc.
 - small wrappers that only call another function with no math or loops
 - prepare functions that only fetch platform info / shapes and store them
   without computing tile sizes (no loops, no non-trivial arithmetic).
 - declarations without bodies, or header-only API declarations.

Your output:
 - Return **ONLY** a JSON object with exactly this shape:
       {"functions": ["name1", "name2", ...]}
 - The list must contain **0 to 4** function names.
 - If there is a call chain where A() simply calls B(), and only B()
   contains the tiling logic, you should select **B**, not A.
 - Choose the 1–4 functions that are **most central** to the tiling
   computation / strategy in this file.
 - If you find no functions with real tiling logic, return:
       {"functions": []}

Naming rules:
 - Each entry in "functions" must be a **bare function name or method name**
   exactly as it appears in the code, e.g.:
     - "DoUbTiling"
     - "BatchNormV3FullReduceTiling::DoUbTiling"
 - Do NOT include parameter lists, return types, templates or extra text.
 - Do NOT include macros, typedefs, or variable names.
 - Do NOT include comments or explanations in the JSON.

File content begins below:
----------------------------------------
{content}
----------------------------------------
Now respond with JSON only, in the form:
{"functions": ["f1", "f2", ...]}
"""

def build_func_messages(content: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": FUNC_PROMPT.replace("{content}", content),
        }
    ]

# ============================================================
# LLM caller (with fixed OpenAI v1 base_url)
# ============================================================

def call_llm_with_retries(
    api_key: str,
    api_host: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    timeout: int,
    retries: int,
) -> Optional[Dict[str, Any]]:
    last_err: Any = None
    for attempt in range(1, retries + 1):
        try:
            return _call_llm_once(api_key, api_host, model, messages, max_tokens, timeout)
        except Exception as e:
            last_err = e
            LOGGER.warning("LLM attempt %d/%d failed: %s", attempt, retries, e)
            time.sleep(min(30, 2**attempt))
    LOGGER.error("LLM failed after %d attempts, last error: %s", retries, last_err)
    return None

def _call_llm_once(
    api_key: str,
    api_host: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    timeout: int,
) -> Optional[Dict[str, Any]]:
    # 1) modern OpenAI client (v1+)
    if OpenAI is not None:
        LOGGER.debug("Using OpenAI v1 client")
        try:
            client = OpenAI(api_key=api_key, base_url=api_host)
        except TypeError:
            # Fallback if this version doesn't support base_url
            client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if not resp.choices:
            return None
        content = resp.choices[0].message.content or ""
        return _extract_json_from_text(content)

    # 2) legacy openai.ChatCompletion
    if openai is not None and hasattr(openai, "ChatCompletion"):
        LOGGER.debug("Using legacy openai.ChatCompletion client")
        if api_host:
            try:
                openai.api_base = api_host
            except Exception:
                pass
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return _extract_json_from_text(content)

    # 3) HTTP fallback using requests to {api_host}/v1/chat/completions
    if requests is not None and api_host:
        LOGGER.debug("Using HTTP fallback to %s", api_host)
        url = api_host.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        obj = resp.json()
        choices = obj.get("choices") or []
        if not choices:
            return None
        first = choices[0]
        msg = first.get("message") or {}
        content = msg.get("content") or first.get("text") or ""
        return _extract_json_from_text(content)

    raise RuntimeError("No LLM client available (OpenAI / openai / requests missing or misconfigured)")

def _extract_json_from_text(t: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed
    except Exception:
        return None

# ============================================================
# Extract function bodies from CPP text
# ============================================================

FUNC_DEF_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_:<>]*\s+[A-Za-z_][A-Za-z0-9_:<>]*\s*\([^)]*\)\s*(const)?\s*)\{",
    re.MULTILINE,
)


def extract_function_body(content: str, fname: str) -> Optional[str]:
    """
    Simple function body extractor that excludes template declarations.
    """
    # 转义函数名
    fname_escaped = re.escape(fname)
    
    # 匹配函数模式：函数名后面跟着参数列表
    pattern = re.compile(rf'\b{fname_escaped}\s*\([^)]*\)\s*(?:const\s*)?\{{', re.MULTILINE)
    
    # 查找所有匹配
    for match in pattern.finditer(content):
        func_start = match.start()
        
        # 向前查找包含函数名的完整行
        line_start = content.rfind('\n', 0, func_start) + 1
        
        # 获取从行开始到匹配开始的内容
        line_content = content[line_start:func_start + len(fname)]
        
        # 检查是否包含template关键字
        if 'template' in line_content:
            # 包含模板声明，需要找到模板行的开始
            template_line_start = content.rfind('\n', 0, line_start - 1) + 1
            template_line = content[template_line_start:line_start]
            
            # 如果模板行确实以template开头，从当前行开始
            if template_line.strip().startswith('template'):
                # 函数签名从当前行（模板行之后的行）开始
                signature_start = line_start
            else:
                signature_start = line_start
        else:
            signature_start = line_start
        
        # 找到左大括号的位置
        brace_pos = content.find('{', match.end() - 1)
        if brace_pos < 0:
            continue
        
        # 使用括号匹配找到函数体结束
        depth = 1
        i = brace_pos + 1
        while i < len(content):
            c = content[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    # 提取函数体，但确保不包含模板行
                    func_content = content[signature_start : i + 1]
                    
                    # 移除任何以template开头的行
                    lines = func_content.split('\n')
                    cleaned_lines = []
                    
                    for line in lines:
                        stripped = line.strip()
                        # 排除以template开头的行
                        if stripped.startswith('template'):
                            continue
                        cleaned_lines.append(line)
                    
                    # 重新组合
                    cleaned_content = '\n'.join(cleaned_lines)
                    return cleaned_content
            i += 1
    
    return None

def add_markers_to_file(path):
    cmd = [
    "python3",
    "/home/l00936201/AscendC-copliot/tools/detector/add_markers.py",
    "-i",
    str(path)  # 确保路径是字符串形式
    ]
    try:
        subprocess.run(cmd, check=True)
        LOGGER.info(f"Add markers: Successfully processed {path}")
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Add markers: Error processing file: {e}")
    except Exception as e:
        LOGGER.error(f"Add markers: Unexpected error: {e}")

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    api_host = args.api_host or os.getenv("API_HOST") or "https://api.openai.com"
    model = args.model or os.getenv("API_MODEL") or "gpt-4"

    if not api_key:
        LOGGER.error("API key not provided. Use --api-key or set API_KEY/OPENAI_API_KEY.")
        return

    # Resolve operator
    try:
        op_base, category, opname = resolve_base(args.base, args.op, args.repo_root)
    except Exception as e:
        LOGGER.error("Failed to resolve operator: %s", e)
        return

    LOGGER.info("Operator base: %s (category=%s opname=%s)", op_base, category, opname)

    op_host = op_base / "op_host"
    if not op_host.exists():
        LOGGER.error("op_host directory not found under %s", op_base)
        return

    scan_dir = op_host / "op_tiling" if (op_host / "op_tiling").exists() else op_host

    files = [
        p
        for p in scan_dir.rglob("*")
        if p.is_file() and p.suffix in {".cpp", ".cc", ".c", ".hpp", ".h"}
    ]

    LOGGER.info("Found %d candidate files to classify", len(files))

    result_root = Path(args.result_root)
    summary: Dict[str, Any] = {
        "op_base": str(op_base),
        "category": category,
        "opname": opname,
        "decisions": [],
    }

    # ------------------------------------------------------------
    # Stage 1: detect tiling (file-level)
    # ------------------------------------------------------------
    for f in files:
        LOGGER.info("LLM judging file: %s", f)
        content = read_text(f, args.max_chars)

        parsed = call_llm_with_retries(
            api_key=api_key,
            api_host=api_host,
            model=model,
            messages=build_messages(content),
            max_tokens=600,
            timeout=args.timeout,
            retries=args.retries,
        )

        if not parsed:
            LOGGER.error("No valid JSON classification from LLM for file %s", f)
            summary["decisions"].append({"file": str(f), "error": "no LLM JSON"})
            continue

        label = str(parsed.get("label", "decl")).strip().lower()
        try:
            conf = int(parsed.get("confidence", 0))
        except Exception:
            conf = 0

        rec: Dict[str, Any] = {
            "file": str(f),
            "label": label,
            "confidence": conf,
            "explanation": parsed.get("explanation", ""),
            "evidence": parsed.get("evidence", []),
        }
        summary["decisions"].append(rec)

        if label != "real":
            continue
        
        timestamp = time.strftime("%Y%m%d%H%M%S")
        LOGGER.info(f"{f} is key tilling file!")
        # copy the file add some markers
        tiling_file_dir = result_root / category / opname / timestamp / "selected_tiling_files"
        tiling_file_dir.mkdir(parents=True, exist_ok=True)
        destination_path = tiling_file_dir / Path(f).name
        shutil.copy2(f, destination_path)
        add_markers_to_file(destination_path)


        # Real tiling file → create output dir
        dest_dir = result_root / category / opname / timestamp / "selected_initial_program" /f.stem
        dest_dir.mkdir(parents=True, exist_ok=True)


        # The ONE output code file: same name as original
        dest_file = dest_dir / f.name

        # ------------------------------------------------------------
        # Stage 2: normalization process(search for key function)
        # ------------------------------------------------------------
        loc = count_lines(f)
        meta: Dict[str, Any] = {"mode": "", "functions": []}

        if loc <= 200:
            # FILE-BASED: keep full file as-is
            LOGGER.info("File %s has %d LOC <= 200 → file-based normalization", f, loc)
            meta["mode"] = "file-based"
            dest_file.write_text(f.read_text(), encoding="utf-8")
            add_markers_to_file(dest_file)
            meta_path = dest_dir / "meta.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rec["exported_to"] = str(dest_file)
            continue

        # FUNCTION-BASED
        LOGGER.info("File %s has %d LOC > 200 → function-based normalization", f, loc)
        meta["mode"] = "function-based"

        # ask LLM for important functions
        fnames_json = call_llm_with_retries(
            api_key=api_key,
            api_host=api_host,
            model=model,
            messages=build_func_messages(content),
            max_tokens=200,
            timeout=args.timeout,
            retries=args.retries,
        )

        fnames: List[str] = []
        if isinstance(fnames_json, dict):
            arr = fnames_json.get("functions", [])
            if isinstance(arr, list):
                fnames = [str(x) for x in arr if isinstance(x, str)]

        # keep at most 4 to be safe (even if model ignores instructions)
        fnames = fnames[:4]
        meta["functions"] = fnames

        extracted_parts: List[str] = []
        for fn in fnames:
            body = extract_function_body(content, fn)
            if body:
                extracted_parts.append(body)
            else:
                LOGGER.warning("Could not extract body for function %r in file %s", fn, f)

        normalized_code = "\n\n\n".join(extracted_parts)
        dest_file.write_text(normalized_code, encoding="utf-8")

        # add markers to normalized tilling function
        add_markers_to_file(dest_file)


        meta_path = dest_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rec["exported_to"] = str(dest_file)

    # ------------------------------------------------------------
    # Write report.json
    # ------------------------------------------------------------
    if args.write_report:
        report_path = result_root / category / opname / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Wrote report to %s", report_path)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
