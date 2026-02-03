from __future__ import annotations

import asyncio
import difflib
import io
import json
import logging
import os
import shutil
import subprocess
import textwrap
import uuid
import zipfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
OPS_ROOT = BASE_DIR / "ops-x"
DETECT_ROOT = BASE_DIR / "tools" / "detector"
DETECT_RESULT_ROOT = DETECT_ROOT / "runs"
UT_OUTPUT_DIR = BASE_DIR / "optimization" / "ut"
OPTIM_ROOT = BASE_DIR / "optimization"
DETECT_SCRIPT = DETECT_ROOT / "main.py"
ADD_MARKER_SCRIPT = DETECT_ROOT / "add_markers.py"
UT_GENERATOR_SCRIPT = BASE_DIR / "tools" / "utGenerator" / "generate_test.py"
OPENEVOLVE_ROOT = BASE_DIR / "optimization"

app = FastAPI(title="AscendC Copilot")

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

logging.basicConfig(
    level=os.environ.get("ASCENDC_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ascendc-copilot")


@dataclass
class OpDetail:
    repo: str
    op: str
    abs_path: Path


class OperatorInput(BaseModel):
    repo: str
    op: str
    shape: List[int] = Field(default_factory=list)
    dtype: str
    platform: str


class UTRequest(BaseModel):
    repo: str
    op: str
    shape: List[int]
    dtype: str


class DetectRequest(BaseModel):
    repo: str
    op: str


class EvolveRequest(BaseModel):
    repo: str
    op: str
    shape: List[int]
    dtype: str
    platform: str


class PackageRequest(BaseModel):
    repo: str
    op: str
    include_generated: bool = True
    output_dir: Optional[str] = None
    best_variant: Optional[str] = None
    best_checkpoint: Optional[str] = None


_sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/repos")
async def list_repos() -> JSONResponse:
    if not OPS_ROOT.exists():
        return JSONResponse({"repos": []})
    repos = sorted([p.name for p in OPS_ROOT.iterdir() if p.is_dir()])
    return JSONResponse({"repos": repos})


def _scan_ops(repo: str) -> List[str]:
    repo_dir = OPS_ROOT / repo
    if not repo_dir.exists():
        return []

    candidates: set[str] = set()
    for root, dirs, files in os.walk(repo_dir):
        root_path = Path(root)
        if root_path.name.startswith("."):
            continue
        # If this directory contains op_host/op_kernel/op_graph, treat it as op root
        if {"op_host", "op_kernel", "op_graph"}.intersection(dirs):
            try:
                rel = root_path.relative_to(repo_dir)
            except ValueError:
                continue
            candidates.add(str(rel))

    return sorted(candidates)


@app.get("/api/operators")
async def list_operators(repo: str = Query(...)) -> JSONResponse:
    return JSONResponse({"repo": repo, "operators": _scan_ops(repo)})


def _resolve_op(repo: str, op: str) -> OpDetail:
    repo_dir = OPS_ROOT / repo
    if not repo_dir.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    abs_path = repo_dir / op
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Operator not found")
    return OpDetail(repo=repo, op=op, abs_path=abs_path)


def _read_first(path: Path, limit: int = 4000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if len(content) > limit:
        return content[:limit].rstrip() + "\n... (truncated)"
    return content


def _collect_files(op_path: Path) -> Dict[str, List[str]]:
    buckets = {
        "tiling_cpp": [],
        "tiling_h": [],
        "op_host": [],
        "op_kernel": [],
        "op_graph": [],
        "tests": [],
        "docs": [],
        "other": [],
    }

    for root, _, files in os.walk(op_path):
        root_path = Path(root)
        rel_root = root_path.relative_to(op_path)
        for name in files:
            rel_path = rel_root / name
            rel_str = str(rel_path)
            lower = name.lower()
            if "tiling" in lower and name.endswith(".cpp"):
                buckets["tiling_cpp"].append(rel_str)
            elif "tiling" in lower and name.endswith(".h"):
                buckets["tiling_h"].append(rel_str)
            elif "op_host" in rel_str.split(os.sep):
                buckets["op_host"].append(rel_str)
            elif "op_kernel" in rel_str.split(os.sep):
                buckets["op_kernel"].append(rel_str)
            elif "op_graph" in rel_str.split(os.sep):
                buckets["op_graph"].append(rel_str)
            elif "tests" in rel_str.split(os.sep):
                buckets["tests"].append(rel_str)
            elif "docs" in rel_str.split(os.sep) or name.lower().endswith(".md"):
                buckets["docs"].append(rel_str)
            else:
                buckets["other"].append(rel_str)

    for key in buckets:
        buckets[key] = sorted(buckets[key])

    return buckets


def _collect_snippets(op_path: Path, files: Iterable[str], max_lines: int = 28) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    for rel_path in files:
        file_path = op_path / rel_path
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = content.splitlines()
        excerpt = "\n".join(lines[:max_lines])
        snippets.append({"path": str(rel_path), "excerpt": excerpt})
    return snippets


def _parse_category_op(op: str, repo: str) -> Tuple[str, str]:
    """Return (category, op_name) using the operator path inside a repo."""
    try:
        rel_parts = (OPS_ROOT / repo / op).relative_to(OPS_ROOT / repo).parts
    except Exception:
        rel_parts = Path(op).parts
    if len(rel_parts) >= 2:
        return rel_parts[0], rel_parts[-1]
    if rel_parts:
        return "", rel_parts[-1]
    return "", op


def _find_tiling_file(op_path: Path) -> Path:
    files = _collect_files(op_path)
    candidates = files["tiling_cpp"] or files["op_host"]
    if not candidates:
        raise FileNotFoundError("No tiling/op_host cpp file found for this operator")
    return op_path / candidates[0]


def _latest_detection_output(category: str, op_name: str) -> Optional[Dict[str, Any]]:
    root = DETECT_RESULT_ROOT / category / op_name
    if not root.exists():
        return None

    timestamp_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not timestamp_dirs:
        return None
    latest = sorted(timestamp_dirs, key=lambda p: p.name)[-1]

    initial_dir = latest / "selected_initial_program"
    initial_file: Optional[Path] = None
    if initial_dir.exists():
        for sub in initial_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.cpp"):
                    initial_file = f
                    break
            if initial_file:
                break
    tiling_dir = latest / "selected_tiling_files"
    tiling_file = None
    if tiling_dir.exists():
        cpp_files = list(tiling_dir.glob("*.cpp"))
        tiling_file = cpp_files[0] if cpp_files else None

    return {
        "timestamp": latest.name,
        "initial_program": initial_file,
        "tiling_file": tiling_file,
        "root": latest,
    }


def _choose_tiling_file(detail: OpDetail, category: str, op_name: str) -> Path:
    detected = _latest_detection_output(category, op_name)
    if detected and detected.get("tiling_file"):
        return Path(detected["tiling_file"])
    return _find_tiling_file(detail.abs_path)


def _ensure_marked_copy(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    try:
        subprocess.run(["python3", str(ADD_MARKER_SCRIPT), "-i", str(dest)], check=True)
    except Exception:
        # markers are best-effort; keep file even if script fails
        pass
    return dest


def _choose_initial_program(detail: OpDetail, category: str, op_name: str) -> Path:
    detected = _latest_detection_output(category, op_name)
    if detected and detected.get("initial_program"):
        return Path(detected["initial_program"])

    fallback = BASE_DIR / "operators" / op_name / "intitial.cpp"
    if fallback.exists():
        return fallback

    tiling_file = _find_tiling_file(detail.abs_path)
    temp_dir = OPTIM_ROOT / "initial_programs" / op_name
    return _ensure_marked_copy(tiling_file, temp_dir)


def _ut_output_path(category: str, op_name: str) -> Path:
    fname = f"test_{op_name.replace('/', '_').replace('-', '_')}.py"
    return UT_OUTPUT_DIR / category / fname if category else UT_OUTPUT_DIR / fname


def _generate_ut_content(op: str, shape: List[int], dtype: str, dim: int = -1) -> Tuple[str, str]:
    """Return (content, mode_used). Falls back to a NumPy stub if generator fails."""
    try:
        from tools.utGenerator import generate_test as ut

        dtype_norm = ut._normalize_dtype(dtype)
        # Try template first for known ops
        try:
            content = ut._render_template(op, shape, dtype_norm, dim)
            return content, "template"
        except NotImplementedError:
            pass

        messages = ut._build_llm_prompt(
            op_name=op,
            shape=shape,
            dtype=dtype_norm,
            dim=dim,
            oneshot_path=ut.Path(UT_GENERATOR_SCRIPT).with_name("test_swi_glu.py"),
        )
        content = ut._llm_request(messages)
        return content, "llm"
    except Exception:
        # Fallback: simple numpy test stub
        stub = textwrap.dedent(
            f"""
            import numpy as np

            def gen_data():
                x = np.random.randn(*{json.dumps(shape)}).astype(np.{dtype} if hasattr(np, "{dtype}") else np.float32)
                return {{"input": x}}

            def run_operator(op_fn, data):
                return op_fn(data["input"])

            if __name__ == "__main__":
                data = gen_data()
                print("sample input shape", data["input"].shape)
            """
        ).strip()
        return stub, "stub"


def _write_ut_file(op: str, shape: List[int], dtype: str, dim: int = -1, category: str = "") -> Tuple[Path, str]:
    (UT_OUTPUT_DIR / category).mkdir(parents=True, exist_ok=True) if category else UT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    op_name = op.split("/")[-1] if op else op
    out_path = _ut_output_path(category, op_name)
    content, mode_used = _generate_ut_content(op, shape, dtype, dim)
    out_path.write_text(content, encoding="utf-8")
    return out_path, mode_used


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _rel_path_for_ui(path_str: str, repo: str) -> str:
    if not path_str:
        return path_str
    p = Path(path_str)
    if not p.is_absolute():
        p = (BASE_DIR / path_str).resolve()
    else:
        p = p.resolve()
    repo_root = (OPS_ROOT / repo).resolve()
    for root in (repo_root, BASE_DIR.resolve()):
        try:
            return str(p.relative_to(root))
        except Exception:
            continue
    return str(p)


def _build_detect_payload(
    repo: str,
    op: str,
    category: str,
    op_name: str,
    result_root: Path,
    duration: float,
    returncode: int,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> Dict[str, Any]:
    report_path = result_root / category / op_name / "report.json"
    report = _load_json(report_path) or {}
    decisions = report.get("decisions", []) if isinstance(report, dict) else []

    findings: List[Dict[str, Any]] = []
    workspace: Dict[str, Any] = {}

    for dec in decisions:
        if dec.get("label") != "real":
            continue
        path = dec.get("file", "")
        exported_to = dec.get("exported_to")
        meta_path = Path(exported_to).parent / "meta.json" if exported_to else None
        meta = _load_json(meta_path) if meta_path and meta_path.exists() else {}
        functions = meta.get("functions", []) if isinstance(meta, dict) else []
        findings.append(
            {
                "path": _rel_path_for_ui(path, repo),
                "functions": functions,
                "mode": meta.get("mode") if isinstance(meta, dict) else None,
                "summary": dec.get("explanation", ""),
                "evidence": dec.get("evidence", []),
            }
        )

        if exported_to and "initial_program" not in workspace:
            workspace["initial_program"] = {
                "path": _rel_path_for_ui(exported_to, repo),
                "functions": functions,
                "mode": meta.get("mode") if isinstance(meta, dict) else None,
            }

    detected_latest = _latest_detection_output(category, op_name)
    if detected_latest and detected_latest.get("tiling_file"):
        workspace["marked_original"] = {
            "path": _rel_path_for_ui(str(detected_latest["tiling_file"]), repo),
        }
    if detected_latest and detected_latest.get("initial_program") and "initial_program" not in workspace:
        workspace["initial_program"] = {
            "path": _rel_path_for_ui(str(detected_latest["initial_program"]), repo),
            "functions": [],
            "mode": None,
        }

    ut_path = _ut_output_path(category, op_name)
    if ut_path.exists():
        workspace["ut_script"] = {
            "path": _rel_path_for_ui(str(ut_path), repo),
        }

    note = f"detect-LLM 完成: 找到 {len(findings)} 个可优化文件" if returncode == 0 else ""
    if returncode != 0:
        note = f"detect-LLM 运行异常 (exit {returncode})"

    return {
        "repo": repo,
        "op": op,
        "category": category,
        "op_name": op_name,
        "findings": findings,
        "note": note,
        "result_root": str(result_root),
        "workspace": workspace,
        "report_path": str(report_path),
        "duration_sec": round(duration, 3),
        "stdout": stdout_tail,
        "stderr": stderr_tail,
    }


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _find_best_program_file(ckpt_dir: Path) -> Optional[Path]:
    for name in ("best_program.cpp", "best_program.cc", "best_program.c", "best_program.py"):
        candidate = ckpt_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = sorted(ckpt_dir.glob("best_program.*"))
    for match in matches:
        if match.is_file():
            return match
    return None


def _scan_checkpoints(output_dir: Path, seen: set[int]) -> List[Dict[str, Any]]:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    events: List[Dict[str, Any]] = []
    for p in sorted(ckpt_dir.glob("checkpoint_*"), key=lambda x: x.name):
        try:
            idx = int(p.name.split("_")[1])
        except Exception:
            continue
        if idx in seen:
            continue
        meta = _load_json(p / "metadata.json") or {}
        best_info = _load_json(p / "best_program_info.json") or {}
        best_metrics = best_info.get("metrics") or {}
        best_score = best_metrics.get("combined_score")
        best_avg_us = _avg_us_from_metrics(best_metrics)
        iteration_index = meta.get("last_iteration", idx)
        latest_info = _latest_program_info(p, iteration_index) or {}
        latest_metrics = latest_info.get("metrics") or {}
        latest_avg_us = _avg_us_from_metrics(latest_metrics)
        latest_score = latest_metrics.get("combined_score")
        avg_us = latest_avg_us if latest_avg_us is not None else best_avg_us
        note_score = latest_score if latest_score is not None else best_score
        events.append(
            {
                "index": iteration_index,
                "variant": f"checkpoint_{idx}",
                "avg_us": round(avg_us, 4) if avg_us is not None else None,
                "best_avg_us": best_avg_us,
                "latest_score": latest_score,
                "latest_program_path": latest_info.get("path"),
                "notes": f"combined_score={note_score}" if note_score is not None else "no score yet",
                "path": str(p),
            }
        )
        seen.add(idx)
    return events


def _drain_log_lines(log_file: Path, pos: int, buffer: str, max_bytes: int = 40000) -> Tuple[int, str, List[str]]:
    if not log_file.exists():
        return pos, buffer, []
    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(pos)
            data = f.read(max_bytes)
            if not data:
                return pos, buffer, []
            pos += len(data)
    except OSError:
        return pos, buffer, []

    text = buffer + data
    lines = text.splitlines()
    if text and not text.endswith(("\n", "\r")):
        buffer = lines.pop() if lines else text
    else:
        buffer = ""
    return pos, buffer, lines


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_*"), key=lambda p: p.name)
    return ckpts[-1] if ckpts else None


def _has_any_checkpoint(output_dir: Path) -> bool:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return False
    return any(ckpt_dir.glob("checkpoint_*"))


def _avg_us_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not metrics:
        return None
    avg_us = metrics.get("avg_us")
    if avg_us is not None:
        try:
            return round(float(avg_us), 4)
        except Exception:
            return None
    score = metrics.get("combined_score")
    if not score:
        return None
    try:
        avg_us = 10.0 / float(score)
        return round(avg_us, 4)
    except Exception:
        return None


def _best_avg_us(ckpt_dir: Path) -> Optional[float]:
    best_info = _load_json(ckpt_dir / "best_program_info.json") or {}
    metrics = best_info.get("metrics") or {}
    return _avg_us_from_metrics(metrics)


def _latest_program_info(ckpt_dir: Path, iteration: Optional[int] = None) -> Optional[Dict[str, Any]]:
    programs_dir = ckpt_dir / "programs"
    if not programs_dir.exists():
        return None
    latest: Optional[Tuple[Tuple[int, float], Path, Dict[str, Any]]] = None
    candidates: List[Tuple[Tuple[int, float], Path, Dict[str, Any]]] = []
    for program_file in programs_dir.glob("*.json"):
        data = _load_json(program_file)
        if not isinstance(data, dict):
            continue
        if "code" not in data:
            continue
        prog_iteration = data.get("iteration_found")
        try:
            iteration_key = int(prog_iteration) if prog_iteration is not None else -1
        except Exception:
            iteration_key = -1
        timestamp = data.get("timestamp") or data.get("saved_at")
        try:
            timestamp_key = float(timestamp) if timestamp is not None else 0.0
        except Exception:
            try:
                timestamp_key = program_file.stat().st_mtime
            except Exception:
                timestamp_key = 0.0
        key = (iteration_key, timestamp_key)
        candidates.append((key, program_file, data))

    if not candidates:
        return None

    if iteration is not None:
        iter_candidates = [item for item in candidates if item[0][0] == iteration]
    else:
        iter_candidates = []

    pool = iter_candidates if iter_candidates else candidates
    latest = max(pool, key=lambda item: item[0])
    _, program_file, data = latest
    return {
        "path": str(program_file),
        "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
        "iteration": data.get("iteration_found"),
        "timestamp": data.get("timestamp") or data.get("saved_at"),
    }


@app.get("/api/operator/detail")
async def operator_detail(repo: str = Query(...), op: str = Query(...)) -> JSONResponse:
    detail = _resolve_op(repo, op)
    files = _collect_files(detail.abs_path)

    readme_path = detail.abs_path / "README.md"
    readme = _read_first(readme_path) if readme_path.exists() else ""

    tiling_snippets = _collect_snippets(detail.abs_path, files["tiling_cpp"])

    payload = {
        "repo": detail.repo,
        "op": detail.op,
        "path": str(detail.abs_path),
        "readme": readme,
        "files": files,
        "tiling_snippets": tiling_snippets,
    }
    return JSONResponse(payload)


@app.post("/api/ut-llm")
async def generate_ut(request: UTRequest) -> JSONResponse:
    _resolve_op(request.repo, request.op)
    category, op_name = _parse_category_op(request.op, request.repo)

    logger.info(
        "ut-LLM request repo=%s op=%s shape=%s dtype=%s category=%s",
        request.repo,
        request.op,
        request.shape,
        request.dtype,
        category,
    )

    try:
        out_path, mode_used = await asyncio.to_thread(
            _write_ut_file, request.op, request.shape, request.dtype, -1, category
        )
        script = out_path.read_text(encoding="utf-8")
        note = f"ut-LLM generated via {mode_used}. Saved to {out_path}"
        logger.info("ut-LLM generated file=%s mode=%s", out_path, mode_used)
    except Exception as exc:  # pragma: no cover
        script = ""
        note = f"生成失败: {exc}"
        logger.exception("ut-LLM generation failed repo=%s op=%s", request.repo, request.op)

    return JSONResponse(
        {
            "repo": request.repo,
            "op": request.op,
            "category": category,
            "op_name": op_name,
            "script": script,
            "note": note,
            "path_hint": str(_ut_output_path(category, op_name)),
        }
    )


@app.post("/api/detect-llm")
async def detect_llm(request: DetectRequest) -> JSONResponse:
    detail = _resolve_op(request.repo, request.op)
    category, op_name = _parse_category_op(request.op, request.repo)

    # detector script will append category/op/timestamp under this root
    result_root = DETECT_RESULT_ROOT
    result_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(DETECT_SCRIPT),
        "--base",
        str(detail.abs_path),
        "--repo-root",
        str(OPS_ROOT / request.repo),
        "--result-root",
        str(result_root),
        "--write-report",
    ]

    start_ts = datetime.utcnow()
    logger.info(
        "detect-LLM start repo=%s op=%s category=%s cmd=\"%s\"",
        request.repo,
        request.op,
        category,
        " ".join(cmd),
    )

    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=900,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("detect-LLM timeout repo=%s op=%s after 900s", request.repo, request.op)
        raise HTTPException(status_code=504, detail=f"detect-LLM timeout: {exc}")

    duration = (datetime.utcnow() - start_ts).total_seconds()
    stdout_tail = (completed.stdout or "")[-4000:]
    stderr_tail = (completed.stderr or "")[-4000:]

    payload = _build_detect_payload(
        request.repo,
        request.op,
        category,
        op_name,
        result_root,
        duration,
        completed.returncode,
        stdout_tail,
        stderr_tail,
    )

    logger.info(
        "detect-LLM done repo=%s op=%s exit=%s findings=%d duration=%.1fs report=%s",
        request.repo,
        request.op,
        completed.returncode,
        len(payload.get("findings", [])),
        duration,
        payload.get("report_path"),
    )
    if stderr_tail:
        logger.warning("detect-LLM stderr (tail):\n%s", stderr_tail)
    elif stdout_tail:
        logger.info("detect-LLM stdout (tail):\n%s", stdout_tail)

    return JSONResponse(payload)


@app.get("/api/detect-llm/stream")
async def detect_llm_stream(repo: str = Query(...), op: str = Query(...)) -> StreamingResponse:
    detail = _resolve_op(repo, op)
    category, op_name = _parse_category_op(op, repo)

    result_root = DETECT_RESULT_ROOT
    result_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(DETECT_SCRIPT),
        "--base",
        str(detail.abs_path),
        "--repo-root",
        str(OPS_ROOT / repo),
        "--result-root",
        str(result_root),
        "--write-report",
    ]

    start_ts = datetime.utcnow()
    logger.info(
        "detect-LLM stream start repo=%s op=%s category=%s cmd=\"%s\"",
        repo,
        op,
        category,
        " ".join(cmd),
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    queue: asyncio.Queue = asyncio.Queue()
    stdout_tail = ""
    stderr_tail = ""

    def _append_tail(current: str, text: str, limit: int = 4000) -> str:
        combined = f"{current}{text}\n"
        if len(combined) > limit:
            return combined[-limit:]
        return combined

    async def _read_stream(stream: Optional[asyncio.StreamReader], name: str) -> None:
        nonlocal stdout_tail, stderr_tail
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="ignore").rstrip("\n")
            if name == "stdout":
                stdout_tail = _append_tail(stdout_tail, text)
            else:
                stderr_tail = _append_tail(stderr_tail, text)
            await queue.put({"type": "log", "stream": name, "message": text})

    async def event_stream() -> Iterable[str]:
        read_tasks = [
            asyncio.create_task(_read_stream(process.stdout, "stdout")),
            asyncio.create_task(_read_stream(process.stderr, "stderr")),
        ]
        process_task = asyncio.create_task(process.wait())
        timeout_task = asyncio.create_task(asyncio.sleep(900))
        try:
            while True:
                if timeout_task.done() and not process_task.done():
                    process.kill()
                    await process.wait()
                    yield f"data: {json.dumps({'type': 'log', 'stream': 'stderr', 'message': 'detect-LLM 超时，已中止。'}, ensure_ascii=False)}\n\n"
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                    continue
                except asyncio.TimeoutError:
                    pass
                if process_task.done() and queue.empty():
                    break

            await asyncio.gather(*read_tasks, return_exceptions=True)
            while not queue.empty():
                item = queue.get_nowait()
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

            duration = (datetime.utcnow() - start_ts).total_seconds()
            returncode = process.returncode if process.returncode is not None else 1
            payload = _build_detect_payload(
                repo,
                op,
                category,
                op_name,
                result_root,
                duration,
                returncode,
                stdout_tail,
                stderr_tail,
            )
            logger.info(
                "detect-LLM stream done repo=%s op=%s exit=%s findings=%d duration=%.1fs",
                repo,
                op,
                returncode,
                len(payload.get("findings", [])),
                duration,
            )
            yield f"data: {json.dumps({'type': 'complete', 'data': payload}, ensure_ascii=False)}\n\n"
        finally:
            for task in read_tasks:
                task.cancel()
            process_task.cancel()
            timeout_task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/workspace/file")
async def workspace_file(path: str = Query(...), repo: Optional[str] = Query(None)) -> JSONResponse:
    raw_path = Path(path)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (BASE_DIR / raw_path).resolve()
        if repo:
            candidate = (OPS_ROOT / repo / raw_path).resolve()
            if candidate.exists():
                resolved = candidate
    allowed_roots = [
        BASE_DIR.resolve(),
        OPS_ROOT.resolve(),
        DETECT_RESULT_ROOT.resolve(),
        OPTIM_ROOT.resolve(),
        UT_OUTPUT_DIR.resolve(),
        OPENEVOLVE_ROOT.resolve(),
    ]
    if not any(_is_under_root(resolved, root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Path not allowed")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    content = resolved.read_text(encoding="utf-8", errors="ignore")
    return JSONResponse({"path": str(resolved), "content": content})


@app.get("/api/checkpoint/diff")
async def checkpoint_diff(checkpoint_a: str = Query(...), checkpoint_b: str = Query(...)) -> JSONResponse:
    def _resolve_path(raw: str) -> Path:
        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
        else:
            resolved = (BASE_DIR / raw_path).resolve()
        allowed_roots = [
            OPTIM_ROOT.resolve(),
            OPS_ROOT.resolve(),
            DETECT_RESULT_ROOT.resolve(),
            BASE_DIR.resolve(),
        ]
        if not any(_is_under_root(resolved, root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="Path not allowed")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        return resolved

    path_a = _resolve_path(checkpoint_a)
    path_b = _resolve_path(checkpoint_b)

    file_a = _find_best_program_file(path_a) if path_a.is_dir() else path_a
    file_b = _find_best_program_file(path_b) if path_b.is_dir() else path_b
    if not file_a or not file_b:
        raise HTTPException(status_code=404, detail="program file not found")
    if not file_a.is_file() or not file_b.is_file():
        raise HTTPException(status_code=404, detail="program file not found")

    text_a = file_a.read_text(encoding="utf-8", errors="ignore").splitlines()
    text_b = file_b.read_text(encoding="utf-8", errors="ignore").splitlines()

    matcher = difflib.SequenceMatcher(a=text_a, b=text_b)
    left_lines: List[Dict[str, Any]] = []
    right_lines: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for a_line, b_line in zip(text_a[i1:i2], text_b[j1:j2]):
                left_lines.append({"text": a_line, "type": "equal"})
                right_lines.append({"text": b_line, "type": "equal"})
        elif tag == "delete":
            for a_line in text_a[i1:i2]:
                left_lines.append({"text": a_line, "type": "remove"})
                right_lines.append({"text": "", "type": "empty"})
        elif tag == "insert":
            for b_line in text_b[j1:j2]:
                left_lines.append({"text": "", "type": "empty"})
                right_lines.append({"text": b_line, "type": "add"})
        elif tag == "replace":
            span = max(i2 - i1, j2 - j1)
            for offset in range(span):
                a_line = text_a[i1 + offset] if (i1 + offset) < i2 else ""
                b_line = text_b[j1 + offset] if (j1 + offset) < j2 else ""
                left_lines.append({"text": a_line, "type": "remove" if a_line else "empty"})
                right_lines.append({"text": b_line, "type": "add" if b_line else "empty"})

    return JSONResponse(
        {
            "file_a": str(file_a),
            "file_b": str(file_b),
            "left": left_lines,
            "right": right_lines,
        }
    )


@app.post("/api/openevolve/start")
async def openevolve_start(request: EvolveRequest) -> JSONResponse:
    detail = _resolve_op(request.repo, request.op)
    category, op_name = _parse_category_op(request.op, request.repo)

    logger.info(
        "OpenEvolve start repo=%s op=%s shape=%s dtype=%s platform=%s category=%s",
        request.repo,
        request.op,
        request.shape,
        request.dtype,
        request.platform,
        category,
    )

    # Prepare UT file
    ut_path, _ = await asyncio.to_thread(
        _write_ut_file, request.op, request.shape, request.dtype, -1, category
    )

    # Resolve target tiling file and initial program
    tiling_file = _choose_tiling_file(detail, category, op_name)
    initial_program = _choose_initial_program(detail, category, op_name)

    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    cmd = [
        "python3",
        str(OPENEVOLVE_ROOT / "start_optimization.py"),
        str(initial_program),
        "--operator-name",
        op_name,
        "--category",
        category,
        "--file-name",
        str(tiling_file),
        "--test-file",
        str(ut_path),
        "--run-id",
        run_id,
        "--iterations",
        "30",
    ]

    test_name = ut_path.stem
    output_dir = OPENEVOLVE_ROOT / category / op_name / f"{test_name}_{run_id}"

    log_dir = OPTIM_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"openevolve_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    logger.info(
        "OpenEvolve launching run_id=%s output_dir=%s cmd=\"%s\" log_file=%s",
        run_id,
        output_dir,
        " ".join(cmd),
        log_file,
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    session_id = uuid.uuid4().hex
    _sessions[session_id] = {
        "request": request.dict(),
        "process": process,
        "log_file": str(log_file),
        "output_dir": output_dir,
        "run_id": run_id,
        "seen": set(),
        "started_at": datetime.utcnow().isoformat(),
        "log_pos": 0,
        "log_buf": "",
        "initial_emitted": False,
    }

    async def _pipe_to_file():
        if process.stdout is None:
            return
        with log_file.open("w", encoding="utf-8") as f:
            async for chunk in process.stdout:
                text = chunk.decode(errors="ignore")
                f.write(text)
                f.flush()
                for line in text.splitlines():
                    if line:
                        logger.info("[OpenEvolve:%s] %s", run_id, line)
        returncode = process.returncode
        logger.info("OpenEvolve process stdout piping ended log_file=%s exit=%s", log_file, returncode)

    asyncio.create_task(_pipe_to_file())

    return JSONResponse(
        {
            "session_id": session_id,
            "log": str(log_file),
            "output_dir": str(output_dir),
            "run_id": run_id,
        }
    )


@app.get("/api/openevolve/stream")
async def openevolve_stream(session_id: str = Query(...)) -> StreamingResponse:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    output_dir: Path = Path(session["output_dir"])
    seen: set[int] = session.get("seen", set())
    process: asyncio.subprocess.Process = session.get("process")
    log_file = session.get("log_file")
    run_id = session.get("run_id")
    log_pos = session.get("log_pos", 0)
    log_buf = session.get("log_buf", "")
    log_path = Path(log_file) if log_file else None
    initial_emitted = session.get("initial_emitted", False)

    async def event_stream() -> Iterable[str]:
        nonlocal log_pos, log_buf, initial_emitted
        while True:
            if log_path:
                log_pos, log_buf, lines = _drain_log_lines(log_path, log_pos, log_buf)
                session["log_pos"] = log_pos
                session["log_buf"] = log_buf
                for line in lines:
                    if (not initial_emitted) and ("END eval" in line) and (not _has_any_checkpoint(output_dir)):
                        initial_emitted = True
                        session["initial_emitted"] = True
                        yield f"data: {json.dumps({'type': 'initial_eval', 'data': {'label': 'Initial evaluation (no replace)'}}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'log', 'data': {'message': line}}, ensure_ascii=False)}\n\n"

            # emit new checkpoints if any
            for evt in _scan_checkpoints(output_dir, seen):
                yield f"data: {json.dumps({'type': 'iteration', 'data': evt})}\n\n"

            # check process status
            done = process is not None and process.returncode is not None
            if process is not None and process.returncode is None:
                await asyncio.sleep(1.0)
                await process.wait() if process.stdout is None else None
                # after wait, loop again to capture final checkpoints
                continue

            if done or (not process):
                exit_code = process.returncode if process is not None else None
                if log_path:
                    log_pos, log_buf, lines = _drain_log_lines(log_path, log_pos, log_buf)
                    session["log_pos"] = log_pos
                    session["log_buf"] = log_buf
                    for line in lines:
                        if (not initial_emitted) and ("END eval" in line) and (not _has_any_checkpoint(output_dir)):
                            initial_emitted = True
                            session["initial_emitted"] = True
                            yield f"data: {json.dumps({'type': 'initial_eval', 'data': {'label': 'Initial evaluation (no replace)'}}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'data': {'message': line}}, ensure_ascii=False)}\n\n"
                for evt in _scan_checkpoints(output_dir, seen):
                    yield f"data: {json.dumps({'type': 'iteration', 'data': evt})}\n\n"
                best_ckpt = _latest_checkpoint(output_dir)
                best_avg_us = _best_avg_us(best_ckpt) if best_ckpt else None
                summary = {
                    "best_variant": best_ckpt.name if best_ckpt else None,
                    "best_avg_us": best_avg_us,
                    "best_latency_ms": round(best_avg_us / 1000.0, 4) if best_avg_us is not None else None,
                    "total_iterations": len(seen),
                    "status": "success" if exit_code in (0, None) else "failed",
                    "exit_code": exit_code,
                    "log_file": log_file,
                    "run_id": run_id,
                    "output_dir": str(output_dir),
                }
                logger.info(
                    "OpenEvolve stream complete session=%s best_variant=%s total_iterations=%d",
                    session_id,
                    summary["best_variant"],
                    summary["total_iterations"],
                )
                yield f"data: {json.dumps({'type': 'complete', 'data': summary})}\n\n"
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/package")
async def download_package(request: PackageRequest) -> StreamingResponse:
    detail = _resolve_op(request.repo, request.op)

    logger.info(
        "package request repo=%s op=%s include_generated=%s",
        request.repo,
        request.op,
        request.include_generated,
    )

    def _resolve_checkpoint_dir() -> Optional[Path]:
        candidates: List[Path] = []
        if request.best_checkpoint:
            candidates.append(Path(request.best_checkpoint))
        if request.output_dir and request.best_variant:
            candidates.append(Path(request.output_dir) / "checkpoints" / request.best_variant)
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except Exception:
                continue
            if not _is_under_root(resolved, OPTIM_ROOT):
                continue
            if resolved.exists() and resolved.is_dir():
                return resolved
        return None

    best_ckpt_dir = _resolve_checkpoint_dir()
    best_program_path: Optional[Path] = None
    best_info_path: Optional[Path] = None
    if best_ckpt_dir:
        for candidate in best_ckpt_dir.glob("best_program.*"):
            if candidate.is_file():
                best_program_path = candidate
                break
        info_candidate = best_ckpt_dir / "best_program_info.json"
        if info_candidate.exists():
            best_info_path = info_candidate

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(detail.abs_path):
            for name in files:
                file_path = Path(root) / name
                rel = file_path.relative_to(detail.abs_path)
                try:
                    if file_path.stat().st_size > 2 * 1024 * 1024:
                        continue
                except OSError:
                    continue
                zipf.write(file_path, arcname=str(Path(detail.op) / rel))

        if request.include_generated:
            gen_dir = Path(detail.op) / "generated"
            ut_script = textwrap.dedent(
                """
                # Placeholder for gen_data.py
                # Use the ut-LLM endpoint to generate full script.
                """
            ).strip()
            zipf.writestr(str(gen_dir / "gen_data.py"), ut_script)
            detect_report = {
                "note": "Placeholder detect-LLM report.",
                "files": [],
            }
            zipf.writestr(str(gen_dir / "detect_report.json"), json.dumps(detect_report, indent=2))
            if best_program_path:
                zipf.write(best_program_path, arcname=str(gen_dir / best_program_path.name))
            if best_info_path:
                zipf.write(best_info_path, arcname=str(gen_dir / best_info_path.name))
            if best_ckpt_dir:
                meta_payload = {
                    "checkpoint_dir": str(best_ckpt_dir),
                    "best_variant": request.best_variant or best_ckpt_dir.name,
                    "output_dir": request.output_dir,
                }
                zipf.writestr(str(gen_dir / "best_checkpoint.json"), json.dumps(meta_payload, indent=2))

    zip_buffer.seek(0)

    suffix = "package"
    if best_ckpt_dir:
        suffix = f"best_{best_ckpt_dir.name}"
    filename = f"{detail.repo}_{detail.op.replace('/', '_')}_{suffix}.zip"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}

    logger.info("package built filename=%s size=%d bytes", filename, len(zip_buffer.getbuffer()))

    return StreamingResponse(zip_buffer, headers=headers, media_type="application/zip")
