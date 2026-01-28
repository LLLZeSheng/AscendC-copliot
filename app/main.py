from __future__ import annotations

import asyncio
import io
import json
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
        score = (best_info.get("metrics") or {}).get("combined_score")
        latency_ms = throughput = None
        if score and score > 0:
            duration_sec = 10.0 / float(score)
            latency_ms = duration_sec * 1000
            throughput = 1.0 / duration_sec
        events.append(
            {
                "index": meta.get("last_iteration", idx),
                "variant": f"checkpoint_{idx}",
                "latency_ms": round(latency_ms, 4) if latency_ms is not None else None,
                "throughput": round(throughput, 4) if throughput is not None else None,
                "notes": f"combined_score={score}" if score is not None else "no score yet",
                "path": str(p),
            }
        )
        seen.add(idx)
    return events


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_*"), key=lambda p: p.name)
    return ckpts[-1] if ckpts else None


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

    try:
        out_path, mode_used = await asyncio.to_thread(
            _write_ut_file, request.op, request.shape, request.dtype, -1, category
        )
        script = out_path.read_text(encoding="utf-8")
        note = f"ut-LLM generated via {mode_used}. Saved to {out_path}"
    except Exception as exc:  # pragma: no cover
        script = ""
        note = f"生成失败: {exc}"

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

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
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

    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=900,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"detect-LLM timeout: {exc}")

    report_path = result_root / category / op_name / "report.json"
    report = _load_json(report_path) or {}
    decisions = report.get("decisions", []) if isinstance(report, dict) else []

    findings: List[Dict[str, Any]] = []
    for dec in decisions:
        if dec.get("label") != "real":
            continue
        path = dec.get("file", "")
        excerpt = _read_first(Path(path)) if path else ""
        findings.append(
            {
                "path": path,
                "summary": dec.get("explanation", ""),
                "evidence": dec.get("evidence", []),
                "excerpt": excerpt,
            }
        )

    note = "detect-LLM 完成"
    if completed.returncode != 0:
        note = f"detect-LLM 运行异常 (exit {completed.returncode})"

    return JSONResponse(
        {
            "repo": request.repo,
            "op": request.op,
            "category": category,
            "op_name": op_name,
            "findings": findings,
            "note": note,
            "result_root": str(result_root),
            "stdout": completed.stdout[-4000:] if completed.stdout else "",
            "stderr": completed.stderr[-4000:] if completed.stderr else "",
        }
    )


@app.post("/api/openevolve/start")
async def openevolve_start(request: EvolveRequest) -> JSONResponse:
    detail = _resolve_op(request.repo, request.op)
    category, op_name = _parse_category_op(request.op, request.repo)

    # Prepare UT file
    ut_path, _ = await asyncio.to_thread(
        _write_ut_file, request.op, request.shape, request.dtype, -1, category
    )

    # Resolve target tiling file and initial program
    tiling_file = _find_tiling_file(detail.abs_path)
    initial_program = _choose_initial_program(detail, category, op_name)

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
        "--iterations",
        "30",
    ]

    log_dir = OPTIM_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"openevolve_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    session_id = uuid.uuid4().hex
    test_name = ut_path.stem
    output_dir = OPENEVOLVE_ROOT / category / op_name / test_name

    _sessions[session_id] = {
        "request": request.dict(),
        "process": process,
        "log_file": str(log_file),
        "output_dir": output_dir,
        "seen": set(),
        "started_at": datetime.utcnow().isoformat(),
    }

    async def _pipe_to_file():
        if process.stdout is None:
            return
        with log_file.open("w", encoding="utf-8") as f:
            async for chunk in process.stdout:
                f.write(chunk.decode(errors="ignore"))
                f.flush()

    asyncio.create_task(_pipe_to_file())

    return JSONResponse({"session_id": session_id, "log": str(log_file), "output_dir": str(output_dir)})


@app.get("/api/openevolve/stream")
async def openevolve_stream(session_id: str = Query(...)) -> StreamingResponse:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    output_dir: Path = Path(session["output_dir"])
    seen: set[int] = session.get("seen", set())
    process: asyncio.subprocess.Process = session.get("process")

    async def event_stream() -> Iterable[str]:
        while True:
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
                best_ckpt = _latest_checkpoint(output_dir)
                summary = {
                    "best_variant": best_ckpt.name if best_ckpt else None,
                    "best_latency_ms": None,
                    "total_iterations": len(seen),
                }
                yield f"data: {json.dumps({'type': 'complete', 'data': summary})}\n\n"
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/package")
async def download_package(request: PackageRequest) -> StreamingResponse:
    detail = _resolve_op(request.repo, request.op)

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

    zip_buffer.seek(0)

    filename = f"{detail.repo}_{detail.op.replace('/', '_')}_package.zip"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}

    return StreamingResponse(zip_buffer, headers=headers, media_type="application/zip")
