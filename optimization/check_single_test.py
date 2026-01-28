#!/usr/bin/env python3
"""
eval_operator.py

eval(category, op_name, kernel_files, host_files, python_code_path, op_type_map) -> float

What it does in tmp copy:
1) Copy / -> /
2) Replace only specified op_kernel/op_host files for category/op_name
3) Delete only build* DIRECTORIES at tmp root (keeps build.sh)
4) Build:
     bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=<op_name>
5) Install generated .run in build_out
6) Create tmp_root/test_duration and copy python_code_path into it
7) Run python once (asserts must pass)
8) Profile:
     /usr/local/Ascend/latest/toolkit/tools/profiler/bin/msprof python3 <python_file>
9) Parse msprof output to find "Data is saved in <DIR>"
10) Parse <DIR>/mindstudio_profiler_output/op_statistic_*.csv
    and return Avg Time(us) for OP Type == op_type_map.get(op_name, op_name)
"""

import csv
import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional, Union


# ----------------- logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eval_operator")


BASE_OPS_NN_ROOT = Path("/home/l00936201/AscendC-copliot/ops-x/ops-nn")
TMP_PARENT = Path("/home/l00936201/AscendC-copliot/tmp")

MSPROF_BIN = Path("/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof")

class EvalError(Exception):
    pass

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to CamelCase, e.g., 'embedding_bag' -> 'EmbeddingBag'"""
    return "".join(word.capitalize() for word in snake_str.split("_") if word)



def _find_operator_dir(base_root: Path, category: str, op_name: str) -> Path:
    op_dir = base_root / category / op_name
    logger.info(f"Locating operator dir: {op_dir}")
    if not op_dir.exists() or not op_dir.is_dir():
        raise FileNotFoundError(f"Operator directory not found: {op_dir}")
    return op_dir


def _find_host_file_recursive(root: Path, filename: str) -> Optional[Path]:
    for path in root.rglob(filename):
        if path.is_file() and path.name == filename:
            return path
    return None


def _verify_files_exist(
    op_dir: Path,
    kernel_files: Dict[str, str],
    host_files: Dict[str, str],
) -> None:
    kernel_dir = op_dir / "op_kernel"
    host_dir = op_dir / "op_host"

    logger.info("Verifying requested replacement files exist in BASE tree...")
    if kernel_files:
        logger.info(f"Kernel dir: {kernel_dir}")
    if host_files:
        logger.info(f"Host dir:   {host_dir}")

    if kernel_files and not kernel_dir.exists():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")
    if host_files and not host_dir.exists():
        raise FileNotFoundError(f"Host directory not found: {host_dir}")

    for fname in kernel_files.keys():
        target = kernel_dir / fname
        logger.info(f"Verify kernel file exists: {target}")
        if not target.exists():
            raise FileNotFoundError(f"Kernel file '{fname}' not found in {kernel_dir}")

    for fname in host_files.keys():
        found = _find_host_file_recursive(host_dir, fname)
        logger.info(f"Verify host file exists (recursive): {fname} -> {found}")
        if not found:
            raise FileNotFoundError(f"Host file '{fname}' not found under {host_dir} (recursive)")


def _copy_base_to_tmp(base_root: Path) -> Path:
    if not base_root.exists() or not base_root.is_dir():
        raise FileNotFoundError(f"Base ops-nn root does not exist: {base_root}")

    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    unique_name = f"ops_nn_{uuid.uuid4().hex}"
    tmp_root = TMP_PARENT / unique_name
    tmp_root.mkdir(parents=True, exist_ok=True)

    dest = tmp_root / "ops-nn-master"
    logger.info(f"Copy base tree:\n  FROM: {base_root}\n  TO:   {dest}")
    shutil.copytree(str(base_root), str(dest), symlinks=True)
    return dest


def _replace_files_in_tmp(
    tmp_ops_root: Path,
    category: str,
    op_name: str,
    kernel_files: Dict[str, str],
    host_files: Dict[str, str],
) -> None:
    tmp_op_dir = tmp_ops_root / category / op_name
    logger.info(f"Replacing files in TMP operator dir: {tmp_op_dir}")
    if not tmp_op_dir.exists():
        raise FileNotFoundError(f"Operator directory not found in tmp copy: {tmp_op_dir}")

    kernel_dir = tmp_op_dir / "op_kernel"
    host_dir = tmp_op_dir / "op_host"

    if kernel_files:
        if not kernel_dir.exists():
            raise FileNotFoundError(f"Kernel directory not found in tmp copy: {kernel_dir}")
        for fname, content in kernel_files.items():
            target = kernel_dir / fname
            logger.info(f"Replace kernel file: {target} (len={len(content)} chars)")
            if not target.exists():
                raise FileNotFoundError(f"Kernel file '{fname}' not found in tmp copy: {target}")
            target.write_text(content, encoding="utf-8")

    if host_files:
        if not host_dir.exists():
            raise FileNotFoundError(f"Host directory not found in tmp copy: {host_dir}")
        for fname, content in host_files.items():
            target = _find_host_file_recursive(host_dir, fname)
            logger.info(f"Replace host file: {fname} -> {target} (len={len(content)} chars)")
            if target is None:
                raise FileNotFoundError(f"Host file '{fname}' not found in tmp copy under {host_dir} (recursive)")
            target.write_text(content, encoding="utf-8")


def _clean_build_artifacts(tmp_ops_root: Path) -> None:
    """
    Delete only build directories, not files (so build.sh is preserved).
    Deletes any directory whose name starts with "build" (e.g. build, build_out, build_swi_glu, etc).
    """
    logger.info(f"Cleaning build directories in TMP root: {tmp_ops_root}")
    removed = []
    for p in tmp_ops_root.glob("build*"):
        if p.exists() and p.is_dir():
            logger.info(f"  Removing directory: {p}")
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
    if not removed:
        logger.info("  No build* directories found to remove.")


def _run_cmd(cmd, cwd: Path, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    logger.info(f"Run cmd:\n  cwd: {cwd}\n  cmd: {' '.join(map(str, cmd))}")
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    logger.info(f"Command return code: {completed.returncode}")
    if completed.stdout:
        logger.info(f"--- STDOUT (head) ---\n{completed.stdout[:2000]}")
    if completed.stderr:
        logger.info(f"--- STDERR (head) ---\n{completed.stderr[:2000]}")

    if completed.returncode != 0:
        raise EvalError(
            "Command failed:\n"
            f"  cmd: {cmd}\n"
            f"  cwd: {cwd}\n"
            f"  returncode: {completed.returncode}\n"
            f"--- STDOUT ---\n{completed.stdout}\n"
            f"--- STDERR ---\n{completed.stderr}\n"
        )
    return completed


def _build_operator(tmp_ops_root: Path, op_name: str, timeout: Optional[int]) -> None:
    cmd = ["bash", "build.sh", "--pkg", "--soc=ascend910b", f"--ops={op_name}", f"--vendor_name={op_name}"]
    logger.info(f"Building operator '{op_name}' ...")
    _run_cmd(cmd, cwd=tmp_ops_root, timeout=timeout)
    logger.info("Build finished.")


def _find_installer(build_out_dir: Path, op_name: str) -> Path:
    pattern = f"cann-ops-nn*{op_name}*linux.aarch64.run"
    matches = list(build_out_dir.glob(pattern))
    logger.info(f"Searching installer in {build_out_dir} with pattern '{pattern}' -> {len(matches)} matches")
    if not matches:
        raise FileNotFoundError(f"Installer not found in {build_out_dir} with pattern: {pattern}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.info(f"Installer chosen: {matches[0]}")
    return matches[0]


def _install_operator(tmp_ops_root: Path, op_name: str, timeout: Optional[int]) -> None:
    build_out_dir = tmp_ops_root / "build_out"
    logger.info(f"Installing operator '{op_name}' from build_out: {build_out_dir}")
    if not build_out_dir.exists():
        raise FileNotFoundError(f"build_out directory not found after build: {build_out_dir}")

    installer = _find_installer(build_out_dir, op_name)
    _run_cmd(["bash", str(installer)], cwd=build_out_dir, timeout=timeout)
    logger.info("Install finished.")


def _stage_python_code(tmp_ops_root: Path, python_code_path: Union[str, Path]) -> Path:
    src = Path(python_code_path)
    logger.info(f"Staging python code into TMP test_duration:\n  src: {src}")
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Python code path does not exist or is not a file: {src}")

    dst_dir = tmp_ops_root / "test_duration"
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    shutil.copy2(str(src), str(dst))
    logger.info(f"  dst: {dst}")
    return dst


def _run_python_once(python_file: Path, timeout: Optional[int]) -> None:
    logger.info(f"Running python once (asserts must pass): {python_file}")
    _run_cmd(["python3", python_file.name], cwd=python_file.parent, timeout=timeout)
    logger.info("Python run OK.")


def _run_msprof(python_file: Path, timeout: Optional[int]) -> subprocess.CompletedProcess:
    if not MSPROF_BIN.exists():
        raise FileNotFoundError(f"msprof not found at: {MSPROF_BIN}")
    logger.info(f"Profiling with msprof: {python_file}")
    return _run_cmd([str(MSPROF_BIN), "python3", python_file.name], cwd=python_file.parent, timeout=timeout)


def _parse_msprof_saved_dir(stdout: str, stderr: str) -> Path:
    text = stdout + "\n" + stderr
    m = re.search(r"Data is saved in\s+(\S+)", text)
    if not m:
        raise EvalError(
            "Failed to find 'Data is saved in <DIR>' in msprof output.\n"
            f"--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}\n"
        )
    saved = Path(m.group(1)).expanduser()
    logger.info(f"msprof saved dir parsed: {saved}")
    return saved


def _find_op_statistic_csv(saved_dir: Path) -> Path:
    out_dir = saved_dir / "mindstudio_profiler_output"
    logger.info(f"Looking for op_statistic_*.csv under: {out_dir}")
    if not out_dir.exists():
        raise FileNotFoundError(f"mindstudio_profiler_output not found under: {saved_dir}")

    candidates = list(out_dir.glob("op_statistic_*.csv"))
    logger.info(f"Found {len(candidates)} candidate op_statistic files.")
    if not candidates:
        raise FileNotFoundError(f"No op_statistic_*.csv found under: {out_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    logger.info(f"Chosen op_statistic csv: {chosen}")
    return chosen


def _read_avg_time_us(op_stat_csv: Path, op_name: str) -> float:
    logger.info(f"Parsing Avg Time(us) from: {op_stat_csv}")

    with open(op_stat_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise EvalError(f"Empty CSV (no header): {op_stat_csv}")

        logger.info(f"CSV columns: {reader.fieldnames}")

        def get_col(d: Dict[str, str], key: str) -> Optional[str]:
            if key in d:
                return d[key]
            for k in d.keys():
                if k.strip().lower() == key.strip().lower():
                    return d[k]
            return None

        rows = list(reader)
        if not rows:
            raise EvalError(f"CSV has header but no data rows: {op_stat_csv}")

        # If there's only one row, return it directly
        if len(rows) == 1:
            avg = get_col(rows[0], "Avg Time(us)")
            if avg is None:
                raise EvalError(f"Missing 'Avg Time(us)' column in: {op_stat_csv}")
            try:
                return float(avg)
            except ValueError as e:
                raise EvalError(f"Failed to parse Avg Time(us)='{avg}': {e}") from e

        op_type_to_find = snake_to_camel(op_name)
        logger.info(f"Target OP Type: '{op_type_to_find}' (from op_name='{op_name}')")

        available_ops = []
        for row in rows:
            op_type = get_col(row, "OP Type")
            if op_type is None:
                raise EvalError(f"Missing 'OP Type' column in: {op_stat_csv}")

            op_type_s = op_type.strip()
            available_ops.append(op_type_s)

            if op_type_s != op_type_to_find:
                continue

            avg = get_col(row, "Avg Time(us)")
            if avg is None:
                raise EvalError(f"Missing 'Avg Time(us)' column in: {op_stat_csv}")

            logger.info(f"Matched row: OP Type={op_type_s} Avg Time(us)={avg} (full row={row})")
            try:
                return float(avg)
            except ValueError as e:
                raise EvalError(
                    f"Failed to parse Avg Time(us)='{avg}' for OP Type='{op_type_to_find}': {e}"
                ) from e

    raise EvalError(
        f"Operator '{op_type_to_find}' (from '{op_name}') not found in CSV. "
        f"Available OP Types: {sorted(set(available_ops))}"
    )



def eval(
    category: str,
    op_name: str,
    kernel_files: Dict[str, str],
    host_files: Dict[str, str],
    python_code_path: Union[str, Path],
    timeout: Optional[int] = None,
) -> float:
    if timeout is None:
        timeout = 300

    kernel_files = kernel_files or {}
    host_files = host_files or {}

    logger.info("=" * 80)
    logger.info(f"BEGIN eval: category={category} op_name={op_name}")
    logger.info(f"BASE_OPS_NN_ROOT: {BASE_OPS_NN_ROOT}")
    logger.info(f"TMP_PARENT:       {TMP_PARENT}")
    logger.info(f"MSPROF_BIN:       {MSPROF_BIN}")
    logger.info(f"python_code_path: {python_code_path}")
    logger.info(f"kernel_files: {list(kernel_files.keys())}")
    logger.info(f"host_files:   {list(host_files.keys())}")

    # 1) verify base files exist
    base_op_dir = _find_operator_dir(BASE_OPS_NN_ROOT, category, op_name)
    _verify_files_exist(base_op_dir, kernel_files, host_files)

    # 2) copy to tmp
    tmp_ops_root = _copy_base_to_tmp(BASE_OPS_NN_ROOT)
    logger.info(f"TMP ops root created: {tmp_ops_root}")

    # 3) replace selected files
    _replace_files_in_tmp(tmp_ops_root, category, op_name, kernel_files, host_files)

    # 4) stage python code in tmp
    python_file = _stage_python_code(tmp_ops_root, python_code_path)

    # 5) delete build* directories in tmp
    _clean_build_artifacts(tmp_ops_root)

    # 6) build + install operator
    _build_operator(tmp_ops_root, op_name, timeout=timeout)
    _install_operator(tmp_ops_root, op_name, timeout=timeout)

    # 7) run python once (asserts must pass)
    _run_python_once(python_file, timeout=timeout)

    # 8) profile the python run
    prof = _run_msprof(python_file, timeout=timeout)

    # 9) parse saved dir from msprof logs
    saved_dir = _parse_msprof_saved_dir(prof.stdout, prof.stderr)

    # 10) parse op_statistic csv
    op_stat_csv = _find_op_statistic_csv(saved_dir)


    # 12) return Avg Time(us)
    avg_us = _read_avg_time_us(op_stat_csv, op_name)

    logger.info(f"RESULT: {op_name} Avg Time(us) = {avg_us}")
    logger.info("END eval")
    logger.info("=" * 80)
    return avg_us


if __name__ == "__main__":
    category = "activation"
    op_name = "swi_glu"
    python_code_path = ""

    t = eval(
        category=category,
        op_name=op_name,
        kernel_files={},
        host_files={},
        python_code_path=python_code_path,
        timeout=None,
    )
    print(f"{op_name} Avg Time(us): {t}")