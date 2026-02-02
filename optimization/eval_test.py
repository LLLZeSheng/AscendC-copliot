# eval.py
import os
import traceback

import check_single_test  # must provide check_single.eval(...)
import replace       # your replace.py (must be importable)


def evaluate(
    program_path: str,
    operator_name: str,
    operator_category: str,
    mode: str,
    file_name: str,          # this is a PATH
    test_file_path: str,
):
    """
    Returns a dict:
      {'combined_score': ...}

    Behavior:
      - Build replaced_code by patching file_name using program_path (function-based)
        or by reading program_path as full content (file-based).
      - Pass host_files with key = basename(file_name).
    """
    try:
        if not os.path.exists(program_path):
            raise FileNotFoundError(f"program_path not found: {program_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"test_file_path not found: {test_file_path}")

        # basename used for host_files dict key
        host_key = os.path.basename(file_name)

        # ---- Build replaced_code ----
        if mode == "file-based":
            with open(program_path, "r", encoding="utf-8") as f:
                replaced_code = f.read()
        else:
            # function-based: patch target file using markers in program_path
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"target file (file_name) not found: {file_name}")

            print("STEP_START: REPLACE")
            replaced_code = replace.replace(
                target_file_path=file_name,
                file_generated_path=program_path,
            )
            print("STEP_OK: REPLACE")

            if replaced_code is None:
                with open(program_path, "r", encoding="utf-8") as f:
                    replaced_code = f.read()

        # ---- Run check ----
        duration = check_single_test.eval(
            op_name=operator_name,
            category=operator_category,
            host_files={
                host_key: replaced_code,   # <-- KEY IS NOW THE EXTRACTED FILE NAME
            },
            kernel_files={},
            python_code_path=test_file_path,
        )

        duration = float(duration)
        if duration <= 0:
            return {"combined_score": 0, "avg_us": 0}

        return {"combined_score": 10.0 / duration, "avg_us": duration}

    except Exception:
        print("[EVAL ERROR] evaluate() failed:")
        traceback.print_exc()
        return {"combined_score": 0, "avg_us": 0}
