#!/usr/bin/env python3
"""
train.py

Run OpenEvolve with explicit CLI arguments (no path parsing, no meta.json).

Required inputs:
  - initial_program_path: path to the initial program to evolve
  - operator_name: operator name
  - operator_category: operator category
  - file_name: the source file name associated with the operator/program
  - test_file_path: path to the python test file used by eval

Notes:
  - You will modify OpenEvolve so that when it imports evaluation_file and calls
    evaluate(), it passes:
        evaluate(program_path, operator_name, operator_category, file_name, test_file_path)
"""

import asyncio
from pathlib import Path

from openevolve.openevolve import OpenEvolve  # Make sure this import matches your environment


OPENEVOLVE_ROOT = "/home/l00936201/AscendC-copliot/optimization"


def run_openevolve(
    initial_program_path: str,
    operator_name: str,
    operator_category: str,
    file_name: str,
    test_file_path: str,
    run_id=None,
    output_dir=None,
    iterations: int = 15,
):
    initial_program = Path(initial_program_path).resolve()
    test_file = Path(test_file_path).resolve()

    if not initial_program.exists():
        raise FileNotFoundError(f"initial_program_path does not exist: {initial_program}")

    if not test_file.exists():
        raise FileNotFoundError(f"test_file_path does not exist: {test_file}")

    print("[INFO] OpenEvolve context:")
    print(f"  operator_name     = {operator_name}")
    print(f"  operator_category = {operator_category}")
    print(f"  file_name         = {file_name}")
    print(f"  initial_program   = {initial_program}")
    print(f"  test_file_path    = {test_file}")
    print(f"  output_dir        = {output_dir}")

    evaluation_file = str(Path(OPENEVOLVE_ROOT) / "eval_test.py")
    config_path = str(Path(OPENEVOLVE_ROOT) / "config.yaml")
    test_name = Path(test_file_path).stem  # e.g., "test_0.py" -> "test_0"
    if output_dir:
        output_path = Path(output_dir)
    else:
        base_dir = Path(OPENEVOLVE_ROOT) / operator_category / operator_name
        suffix = f"{test_name}_{run_id}" if run_id else test_name
        output_path = base_dir / suffix
    output_dir = str(output_path)


    # Pass only what you want OpenEvolve to carry through to eval().
    evolve = OpenEvolve(
        initial_program_path=str(initial_program),
        evaluation_file=evaluation_file,
        config_path=config_path,
        output_dir=output_dir,
        operator_name=operator_name,
        operator_category=operator_category,
        file_name=file_name,
        test_file_path=str(test_file),
        mode = "function_based"
    )

    print("[INFO] Starting evolution process...")
    best_program = asyncio.run(evolve.run(iterations=iterations))
    print("[INFO] Evolution completed successfully.")
    print(f"[RESULT] Best program path: {best_program}")

    return best_program


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenEvolve with explicit arguments")
    parser.add_argument(
        "initial_program_path",
        help="Path to the initial program to evolve",
    )
    parser.add_argument(
        "--operator-name",
        required=True,
        help="Operator name",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Operator category",
    )
    parser.add_argument(
        "--file-name",
        required=True,
        help="Associated file name (as expected by eval)",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to the python test file to use during evaluation",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id appended to the output directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (absolute or relative to CWD)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of evolution iterations (default: 15)",
    )

    args = parser.parse_args()

    run_openevolve(
        initial_program_path=args.initial_program_path,
        operator_name=args.operator_name,
        operator_category=args.category,
        file_name=args.file_name,
        test_file_path=args.test_file,
        run_id=args.run_id,
        output_dir=args.output_dir,
        iterations=args.iterations,
    )
