#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SHAPE="${1:-[48,1,9216]}"
DTYPE="${2:-bfloat16}"

python3 "${ROOT_DIR}/generate_test.py" \
  --op activation/gelu_quant \
  --shape "${SHAPE}" \
  --dtype "${DTYPE}" \
  --mode llm \
  --oneshot-path "${ROOT_DIR}/test_swi_glu.py" \
  --out "${SCRIPT_DIR}/test_gelu_quant.py"
