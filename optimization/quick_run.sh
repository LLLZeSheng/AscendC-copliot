#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

ts() { date +"%Y%m%d_%H%M%S"; }

run() {
  local name="$1"; shift
  local log="${LOG_DIR}/$(ts)_${name}.log"
  echo "==> Running: $name"
  echo "    Log: $log"
  # Capture stdout+stderr, also show on console
  "$@" 2>&1 | tee "$log"
}

python3 /home/l00936201/AscendC-copliot/optimization/start_optimization.py /home/l00936201/AscendC-copliot/operators/swi_glu/intitial.cpp \
  --operator-name swi_glu --category activation \
  --file-name /home/l00936201/AscendC-copliot/operators/swi_glu/swi_glu_tiling.cpp\
  --test-file /home/l00936201/AscendC-copliot/tools/utGenerator/test_swi_glu.py

echo "All commands completed successfully. Logs in: $LOG_DIR"