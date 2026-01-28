#!/usr/bin/env bash

SCRIPT="/home/l00936201/AscendC-copliot/tools/detector/main.py"
MAX_JOBS=1

# List of operators in the form category/op_name
ops=(
"activation/swi_glu"
)

running=0

for op in "${ops[@]}"; do
    echo "Starting: $op"
    python3 "$SCRIPT" --op "$op" &

    ((running++))
    if (( running == MAX_JOBS )); then
        # Wait for this batch of 5
        wait
        running=0
    fi
done

# Wait for any leftover jobs (<5)
wait

echo "All done."
