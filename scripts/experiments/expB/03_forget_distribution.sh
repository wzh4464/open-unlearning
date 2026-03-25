#!/bin/bash
# RebuttalB Step 3: Analyze forget sample distribution for all B values

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../../env.sh"

OUTPUT_DIR="${SAVES_BASE}/experiments/expB/stats"
mkdir -p "$OUTPUT_DIR"

echo "=== RebuttalB: Forget Distribution Analysis ==="

for B in 8 16 32 64 128 256; do
    TRAIN_LOG="${TRAIN_LOG_DIRS[$B]}"
    if [ ! -d "$TRAIN_LOG" ]; then
        echo "SKIP: B=$B (no training logs)"
        continue
    fi

    echo "Analyzing B=$B..."
    $PYTHON_CMD scripts/rebuttalB_forget_distribution.py \
        --train-log-dir "$TRAIN_LOG" \
        --forget-split "${FORGET_SPLIT}" \
        --effective-batch-size "$B" \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "=== Distribution Analysis Complete ==="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null
