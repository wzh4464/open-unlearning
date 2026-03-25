#!/bin/bash
# RebuttalB Step 2: Evaluate all unlearned models from the (K,B) sweep
#
# Usage:
#   bash scripts/experiments/expB/02_eval_sweep.sh
#   bash scripts/experiments/expB/02_eval_sweep.sh --B 64

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

FILTER_B=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --B) FILTER_B="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

B_LIST=(8 16 32 64 128 256)
[ -n "$FILTER_B" ] && B_LIST=($FILTER_B)

echo "=== RebuttalB: Eval Sweep ==="

TOTAL=0
SKIPPED=0

for B in "${B_LIST[@]}"; do
    for K in "${K_VALUES[@]}"; do
        if ! valid_k_for_b "$B" "$K"; then
            continue
        fi

        TASK="expB_B${B}_K${K}"
        UNLEARN_DIR="${SAVES_BASE}/unlearn/${TASK}"
        EVAL_DIR="${SAVES_BASE}/eval/${TASK}"

        # Check unlearned model exists
        if [ ! -d "$UNLEARN_DIR" ]; then
            echo "SKIP: $TASK (no unlearned model)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Skip if already evaluated
        if [ -f "$EVAL_DIR/TOFU_SUMMARY.json" ]; then
            echo "SKIP: $TASK (already evaluated)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo ""
        echo "====== Eval B=$B K=$K ======"

        $PYTHON_CMD src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path="${UNLEARN_DIR}" \
            model.tokenizer_args.pretrained_model_name_or_path="${UNLEARN_DIR}" \
            forget_split=${FORGET_SPLIT} \
            holdout_split=${HOLDOUT_SPLIT} \
            paths.output_dir="${EVAL_DIR}" \
            task_name="${TASK}" \
            2>&1 | tee "${EVAL_DIR}.log" || {
                echo "FAILED eval: $TASK"
                continue
            }

        TOTAL=$((TOTAL + 1))
        echo "====== Eval $TASK done ======"
    done
done

echo ""
echo "=== Eval Sweep Complete ==="
echo "Evaluated: $TOTAL, Skipped: $SKIPPED"
