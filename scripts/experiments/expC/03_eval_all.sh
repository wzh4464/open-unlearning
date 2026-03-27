#!/bin/bash
# Experiment C: Eval all unlearned models

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Experiment C: Eval All ==="

for dir in ${SAVES_BASE}/unlearn/expC_*; do
    [ ! -d "$dir" ] && continue
    TASK=$(basename "$dir")
    EVAL_DIR="${SAVES_BASE}/eval/${TASK}"

    if [ -f "$EVAL_DIR/TOFU_SUMMARY.json" ]; then
        echo "SKIP: $TASK (already eval'd)"
        continue
    fi

    if [ ! -f "$dir/model.safetensors" ]; then
        echo "SKIP: $TASK (no model)"
        continue
    fi

    echo ""
    echo "====== Eval $TASK ======"

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="$dir" \
        model.tokenizer_args.pretrained_model_name_or_path="$dir" \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        paths.output_dir="${EVAL_DIR}" \
        task_name="${TASK}"

    echo "====== Eval $TASK done ======"
done

echo ""
echo "=== Eval complete ==="
