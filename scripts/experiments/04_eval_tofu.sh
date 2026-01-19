#!/bin/bash
# Step 4: TOFU Basic Evaluation
# Evaluates unlearned models on TOFU benchmark
#
# Usage: ./04_eval_tofu.sh [GPU_ID] [MODEL_PATH]
# Example: ./04_eval_tofu.sh 0
#          ./04_eval_tofu.sh 0 saves/unlearn/lmcleaner_llama32_1b_epoch1_K1000
#
# If MODEL_PATH is not provided, evaluates all unlearned models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# GPU configuration
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

# Optional: specific model path
SPECIFIC_MODEL_PATH="${2:-}"

echo "=============================================="
echo "Step 4: TOFU Basic Evaluation"
echo "=============================================="
print_config
echo "GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Function to evaluate a single model
evaluate_model() {
    local model_path=$1
    local task_name=$2
    local eval_output_dir="${model_path}/evals"

    echo "=============================================="
    echo "Evaluating: ${task_name}"
    echo "Model: ${model_path}"
    echo "Output: ${eval_output_dir}"
    echo "=============================================="

    if [ ! -d "${model_path}" ]; then
        echo "WARNING: Model not found: ${model_path}"
        echo "Skipping..."
        return
    fi

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_eval" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        paths.output_dir="${eval_output_dir}"

    echo "Evaluation complete for ${task_name}"
    echo ""
}

if [ -n "${SPECIFIC_MODEL_PATH}" ]; then
    # Evaluate specific model
    task_name=$(basename "${SPECIFIC_MODEL_PATH}")
    evaluate_model "${SPECIFIC_MODEL_PATH}" "${task_name}"
else
    # Evaluate all unlearned models
    echo "Evaluating all unlearned models..."
    echo ""

    # LMCleaner models
    for EPOCH in "${EPOCHS[@]}"; do
        TASK_NAME=$(get_lmcleaner_task_name $EPOCH)
        MODEL_PATH="saves/unlearn/${TASK_NAME}"
        evaluate_model "${MODEL_PATH}" "${TASK_NAME}"
    done

    # Baseline models
    for METHOD in "${BASELINE_METHODS[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
            TASK_NAME=$(get_baseline_task_name $METHOD $EPOCH)
            MODEL_PATH="saves/unlearn/${TASK_NAME}"
            evaluate_model "${MODEL_PATH}" "${TASK_NAME}"
        done
    done
fi

echo "=============================================="
echo "All evaluations complete!"
echo "=============================================="
echo ""
echo "Results summary:"
echo "  - LMCleaner results: saves/unlearn/lmcleaner_*/evals/"
echo "  - Baseline results: saves/unlearn/baseline_*/evals/"
echo ""
echo "Key metrics to check in TOFU_EVAL.json:"
echo "  - Forget Quality: extraction_strength, truth_ratio"
echo "  - Retain Performance: ROUGE scores on retain set"
echo "  - Model Utility: general capability scores"
