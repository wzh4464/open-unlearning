#!/bin/bash
# Step 5: TOFU MIA (Membership Inference Attack) Evaluation
# Evaluates unlearned models on privacy metrics
#
# Usage: ./05_eval_tofu_mia.sh [GPU_ID] [MODEL_PATH]
# Example: ./05_eval_tofu_mia.sh 0
#          ./05_eval_tofu_mia.sh 0 saves/unlearn/lmcleaner_llama32_1b_epoch1_K1000
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
echo "Step 5: TOFU MIA Evaluation"
echo "=============================================="
print_config
echo "GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Function to evaluate MIA for a single model
evaluate_mia() {
    local model_path=$1
    local task_name=$2
    local eval_output_dir="${model_path}/evals_mia"

    echo "=============================================="
    echo "MIA Evaluation: ${task_name}"
    echo "Model: ${model_path}"
    echo "Output: ${eval_output_dir}"
    echo "=============================================="

    if [ ! -d "${model_path}" ]; then
        echo "WARNING: Model not found: ${model_path}"
        echo "Skipping..."
        return
    fi

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/mia \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_mia" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        paths.output_dir="${eval_output_dir}"

    echo "MIA evaluation complete for ${task_name}"
    echo ""
}

if [ -n "${SPECIFIC_MODEL_PATH}" ]; then
    # Evaluate specific model
    task_name=$(basename "${SPECIFIC_MODEL_PATH}")
    evaluate_mia "${SPECIFIC_MODEL_PATH}" "${task_name}"
else
    # Evaluate all unlearned models
    echo "Running MIA evaluation on all unlearned models..."
    echo ""

    # LMCleaner models
    for EPOCH in "${EPOCHS[@]}"; do
        TASK_NAME=$(get_lmcleaner_task_name $EPOCH)
        MODEL_PATH="saves/unlearn/${TASK_NAME}"
        evaluate_mia "${MODEL_PATH}" "${TASK_NAME}"
    done

    # Baseline models
    for METHOD in "${BASELINE_METHODS[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
            TASK_NAME=$(get_baseline_task_name $METHOD $EPOCH)
            MODEL_PATH="saves/unlearn/${TASK_NAME}"
            evaluate_mia "${MODEL_PATH}" "${TASK_NAME}"
        done
    done
fi

echo "=============================================="
echo "All MIA evaluations complete!"
echo "=============================================="
echo ""
echo "MIA Results summary:"
echo "  - LMCleaner MIA: saves/unlearn/lmcleaner_*/evals_mia/"
echo "  - Baseline MIA: saves/unlearn/baseline_*/evals_mia/"
echo ""
echo "Key MIA metrics:"
echo "  - Attack success rate (lower is better for privacy)"
echo "  - ROC AUC scores"
echo "  - Forget set vs holdout set distinguishability"
