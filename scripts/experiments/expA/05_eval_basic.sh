#!/bin/bash
# Experiment A - Step 5: Basic TOFU Evaluation
# Evaluates all models: original finetuned, retrain, LMCleaner, baselines
# Evaluates retrain FIRST to get retain_logs_path for forget_quality and privleak
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_ROOT}"

echo "=============================================="
echo "ExpA Step 5: TOFU Basic Evaluation"
echo "=============================================="
print_config

# Track eval output paths for reference-based metrics
RETRAIN_EVAL_JSON=""
ORIGINAL_EVAL_JSON=""

evaluate_model() {
    local model_path=$1
    local task_name=$2
    local retain_logs=${3:-}
    local original_logs=${4:-}
    local eval_output_dir="${model_path}/evals"

    echo "----------------------------------------------"
    echo "Evaluating: ${task_name}"
    echo "Model: ${model_path}"
    echo "retain_logs_path: ${retain_logs:-<config default>}"
    echo "original_logs_path: ${original_logs:-<config default>}"
    echo "----------------------------------------------"

    if [ ! -d "${model_path}" ]; then
        echo "WARNING: Model not found: ${model_path}. Skipping."
        return
    fi

    # Build retain_logs_path override only when a real path is provided
    local retain_override=()
    if [ -n "${retain_logs}" ]; then
        retain_override=("retain_logs_path=${retain_logs}")
    fi
    local original_override=()
    if [ -n "${original_logs}" ]; then
        original_override=("eval.tofu.original_logs_path=${original_logs}")
    fi

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_eval" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        model.model_args.attn_implementation=${ATTN_IMPL} \
        paths.output_dir="${eval_output_dir}" \
        "${retain_override[@]}" \
        "${original_override[@]}"

    echo "Done: ${task_name}"
}

# 1. Evaluate retrain model FIRST (gold standard, needed for retain_logs_path)
echo ""
echo ">>> Evaluating retrain model first (for retain_logs_path) <<<"
evaluate_model "${RETRAIN_DIR}" "expA_retrain"
RETRAIN_EVAL_JSON="${RETRAIN_DIR}/evals/TOFU_EVAL.json"

if [ ! -f "${RETRAIN_EVAL_JSON}" ]; then
    echo "WARNING: Retrain eval output not found at ${RETRAIN_EVAL_JSON}"
    echo "forget_quality and privleak will use config defaults."
    RETRAIN_EVAL_JSON=""
fi

# 2. Evaluate original finetuned model (reference)
evaluate_model "${FINETUNE_DIR}" "expA_original" "${RETRAIN_EVAL_JSON}"
ORIGINAL_EVAL_JSON="${FINETUNE_DIR}/evals/TOFU_EVAL.json"

if [ ! -f "${ORIGINAL_EVAL_JSON}" ]; then
    echo "WARNING: Original eval output not found at ${ORIGINAL_EVAL_JSON}"
    echo "selectivity will use config default and resolve to None."
    ORIGINAL_EVAL_JSON=""
fi

# 3. Evaluate LMCleaner
TASK_NAME=$(get_unlearn_task_name "lmcleaner")
evaluate_model "$(get_unlearn_output_dir "lmcleaner")" "${TASK_NAME}" "${RETRAIN_EVAL_JSON}" "${ORIGINAL_EVAL_JSON}"

# 3b. Evaluate LMCleaner post-finetune
POSTFT_DIR=$(get_postft_output_dir "lmcleaner")
if [ -d "${POSTFT_DIR}" ]; then
    evaluate_model "${POSTFT_DIR}" "$(get_postft_task_name "lmcleaner")" "${RETRAIN_EVAL_JSON}" "${ORIGINAL_EVAL_JSON}"
fi

# 4. Evaluate all baselines
for METHOD in "${BASELINE_METHODS[@]}"; do
    METHOD_LOWER="${METHOD,,}"
    TASK_NAME=$(get_unlearn_task_name "${METHOD_LOWER}")
    evaluate_model "$(get_unlearn_output_dir "${METHOD_LOWER}")" "${TASK_NAME}" "${RETRAIN_EVAL_JSON}" "${ORIGINAL_EVAL_JSON}"
done

echo ""
echo "=============================================="
echo "All basic evaluations complete!"
echo "=============================================="
echo "Retrain eval JSON: ${RETRAIN_EVAL_JSON}"
echo "Original eval JSON: ${ORIGINAL_EVAL_JSON}"
