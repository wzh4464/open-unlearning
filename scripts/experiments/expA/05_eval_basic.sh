#!/bin/bash
# Experiment A - Step 5: Basic TOFU Evaluation
# Evaluates all models: original finetuned, retrain, LMCleaner, baselines
# Evaluates retrain FIRST to get retain_logs_path for forget_quality and privleak
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 5: TOFU Basic Evaluation"
echo "=============================================="
print_config

# Track retrain eval output path for forget_quality/privleak reference
RETRAIN_EVAL_JSON=""

evaluate_model() {
    local model_path=$1
    local task_name=$2
    local retain_logs=${3:-"null"}
    local eval_output_dir="${model_path}/evals"

    echo "----------------------------------------------"
    echo "Evaluating: ${task_name}"
    echo "Model: ${model_path}"
    echo "retain_logs_path: ${retain_logs}"
    echo "----------------------------------------------"

    if [ ! -d "${model_path}" ]; then
        echo "WARNING: Model not found: ${model_path}. Skipping."
        return
    fi

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_eval" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${eval_output_dir}" \
        retain_logs_path="${retain_logs}"

    echo "Done: ${task_name}"
}

# 1. Evaluate retrain model FIRST (gold standard, needed for retain_logs_path)
echo ""
echo ">>> Evaluating retrain model first (for retain_logs_path) <<<"
evaluate_model "${RETRAIN_DIR}" "expA_retrain"
RETRAIN_EVAL_JSON="${RETRAIN_DIR}/evals/TOFU_EVAL.json"

if [ ! -f "${RETRAIN_EVAL_JSON}" ]; then
    echo "WARNING: Retrain eval output not found at ${RETRAIN_EVAL_JSON}"
    echo "forget_quality and privleak will use fallback values."
    RETRAIN_EVAL_JSON="null"
fi

# 2. Evaluate original finetuned model (reference)
evaluate_model "${FINETUNE_DIR}" "expA_original" "${RETRAIN_EVAL_JSON}"

# 3. Evaluate LMCleaner
TASK_NAME=$(get_unlearn_task_name "lmcleaner")
evaluate_model "$(get_unlearn_output_dir "lmcleaner")" "${TASK_NAME}" "${RETRAIN_EVAL_JSON}"

# 4. Evaluate all baselines
for METHOD in "${BASELINE_METHODS[@]}"; do
    METHOD_LOWER="${METHOD,,}"
    TASK_NAME=$(get_unlearn_task_name "${METHOD_LOWER}")
    evaluate_model "$(get_unlearn_output_dir "${METHOD_LOWER}")" "${TASK_NAME}" "${RETRAIN_EVAL_JSON}"
done

echo ""
echo "=============================================="
echo "All basic evaluations complete!"
echo "=============================================="
echo "Retrain eval JSON: ${RETRAIN_EVAL_JSON}"
