#!/bin/bash
# Experiment A - Step 6: MIA + Privacy Evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_ROOT}"

echo "=============================================="
echo "ExpA Step 6: MIA + Privacy Evaluation"
echo "=============================================="

# Get retrain eval path for privleak reference
RETRAIN_EVAL_JSON="${RETRAIN_DIR}/evals/TOFU_EVAL.json"
if [ ! -f "${RETRAIN_EVAL_JSON}" ]; then
    echo "WARNING: Retrain eval not found at ${RETRAIN_EVAL_JSON}"
    echo "privleak will use config default ref_value=0.5"
    RETRAIN_EVAL_JSON=""
fi

evaluate_mia() {
    local model_path=$1
    local task_name=$2
    local retain_logs=${3:-}
    local eval_output_dir="${model_path}/evals_mia"

    echo "----------------------------------------------"
    echo "MIA Eval: ${task_name}"
    echo "Model: ${model_path}"
    echo "retain_logs_path: ${retain_logs:-<config default>}"
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

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/mia \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_mia" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${eval_output_dir}" \
        "${retain_override[@]}"

    echo "Done: ${task_name}"
}

# Original finetuned
evaluate_mia "${FINETUNE_DIR}" "expA_original" "${RETRAIN_EVAL_JSON}"

# Retrain
evaluate_mia "${RETRAIN_DIR}" "expA_retrain" "${RETRAIN_EVAL_JSON}"

# LMCleaner
evaluate_mia "$(get_unlearn_output_dir "lmcleaner")" "$(get_unlearn_task_name "lmcleaner")" "${RETRAIN_EVAL_JSON}"

# Baselines
for METHOD in "${BASELINE_METHODS[@]}"; do
    METHOD_LOWER="${METHOD,,}"
    evaluate_mia "$(get_unlearn_output_dir "${METHOD_LOWER}")" "$(get_unlearn_task_name "${METHOD_LOWER}")" "${RETRAIN_EVAL_JSON}"
done

echo ""
echo "All MIA evaluations complete!"
