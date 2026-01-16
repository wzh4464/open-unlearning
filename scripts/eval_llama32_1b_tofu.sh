#!/bin/bash
# Evaluate Llama-3.2-1B on TOFU benchmark
# Uses the official open-unlearning finetuned model

set -e
source "$(dirname "$0")/env.sh"

# Default values
FORGET_SPLIT=${FORGET_SPLIT:-"forget10"}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-"holdout10"}
TASK_NAME=${TASK_NAME:-"eval_llama32_1b_tofu"}
MODEL_PATH=${MODEL_PATH:-"open-unlearning/tofu_Llama-3.2-1B-Instruct_full"}

echo "=== Evaluating Llama-3.2-1B on TOFU ==="
echo "Model: ${MODEL_PATH}"
echo "Forget split: ${FORGET_SPLIT}"
echo "Holdout split: ${HOLDOUT_SPLIT}"
echo "Task name: ${TASK_NAME}"

$PYTHON_CMD src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    task_name="${TASK_NAME}"

echo "=== Evaluation complete ==="
echo "Results saved to: saves/eval/${TASK_NAME}"
