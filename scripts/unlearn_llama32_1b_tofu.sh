#!/bin/bash
# Unlearn Llama-3.2-1B on TOFU using GradAscent
# Uses the official open-unlearning finetuned model

set -e

# Default values
FORGET_SPLIT=${FORGET_SPLIT:-"forget10"}
RETAIN_SPLIT=${RETAIN_SPLIT:-"retain90"}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-"holdout10"}
TASK_NAME=${TASK_NAME:-"unlearn_llama32_1b_gradasc"}
MODEL_PATH=${MODEL_PATH:-"open-unlearning/tofu_Llama-3.2-1B-Instruct_full"}
TRAINER=${TRAINER:-"GradAscent"}

echo "=== Unlearning Llama-3.2-1B on TOFU ==="
echo "Model: ${MODEL_PATH}"
echo "Trainer: ${TRAINER}"
echo "Forget split: ${FORGET_SPLIT}"
echo "Retain split: ${RETAIN_SPLIT}"
echo "Task name: ${TASK_NAME}"

uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
    trainer="${TRAINER}" \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    task_name="${TASK_NAME}"

echo "=== Unlearning complete ==="
echo "Model saved to: saves/unlearn/${TASK_NAME}"

# Run evaluation
echo "=== Running evaluation ==="
uv run python src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${TASK_NAME}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    task_name="${TASK_NAME}_eval"

echo "=== Evaluation complete ==="
echo "Results saved to: saves/eval/${TASK_NAME}_eval"
