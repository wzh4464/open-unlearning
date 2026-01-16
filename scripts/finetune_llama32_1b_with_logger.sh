#!/bin/bash
# Finetune Llama-3.2-1B on TOFU with TrainingLogger enabled (for LMCleaner)
# Uses the official open-unlearning finetuned model as base

set -e
source "$(dirname "$0")/env.sh"

# Default values
FORGET_SPLIT=${FORGET_SPLIT:-"forget10"}
RETAIN_SPLIT=${RETAIN_SPLIT:-"retain90"}
TASK_NAME=${TASK_NAME:-"finetune_llama32_1b_with_logger"}
MODEL_PATH=${MODEL_PATH:-"open-unlearning/tofu_Llama-3.2-1B-Instruct_full"}
LOG_DIR=${LOG_DIR:-"saves/train_logs/${TASK_NAME}"}

echo "=== Finetuning Llama-3.2-1B with TrainingLogger ==="
echo "Model: ${MODEL_PATH}"
echo "Forget split: ${FORGET_SPLIT}"
echo "Retain split: ${RETAIN_SPLIT}"
echo "Task name: ${TASK_NAME}"
echo "Log dir: ${LOG_DIR}"

$PYTHON_CMD src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
    trainer=FinetuneTrainer \
    trainer.handler=FinetuneTrainer \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    trainer.args.training_logger.enabled=true \
    trainer.args.training_logger.log_dir="${LOG_DIR}" \
    trainer.args.training_logger.max_steps=1000 \
    trainer.args.training_logger.mode=batch \
    trainer.args.training_logger.save_interval=100 \
    trainer.args.training_logger.save_batch_data=false \
    trainer.args.training_logger.save_indices_only=true \
    trainer.args.training_logger.save_rng_state=true \
    task_name="${TASK_NAME}"

echo "=== Finetuning complete ==="
echo "Model saved to: saves/unlearn/${TASK_NAME}"
echo "Training logs saved to: ${LOG_DIR}"
