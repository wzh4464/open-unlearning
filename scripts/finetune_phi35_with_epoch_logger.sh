#!/bin/bash
# Finetune Phi-3.5-mini-instruct on TOFU with epoch-aware TrainingLogger
# This script ensures:
# 1. Model checkpoints are saved at the end of each epoch
# 2. Each epoch has the same number of intermediate checkpoints
#
# Usage:
#   bash scripts/finetune_phi35_with_epoch_logger.sh
#
# Environment variables:
#   NUM_EPOCHS: Number of training epochs (default: 5)
#   CHECKPOINTS_PER_EPOCH: Number of intermediate checkpoints per epoch (default: 4)
#   BATCH_SIZE: Per-device batch size (default: 4)
#   GRAD_ACCUM: Gradient accumulation steps (default: 4)

set -e

# Default values
NUM_EPOCHS=${NUM_EPOCHS:-5}
CHECKPOINTS_PER_EPOCH=${CHECKPOINTS_PER_EPOCH:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TASK_NAME=${TASK_NAME:-"phi35_tofu_finetune_with_epoch_log"}
LOG_DIR=${LOG_DIR:-"saves/train_logs/${TASK_NAME}"}

echo "=== Finetuning Phi-3.5-mini-instruct with Epoch-Aware TrainingLogger ==="
echo "Epochs: ${NUM_EPOCHS}"
echo "Checkpoints per epoch: ${CHECKPOINTS_PER_EPOCH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRAD_ACCUM}"
echo "Task name: ${TASK_NAME}"
echo "Log dir: ${LOG_DIR}"

# Calculate steps per epoch
# TOFU full dataset has 4000 samples
# steps_per_epoch = dataset_size / (batch_size * grad_accum * num_gpus)
# For single GPU: steps_per_epoch = 4000 / (4 * 4) = 250
DATASET_SIZE=4000
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH_SIZE))

echo "Calculated steps per epoch: ${STEPS_PER_EPOCH}"
echo "Total steps: $((STEPS_PER_EPOCH * NUM_EPOCHS))"
echo ""

# Run finetuning with epoch-aware logging
uv run python src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=Phi-3.5-mini-instruct \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${LOG_DIR}" \
    +trainer.args.training_logger.max_steps=10000 \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.steps_per_epoch=${STEPS_PER_EPOCH} \
    +trainer.args.training_logger.checkpoints_per_epoch=${CHECKPOINTS_PER_EPOCH} \
    +trainer.args.training_logger.save_at_epoch_end=true \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_rng_state=true \
    task_name="${TASK_NAME}"

echo ""
echo "=== Finetuning complete ==="
echo "Model saved to: saves/finetune/${TASK_NAME}"
echo "Training logs saved to: ${LOG_DIR}"
echo ""
echo "Checkpoint structure:"
echo "  - ${LOG_DIR}/model_checkpoints/epoch_*/  : Model checkpoints at epoch end"
echo "  - ${LOG_DIR}/step_records_*.pkl          : Training step records"
echo "  - ${LOG_DIR}/meta.json                   : Training metadata"
echo ""
echo "Each epoch has ${CHECKPOINTS_PER_EPOCH} intermediate checkpoints + 1 epoch-end checkpoint"
