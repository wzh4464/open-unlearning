#!/bin/bash
# Safe finetuning Llama-3.2-1B with frequent saves to prevent OOM

set -e
source "$(dirname "$0")/env.sh"

NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TASK_NAME=${TASK_NAME:-"llama32_1b_tofu_safe"}
LOG_DIR=${LOG_DIR:-"saves/train_logs/${TASK_NAME}"}

# Calculate steps per epoch
DATASET_SIZE=4000
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH_SIZE))

echo "=== Safe Finetuning Llama-3.2-1B ==="
echo "Epochs: ${NUM_EPOCHS}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Saving every 10 steps to prevent memory accumulation"
echo "Task name: ${TASK_NAME}"
echo ""

$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=Llama-3.2-1B-Instruct \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${LOG_DIR}" \
    +trainer.args.training_logger.max_steps=10000 \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.save_interval=10 \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_rng_state=true \
    +trainer.args.training_logger.steps_per_epoch=${STEPS_PER_EPOCH} \
    +trainer.args.training_logger.save_at_epoch_end=true \
    task_name="${TASK_NAME}"

echo ""
echo "=== Training complete ==="
echo "Logs: ${LOG_DIR}"
