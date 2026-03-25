#!/bin/bash
# Plain SGD finetuning for LMCleaner paper compatibility
# Usage: bash scripts/finetune_sgd.sh [--lr 1e-4] [--epochs 1]

set -e
source "$(dirname "$0")/env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Defaults
LR=${LR:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) LR="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

TASK_NAME="sgd_llama32_1b_tofu_lr${LR}"
LOG_DIR="saves/train_logs/${TASK_NAME}"

DATASET_SIZE=4000
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH_SIZE))

echo "=== Plain SGD Finetuning ==="
echo "LR: ${LR}, Epochs: ${NUM_EPOCHS}"
echo "Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Task: ${TASK_NAME}"
echo ""

$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/sgd \
    model=Llama-3.2-1B-Instruct \
    trainer=finetune_sgd \
    collator=DataCollatorForSupervisedDatasetwithIndex \
    +trainer.args.remove_unused_columns=False \
    trainer.args.learning_rate=${LR} \
    trainer.args.max_grad_norm=0 \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
    trainer.args.save_strategy=epoch \
    trainer.args.save_total_limit=5 \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${LOG_DIR}" \
    +trainer.args.training_logger.max_steps=10000 \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.sync_mode=true \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_rng_state=true \
    +trainer.args.training_logger.steps_per_epoch=${STEPS_PER_EPOCH} \
    +trainer.args.training_logger.save_at_epoch_end=true \
    +trainer.args.spectral_norm.enabled=true \
    +trainer.args.spectral_norm.interval=50 \
    +trainer.args.spectral_norm.num_power_iters=20 \
    +trainer.args.spectral_norm.output_dir="${LOG_DIR}" \
    task_name="${TASK_NAME}"

echo ""
echo "=== SGD Training complete ==="
echo "Model: saves/finetune/${TASK_NAME}"
echo "Logs: ${LOG_DIR}"
