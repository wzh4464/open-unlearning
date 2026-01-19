#!/bin/bash
# Step 1: Finetune model on TOFU dataset with TrainingLogger
#
# Usage: ./01_finetune.sh [GPU_ID]
# Example: ./01_finetune.sh 0
#          CUDA_VISIBLE_DEVICES=0,1 ./01_finetune.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# GPU configuration
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

echo "=============================================="
echo "Step 1: Finetune ${MODEL_NAME} on TOFU"
echo "=============================================="
print_config
echo "GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Get random master port
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

TASK_NAME="${MODEL_SHORT}_tofu_safe"

echo "[1/1] Starting finetuning with TrainingLogger..."
echo "Task: ${TASK_NAME}"
echo "Output: ${FINETUNE_DIR}"
echo "Training logs: ${TRAINING_LOG_DIR}"
echo ""

$ACCELERATE_CMD launch \
    --config_file configs/accelerate/default_config.yaml \
    --main_process_port $MASTER_PORT \
    src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${TRAINING_LOG_DIR}" \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_batch_data=false \
    +trainer.args.training_logger.save_rng_state=true \
    +trainer.args.training_logger.save_interval=1 \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    ++trainer.args.bf16=true \
    ++trainer.args.save_strategy=steps \
    ++trainer.args.save_steps=250

echo ""
echo "=============================================="
echo "Finetuning complete!"
echo "=============================================="
echo "Model saved to: ${FINETUNE_DIR}"
echo "Training logs saved to: ${TRAINING_LOG_DIR}"
echo ""
echo "Checkpoints available:"
for step in "${CHECKPOINTS[@]}"; do
    epoch=$((step / 250))
    if [ -d "${FINETUNE_DIR}/checkpoint-${step}" ]; then
        echo "  - Epoch ${epoch}: ${FINETUNE_DIR}/checkpoint-${step}"
    fi
done
