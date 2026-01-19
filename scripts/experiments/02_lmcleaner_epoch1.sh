#!/bin/bash
# Step 2: LMCleaner Unlearn - Epoch 1 (checkpoint-250)
#
# Usage: ./02_lmcleaner_epoch1.sh [GPU_ID]
# Example: ./02_lmcleaner_epoch1.sh 0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

EPOCH=1
STEP=$((EPOCH * STEPS_PER_EPOCH))

# GPU configuration
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

TASK_NAME=$(get_lmcleaner_task_name $EPOCH)
CHECKPOINT_PATH=$(get_checkpoint_path $EPOCH)
OUTPUT_DIR="saves/unlearn/${TASK_NAME}"

echo "=============================================="
echo "Step 2: LMCleaner Unlearn - Epoch ${EPOCH}"
echo "=============================================="
print_config
echo "Epoch: ${EPOCH} (Step ${STEP})"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    echo "Please run 01_finetune.sh first."
    exit 1
fi

# Check if training logs exist
if [ ! -d "${TRAINING_LOG_DIR}" ]; then
    echo "ERROR: Training log directory not found: ${TRAINING_LOG_DIR}"
    echo "Please run 01_finetune.sh first."
    exit 1
fi

echo "[1/1] Running LMCleaner Unlearn..."
$PYTHON_CMD src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${CHECKPOINT_PATH}" \
    trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
    trainer.method_args.K="${K}" \
    trainer.method_args.max_step="${STEP}" \
    trainer.method_args.hessian_mode="${HESSIAN_MODE}" \
    trainer.method_args.damping="${DAMPING}" \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    ++trainer.args.bf16=true

echo ""
echo "=============================================="
echo "LMCleaner Unlearn Epoch ${EPOCH} complete!"
echo "=============================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate, run: ./04_eval_tofu.sh ${CUDA_VISIBLE_DEVICES:-0} ${OUTPUT_DIR}"
