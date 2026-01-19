#!/bin/bash
# Step 3: Baseline Unlearn - Epoch 5 (checkpoint-1250)
# Runs GradDiff and NPO methods for comparison
#
# Usage: ./03_baseline_epoch5.sh [GPU_ID]
# Example: ./03_baseline_epoch5.sh 0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

EPOCH=5
STEP=1250

# GPU configuration
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

CHECKPOINT_PATH=$(get_checkpoint_path $EPOCH)

echo "=============================================="
echo "Step 3: Baseline Unlearn - Epoch ${EPOCH}"
echo "=============================================="
print_config
echo "Epoch: ${EPOCH} (Step ${STEP})"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Methods: ${BASELINE_METHODS[*]}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    echo "Please run 01_finetune.sh first."
    exit 1
fi

for METHOD in "${BASELINE_METHODS[@]}"; do
    TASK_NAME=$(get_baseline_task_name $METHOD $EPOCH)
    OUTPUT_DIR="saves/unlearn/${TASK_NAME}"

    echo "=============================================="
    echo "Running ${METHOD} Unlearn - Epoch ${EPOCH}"
    echo "Task: ${TASK_NAME}"
    echo "Output: ${OUTPUT_DIR}"
    echo "=============================================="

    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer="${METHOD}" \
        task_name="${TASK_NAME}" \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        model.model_args.pretrained_model_name_or_path="${CHECKPOINT_PATH}" \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        ++trainer.args.bf16=true

    echo "${METHOD} complete for Epoch ${EPOCH}"
    echo ""
done

echo "=============================================="
echo "All Baselines for Epoch ${EPOCH} complete!"
echo "=============================================="
echo "Models saved to:"
for METHOD in "${BASELINE_METHODS[@]}"; do
    TASK_NAME=$(get_baseline_task_name $METHOD $EPOCH)
    echo "  - saves/unlearn/${TASK_NAME}"
done
