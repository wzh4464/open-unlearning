#!/bin/bash
# Run reduced LMCleaner configurations to save time
# - K values: 10, 50, 100
# - Epochs: 1 and 5 (start and end points)

set -e

# Use environment variables for portability
SAVES_DIR="${SAVES_DIR:-./saves}"
MODEL_PATH="${MODEL_PATH:-${SAVES_DIR}/finetune/tofu_finetune_gpt2}"
LOG_DIR="${LOG_DIR:-${SAVES_DIR}/train_logs/tofu_finetune_gpt2}"

echo "Starting reduced LMCleaner runs at $(date)"

# Run configurations
run_lmcleaner() {
    local K=$1
    local EPOCH=$2
    local MAX_STEP=$3

    TASK_NAME="gpt2_LMCleaner_epoch${EPOCH}_K${K}"
    echo ""
    echo "========================================"
    echo "Running: Epoch ${EPOCH} (max_step=${MAX_STEP}), K=${K}"
    echo "Task: ${TASK_NAME}"
    echo "Started at: $(date)"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        model=gpt2 \
        model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH} \
        trainer=LMCleanerBatch \
        trainer.method_args.K=${K} \
        trainer.method_args.training_log_dir=${LOG_DIR} \
        trainer.method_args.max_step=${MAX_STEP} \
        task_name=${TASK_NAME}

    echo "Completed: ${TASK_NAME} at $(date)"
}

# Epoch 1 runs (max_step=125) - should be fast
echo "=== Running Epoch 1 configurations ==="
run_lmcleaner 10 1 125
run_lmcleaner 50 1 125
run_lmcleaner 100 1 125

# Epoch 5 runs (max_step=625 = full training)
echo ""
echo "=== Running Epoch 5 configurations ==="
run_lmcleaner 50 5 625
run_lmcleaner 100 5 625

echo ""
echo "========================================"
echo "All reduced LMCleaner runs completed at $(date)"
echo "========================================"
