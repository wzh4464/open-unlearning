#!/bin/bash
# Run all LMCleaner configurations with different K values and epochs
# Epochs are controlled by max_step parameter (training has 625 steps total)

set -e

MODEL_PATH="/app/saves/finetune/tofu_finetune_gpt2"
LOG_DIR="/app/saves/train_logs/tofu_finetune_gpt2"
OUTPUT_BASE="saves/unlearn"

# K values to test
K_VALUES=(10 50 100 250 500)

# Epoch max_step values (625 total steps, divided into 5 epochs)
# Epoch 1: 125 steps, Epoch 2: 250 steps, etc.
EPOCH_STEPS=(125 250 375 500 625)
EPOCH_NAMES=(1 2 3 4 5)

echo "Starting LMCleaner batch runs at $(date)"
echo "Total configurations: ${#K_VALUES[@]} K values x ${#EPOCH_STEPS[@]} epochs = $((${#K_VALUES[@]} * ${#EPOCH_STEPS[@]})) runs"

for i in "${!EPOCH_STEPS[@]}"; do
    MAX_STEP=${EPOCH_STEPS[$i]}
    EPOCH=${EPOCH_NAMES[$i]}

    for K in "${K_VALUES[@]}"; do
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
        echo ""
    done
done

echo ""
echo "========================================"
echo "All LMCleaner runs completed at $(date)"
echo "========================================"
