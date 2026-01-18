#!/bin/bash
# Quick test for LMCleaner lazy loading fix
# Usage: bash scripts/test_lazy_loading.sh

set -e

LOG_DIR="saves/train_logs/llama32_1b_tofu_safe"
CHECKPOINT="saves/finetune/llama32_1b_tofu_safe/checkpoint-250"

# Check prerequisites
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Training log dir not found: $LOG_DIR"
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=== Testing LMCleaner Lazy Loading ==="

python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name=lmcleaner_quick_test \
    model=Llama-3.2-1B-Instruct \
    forget_split=forget10 \
    retain_split=retain90 \
    model.model_args.pretrained_model_name_or_path="$CHECKPOINT" \
    trainer.method_args.training_log_dir="$LOG_DIR" \
    trainer.method_args.max_step=250 \
    trainer.method_args.K=5

echo "=== Test Complete ==="
