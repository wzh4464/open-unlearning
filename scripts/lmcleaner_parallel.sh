#!/bin/bash
# Parallel LMCleaner Unlearn on Multiple GPUs
set -e

# 直接使用系统 python
PYTHON_CMD="python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

MODEL_NAME="Llama-3.2-1B-Instruct"
TRAINING_LOG_DIR="saves/train_logs/llama32_1b_tofu_safe"
FINETUNE_BASE="saves/finetune/llama32_1b_tofu_safe"
K=${K:-1000}

run_unlearn() {
    local GPU=$1
    local STEP=$2
    local EPOCH=$((STEP / 250))
    local TASK_NAME="lmcleaner_llama32_epoch${EPOCH}_K${K}"
    local CHECKPOINT_PATH="${FINETUNE_BASE}/checkpoint-${STEP}"
    local OUTPUT_DIR="saves/unlearn/${TASK_NAME}"
    local LOG_FILE="/tmp/lmcleaner_epoch${EPOCH}.log"

    echo "[GPU $GPU] Starting Epoch $EPOCH (Step $STEP)..."

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LMCleanerBatch \
        task_name="${TASK_NAME}" \
        model="${MODEL_NAME}" \
        forget_split=forget10 \
        retain_split=retain90 \
        model.model_args.pretrained_model_name_or_path="${CHECKPOINT_PATH}" \
        trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
        trainer.method_args.K="${K}" \
        trainer.method_args.max_step="${STEP}" \
        trainer.method_args.hessian_mode=GGN \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.gradient_checkpointing=true \
        ++trainer.args.bf16=true 2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU] Completed Epoch $EPOCH unlearn!"

    # Run eval
    echo "[GPU $GPU] Running eval for Epoch $EPOCH..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split=forget10 \
        holdout_split=holdout10 \
        task_name="${TASK_NAME}_eval" \
        model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
        paths.output_dir="${OUTPUT_DIR}/evals" 2>&1 | tee -a "$LOG_FILE"

    echo "[GPU $GPU] Completed Epoch $EPOCH!"
}

echo "=============================================="
echo "Parallel LMCleaner Unlearn (4 GPUs)"
echo "=============================================="

# Run first 4 checkpoints in parallel
run_unlearn 0 250 &
PID1=$!
run_unlearn 1 500 &
PID2=$!
run_unlearn 2 750 &
PID3=$!
run_unlearn 3 1000 &
PID4=$!

echo "Started 4 parallel jobs: PIDs $PID1 $PID2 $PID3 $PID4"
echo "Waiting for completion..."

# Wait for all to complete
wait $PID1 && echo "Epoch 1 done" || echo "Epoch 1 failed"
wait $PID2 && echo "Epoch 2 done" || echo "Epoch 2 failed"
wait $PID3 && echo "Epoch 3 done" || echo "Epoch 3 failed"
wait $PID4 && echo "Epoch 4 done" || echo "Epoch 4 failed"

# Run epoch 5 on GPU 0
echo ""
echo "Running Epoch 5 (Step 1250)..."
run_unlearn 0 1250

echo ""
echo "=============================================="
echo "All checkpoints completed!"
echo "=============================================="
