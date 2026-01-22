#!/bin/bash
# LMCleaner Parameter Optimization Experiments
# Run 4 experiments in parallel on 4 GPUs
set -e

PYTHON_CMD="python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

MODEL_NAME="Llama-3.2-1B-Instruct"
TRAINING_LOG_DIR="saves/train_logs/llama32_1b_tofu_safe"
FINETUNE_BASE="saves/finetune/llama32_1b_tofu_safe"
CHECKPOINT_PATH="${FINETUNE_BASE}/checkpoint-1250"  # Epoch 5
K=100

# Experiment 1: damping=0.01
run_damping_exp() {
    local GPU=0
    local DAMPING=0.01
    local TASK_NAME="lmcleaner_damping${DAMPING}_epoch5_K${K}"
    local OUTPUT_DIR="saves/unlearn/${TASK_NAME}"
    local LOG_FILE="saves/unlearn/${TASK_NAME}.log"

    echo "[GPU $GPU] Starting damping=$DAMPING experiment..."

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
        trainer.method_args.max_step=1250 \
        trainer.method_args.hessian_mode=GGN \
        trainer.method_args.damping="${DAMPING}" \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.gradient_checkpointing=true \
        ++trainer.args.bf16=true 2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU] Running eval..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split=forget10 \
        holdout_split=holdout10 \
        task_name="${TASK_NAME}_eval" \
        model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
        paths.output_dir="${OUTPUT_DIR}/evals" 2>&1 | tee -a "$LOG_FILE"

    echo "[GPU $GPU] Completed damping experiment!"
}

# Experiment 2: epsilon=0.5 (Phase 4 privacy noise)
run_epsilon_exp() {
    local GPU=1
    local EPSILON=0.5
    local TASK_NAME="lmcleaner_epsilon${EPSILON}_epoch5_K${K}"
    local OUTPUT_DIR="saves/unlearn/${TASK_NAME}"
    local LOG_FILE="saves/unlearn/${TASK_NAME}.log"

    echo "[GPU $GPU] Starting epsilon=$EPSILON experiment..."

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
        trainer.method_args.max_step=1250 \
        trainer.method_args.hessian_mode=GGN \
        ++trainer.method_args.epsilon="${EPSILON}" \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.gradient_checkpointing=true \
        ++trainer.args.bf16=true 2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU] Running eval..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split=forget10 \
        holdout_split=holdout10 \
        task_name="${TASK_NAME}_eval" \
        model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
        paths.output_dir="${OUTPUT_DIR}/evals" 2>&1 | tee -a "$LOG_FILE"

    echo "[GPU $GPU] Completed epsilon experiment!"
}

# Experiment 3: Re-finetune 3 epochs (use existing GGN K100)
run_refinetune3_exp() {
    local GPU=2
    local BASE_MODEL="saves/unlearn/lmcleaner_llama32_epoch5_K100"
    local TASK_NAME="lmcleaner_epoch5_K100_refinetune3"
    local OUTPUT_DIR="saves/unlearn/${TASK_NAME}"
    local LOG_FILE="saves/unlearn/${TASK_NAME}.log"

    echo "[GPU $GPU] Starting re-finetune 3 epochs experiment..."

    # Check if base model exists
    if [ ! -d "$BASE_MODEL" ]; then
        echo "Base model not found: $BASE_MODEL"
        echo "Running LMCleaner first..."
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=LMCleanerBatch \
            task_name="lmcleaner_llama32_epoch5_K100" \
            model="${MODEL_NAME}" \
            forget_split=forget10 \
            retain_split=retain90 \
            model.model_args.pretrained_model_name_or_path="${CHECKPOINT_PATH}" \
            trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
            trainer.method_args.K="${K}" \
            trainer.method_args.max_step=1250 \
            trainer.method_args.hessian_mode=GGN \
            trainer.args.per_device_train_batch_size=2 \
            trainer.args.gradient_accumulation_steps=8 \
            trainer.args.gradient_checkpointing=true \
            ++trainer.args.bf16=true 2>&1 | tee "$LOG_FILE"
        BASE_MODEL="saves/unlearn/lmcleaner_llama32_epoch5_K100"
    fi

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD scripts/refiletune_retain.py \
        --model_path "$BASE_MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs 3 \
        --learning_rate 5e-6 \
        --batch_size 4 2>&1 | tee -a "$LOG_FILE"

    echo "[GPU $GPU] Running eval..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split=forget10 \
        holdout_split=holdout10 \
        task_name="${TASK_NAME}_eval" \
        model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
        paths.output_dir="${OUTPUT_DIR}/evals" 2>&1 | tee -a "$LOG_FILE"

    echo "[GPU $GPU] Completed re-finetune 3 epochs experiment!"
}

# Experiment 4: Combined damping + epsilon
run_combined_exp() {
    local GPU=3
    local DAMPING=0.01
    local EPSILON=0.5
    local TASK_NAME="lmcleaner_damping${DAMPING}_epsilon${EPSILON}_epoch5_K${K}"
    local OUTPUT_DIR="saves/unlearn/${TASK_NAME}"
    local LOG_FILE="saves/unlearn/${TASK_NAME}.log"

    echo "[GPU $GPU] Starting combined damping=$DAMPING epsilon=$EPSILON experiment..."

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
        trainer.method_args.max_step=1250 \
        trainer.method_args.hessian_mode=GGN \
        trainer.method_args.damping="${DAMPING}" \
        ++trainer.method_args.epsilon="${EPSILON}" \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.gradient_checkpointing=true \
        ++trainer.args.bf16=true 2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU] Running eval..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split=forget10 \
        holdout_split=holdout10 \
        task_name="${TASK_NAME}_eval" \
        model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
        paths.output_dir="${OUTPUT_DIR}/evals" 2>&1 | tee -a "$LOG_FILE"

    echo "[GPU $GPU] Completed combined experiment!"
}

echo "=============================================="
echo "LMCleaner Parameter Optimization (4 GPUs)"
echo "=============================================="
echo "Exp 1 (GPU 0): damping=0.01"
echo "Exp 2 (GPU 1): epsilon=0.5"
echo "Exp 3 (GPU 2): re-finetune 3 epochs"
echo "Exp 4 (GPU 3): damping+epsilon combined"
echo "=============================================="

# Run all 4 experiments in parallel
run_damping_exp &
PID1=$!
run_epsilon_exp &
PID2=$!
run_refinetune3_exp &
PID3=$!
run_combined_exp &
PID4=$!

echo "Started 4 parallel jobs: PIDs $PID1 $PID2 $PID3 $PID4"
echo "Waiting for completion..."

# Wait and report
wait $PID1 && echo "Exp 1 (damping) done" || echo "Exp 1 (damping) failed"
wait $PID2 && echo "Exp 2 (epsilon) done" || echo "Exp 2 (epsilon) failed"
wait $PID3 && echo "Exp 3 (re-finetune3) done" || echo "Exp 3 (re-finetune3) failed"
wait $PID4 && echo "Exp 4 (combined) done" || echo "Exp 4 (combined) failed"

echo ""
echo "=============================================="
echo "All parameter optimization experiments completed!"
echo "=============================================="
