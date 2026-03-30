#!/bin/bash
# Experiment A: Llama-2-7B Single-Epoch Configuration
# All shared parameters for the experiment

# ============================================
# Model Configuration
# ============================================
MODEL_NAME="Llama-3.2-3B-Instruct"
MODEL_SHORT="llama32_3b"
BASE_MODEL_PATH="unsloth/Llama-3.2-3B-Instruct"

# ============================================
# Path Configuration
# ============================================
FINETUNE_DIR="saves/finetune/${MODEL_SHORT}_tofu_1epoch"
RETRAIN_DIR="saves/finetune/${MODEL_SHORT}_tofu_retrain"
TRAINING_LOG_DIR="saves/train_logs/${MODEL_SHORT}_tofu_1epoch"
UNLEARN_BASE_DIR="saves/unlearn"
RESULTS_DIR="saves/results/expA"

# ============================================
# Training Configuration
# ============================================
NUM_EPOCHS=1
# 3B bf16 u[t] ~6GB/step. batch=8*accum=8=64 -> 63 steps -> ~378GB total
PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
STEPS_PER_EPOCH=63  # 4000 samples / effective_batch_64
LEARNING_RATE="1e-3"
WEIGHT_DECAY="0.01"
OPTIMIZER="sgd"
WARMUP_EPOCHS="0.1"  # 10% warmup for single-epoch training

# ============================================
# Data Splits
# ============================================
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
FORGET_INDICES_FILE="configs/forget_indices/forget10.json"

# ============================================
# LMCleaner Parameters
# ============================================
K=50
HESSIAN_MODE="fisher"
DAMPING="0.0001"

# ============================================
# Post-Finetune Parameters (retain-set recovery)
# ============================================
POSTFT_LEARNING_RATE="1e-5"
POSTFT_EPOCHS=1
POSTFT_WARMUP_EPOCHS="0.1"

# ============================================
# PDU Parameters
# ============================================
PDU_RETAIN_LOSS_EPS="0.3"
PDU_DUAL_STEP_SIZE="1.0"
PDU_DUAL_WARMUP_EPOCHS="0"

# ============================================
# Baseline Methods
# ============================================
BASELINE_METHODS=("GradDiff" "NPO" "PDU" "UNDIAL")

# ============================================
# Seed
# ============================================
SEED=${SEED:-0}

# ============================================
# Environment
# ============================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1
PYTHON_CMD="python"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"  # sdpa for B200, flash_attention_2 for A100/H100

# ============================================
# Helper Functions
# ============================================
get_unlearn_task_name() {
    local method=$1
    local seed=${2:-$SEED}
    echo "expA_${method}_s${seed}"
}

get_unlearn_output_dir() {
    local method=$1
    local seed=${2:-$SEED}
    echo "${UNLEARN_BASE_DIR}/$(get_unlearn_task_name $method $seed)"
}

get_postft_task_name() {
    local method=$1
    local seed=${2:-$SEED}
    echo "expA_${method}_postft_s${seed}"
}

get_postft_output_dir() {
    local method=$1
    local seed=${2:-$SEED}
    echo "saves/finetune/$(get_postft_task_name $method $seed)"
}

# Post-finetune an unlearned model on retain set
# Usage: run_postft <method_name> <unlearned_model_dir>
run_postft() {
    local method=$1
    local model_dir=$2
    local task_name=$(get_postft_task_name "$method")
    local output_dir=$(get_postft_output_dir "$method")

    echo "----------------------------------------------"
    echo "Post-finetune: ${task_name}"
    echo "Source: ${model_dir}"
    echo "Output: ${output_dir}"
    echo "----------------------------------------------"

    if [ ! -d "${model_dir}" ]; then
        echo "WARNING: Model not found: ${model_dir}. Skipping post-finetune."
        return
    fi

    $PYTHON_CMD src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default \
        task_name="${task_name}" \
        model="${MODEL_NAME}" \
        model.model_args.pretrained_model_name_or_path="${model_dir}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        model.model_args.attn_implementation=sdpa \
        'data.train.TOFU_QA_full.args.hf_args.name=retain90' \
        trainer.args.num_train_epochs=${POSTFT_EPOCHS} \
        trainer.args.learning_rate=${POSTFT_LEARNING_RATE} \
        trainer.args.weight_decay=${WEIGHT_DECAY} \
        trainer.args.warmup_epochs=${POSTFT_WARMUP_EPOCHS} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.bf16=true \
        ++trainer.args.save_strategy=epoch

    echo "Post-finetune complete: ${task_name}"
}

print_config() {
    echo "=============================================="
    echo "Experiment A Configuration"
    echo "=============================================="
    echo "Model: ${MODEL_NAME} (${MODEL_SHORT})"
    echo "Base model: ${BASE_MODEL_PATH}"
    echo "Finetune dir: ${FINETUNE_DIR}"
    echo "Retrain dir: ${RETRAIN_DIR}"
    echo "Training logs: ${TRAINING_LOG_DIR}"
    echo "Epochs: ${NUM_EPOCHS}"
    echo "Steps/epoch: ${STEPS_PER_EPOCH}"
    echo "Effective batch: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
    echo "Warmup: ${WARMUP_EPOCHS} epochs"
    echo "Splits: ${FORGET_SPLIT}/${RETAIN_SPLIT}"
    echo "LMCleaner K: ${K}"
    echo "Seed: ${SEED}"
    echo "=============================================="
}
