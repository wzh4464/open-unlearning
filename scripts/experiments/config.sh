#!/bin/bash
# Shared configuration for experiment scripts
# Modify this file to switch models or change experiment settings

# ============================================
# Model Configuration - Modify here to switch models
# ============================================
MODEL_NAME="Llama-3.2-1B-Instruct"
MODEL_SHORT="llama32_1b"

# Paths - these are derived from MODEL_SHORT
FINETUNE_DIR="saves/finetune/${MODEL_SHORT}_tofu_safe"
TRAINING_LOG_DIR="saves/train_logs/${MODEL_SHORT}_tofu_safe"

# ============================================
# Experiment Configuration
# ============================================
# Epoch to checkpoint step mapping (250 steps per epoch)
EPOCHS=(1 2 3 4 5)
CHECKPOINTS=(250 500 750 1000 1250)

# Data splits
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"

# ============================================
# LMCleaner Parameters
# ============================================
K=${K:-1000}
HESSIAN_MODE=${HESSIAN_MODE:-"GGN"}
DAMPING=${DAMPING:-0.0001}

# ============================================
# Baseline Methods
# ============================================
BASELINE_METHODS=("GradDiff" "NPO")

# ============================================
# Training Parameters
# ============================================
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

# ============================================
# Environment Settings
# ============================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1

# Python command (docker vs local)
if [ -n "$IN_DOCKER" ]; then
    PYTHON_CMD="python"
    ACCELERATE_CMD="accelerate"
else
    PYTHON_CMD="python"
    ACCELERATE_CMD="accelerate"
fi

# ============================================
# Helper Functions
# ============================================

# Get checkpoint path for a given epoch
get_checkpoint_path() {
    local epoch=$1
    local step=$((epoch * 250))
    echo "${FINETUNE_DIR}/checkpoint-${step}"
}

# Get task name for LMCleaner
get_lmcleaner_task_name() {
    local epoch=$1
    echo "lmcleaner_${MODEL_SHORT}_epoch${epoch}_K${K}"
}

# Get task name for baseline
get_baseline_task_name() {
    local method=$1
    local epoch=$2
    echo "baseline_${MODEL_SHORT}_epoch${epoch}_${method}"
}

# Print configuration summary
print_config() {
    echo "=============================================="
    echo "Experiment Configuration"
    echo "=============================================="
    echo "Model: ${MODEL_NAME} (${MODEL_SHORT})"
    echo "Finetune dir: ${FINETUNE_DIR}"
    echo "Training log dir: ${TRAINING_LOG_DIR}"
    echo "Epochs: ${EPOCHS[*]}"
    echo "Checkpoints: ${CHECKPOINTS[*]}"
    echo "Forget split: ${FORGET_SPLIT}"
    echo "Retain split: ${RETAIN_SPLIT}"
    echo "Holdout split: ${HOLDOUT_SPLIT}"
    echo "LMCleaner K: ${K}"
    echo "Hessian mode: ${HESSIAN_MODE}"
    echo "=============================================="
}
