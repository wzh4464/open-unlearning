#!/bin/bash
# RebuttalB (K,B) ablation - shared configuration
# Source this file from all expB scripts

# === Model ===
MODEL="Llama-3.2-1B-Instruct"

# === Forget/Retain splits ===
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"

# === K values to sweep ===
K_VALUES=(10 20 30 40 50)

# === B values and their training log directories ===
# Each B has a pre-trained model checkpoint + training logs
declare -A TRAIN_LOG_DIRS
TRAIN_LOG_DIRS[8]="/workspace/saves/train_logs/rebuttalB_B8_seed0"
TRAIN_LOG_DIRS[16]="/workspace/saves/train_logs/llama32_1b_tofu_safe"
TRAIN_LOG_DIRS[32]="/workspace/saves/train_logs/rebuttalB_B32_seed0"
TRAIN_LOG_DIRS[64]="/workspace/saves/train_logs/rebuttalB_B64_seed0"
TRAIN_LOG_DIRS[128]="/workspace/saves/train_logs/rebuttalB_B128_seed0"
TRAIN_LOG_DIRS[256]="/workspace/saves/train_logs/rebuttalB_B256_seed0"

declare -A MODEL_DIRS
MODEL_DIRS[8]="/workspace/saves/finetune/rebuttalB_B8_seed0"
MODEL_DIRS[16]="/workspace/saves/finetune/llama32_1b_tofu_safe/checkpoint-250"
MODEL_DIRS[32]="/workspace/saves/finetune/rebuttalB_B32_seed0"
MODEL_DIRS[64]="/workspace/saves/finetune/rebuttalB_B64_seed0"
MODEL_DIRS[128]="/workspace/saves/finetune/rebuttalB_B128_seed0"
MODEL_DIRS[256]="/workspace/saves/finetune/rebuttalB_B256_seed0"

declare -A STEPS_PER_EPOCH
STEPS_PER_EPOCH[8]=500
STEPS_PER_EPOCH[16]=250
STEPS_PER_EPOCH[32]=125
STEPS_PER_EPOCH[64]=62
STEPS_PER_EPOCH[128]=31
STEPS_PER_EPOCH[256]=15

# === LMCleaner defaults ===
HESSIAN_MODE="GGN"
DAMPING="1e-4"

# === Output base ===
SAVES_BASE="/workspace/saves"

# === Helper ===
valid_k_for_b() {
    local B=$1 K=$2
    local max_k=${STEPS_PER_EPOCH[$B]}
    [ "$K" -le "$max_k" ]
}
