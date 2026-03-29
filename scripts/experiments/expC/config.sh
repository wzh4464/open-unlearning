#!/bin/bash
# Experiment C: Influence Removal vs Noise Injection Ablation
# Shared configuration

MODEL="Llama-3.2-1B-Instruct"
MODEL_SHORT="llama32_1b"
BASE_MODEL_PATH="unsloth/Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout01"

# Training config (1 epoch, SGD)
NUM_EPOCHS=1
PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE="1e-3"
WEIGHT_DECAY="0.01"
OPTIMIZER="sgd"
WARMUP_EPOCHS="0.1"
SEED=${SEED:-0}

# Paths
FINETUNE_DIR="saves/finetune/${MODEL_SHORT}_tofu_1epoch"
RETRAIN_DIR="saves/finetune/${MODEL_SHORT}_tofu_retrain_f01"
TRAIN_LOG_DIR="saves/train_logs/${MODEL_SHORT}_tofu_1epoch"
MODEL_DIR="${FINETUNE_DIR}"
MAX_STEP=63  # 4000/(8*8) = 63 optimizer steps for 1 epoch

# Fixed LMCleaner params
K=10
HESSIAN_MODE="fisher"
DAMPING="1e-4"

# Paper-faithful noise params
DELTA_CERT=0.03  # Public bound: max ||u[t]|| = 0.0275 for B=16, rounded up
BETA=0.1         # Concentration factor
PROJ_RANK=10     # Π_k rank (k=10 → ~49GB RAM for 1.24B params)
PROJ_SEED=42     # Public randomness seed

# Privacy defaults
DEFAULT_EPSILON=1.0
DEFAULT_DELTA=1e-5

# Epsilon sweep values
EPSILON_VALUES=(0.25 0.5 1.0 2.0 4.0)

# Seeds for multi-seed runs
SEEDS=(0)

SAVES_BASE="saves"
PYTHON_CMD="python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
