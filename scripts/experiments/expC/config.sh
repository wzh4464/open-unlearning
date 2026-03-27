#!/bin/bash
# Experiment C: Influence Removal vs Noise Injection Ablation
# Shared configuration

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
HOLDOUT_SPLIT="holdout10"

# Use B=16 training (existing, well-validated)
TRAIN_LOG_DIR="/workspace/saves/train_logs/llama32_1b_tofu_safe"
MODEL_DIR="/workspace/saves/finetune/llama32_1b_tofu_safe/checkpoint-250"
MAX_STEP=250

# Fixed LMCleaner params
K=50
HESSIAN_MODE="GGN"
DAMPING="1e-4"

# Paper-faithful noise params
DELTA_CERT=0.03  # Public bound: max ||u[t]|| = 0.0275 for B=16, rounded up
BETA=0.1         # Concentration factor
PROJ_RANK=100    # Π_k rank
PROJ_SEED=42     # Public randomness seed

# Privacy defaults
DEFAULT_EPSILON=1.0
DEFAULT_DELTA=1e-5

# Epsilon sweep values
EPSILON_VALUES=(0.25 0.5 1.0 2.0 4.0)

# Seeds for multi-seed runs
SEEDS=(0 1 2)

SAVES_BASE="/workspace/saves"
