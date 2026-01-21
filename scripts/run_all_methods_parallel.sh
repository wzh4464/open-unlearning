#!/bin/bash
# Run all unlearning methods in parallel using 4 GPUs

set -e

# Configuration
MODEL="Llama-3.2-1B-Instruct"
MODEL_SHORT="llama32"
FINETUNE_DIR="saves/finetune/llama32_1b_tofu_safe"
TRAINING_LOG_DIR="saves/train_logs/llama32_1b_tofu_safe"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"

# Epochs and their checkpoints
declare -A EPOCH_CHECKPOINTS=(
    [1]=250
    [2]=500
    [3]=750
    [4]=1000
    [5]=1250
)

# Methods that don't need special config
SIMPLE_METHODS="GradAscent GradDiff NPO CEU DPO RMU SatImp SimNPO UNDIAL WGA"

# LMCleaner K values
LMCLEANER_K_VALUES="50 100 500 1000 1250"

# Generate job list
JOB_FILE="/tmp/unlearn_jobs.txt"
> "$JOB_FILE"

echo "Generating job list..."

# Check existing experiments
check_exists() {
    local name="$1"
    if [[ -f "saves/unlearn/${name}/evals/TOFU_SUMMARY.json" ]]; then
        return 0  # exists
    fi
    return 1  # does not exist
}

# Simple methods (epochs 1-5)
for method in $SIMPLE_METHODS; do
    for epoch in 1 2 3 4 5; do
        ckpt=${EPOCH_CHECKPOINTS[$epoch]}
        exp_name="${method,,}_${MODEL_SHORT}_epoch${epoch}"

        if check_exists "$exp_name"; then
            echo "SKIP: $exp_name (already exists)"
            continue
        fi

        echo "$method $epoch $ckpt $exp_name" >> "$JOB_FILE"
    done
done

# LMCleaner with different K values
for K in $LMCLEANER_K_VALUES; do
    for epoch in 1 2 3 4 5; do
        ckpt=${EPOCH_CHECKPOINTS[$epoch]}
        exp_name="lmcleaner_${MODEL_SHORT}_epoch${epoch}_K${K}"

        if check_exists "$exp_name"; then
            echo "SKIP: $exp_name (already exists)"
            continue
        fi

        echo "LMCleanerBatch $epoch $ckpt $exp_name $K" >> "$JOB_FILE"
    done
done

echo ""
echo "Jobs to run:"
cat "$JOB_FILE"
echo ""
echo "Total jobs: $(wc -l < "$JOB_FILE")"
echo ""

# Function to run a single job
run_job() {
    local gpu=$1
    local method=$2
    local epoch=$3
    local ckpt=$4
    local exp_name=$5
    local K=$6

    local model_path="saves/finetune/llama32_1b_tofu_safe/checkpoint-${ckpt}"
    local log_file="saves/unlearn/${exp_name}.log"

    echo "[GPU $gpu] Starting: $exp_name"

    if [[ "$method" == "LMCleanerBatch" ]]; then
        CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
            --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path="$model_path" \
            trainer=LMCleanerBatch \
            trainer.method_args.training_log_dir="saves/train_logs/llama32_1b_tofu_safe" \
            trainer.method_args.K=$K \
            forget_split=forget10 \
            retain_split=retain90 \
            task_name="$exp_name" \
            > "$log_file" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
            --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path="$model_path" \
            trainer=$method \
            forget_split=forget10 \
            retain_split=retain90 \
            task_name="$exp_name" \
            > "$log_file" 2>&1
    fi

    local status=$?
    if [[ $status -eq 0 ]]; then
        echo "[GPU $gpu] Completed: $exp_name"
    else
        echo "[GPU $gpu] FAILED: $exp_name (exit code: $status)"
    fi
    return $status
}

export -f run_job

# Run jobs in parallel on 4 GPUs
echo "Starting parallel execution on 4 GPUs..."
cat "$JOB_FILE" | parallel --jobs 4 --colsep ' ' \
    'gpu=$(( ({#} - 1) % 4 )); run_job $gpu {1} {2} {3} {4} {5}'

echo ""
echo "All jobs completed!"
