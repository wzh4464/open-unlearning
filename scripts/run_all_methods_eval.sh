#!/bin/bash
# Run all unlearning methods and evaluate ALL metrics (including MIA) for Phi-3.5
# Also tracks efficiency metrics (time, GPU memory, throughput)
# Usage: CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_all_methods_eval.sh

set -e
set -o pipefail
source "$(dirname "$0")/env.sh"

# ============================================
# Configuration
# ============================================
MODEL="Phi-3.5-mini-instruct"
FINETUNED_MODEL="saves/finetune/phi35_tofu_finetune_with_log"
TRAINING_LOG_DIR="saves/train_logs/phi35_tofu_finetune_with_log"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"

# Skip options
SKIP_BASELINE=true  # Set to true to skip baseline evaluation
SKIP_METHODS=(
    "GradAscent"
    # "GradDiff"
    # "NPO"
)
SKIP_UNLEARN=(  # Skip only unlearn phase, still run eval
    "GradDiff"
)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1
export MASTER_PORT=$($PYTHON_CMD -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# All unlearning methods
METHODS=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "DPO"
    "RMU"
    "UNDIAL"
    "LMCleanerBatch"
)

RESULTS_DIR="saves/results_phi35"
mkdir -p "$RESULTS_DIR"

# ============================================
# Logging setup
# ============================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${RESULTS_DIR}/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Log function: runs command and logs stdout/stderr separately
run_with_log() {
    local name="$1"
    shift
    local stdout_log="${LOG_DIR}/${name}.stdout.log"
    local stderr_log="${LOG_DIR}/${name}.stderr.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $name"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] stdout -> $stdout_log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] stderr -> $stderr_log"

    # Run command, tee stdout and stderr separately, preserve exit code
    { "$@" 2> >(tee "$stderr_log" >&2); } > >(tee "$stdout_log")
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $name failed with exit code $exit_code"
        echo "Check logs: $stderr_log"
        exit $exit_code
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $name"
}

echo "============================================"
echo "Running all unlearning methods for Phi-3.5"
echo "With ALL metrics (including MIA) + Efficiency tracking"
echo "============================================"
echo "Finetuned model: $FINETUNED_MODEL"
echo "Training logs: $TRAINING_LOG_DIR"
echo "Forget split: $FORGET_SPLIT"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================
# Step 1: Evaluate finetuned model (baseline)
# ============================================
if [ "$SKIP_BASELINE" = true ]; then
    echo ">>> Step 1: Skipping baseline evaluation (SKIP_BASELINE=true)"
else
    echo ">>> Step 1: Evaluating finetuned model (baseline)..."
    TASK_NAME="phi35_finetuned_eval"

    run_with_log "${TASK_NAME}" \
        $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        eval=tofu_full \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
        model.tokenizer_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        task_name=${TASK_NAME}

    echo "Baseline evaluation complete."
fi
echo ""

# Helper function to check if method should be skipped
should_skip_method() {
    local method="$1"
    for skip in "${SKIP_METHODS[@]}"; do
        if [ "$skip" == "$method" ]; then
            return 0
        fi
    done
    return 1
}

# Helper function to check if unlearn phase should be skipped
should_skip_unlearn() {
    local method="$1"
    for skip in "${SKIP_UNLEARN[@]}"; do
        if [ "$skip" == "$method" ]; then
            return 0
        fi
    done
    return 1
}

# ============================================
# Step 2: Run each unlearning method
# ============================================
for METHOD in "${METHODS[@]}"; do
    echo "============================================"
    echo ">>> Running method: $METHOD"
    echo "============================================"

    # Check if method should be skipped
    if should_skip_method "$METHOD"; then
        echo "Skipping $METHOD (in SKIP_METHODS)"
        continue
    fi

    TASK_NAME="phi35_${METHOD}_${FORGET_SPLIT}"
    UNLEARN_OUTPUT="saves/unlearn/${TASK_NAME}"

    # Skip if already completed
    if [ -f "${UNLEARN_OUTPUT}/TOFU_EVAL.json" ]; then
        echo "Skipping $METHOD - already completed"
        continue
    fi

    # Check if unlearn phase should be skipped
    if should_skip_unlearn "$METHOD"; then
        echo "Skipping unlearn phase for $METHOD (in SKIP_UNLEARN)"
    else
        # Run unlearning with efficiency tracking
        echo "Running unlearning with efficiency tracking..."

        if [ "$METHOD" == "LMCleanerBatch" ] || [ "$METHOD" == "LMCleanerSample" ]; then
            # LMCleaner needs training_log_dir
            run_with_log "${TASK_NAME}_unlearn" \
                $ACCELERATE_CMD launch \
                --config_file configs/accelerate/default_config.yaml \
                --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default \
                trainer=${METHOD} \
                model=${MODEL} \
                model.model_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
                model.tokenizer_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
                forget_split=${FORGET_SPLIT} \
                retain_split=${RETAIN_SPLIT} \
                trainer.method_args.training_log_dir=${TRAINING_LOG_DIR} \
                trainer.args.per_device_train_batch_size=2 \
                trainer.args.gradient_accumulation_steps=4 \
                trainer.args.gradient_checkpointing=true \
                ++trainer.args.bf16=true \
                +efficiency.enabled=true \
                task_name=${TASK_NAME}
        else
            # Standard unlearning methods
            run_with_log "${TASK_NAME}_unlearn" \
                $ACCELERATE_CMD launch \
                --config_file configs/accelerate/default_config.yaml \
                --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default \
                trainer=${METHOD} \
                model=${MODEL} \
                model.model_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
                model.tokenizer_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
                forget_split=${FORGET_SPLIT} \
                retain_split=${RETAIN_SPLIT} \
                trainer.args.per_device_train_batch_size=2 \
                trainer.args.gradient_accumulation_steps=4 \
                trainer.args.gradient_checkpointing=true \
                ++trainer.args.bf16=true \
                +efficiency.enabled=true \
                task_name=${TASK_NAME}
        fi

        # Update port for next run
        export MASTER_PORT=$($PYTHON_CMD -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    fi

    # Run evaluation with ALL metrics (including MIA)
    echo "Running full evaluation (including MIA)..."
    run_with_log "${TASK_NAME}_eval" \
        $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        eval=tofu_full \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${UNLEARN_OUTPUT} \
        model.tokenizer_args.pretrained_model_name_or_path=${UNLEARN_OUTPUT} \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        retain_logs_path=saves/eval/phi35_finetuned_eval/TOFU_EVAL.json \
        task_name=${TASK_NAME}_eval

    echo "Completed: $METHOD"
    echo ""
done

# ============================================
# Step 3: Summarize results
# ============================================
echo "============================================"
echo "All methods completed!"
echo "============================================"
echo ""
echo "Results saved in:"
echo "- Unlearning outputs: saves/unlearn/"
echo "- Evaluation results: saves/eval/"
echo "- Efficiency metrics: saves/unlearn/*/efficiency_metrics.json"
echo "- Logs: ${LOG_DIR}/"
echo "  - *.stdout.log: standard output"
echo "  - *.stderr.log: standard error"
echo ""
echo "Metrics included:"
echo "- Forget quality: Truth Ratio, Q_A_Prob, Q_A_ROUGE"
echo "- Model utility: MMLU scores"
echo "- Privacy: privleak, extraction_strength, exact_memorization"
echo "- MIA attacks: min_k, min_k++, loss, zlib, gradnorm"
echo "- Retain/RA/WF metrics"
echo "- Efficiency: time, GPU memory, throughput"
