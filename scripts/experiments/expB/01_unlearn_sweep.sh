#!/bin/bash
# RebuttalB Step 1: Run LMCleaner unlearning for all valid (K, B) combinations
#
# Usage:
#   bash scripts/experiments/expB/01_unlearn_sweep.sh          # all (K,B)
#   bash scripts/experiments/expB/01_unlearn_sweep.sh --B 64   # single B
#   bash scripts/experiments/expB/01_unlearn_sweep.sh --K 50   # single K

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../../env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Parse optional filters
FILTER_B=""
FILTER_K=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --B) FILTER_B="$2"; shift 2 ;;
        --K) FILTER_K="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

B_LIST=(8 16 32 64 128 256)
[ -n "$FILTER_B" ] && B_LIST=($FILTER_B)

K_LIST=("${K_VALUES[@]}")
[ -n "$FILTER_K" ] && K_LIST=($FILTER_K)

echo "=== RebuttalB: LMCleaner (K,B) Sweep ==="
echo "B values: ${B_LIST[*]}"
echo "K values: ${K_LIST[*]}"
echo ""

TOTAL=0
SKIPPED=0
FAILED=0

for B in "${B_LIST[@]}"; do
    TRAIN_LOG="${TRAIN_LOG_DIRS[$B]}"
    MODEL_PATH="${MODEL_DIRS[$B]}"
    MAX_STEP="${STEPS_PER_EPOCH[$B]}"

    if [ ! -d "$TRAIN_LOG" ]; then
        echo "ERROR: training logs not found for B=$B: $TRAIN_LOG"
        continue
    fi
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: model not found for B=$B: $MODEL_PATH"
        continue
    fi

    for K in "${K_LIST[@]}"; do
        # Check K <= max steps for this B
        if ! valid_k_for_b "$B" "$K"; then
            echo "SKIP: B=$B K=$K (K > max_step=$MAX_STEP)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        TASK="expB_B${B}_K${K}"
        OUT_DIR="${SAVES_BASE}/unlearn/${TASK}"

        # Skip if already completed
        if [ -f "$OUT_DIR/efficiency_metrics.json" ]; then
            echo "SKIP: $TASK (already done)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo ""
        echo "====== B=$B K=$K (max_step=$MAX_STEP) ======"

        $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=LMCleanerBatch \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
            model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}" \
            forget_split=${FORGET_SPLIT} \
            retain_split=${RETAIN_SPLIT} \
            holdout_split=${HOLDOUT_SPLIT} \
            trainer.method_args.training_log_dir="${TRAIN_LOG}" \
            trainer.method_args.K=${K} \
            trainer.method_args.max_step=${MAX_STEP} \
            trainer.method_args.hessian_mode=${HESSIAN_MODE} \
            trainer.method_args.damping=${DAMPING} \
            trainer.args.num_train_epochs=0 \
            paths.output_dir="${OUT_DIR}" \
            task_name="${TASK}" \
            2>&1 | tee "${OUT_DIR}.log" || {
                echo "FAILED: $TASK"
                FAILED=$((FAILED + 1))
                continue
            }

        TOTAL=$((TOTAL + 1))
        echo "====== $TASK done ======"
    done
done

echo ""
echo "=== Sweep Complete ==="
echo "Completed: $TOTAL"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
