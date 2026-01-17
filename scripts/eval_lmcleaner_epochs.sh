#!/bin/bash
# LMCleaner epoch-based evaluation
# Usage: bash scripts/eval_lmcleaner_epochs.sh

set -e
source "$(dirname "$0")/env.sh"

MODEL="Llama-3.2-1B-Instruct"
MODEL_NAME="llama32_1b"
TASK_NAME="${MODEL_NAME}_tofu_safe"
LOG_DIR="saves/train_logs/${TASK_NAME}"
CHECKPOINTS_DIR="saves/finetune/${TASK_NAME}"

# Forget/retain splits
FORGET_SPLIT=${FORGET_SPLIT:-"forget10"}
RETAIN_SPLIT=${RETAIN_SPLIT:-"retain90"}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-"holdout10"}

# LMCleaner params
K=${K:-800}
HESSIAN_MODE=${HESSIAN_MODE:-"GGN"}

# Dynamically read epoch boundaries from training meta.json
META_FILE="${LOG_DIR}/meta.json"
EPOCH_STEPS=()

if [ -f "${META_FILE}" ]; then
    # Read epoch_end_steps from meta.json
    mapfile -t EPOCH_STEPS < <(
        python3 -c "
import json
import sys
try:
    with open('${META_FILE}', 'r') as f:
        meta = json.load(f)
    for step in meta.get('epoch_end_steps', []):
        print(int(step))
except Exception as e:
    sys.exit(0)
"
    )
fi

# Fallback: derive steps from existing checkpoint directories
if [ ${#EPOCH_STEPS[@]} -eq 0 ]; then
    if compgen -G "${CHECKPOINTS_DIR}/checkpoint-*" > /dev/null 2>&1; then
        mapfile -t EPOCH_STEPS < <(
            for ckpt in "${CHECKPOINTS_DIR}"/checkpoint-*; do
                [ -d "${ckpt}" ] || continue
                name="$(basename "${ckpt}")"
                step="${name#checkpoint-}"
                # Only keep numeric step identifiers
                case "${step}" in
                    ''|*[!0-9]*) continue ;;
                    *) echo "${step}" ;;
                esac
            done | sort -n
        )
    fi
fi

if [ ${#EPOCH_STEPS[@]} -eq 0 ]; then
    echo "Error: Unable to determine epoch step boundaries."
    echo "Please ensure ${META_FILE} exists or checkpoints are present in ${CHECKPOINTS_DIR}."
    exit 1
fi

echo "=== LMCleaner Epoch Evaluation ==="
echo "Model: ${MODEL}"
echo "Training logs: ${LOG_DIR}"
echo "Checkpoints: ${CHECKPOINTS_DIR}"
echo "Epoch steps: ${EPOCH_STEPS[*]}"
echo ""

for i in "${!EPOCH_STEPS[@]}"; do
    step=${EPOCH_STEPS[$i]}
    epoch=$((i + 1))
    checkpoint_path="${CHECKPOINTS_DIR}/checkpoint-${step}"

    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "Warning: checkpoint-${step} not found, skipping epoch ${epoch}"
        continue
    fi

    UNLEARN_TASK="lmcleaner_${TASK_NAME}_epoch${epoch}"
    EVAL_TASK="${UNLEARN_TASK}_eval"

    echo "=== Epoch ${epoch} (step ${step}) ==="

    # Step 1: Run LMCleaner unlearning
    echo "Running LMCleaner unlearning..."
    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LMCleanerBatch \
        task_name=${UNLEARN_TASK} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${checkpoint_path} \
        trainer.method_args.training_log_dir=${LOG_DIR} \
        trainer.method_args.max_step=${step} \
        trainer.method_args.K=${K} \
        trainer.method_args.hessian_mode=${HESSIAN_MODE}

    # Step 2: Evaluate
    echo "Evaluating..."
    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        task_name=${EVAL_TASK} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${UNLEARN_TASK} \
        paths.output_dir=saves/unlearn/${UNLEARN_TASK}/evals

    echo "Epoch ${epoch} complete. Results: saves/unlearn/${UNLEARN_TASK}/evals/"
    echo ""
done

echo "=== All epochs evaluated ==="
echo "Compare results in saves/unlearn/lmcleaner_*/evals/TOFU_EVAL.json"
