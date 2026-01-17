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

# Epoch boundaries (from training meta.json)
EPOCH_STEPS=(63 126 189 252 315)

echo "=== LMCleaner Epoch Evaluation ==="
echo "Model: ${MODEL}"
echo "Training logs: ${LOG_DIR}"
echo "Checkpoints: ${CHECKPOINTS_DIR}"
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
