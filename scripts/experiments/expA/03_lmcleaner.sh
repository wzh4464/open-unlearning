#!/bin/bash
# Experiment A - Step 3: LMCleaner Batch-Level Unlearning
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_ROOT}"

TASK_NAME=$(get_unlearn_task_name "lmcleaner")
OUTPUT_DIR=$(get_unlearn_output_dir "lmcleaner")

echo "=============================================="
echo "ExpA Step 3: LMCleaner Unlearn"
echo "=============================================="
print_config
echo "K: ${K}, Hessian: ${HESSIAN_MODE}, Damping: ${DAMPING}"
echo "Output: ${OUTPUT_DIR}"

# Verify finetuned model exists
if [ ! -d "${FINETUNE_DIR}" ]; then
    echo "ERROR: Finetuned model not found at ${FINETUNE_DIR}"
    echo "Run 01_finetune_full.sh first."
    exit 1
fi

# Verify training logs exist
if [ ! -d "${TRAINING_LOG_DIR}" ]; then
    echo "ERROR: Training logs not found at ${TRAINING_LOG_DIR}"
    exit 1
fi

echo "Running LMCleaner Unlearn..."
$PYTHON_CMD src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${FINETUNE_DIR}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
    trainer.method_args.K=${K} \
    trainer.method_args.max_step=${STEPS_PER_EPOCH} \
    trainer.method_args.hessian_mode="${HESSIAN_MODE}" \
    trainer.method_args.damping=${DAMPING} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.seed=${SEED} \
    ++trainer.args.bf16=true \
    trainer.args.efficiency_tracking.enabled=true \
    model.model_args.attn_implementation=eager \
    trainer.method_args.use_historical_params=false

echo ""
echo "LMCleaner unlearning complete!"
echo "Model: ${OUTPUT_DIR}"
