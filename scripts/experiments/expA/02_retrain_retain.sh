#!/bin/bash
# Experiment A - Step 2: Retrain on retain90 only (gold standard)
# Finetunes base model on retain set only, simulating perfect unlearning.
# NOTE: We reuse the finetune/tofu/default experiment config but override
# the dataset split from "full" to "retain90" via the TOFU_QA_full key.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 2: Retrain ${MODEL_NAME} on retain90 only"
echo "=============================================="
print_config

# Verify base model (skip check for HF model IDs like org/name)
if [[ "${BASE_MODEL_PATH}" != */* ]] && [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "ERROR: Base model not found at ${BASE_MODEL_PATH}"
    exit 1
fi

TASK_NAME="${MODEL_SHORT}_tofu_retrain"

echo "Starting retrain (retain90 only)..."
echo "Task: ${TASK_NAME}"
echo "Output: ${RETRAIN_DIR}"

$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    model.model_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    'data.train.TOFU_QA_full.args.hf_args.name=retain90' \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.seed=${SEED} \
    ++trainer.args.optim=${OPTIMIZER} \
    ++trainer.args.bf16=true \
    ++trainer.args.save_strategy=epoch \
    ++trainer.args.efficiency_tracking.enabled=true

echo ""
echo "Retrain complete!"
echo "Model: ${RETRAIN_DIR}"
