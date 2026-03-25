#!/bin/bash
# Experiment A - Step 4: Baseline Unlearning Methods
# Runs: GradDiff, NPO, PDU, UNDIAL
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_ROOT}"

echo "=============================================="
echo "ExpA Step 4: Baseline Unlearning Methods"
echo "=============================================="
print_config
echo "Methods: ${BASELINE_METHODS[*]}"

# Verify finetuned model exists
if [ ! -d "${FINETUNE_DIR}" ]; then
    echo "ERROR: Finetuned model not found at ${FINETUNE_DIR}"
    echo "Run 01_finetune_full.sh first."
    exit 1
fi

for METHOD in "${BASELINE_METHODS[@]}"; do
    METHOD_LOWER="${METHOD,,}"
    TASK_NAME=$(get_unlearn_task_name "${METHOD_LOWER}")
    OUTPUT_DIR=$(get_unlearn_output_dir "${METHOD_LOWER}")

    echo ""
    echo "=============================================="
    echo "Running ${METHOD}"
    echo "Task: ${TASK_NAME}"
    echo "Output: ${OUTPUT_DIR}"
    echo "=============================================="

    # Build method-specific overrides as an array (avoids whitespace issues)
    METHOD_OVERRIDES=()
    if [ "${METHOD}" = "PDU" ]; then
        METHOD_OVERRIDES=(
            "trainer.method_args.retain_loss_eps=${PDU_RETAIN_LOSS_EPS}"
            "trainer.method_args.dual_step_size=${PDU_DUAL_STEP_SIZE}"
            "trainer.method_args.dual_warmup_epochs=${PDU_DUAL_WARMUP_EPOCHS}"
        )
    fi

    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer="${METHOD}" \
        task_name="${TASK_NAME}" \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        model.model_args.pretrained_model_name_or_path="${FINETUNE_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.bf16=true \
        trainer.args.efficiency_tracking.enabled=true \
        "${METHOD_OVERRIDES[@]}"

    echo "${METHOD} complete!"
done

echo ""
echo "=============================================="
echo "All baselines complete!"
echo "=============================================="
