#!/bin/bash
# Safe finetuning with frequent saves to prevent OOM
# Usage: bash scripts/finetune.sh --llama   # for Llama-3.2-1B
#        bash scripts/finetune.sh --phi     # for Phi-3.5-mini

set -e
source "$(dirname "$0")/env.sh"

# Parse model selection
MODEL=""
MODEL_NAME=""
RESUME_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --llama)
            MODEL="Llama-3.2-1B-Instruct"
            MODEL_NAME="llama32_1b"
            shift
            ;;
        --phi)
            MODEL="Phi-3.5-mini-instruct"
            MODEL_NAME="phi35"
            shift
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --llama | --phi [--resume <path|auto>]"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: Must specify --llama or --phi"
    echo "Usage: $0 --llama | --phi"
    exit 1
fi

NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TASK_NAME=${TASK_NAME:-"${MODEL_NAME}_tofu_safe"}
LOG_DIR=${LOG_DIR:-"saves/train_logs/${TASK_NAME}"}

# Build resume argument
RESUME_ARG=""
if [ -n "$RESUME_PATH" ]; then
    if [ "$RESUME_PATH" = "auto" ]; then
        # Auto-find the latest checkpoint
        LATEST_CKPT=$(ls -td saves/finetune/${TASK_NAME}/checkpoint-* 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_ARG="+trainer.args.resume_from_checkpoint=${LATEST_CKPT}"
            echo "Auto-resuming from: ${LATEST_CKPT}"
        else
            echo "Warning: No checkpoint found for auto-resume, starting fresh"
        fi
    else
        RESUME_ARG="+trainer.args.resume_from_checkpoint=${RESUME_PATH}"
        echo "Resuming from: ${RESUME_PATH}"
    fi
fi

# Calculate steps per epoch
DATASET_SIZE=4000
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH_SIZE))

echo "=== Safe Finetuning ${MODEL} ==="
echo "Epochs: ${NUM_EPOCHS}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Saving every 10 steps"
echo "Task name: ${TASK_NAME}"
echo ""

$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=${MODEL} \
    collator=DataCollatorForSupervisedDatasetwithIndex \
    +trainer.args.remove_unused_columns=False \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
    trainer.args.save_strategy=epoch \
    trainer.args.save_total_limit=5 \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${LOG_DIR}" \
    +trainer.args.training_logger.max_steps=10000 \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.sync_mode=true \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_rng_state=true \
    +trainer.args.training_logger.steps_per_epoch=${STEPS_PER_EPOCH} \
    +trainer.args.training_logger.save_at_epoch_end=true \
    +trainer.args.spectral_norm.enabled=true \
    +trainer.args.spectral_norm.interval=50 \
    +trainer.args.spectral_norm.num_power_iters=20 \
    +trainer.args.spectral_norm.output_dir="${LOG_DIR}" \
    ${RESUME_ARG:+$RESUME_ARG} \
    task_name="${TASK_NAME}"

echo ""
echo "=== Training complete ==="
echo "Logs: ${LOG_DIR}"
