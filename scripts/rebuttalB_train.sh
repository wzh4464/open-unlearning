#!/bin/bash
# RebuttalB: Train with different effective batch sizes B
# Each B produces a separate training trajectory for (K,B) ablation
#
# Usage:
#   bash scripts/rebuttalB_train.sh --batch-size 64
#   bash scripts/rebuttalB_train.sh --batch-size 128
#   bash scripts/rebuttalB_train.sh --batch-size 256
#   bash scripts/rebuttalB_train.sh --batch-size 512
#
# Or run all:
#   for B in 64 128 256 512; do bash scripts/rebuttalB_train.sh --batch-size $B; done

set -e
source "$(dirname "$0")/env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Defaults
EFFECTIVE_B=${EFFECTIVE_B:-256}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SEED=${SEED:-0}
LR=${LR:-1e-5}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size|-B) EFFECTIVE_B="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Determine micro-batch size and gradient accumulation
# H200 can fit ~64 samples of seq_len=512 for Llama-3.2-1B in bf16
MAX_MICRO_BATCH=64

if [ "$EFFECTIVE_B" -le "$MAX_MICRO_BATCH" ]; then
    MICRO_BATCH=$EFFECTIVE_B
    GRAD_ACCUM=1
else
    MICRO_BATCH=$MAX_MICRO_BATCH
    GRAD_ACCUM=$((EFFECTIVE_B / MICRO_BATCH))
fi

TASK_NAME="rebuttalB_B${EFFECTIVE_B}_seed${SEED}"
LOG_DIR="saves/train_logs/${TASK_NAME}"

DATASET_SIZE=4000
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_B))

echo "=== RebuttalB Training ==="
echo "Effective B: ${EFFECTIVE_B} (micro=${MICRO_BATCH}, accum=${GRAD_ACCUM})"
echo "Epochs: ${NUM_EPOCHS}, LR: ${LR}, Seed: ${SEED}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Task: ${TASK_NAME}"
echo ""

$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=Llama-3.2-1B-Instruct \
    collator=DataCollatorForSupervisedDatasetwithIndex \
    +trainer.args.remove_unused_columns=False \
    trainer.args.optim=adamw_torch \
    trainer.args.learning_rate=${LR} \
    trainer.args.weight_decay=0.01 \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.per_device_train_batch_size=${MICRO_BATCH} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
    trainer.args.save_strategy=epoch \
    trainer.args.save_total_limit=5 \
    trainer.args.seed=${SEED} \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${LOG_DIR}" \
    +trainer.args.training_logger.max_steps=10000 \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.sync_mode=true \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_rng_state=true \
    +trainer.args.training_logger.steps_per_epoch=${STEPS_PER_EPOCH} \
    +trainer.args.training_logger.save_at_epoch_end=true \
    task_name="${TASK_NAME}"

echo ""
echo "=== Training complete ==="
echo "Model: saves/finetune/${TASK_NAME}"
echo "Logs: ${LOG_DIR}"
