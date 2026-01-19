#!/bin/bash
# LMCleaner Unlearn + Eval on Multiple Checkpoints
#
# 对每个 epoch checkpoint 运行 LMCleaner unlearn 并评估
# Checkpoints: 250 (epoch 1), 500 (epoch 2), 750 (epoch 3), 1000 (epoch 4), 1250 (epoch 5)

set -e
source "$(dirname "$0")/env.sh"

# 配置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

MODEL_NAME="Llama-3.2-1B-Instruct"
TRAINING_LOG_DIR="saves/train_logs/llama32_1b_tofu_safe"
FINETUNE_BASE="saves/finetune/llama32_1b_tofu_safe"

# LMCleaner 参数
K=${K:-1000}
HESSIAN_MODE=${HESSIAN_MODE:-"GGN"}
DAMPING=${DAMPING:-0.0001}

# 数据切分
FORGET_SPLIT=${FORGET_SPLIT:-"forget10"}
RETAIN_SPLIT=${RETAIN_SPLIT:-"retain90"}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-"holdout10"}

# Checkpoints (epoch -> step)
CHECKPOINTS=(250 500 750 1000 1250)

echo "=============================================="
echo "LMCleaner Multi-Checkpoint Unlearn + Eval"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Forget split: ${FORGET_SPLIT}"
echo "K: ${K}, Hessian mode: ${HESSIAN_MODE}"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo ""

for STEP in "${CHECKPOINTS[@]}"; do
    EPOCH=$((STEP / 250))
    TASK_NAME="lmcleaner_llama32_epoch${EPOCH}_K${K}"
    CHECKPOINT_PATH="${FINETUNE_BASE}/checkpoint-${STEP}"
    OUTPUT_DIR="saves/unlearn/${TASK_NAME}"

    echo "=============================================="
    echo "Processing Epoch ${EPOCH} (Step ${STEP})"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    echo "=============================================="

    # 检查 checkpoint 是否存在
    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "Warning: Checkpoint not found: ${CHECKPOINT_PATH}"
        echo "Skipping..."
        continue
    fi

    # Step 1: LMCleaner Unlearn
    echo ""
    echo "[1/2] Running LMCleaner Unlearn..."
    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LMCleanerBatch \
        task_name="${TASK_NAME}" \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        model.model_args.pretrained_model_name_or_path="${CHECKPOINT_PATH}" \
        trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
        trainer.method_args.K="${K}" \
        trainer.method_args.max_step="${STEP}" \
        trainer.method_args.hessian_mode="${HESSIAN_MODE}" \
        trainer.method_args.damping="${DAMPING}" \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.gradient_checkpointing=true \
        ++trainer.args.bf16=true

    # Step 2: Evaluation
    echo ""
    echo "[2/2] Running Evaluation..."
    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${TASK_NAME}_eval" \
        model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}" \
        paths.output_dir="${OUTPUT_DIR}/evals"

    echo ""
    echo "Completed Epoch ${EPOCH}!"
    echo ""
done

echo "=============================================="
echo "All checkpoints processed!"
echo "=============================================="
echo ""
echo "Results saved in:"
for STEP in "${CHECKPOINTS[@]}"; do
    EPOCH=$((STEP / 250))
    echo "  - saves/unlearn/lmcleaner_llama32_epoch${EPOCH}_K${K}/evals/"
done
