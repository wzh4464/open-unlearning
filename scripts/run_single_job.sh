#!/bin/bash
# Single job runner for GNU Parallel
# Usage: ./run_single_job.sh <gpu_id> <method> <epoch_name> <max_step> <k_value>

set -e

GPU=$1
METHOD=$2
EPOCH_NAME=$3
MAX_STEP=$4
K_VALUE=$5

MODEL_PATH="saves/finetune/tofu_finetune_gpt2"
TRAINING_LOG_DIR="saves/train_logs/tofu_finetune_gpt2"
OUTPUT_DIR="saves/unlearn/tofu_gpt2_eval"

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p "$OUTPUT_DIR"

if [[ "$METHOD" == "LMCleanerBatch" ]]; then
    TASK_NAME="gpt2_LMCleaner_${EPOCH_NAME}_K${K_VALUE}"
    echo "[GPU $GPU] LMCleaner K=$K_VALUE $EPOCH_NAME (step=$MAX_STEP)"

    # Unlearn
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        model=gpt2 \
        model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
        model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
        trainer=LMCleanerBatch \
        trainer.method_args.training_log_dir="$TRAINING_LOG_DIR" \
        trainer.method_args.K=$K_VALUE \
        trainer.method_args.max_step=$MAX_STEP \
        trainer.method_args.apply_immediately=true \
        trainer.args.num_train_epochs=0 \
        task_name="$TASK_NAME" \
        2>&1 | tee "$OUTPUT_DIR/${TASK_NAME}_unlearn.log"

    # Eval
    python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=gpt2 \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/$TASK_NAME" \
        model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/$TASK_NAME" \
        task_name="${TASK_NAME}_eval" \
        2>&1 | tee "$OUTPUT_DIR/${TASK_NAME}_eval.log"
else
    TASK_NAME="gpt2_${METHOD}"
    echo "[GPU $GPU] $METHOD"

    # Unlearn
    python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        model=gpt2 \
        model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
        model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
        trainer=$METHOD \
        trainer.args.num_train_epochs=5 \
        task_name="$TASK_NAME" \
        2>&1 | tee "$OUTPUT_DIR/${TASK_NAME}_unlearn.log"

    # Eval
    python src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=gpt2 \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/$TASK_NAME" \
        model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/$TASK_NAME" \
        task_name="${TASK_NAME}_eval" \
        2>&1 | tee "$OUTPUT_DIR/${TASK_NAME}_eval.log"
fi

echo "[GPU $GPU] Done: $TASK_NAME"
