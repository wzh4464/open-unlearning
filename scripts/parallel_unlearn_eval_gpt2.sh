#!/bin/bash
# Parallel unlearning and evaluation script for GPT-2 TOFU experiment
# Runs multiple methods across 4 GPUs

set -e

# Configuration
MODEL_PATH="saves/finetune/tofu_finetune_gpt2"
TRAINING_LOG_DIR="saves/train_logs/tofu_finetune_gpt2"
BASE_OUTPUT_DIR="saves/unlearn/tofu_gpt2_eval"

# Epochs (step numbers at end of each epoch)
EPOCHS=(125 250 375 500 625)
EPOCH_NAMES=(epoch1 epoch2 epoch3 epoch4 epoch5)

# LMCleaner K values
K_VALUES=(10 50 100 250 500)

# Other unlearning methods to test
OTHER_METHODS=(GradAscent GradDiff NPO SimNPO)

# GPU assignment - use environment variable or default to 4
NUM_GPUS=${NUM_GPUS:-4}

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Function to run unlearn + eval
run_unlearn_eval() {
    local gpu=$1
    local method=$2
    local epoch_step=$3
    local epoch_name=$4
    local k_value=$5
    local task_name=$6

    export CUDA_VISIBLE_DEVICES=$gpu

    if [[ "$method" == "LMCleanerBatch" ]]; then
        # LMCleaner: run unlearn with specific K and max_step, then eval
        echo "[GPU $gpu] Running LMCleaner K=$k_value epoch=$epoch_name (step=$epoch_step)"

        python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            model=gpt2 \
            model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
            model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
            trainer=LMCleanerBatch \
            trainer.method_args.training_log_dir="$TRAINING_LOG_DIR" \
            trainer.method_args.K=$k_value \
            trainer.method_args.max_step=$epoch_step \
            trainer.method_args.apply_immediately=true \
            trainer.args.num_train_epochs=0 \
            task_name="$task_name" \
            2>&1 | tee "$BASE_OUTPUT_DIR/${task_name}_unlearn.log"

        # Run evaluation
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=gpt2 \
            model.model_args.pretrained_model_name_or_path="saves/unlearn/$task_name" \
            model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/$task_name" \
            task_name="${task_name}_eval" \
            2>&1 | tee "$BASE_OUTPUT_DIR/${task_name}_eval.log"
    else
        # Other methods: run unlearn training, then eval
        echo "[GPU $gpu] Running $method epoch=$epoch_name"

        # For other methods, we use the finetuned model as starting point
        # and run unlearning for a few epochs
        python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            model=gpt2 \
            model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
            model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
            trainer=$method \
            trainer.args.num_train_epochs=5 \
            task_name="$task_name" \
            2>&1 | tee "$BASE_OUTPUT_DIR/${task_name}_unlearn.log"

        # Run evaluation
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=gpt2 \
            model.model_args.pretrained_model_name_or_path="saves/unlearn/$task_name" \
            model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/$task_name" \
            task_name="${task_name}_eval" \
            2>&1 | tee "$BASE_OUTPUT_DIR/${task_name}_eval.log"
    fi

    echo "[GPU $gpu] Completed: $task_name"
}

export -f run_unlearn_eval
export MODEL_PATH TRAINING_LOG_DIR BASE_OUTPUT_DIR

# Build job list
JOBS=()

# LMCleaner jobs: epoch × K combinations
for i in "${!EPOCHS[@]}"; do
    epoch_step=${EPOCHS[$i]}
    epoch_name=${EPOCH_NAMES[$i]}
    for k in "${K_VALUES[@]}"; do
        task_name="gpt2_LMCleaner_${epoch_name}_K${k}"
        JOBS+=("LMCleanerBatch:$epoch_step:$epoch_name:$k:$task_name")
    done
done

# Other methods jobs (run once each, using final model)
for method in "${OTHER_METHODS[@]}"; do
    task_name="gpt2_${method}"
    JOBS+=("$method:625:epoch5:0:$task_name")
done

echo "Total jobs: ${#JOBS[@]}"
echo "Jobs: ${JOBS[@]}"

# Run jobs in parallel across 4 GPUs
job_idx=0
while [ $job_idx -lt ${#JOBS[@]} ]; do
    pids=()

    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $job_idx -lt ${#JOBS[@]} ]; then
            job=${JOBS[$job_idx]}
            IFS=':' read -r method epoch_step epoch_name k_value task_name <<< "$job"

            run_unlearn_eval $gpu "$method" "$epoch_step" "$epoch_name" "$k_value" "$task_name" &
            pids+=($!)

            ((job_idx++))
        fi
    done

    # Wait for this batch to complete
    for pid in "${pids[@]}"; do
        wait $pid || true
    done
done

echo "All jobs completed!"
echo "Results saved to: $BASE_OUTPUT_DIR"
