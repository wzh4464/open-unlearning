#!/bin/bash
# Run refinetune 4 and 5 epochs for all K values

K_VALUES=(50 500 1000 1250)  # K=100 already done
EPOCHS=(4 5)

# Function to run refinetune + eval
run_experiment() {
    local K=$1
    local EPOCH=$2
    local GPU=$3

    MODEL_PATH="/app/saves/unlearn/lmcleaner_llama32_epoch5_K${K}"
    OUTPUT_DIR="/app/saves/unlearn/lmcleaner_epoch5_K${K}_refinetune${EPOCH}"

    echo "[GPU $GPU] Starting K=$K, refinetune epochs=$EPOCH"

    # Run refinetune
    CUDA_VISIBLE_DEVICES=$GPU python /app/scripts/refiletune_retain.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs $EPOCH \
        --learning_rate 1e-5 \
        --batch_size 4

    # Run evaluation
    CUDA_VISIBLE_DEVICES=$GPU python /app/src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model.model_args.pretrained_model_name_or_path="$OUTPUT_DIR" \
        task_name="lmcleaner_epoch5_K${K}_refinetune${EPOCH}"

    echo "[GPU $GPU] Completed K=$K, refinetune epochs=$EPOCH"
}

export -f run_experiment

# Round 1: K=50,500,1000,1250 with epochs=4 on GPU 0,1,2,3
echo "=== Round 1: refinetune 4 epochs ==="
run_experiment 50 4 0 &
run_experiment 500 4 1 &
run_experiment 1000 4 2 &
run_experiment 1250 4 3 &
wait

# Round 2: K=50,500,1000,1250 with epochs=5 on GPU 0,1,2,3
echo "=== Round 2: refinetune 5 epochs ==="
run_experiment 50 5 0 &
run_experiment 500 5 1 &
run_experiment 1000 5 2 &
run_experiment 1250 5 3 &
wait

echo "=== All experiments completed ==="
