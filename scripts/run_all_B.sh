#!/bin/bash
set -e
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=0

# Override save paths to /workspace
export SAVES_BASE="/workspace/saves"

for B in 64 128 256 512; do
    TASK="rebuttalB_B${B}_seed0"
    LOG_DIR="${SAVES_BASE}/train_logs/${TASK}"
    
    # Determine micro-batch and grad accum
    if [ "$B" -le 64 ]; then
        MICRO=$B
        ACCUM=1
    else
        MICRO=64
        ACCUM=$((B / 64))
    fi
    
    STEPS=$((4000 / B))
    
    echo ""
    echo "=========================================="
    echo "Training B=${B} (micro=${MICRO}, accum=${ACCUM}, steps=${STEPS})"
    echo "=========================================="
    
    $PYTHON_CMD src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default \
        model=Llama-3.2-1B-Instruct \
        collator=DataCollatorForSupervisedDatasetwithIndex \
        +trainer.args.remove_unused_columns=False \
        trainer.args.optim=adamw_torch \
        trainer.args.learning_rate=1e-5 \
        trainer.args.weight_decay=0.01 \
        trainer.args.num_train_epochs=1 \
        trainer.args.per_device_train_batch_size=${MICRO} \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        trainer.args.save_strategy=epoch \
        trainer.args.save_total_limit=1 \
        trainer.args.seed=0 \
        +trainer.args.training_logger.enabled=true \
        +trainer.args.training_logger.log_dir="${LOG_DIR}" \
        +trainer.args.training_logger.max_steps=10000 \
        +trainer.args.training_logger.mode=batch \
        +trainer.args.training_logger.sync_mode=true \
        +trainer.args.training_logger.save_indices_only=true \
        +trainer.args.training_logger.save_rng_state=true \
        +trainer.args.training_logger.steps_per_epoch=${STEPS} \
        +trainer.args.training_logger.save_at_epoch_end=true \
        paths.output_dir="${SAVES_BASE}/finetune/${TASK}" \
        task_name="${TASK}"
    
    echo "=== B=${B} done ==="
done

echo ""
echo "All 4 batch sizes complete!"
