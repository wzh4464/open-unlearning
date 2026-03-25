#!/bin/bash
# Run B={32,64,128,256} sequentially, 1 epoch each, output to /workspace
set -e
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=0

SAVES="/workspace/saves"

for B in 32 64 128 256; do
    TASK="rebuttalB_B${B}_seed0"
    LOG_DIR="${SAVES}/train_logs/${TASK}"

    # micro_batch capped at 64 (GPU limit), rest via grad_accum
    if [ "$B" -le 64 ]; then MICRO=$B; ACCUM=1
    else
        if [ $((B % 64)) -ne 0 ]; then echo "Error: B=$B not divisible by 64"; exit 1; fi
        MICRO=64; ACCUM=$((B / 64))
    fi

    STEPS=$((4000 / B))

    echo ""
    echo "====== B=${B}  micro=${MICRO} accum=${ACCUM} steps=${STEPS} ======"

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
        paths.output_dir="${SAVES}/finetune/${TASK}" \
        task_name="${TASK}"

    # Verify sample_indices were saved
    SI="${LOG_DIR}/sample_indices.json"
    if [ -f "$SI" ]; then
        N=$(python3 -c "import json; d=json.load(open('$SI')); s=next(iter(d.values())); print(len(s))")
        echo "  sample_indices OK: ${N} indices/step (expected ${B})"
    else
        echo "  WARNING: sample_indices.json not found!"
    fi

    echo "====== B=${B} done ======"
done

echo ""
echo "All done. Storage used:"
du -sh ${SAVES}/train_logs/rebuttalB_B* 2>/dev/null
