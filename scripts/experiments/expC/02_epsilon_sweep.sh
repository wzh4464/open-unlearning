#!/bin/bash
# Experiment C: Full method ε sweep
# Fixed K=50, δ=1e-5, sweep ε ∈ {0.25, 0.5, 1.0, 2.0, 4.0}

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SEED=${1:-0}

echo "=== Experiment C: ε Sweep (seed=${SEED}) ==="

for EPS in "${EPSILON_VALUES[@]}"; do
    TASK="expC_full_eps${EPS}_seed${SEED}"
    OUT_DIR="${SAVES_BASE}/unlearn/${TASK}"

    if [ -f "$OUT_DIR/audit/audit_records.json" ]; then
        echo "SKIP: $TASK (already done)"
        continue
    fi

    echo ""
    echo "====== ε=${EPS} ======"

    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LMCleanerBatch \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${MODEL_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${MODEL_DIR}" \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        trainer.method_args.training_log_dir="${TRAIN_LOG_DIR}" \
        trainer.method_args.K=${K} \
        trainer.method_args.max_step=${MAX_STEP} \
        trainer.method_args.hessian_mode=${HESSIAN_MODE} \
        trainer.method_args.damping=${DAMPING} \
        trainer.method_args.epsilon=${EPS} \
        trainer.method_args.delta=${DEFAULT_DELTA} \
        trainer.method_args.noise_mode=subspace \
        trainer.method_args.beta=${BETA} \
        trainer.method_args.projector_rank=${PROJ_RANK} \
        trainer.method_args.projector_seed=${PROJ_SEED} \
        trainer.method_args.delta_cert_public=${DELTA_CERT} \
        trainer.method_args.use_historical_params=false \
        trainer.args.num_train_epochs=0 \
        trainer.args.seed=${SEED} \
        paths.output_dir="${OUT_DIR}" \
        task_name="${TASK}"

    echo "====== ε=${EPS} done ======"
done

echo ""
echo "=== ε Sweep complete ==="
