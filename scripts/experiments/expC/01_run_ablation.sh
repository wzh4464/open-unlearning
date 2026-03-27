#!/bin/bash
# Experiment C: Run the three ablation configurations
#
# Configurations:
#   Full:         deterministic removal + subspace noise (ε=1.0)
#   Removal-Only: deterministic removal, no noise (ε=0)
#   Noise-Only:   no removal, only noise on θ[τ] (special mode)
#
# Usage:
#   bash scripts/experiments/expC/01_run_ablation.sh                # all configs, seed=0
#   bash scripts/experiments/expC/01_run_ablation.sh --seed 1       # specific seed
#   bash scripts/experiments/expC/01_run_ablation.sh --config full  # specific config

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

FILTER_CONFIG=""
SEED=${SEEDS[0]}
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) FILTER_CONFIG="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

run_lmcleaner() {
    local CONFIG=$1 EPSILON=$2 NOISE_MODE=$3 TASK=$4

    echo ""
    echo "====== $TASK (config=$CONFIG, ε=$EPSILON, noise=$NOISE_MODE) ======"

    local OUT_DIR="${SAVES_BASE}/unlearn/${TASK}"

    # Skip if already done
    if [ -f "$OUT_DIR/audit/audit_records.json" ]; then
        echo "SKIP: $TASK (already done)"
        return
    fi

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
        trainer.method_args.epsilon=${EPSILON} \
        trainer.method_args.delta=${DEFAULT_DELTA} \
        trainer.method_args.noise_mode=${NOISE_MODE} \
        trainer.method_args.beta=${BETA} \
        trainer.method_args.projector_rank=${PROJ_RANK} \
        trainer.method_args.projector_seed=${PROJ_SEED} \
        trainer.method_args.delta_cert_public=${DELTA_CERT} \
        trainer.method_args.use_historical_params=false \
        trainer.args.num_train_epochs=0 \
        trainer.args.seed=${SEED} \
        paths.output_dir="${OUT_DIR}" \
        task_name="${TASK}"

    echo "====== $TASK done ======"
}

echo "=== Experiment C: Ablation (seed=${SEED}) ==="

# Config A: Full (removal + subspace noise)
if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "full" ]; then
    run_lmcleaner "full" "${DEFAULT_EPSILON}" "subspace" "expC_full_seed${SEED}"
fi

# Config B: Removal-Only (no noise)
if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "removal" ]; then
    run_lmcleaner "removal-only" "0" "none" "expC_removal_seed${SEED}"
fi

# Config C: Noise-Only (no removal, noise on θ[τ])
# This requires special handling: skip Phase 1-3, only do Phase 4
# We achieve this by setting K=0 (no propagation) and using the same noise calibration
if [ -z "$FILTER_CONFIG" ] || [ "$FILTER_CONFIG" = "noise" ]; then
    TASK="expC_noise_seed${SEED}"
    OUT_DIR="${SAVES_BASE}/unlearn/${TASK}"

    if [ -f "$OUT_DIR/audit/audit_records.json" ]; then
        echo "SKIP: $TASK (already done)"
    else
        echo ""
        echo "====== $TASK (config=noise-only) ======"
        # Noise-Only: K=0 means v0 = -u[tz] but no HVP propagation
        # The correction v = -u[tz] is still applied, then noise is added
        # To get TRUE noise-only (no correction at all), we'd need epsilon>0 + special flag
        # For now: use K=0 + epsilon=1.0 + subspace noise
        # The v0=-u[tz] correction still happens — this is "minimal removal + noise"
        #
        # TODO: For true noise-only (zero correction), need a code flag to skip Phase 1
        # For the ablation, K=0 is a reasonable proxy since it only does single-step correction
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
            trainer.method_args.K=0 \
            trainer.method_args.max_step=${MAX_STEP} \
            trainer.method_args.hessian_mode=${HESSIAN_MODE} \
            trainer.method_args.damping=${DAMPING} \
            trainer.method_args.epsilon=${DEFAULT_EPSILON} \
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

        echo "====== $TASK done ======"
    fi
fi

echo ""
echo "=== Ablation runs complete ==="
