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

    # Determine skip_correction flag
    local SKIP_CORR="false"
    if [ "$CONFIG" = "noise-only" ]; then
        SKIP_CORR="true"
    fi

    echo ""
    echo "====== $TASK (config=$CONFIG, ε=$EPSILON, noise=$NOISE_MODE, skip_corr=$SKIP_CORR) ======"

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
        trainer.method_args.noise_mode=isotropic \
        trainer.method_args.beta=${BETA} \
        trainer.method_args.projector_rank=${PROJ_RANK} \
        trainer.method_args.projector_seed=${PROJ_SEED} \
        trainer.method_args.delta_cert_public=${DELTA_CERT} \
        trainer.method_args.skip_correction=${SKIP_CORR} \
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
    # True Noise-Only: skip_correction=true zeros out v before noise injection.
    # This gives θ_noise = θ[τ] + ξ (no deterministic removal at all).
    run_lmcleaner "noise-only" "${DEFAULT_EPSILON}" "subspace" "expC_noise_seed${SEED}"
fi

echo ""
echo "=== Ablation runs complete ==="
