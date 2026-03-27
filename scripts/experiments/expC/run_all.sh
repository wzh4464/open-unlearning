#!/bin/bash
# Experiment C: Full pipeline
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../env.sh"

echo "=========================================="
echo "  Experiment C: Removal vs Noise Ablation"
echo "  K=${K}, β=${BETA}, Δ̄_cert=${DELTA_CERT}"
echo "=========================================="

STEP=${1:-""}

if [ -z "$STEP" ] || [ "$STEP" = "1" ]; then
    echo ">>> Step 1: Three-config ablation (seed=0)"
    bash "$SCRIPT_DIR/01_run_ablation.sh" --seed 0
fi

if [ -z "$STEP" ] || [ "$STEP" = "2" ]; then
    echo ">>> Step 2: ε sweep (seed=0)"
    bash "$SCRIPT_DIR/02_epsilon_sweep.sh" 0
fi

if [ -z "$STEP" ] || [ "$STEP" = "3" ]; then
    echo ">>> Step 3: Eval all"
    bash "$SCRIPT_DIR/03_eval_all.sh"
fi

echo "=========================================="
echo "  Experiment C Complete"
echo "=========================================="
