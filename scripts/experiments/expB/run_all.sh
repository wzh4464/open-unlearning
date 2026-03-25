#!/bin/bash
# RebuttalB: Run full (K,B) ablation pipeline
#
# Steps:
#   1. LMCleaner unlearning sweep (all valid K,B)
#   2. TOFU eval sweep
#   3. Forget distribution analysis
#   4. Aggregate results to CSV
#
# Usage:
#   bash scripts/experiments/expB/run_all.sh
#   bash scripts/experiments/expB/run_all.sh --step 2   # run only step 2

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$(dirname "$SCRIPT_DIR")/../env.sh"

STEP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

run_step() {
    local n=$1
    [ -n "$STEP" ] && [ "$STEP" != "$n" ] && return 0
    return 1
}

echo "=========================================="
echo "  RebuttalB: (K,B) Joint Ablation"
echo "  K = {${K_VALUES[*]}}"
echo "  B = {8, 16, 32, 64, 128, 256}"
echo "=========================================="
echo ""

if [ -z "$STEP" ] || [ "$STEP" = "1" ]; then
    echo ">>> Step 1: Unlearning sweep"
    bash "$SCRIPT_DIR/01_unlearn_sweep.sh"
    echo ""
fi

if [ -z "$STEP" ] || [ "$STEP" = "2" ]; then
    echo ">>> Step 2: Eval sweep"
    bash "$SCRIPT_DIR/02_eval_sweep.sh"
    echo ""
fi

if [ -z "$STEP" ] || [ "$STEP" = "3" ]; then
    echo ">>> Step 3: Forget distribution analysis"
    bash "$SCRIPT_DIR/03_forget_distribution.sh"
    echo ""
fi

if [ -z "$STEP" ] || [ "$STEP" = "4" ]; then
    echo ">>> Step 4: Aggregate results"
    $PYTHON_CMD "$SCRIPT_DIR/04_aggregate_results.py"
    echo ""
fi

echo "=========================================="
echo "  RebuttalB Pipeline Complete"
echo "=========================================="
