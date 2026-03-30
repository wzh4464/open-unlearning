#!/bin/bash
# Experiment A: Master Orchestration Script
# Runs all steps sequentially. Use in tmux:
#   tmux new-session -d -s expA "cd /app && bash scripts/experiments/expA/run_all.sh 2>&1 | tee saves/results/expA/run_all.log"
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Resume from a specific step (default: 1)
START_FROM=${START_FROM:-1}

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Experiment A: Full Pipeline"
echo "=============================================="
print_config
echo "Start from step: ${START_FROM}"
echo "Log: ${RESULTS_DIR}/run_all.log"
echo ""

run_step() {
    local step_num=$1
    local script=$2
    local description=$3

    if [ "${step_num}" -lt "${START_FROM}" ]; then
        echo "[SKIP] Step ${step_num}: ${description} (START_FROM=${START_FROM})"
        return
    fi

    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step ${step_num}: ${description}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    local step_start=$(date +%s)
    bash "${SCRIPT_DIR}/${script}"
    local step_end=$(date +%s)
    local step_duration=$((step_end - step_start))

    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step ${step_num} complete: ${description} (${step_duration}s)"
    echo ""
}

PIPELINE_START=$(date +%s)

# Step 1: Finetune on full TOFU (1 epoch)
run_step 1 "01_finetune_full.sh" "Finetune on full TOFU (1 epoch)"

# Step 2: Retrain on retain90 (gold standard)
run_step 2 "02_retrain_retain.sh" "Retrain on retain90 (gold standard)"

# Step 3: LMCleaner + Post-finetune on retain set
run_step 3 "03_lmcleaner.sh" "LMCleaner Unlearning + Post-Finetune"

# Step 4: Baselines (GradDiff, NPO, PDU, UNDIAL)
run_step 4 "04_baselines.sh" "Baseline Unlearning (GradDiff, NPO, PDU, UNDIAL)"

# Step 5: Basic TOFU Eval (retrain first for retain_logs_path)
run_step 5 "05_eval_basic.sh" "TOFU Basic Evaluation"

# Step 6: MIA + Privacy Eval
run_step 6 "06_eval_mia.sh" "MIA + Privacy Evaluation"

# Step 7: Aggregate Results
if [ "7" -ge "${START_FROM}" ]; then
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 7: Aggregate Results"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    $PYTHON_CMD "${SCRIPT_DIR}/08_aggregate_results.py" ${SEED}
fi

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "=============================================="
echo "EXPERIMENT A COMPLETE"
echo "=============================================="
echo "Total time: ${PIPELINE_DURATION}s ($((PIPELINE_DURATION / 60))m)"
echo ""
echo "Results: ${RESULTS_DIR}/"
echo "Models:"
echo "  Original:  ${FINETUNE_DIR}"
echo "  Retrain:   ${RETRAIN_DIR}"
echo "  LMCleaner: $(get_unlearn_output_dir lmcleaner)"
for METHOD in "${BASELINE_METHODS[@]}"; do
    echo "  ${METHOD}: $(get_unlearn_output_dir ${METHOD,,})"
done
