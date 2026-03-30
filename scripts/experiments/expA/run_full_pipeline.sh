#!/bin/bash
# Experiment A: Full pipeline with sparse checkpoints + CheckpointAwareUProvider
# forget10/retain90, SGD lr=1e-3, Fisher HVP, use_historical_params=true
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_ROOT}"

PIPELINE_START=$(date +%s)

echo "=============================================="
echo "Experiment A: Full Pipeline (historical params)"
echo "=============================================="
print_config
echo "use_historical_params: true (CheckpointAwareUProvider)"

# =============================================
# Stage 1: Finetune with sparse checkpoints
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Stage 1: Finetune (sparse checkpoints stride=10)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
STAGE_START=$(date +%s)
bash "${SCRIPT_DIR}/01_finetune_full.sh"
echo "--- Stage 1 VERIFY ---"
CKPT_COUNT=$(ls saves/train_logs/llama32_3b_tofu_1epoch/sparse_checkpoints/step_*.pt 2>/dev/null | wc -l)
echo "Sparse checkpoints: ${CKPT_COUNT} files"
ls -lh saves/train_logs/llama32_3b_tofu_1epoch/sparse_checkpoints/ 2>/dev/null | head -3
test -f saves/train_logs/llama32_3b_tofu_1epoch/sample_indices.json && echo "sample_indices.json: ✓" || echo "sample_indices.json: ✗ FAIL"
test -f saves/finetune/llama32_3b_tofu_1epoch/model.safetensors && echo "model.safetensors: ✓" || echo "model.safetensors: ✗ FAIL"
echo "Stage 1 done ($(($(date +%s) - STAGE_START))s)"

# Build eta_cache from step_meta if needed
echo "Building eta_cache..."
$PYTHON_CMD -c "
import pickle, json
from pathlib import Path
log_dir = Path('saves/train_logs/llama32_3b_tofu_1epoch')
eta = {}
for f in sorted(log_dir.glob('step_meta_*.pkl')):
    with open(f, 'rb') as fp:
        r = pickle.load(fp)
    eta[r['step_id']] = r['eta']
if not eta:
    for f in sorted(log_dir.glob('step_records_chunk_*.pkl')):
        import re
        step_id = int(re.search(r'(\d+)', f.stem).group())
        with open(f, 'rb') as fp:
            records = pickle.load(fp)
        for rec in records:
            eta[rec['step_id']] = rec['eta']
        del records
with open(log_dir / 'eta_cache.json', 'w') as fp:
    json.dump(eta, fp)
print(f'eta_cache: {len(eta)} entries')
"

# Reconstruct sample_indices if needed
if [ ! -f saves/train_logs/llama32_3b_tofu_1epoch/sample_indices.json ]; then
    echo "Reconstructing sample_indices..."
    $PYTHON_CMD -c "
import json, torch
g = torch.Generator(); g.manual_seed(${SEED})
perm = torch.randperm(4000, generator=g).tolist()
si = {}
for s in range(0, len(perm), ${PER_DEVICE_BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS}):
    si[s // (${PER_DEVICE_BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS})] = perm[s:s+${PER_DEVICE_BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}]
with open('saves/train_logs/llama32_3b_tofu_1epoch/sample_indices.json', 'w') as f:
    json.dump(si, f)
print(f'sample_indices: {len(si)} steps')
"
fi

# =============================================
# Stage 2: Retrain on retain90
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Stage 2: Retrain (retain90)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
STAGE_START=$(date +%s)
bash "${SCRIPT_DIR}/02_retrain_retain.sh"
echo "--- Stage 2 VERIFY ---"
test -f saves/finetune/llama32_3b_tofu_retrain/model.safetensors && echo "retrain model: ✓" || echo "retrain model: ✗ FAIL"
echo "Stage 2 done ($(($(date +%s) - STAGE_START))s)"

# =============================================
# Stage 3: LMCleaner K-sweep (CheckpointAwareUProvider)
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Stage 3: LMCleaner K-sweep (historical params)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
bash "${SCRIPT_DIR}/run_k_sweep.sh"

# =============================================
# Stage 4: Final verify
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Final Verification"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

echo "--- Models ---"
for K in 10 20 30 40 50; do
    test -f "saves/unlearn/expA_lmcleaner_k${K}_s${SEED}/model.safetensors" && echo "  LMC_K${K}: ✓" || echo "  LMC_K${K}: ✗"
    # Check for sharded safetensors (model-00001-of-*.safetensors) as well
    POSTFT="saves/finetune/expA_lmcleaner_k${K}_postft_s${SEED}"
    (test -f "${POSTFT}/model.safetensors" || ls ${POSTFT}/model-00001-of-*.safetensors >/dev/null 2>&1) && echo "  LMC_K${K}+PostFT: ✓" || echo "  LMC_K${K}+PostFT: ✗"
done
for M in graddiff npo pdu undial; do
    test -f "saves/unlearn/expA_${M}_s${SEED}/model.safetensors" && echo "  ${M}: ✓" || echo "  ${M}: ✗"
done

echo "--- Evals ---"
EVAL_COUNT=$(find saves -name "TOFU_EVAL.json" -path "*/evals/*" 2>/dev/null | wc -l)
echo "  TOFU_EVAL.json files: ${EVAL_COUNT}"

echo "--- CSV ---"
test -f saves/results/expA/k_sweep_results_s${SEED}.csv && echo "  Results CSV: ✓" || echo "  Results CSV: ✗"

PIPELINE_END=$(date +%s)
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE: $(( PIPELINE_END - PIPELINE_START ))s ($(( (PIPELINE_END - PIPELINE_START) / 60 ))m)"
echo "=============================================="
