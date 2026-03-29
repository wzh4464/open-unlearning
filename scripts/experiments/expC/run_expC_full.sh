#!/bin/bash
# Experiment C: Full pipeline (1B, 1 epoch, forget01, K=10)
# 1. Finetune 1B with sparse checkpoints
# 2. Retrain on retain99
# 3. Three-config ablation (Full, Removal-Only, Noise-Only)
# 4. Epsilon sweep
# 5. Eval all
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "Experiment C: 1B model, forget01, K=${K}"
echo "=============================================="

# =============================================
# Step 1: Finetune 1B with sparse checkpoints
# =============================================
if [ ! -f "${FINETUNE_DIR}/model.safetensors" ]; then
    echo "[$(date '+%H:%M:%S')] Step 1: Finetune ${MODEL}"
    $PYTHON_CMD src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default \
        task_name="${MODEL_SHORT}_tofu_1epoch" \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        trainer.args.num_train_epochs=${NUM_EPOCHS} \
        trainer.args.learning_rate=${LEARNING_RATE} \
        trainer.args.weight_decay=${WEIGHT_DECAY} \
        trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.optim=${OPTIMIZER} \
        ++trainer.args.bf16=true \
        ++trainer.args.save_strategy=no \
        +trainer.args.training_logger.enabled=true \
        +trainer.args.training_logger.log_dir="${TRAIN_LOG_DIR}" \
        +trainer.args.training_logger.mode=batch \
        +trainer.args.training_logger.save_indices_only=true \
        +trainer.args.training_logger.save_batch_data=false \
        +trainer.args.training_logger.save_rng_state=false \
        +trainer.args.training_logger.save_interval=0 \
        +trainer.args.training_logger.save_sparse_checkpoints=true \
        +trainer.args.training_logger.checkpoint_stride=10
    echo "--- Finetune VERIFY ---"
    echo "model: $(test -f ${FINETUNE_DIR}/model.safetensors && echo ✓ || echo ✗)"
    echo "checkpoints: $(ls ${TRAIN_LOG_DIR}/sparse_checkpoints/*.pt 2>/dev/null | wc -l) files"
else
    echo "Step 1: SKIP (finetune exists)"
fi

# Build sample_indices + eta_cache
echo "[$(date '+%H:%M:%S')] Building indices..."
$PYTHON_CMD -c "
import json, torch, pickle
from pathlib import Path

log_dir = Path('${TRAIN_LOG_DIR}')

# sample_indices
if not (log_dir / 'sample_indices.json').exists():
    g = torch.Generator(); g.manual_seed(${SEED})
    perm = torch.randperm(4000, generator=g).tolist()
    si = {}
    bs = ${PER_DEVICE_BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS}
    for s in range(0, len(perm), bs):
        si[s // bs] = perm[s:s+bs]
    with open(log_dir / 'sample_indices.json', 'w') as f:
        json.dump(si, f)
    print(f'sample_indices: {len(si)} steps')

# eta_cache
if not (log_dir / 'eta_cache.json').exists():
    eta = {}
    for f in sorted(log_dir.glob('step_meta_*.pkl')):
        with open(f, 'rb') as fp:
            r = pickle.load(fp)
        eta[r['step_id']] = r['eta']
    if not eta:
        for f in sorted(log_dir.glob('step_records_chunk_*.pkl')):
            import re
            with open(f, 'rb') as fp:
                records = pickle.load(fp)
            for rec in records:
                eta[rec['step_id']] = rec['eta']
            del records
    with open(log_dir / 'eta_cache.json', 'w') as f:
        json.dump(eta, f)
    print(f'eta_cache: {len(eta)} entries')

# θ[0] checkpoint
ckpt_dir = log_dir / 'sparse_checkpoints'
if ckpt_dir.exists() and not (ckpt_dir / 'step_000000.pt').exists():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('${BASE_MODEL_PATH}', torch_dtype=torch.bfloat16)
    state = {n: p.detach().cpu() for n, p in model.named_parameters()}
    torch.save(state, ckpt_dir / 'step_000000.pt')
    print('Saved θ[0] checkpoint')
    del model, state
"

# =============================================
# Step 2: Retrain on retain99
# =============================================
if [ ! -f "${RETRAIN_DIR}/model.safetensors" ]; then
    echo "[$(date '+%H:%M:%S')] Step 2: Retrain (retain99)"
    $PYTHON_CMD src/train.py --config-name=train.yaml \
        experiment=finetune/tofu/default \
        task_name="${MODEL_SHORT}_tofu_retrain_f01" \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        "data.train.TOFU_QA_full.args.hf_args.name=${RETAIN_SPLIT}" \
        trainer.args.num_train_epochs=${NUM_EPOCHS} \
        trainer.args.learning_rate=${LEARNING_RATE} \
        trainer.args.weight_decay=${WEIGHT_DECAY} \
        trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.optim=${OPTIMIZER} \
        ++trainer.args.bf16=true
    echo "Retrain done"
else
    echo "Step 2: SKIP (retrain exists)"
fi

# Eval retrain
RETRAIN_EVAL_JSON="${RETRAIN_DIR}/evals/TOFU_EVAL.json"
if [ ! -f "${RETRAIN_EVAL_JSON}" ]; then
    echo "[$(date '+%H:%M:%S')] Eval retrain"
    $PYTHON_CMD src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
        model="${MODEL}" forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" \
        task_name="expC_retrain_eval" \
        model.model_args.pretrained_model_name_or_path="${RETRAIN_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${RETRAIN_DIR}/evals"
fi

# =============================================
# Step 3: Three-config ablation
# =============================================
echo "[$(date '+%H:%M:%S')] Step 3: Ablation (Full, Removal-Only, Noise-Only)"

run_lmc() {
    local TASK=$1 EPSILON=$2 NOISE_MODE=$3 SKIP_CORR=$4
    local OUT="${SAVES_BASE}/unlearn/${TASK}"
    if [ -f "${OUT}/model.safetensors" ]; then
        echo "SKIP: ${TASK}"
        return
    fi
    echo "[$(date '+%H:%M:%S')] ${TASK} (ε=${EPSILON}, noise=${NOISE_MODE}, skip_corr=${SKIP_CORR})"
    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LMCleanerBatch \
        model="${MODEL}" \
        model.model_args.pretrained_model_name_or_path="${MODEL_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        forget_split="${FORGET_SPLIT}" retain_split="${RETAIN_SPLIT}" \
        trainer.method_args.training_log_dir="${TRAIN_LOG_DIR}" \
        trainer.method_args.K=${K} \
        trainer.method_args.max_step=${MAX_STEP} \
        trainer.method_args.hessian_mode="${HESSIAN_MODE}" \
        trainer.method_args.damping=${DAMPING} \
        trainer.method_args.use_historical_params=false \
        trainer.method_args.epsilon=${EPSILON} \
        trainer.method_args.delta=${DEFAULT_DELTA} \
        ++trainer.method_args.noise_mode="${NOISE_MODE}" \
        ++trainer.method_args.skip_correction=${SKIP_CORR} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.bf16=true \
        trainer.args.efficiency_tracking.enabled=true \
        model.model_args.attn_implementation=eager \
        task_name="${TASK}"
    echo "${TASK} done"
}

# Full: removal + noise (ε=1.0)
run_lmc "expC_full_s${SEED}" "${DEFAULT_EPSILON}" "subspace" "false"

# Removal-Only: no noise (ε=0)
run_lmc "expC_removal_only_s${SEED}" "0" "none" "false"

# Noise-Only: no removal, only noise
run_lmc "expC_noise_only_s${SEED}" "${DEFAULT_EPSILON}" "subspace" "true"

# =============================================
# Step 4: Epsilon sweep
# =============================================
echo "[$(date '+%H:%M:%S')] Step 4: Epsilon sweep"
for EPS in "${EPSILON_VALUES[@]}"; do
    run_lmc "expC_eps${EPS}_s${SEED}" "${EPS}" "subspace" "false"
done

# =============================================
# Step 5: Eval all
# =============================================
echo "[$(date '+%H:%M:%S')] Step 5: Eval all"

eval_model() {
    local MP=$1 TN=$2
    [ ! -d "$MP" ] && return
    if [ ! -f "$MP/evals/TOFU_SUMMARY.json" ]; then
        echo "Eval: $TN"
        $PYTHON_CMD src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
            model="${MODEL}" forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" \
            task_name="${TN}_eval" \
            model.model_args.pretrained_model_name_or_path="$MP" \
            model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
            paths.output_dir="$MP/evals" retain_logs_path="${RETRAIN_EVAL_JSON}"
    fi
    if [ ! -f "$MP/evals_mia/TOFU_SUMMARY.json" ]; then
        echo "MIA: $TN"
        $PYTHON_CMD src/eval.py --config-name=eval.yaml experiment=eval/tofu/mia \
            model="${MODEL}" forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" \
            task_name="${TN}_mia" \
            model.model_args.pretrained_model_name_or_path="$MP" \
            model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
            paths.output_dir="$MP/evals_mia" retain_logs_path="${RETRAIN_EVAL_JSON}"
    fi
}

# Original
eval_model "${FINETUNE_DIR}" "expC_original"
# Ablation configs
eval_model "${SAVES_BASE}/unlearn/expC_full_s${SEED}" "expC_full"
eval_model "${SAVES_BASE}/unlearn/expC_removal_only_s${SEED}" "expC_removal_only"
eval_model "${SAVES_BASE}/unlearn/expC_noise_only_s${SEED}" "expC_noise_only"
# Epsilon sweep
for EPS in "${EPSILON_VALUES[@]}"; do
    eval_model "${SAVES_BASE}/unlearn/expC_eps${EPS}_s${SEED}" "expC_eps${EPS}"
done

echo "[$(date '+%H:%M:%S')] ALL DONE"
