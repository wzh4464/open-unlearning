#!/bin/bash
# Experiment A: K-sweep for LMCleaner + baselines + eval
# Runs LMCleaner with K=10,20,30,40,50, then baselines, then evals all
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_ROOT}"

PIPELINE_START=$(date +%s)

echo "=============================================="
echo "Experiment A: K-sweep (forget01, K=10,20,30,40,50)"
echo "=============================================="
print_config

# =============================================
# Step 1: LMCleaner with multiple K values
# =============================================
for K_VAL in 10 20 30 40 50; do
    TASK="expA_lmcleaner_k${K_VAL}_s${SEED}"
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "[$(date '+%H:%M:%S')] LMCleaner K=${K_VAL}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    STEP_START=$(date +%s)
    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=LMCleanerBatch \
        task_name="${TASK}" \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        model.model_args.pretrained_model_name_or_path="${FINETUNE_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
        trainer.method_args.K=${K_VAL} \
        trainer.method_args.max_step=${STEPS_PER_EPOCH} \
        trainer.method_args.hessian_mode="${HESSIAN_MODE}" \
        trainer.method_args.damping=${DAMPING} \
        trainer.method_args.use_historical_params=false \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.bf16=true \
        trainer.args.efficiency_tracking.enabled=true \
        model.model_args.attn_implementation=eager
    STEP_END=$(date +%s)
    echo "LMCleaner K=${K_VAL} done ($(( STEP_END - STEP_START ))s)"
done

# =============================================
# Step 2: Baselines
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Baselines"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
bash "${SCRIPT_DIR}/04_baselines.sh"

# =============================================
# Step 3: Eval all models
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Evaluation"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

RETRAIN_EVAL_JSON="${RETRAIN_DIR}/evals/TOFU_EVAL.json"

# Eval retrain first if not done
if [ ! -f "${RETRAIN_EVAL_JSON}" ]; then
    echo "Evaluating retrain model first..."
    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" \
        task_name="expA_retrain_eval" \
        model.model_args.pretrained_model_name_or_path="${RETRAIN_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${RETRAIN_DIR}/evals"
fi

eval_model() {
    local model_path=$1
    local task_name=$2
    local eval_dir="${model_path}/evals"

    if [ ! -d "${model_path}" ]; then
        echo "SKIP: ${task_name} (not found)"
        return
    fi

    echo "Eval: ${task_name}"
    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_eval" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${eval_dir}" \
        retain_logs_path="${RETRAIN_EVAL_JSON}"
}

# Eval original
eval_model "${FINETUNE_DIR}" "expA_original"

# Eval LMCleaner variants
for K_VAL in 10 20 30 40 50; do
    eval_model "saves/unlearn/expA_lmcleaner_k${K_VAL}_s${SEED}" "expA_lmcleaner_k${K_VAL}"
done

# Eval baselines
for METHOD in "${BASELINE_METHODS[@]}"; do
    METHOD_LOWER="${METHOD,,}"
    eval_model "$(get_unlearn_output_dir "${METHOD_LOWER}")" "$(get_unlearn_task_name "${METHOD_LOWER}")"
done

# =============================================
# Step 4: Aggregate
# =============================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "[$(date '+%H:%M:%S')] Aggregate Results"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# Custom aggregator that includes K variants
AGGREGATE_SEED=${SEED} AGGREGATE_FINETUNE_DIR=${FINETUNE_DIR} AGGREGATE_RETRAIN_DIR=${RETRAIN_DIR} \
$PYTHON_CMD - << 'PYEOF'
import json, csv, sys, os
from pathlib import Path

seed = int(os.environ.get("AGGREGATE_SEED", "0"))
finetune_dir = os.environ.get("AGGREGATE_FINETUNE_DIR", "saves/finetune/llama32_3b_tofu_1epoch")
retrain_dir = os.environ.get("AGGREGATE_RETRAIN_DIR", "saves/finetune/llama32_3b_tofu_retrain")
results_dir = Path("saves/results/expA")
results_dir.mkdir(parents=True, exist_ok=True)

def load_json(path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: {path}: {e}")
    return {}

def extract_metrics(model_dir):
    result = {}
    for subdir in ["evals", "."]:
        found = False
        for fname in ["TOFU_SUMMARY.json", "TOFU_EVAL.json"]:
            p = model_dir / subdir / fname
            if p.exists():
                found = True
                data = load_json(p)
                if fname == "TOFU_SUMMARY.json":
                    result.update(data)
                elif fname == "TOFU_EVAL.json":
                    for k, v in data.items():
                        if isinstance(v, dict) and "agg_value" in v:
                            result[k] = v["agg_value"]
        if found:
            break
    eff = load_json(model_dir / "efficiency_metrics.json")
    if eff:
        result["time_seconds"] = eff.get("unlearning_time_seconds")
        result["peak_gpu_mb"] = eff.get("peak_gpu_memory_mb")
    return result

methods = {}
methods["Original"] = Path(finetune_dir)
methods["Retrain"] = Path(retrain_dir)
for k in [10, 20, 30, 40, 50]:
    d = Path(f"saves/unlearn/expA_lmcleaner_k{k}_s{seed}")
    if d.exists():
        methods[f"LMCleaner_K{k}"] = d
for m in ["graddiff", "npo", "pdu", "undial"]:
    d = Path(f"saves/unlearn/expA_{m}_s{seed}")
    if d.exists():
        methods[m.upper()] = d

rows = []
for name, d in methods.items():
    if not d.exists():
        continue
    metrics = extract_metrics(d)
    row = {"Method": name}
    row.update(metrics)
    rows.append(row)

# CSV
all_keys = sorted(set(k for r in rows for k in r if k != "Method"))
csv_path = results_dir / f"k_sweep_results_s{seed}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Method"] + all_keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Saved: {csv_path}")

# Print table
key_metrics = ["model_utility", "forget_quality", "forget_truth_ratio", "time_seconds"]
available = [k for k in key_metrics if any(k in r for r in rows)]
if available:
    header = f"{'Method':<20}" + "".join(f"{k:>18}" for k in available)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['Method']:<20}"
        for k in available:
            v = row.get(k, "N/A")
            if isinstance(v, float):
                line += f"{v:>18.4f}"
            elif v is None:
                line += f"{'N/A':>18}"
            else:
                line += f"{str(v):>18}"
        print(line)
PYEOF

PIPELINE_END=$(date +%s)
echo ""
echo "=============================================="
echo "K-SWEEP COMPLETE: $(( PIPELINE_END - PIPELINE_START ))s"
echo "=============================================="
