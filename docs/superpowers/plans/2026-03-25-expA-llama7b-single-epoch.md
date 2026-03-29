# Experiment A: Llama-2-7B Single-Epoch Full Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run single-epoch TOFU unlearning on Llama-2-7B with 6 methods (LMCleaner, GradDiff, NPO, PDU, UNDIAL, Retrain) + full evaluation (utility, forget quality, MIA, PrivLeak, efficiency) to answer 6 core research questions about LMCleaner's theoretical validity at scale.

**Architecture:** Create a self-contained experiment suite under `scripts/experiments/expA/` modeled after the existing `scripts/experiments/` infrastructure. A shared `config.sh` defines all paths, hyperparams, and methods. Individual scripts handle finetune, retrain, LMCleaner, baselines, eval, and aggregation. All experiments use tmux for execution, efficiency tracking enabled for all methods.

**Tech Stack:** Hydra configs, bash scripts, Python (torch, HuggingFace Transformers, Accelerate), 1x H200 GPU (143GB), tmux for session management.

---

## Environment & Hardware

- **GPU:** 1x NVIDIA H200 (143GB VRAM) — sufficient for 7B bf16 + gradient checkpointing
- **Base model:** Llama-2-7b-hf (downloaded via ModelScope → `~/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf`)
- **Dataset:** TOFU (4000 samples full, forget10/retain90 split)
- **Single epoch:** 4000 samples / effective_batch_16 = 250 steps

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Llama-2-7b-hf | 7B scale experiment |
| Epochs | 1 | Single-pass, theory-aligned |
| Batch size | 4 per device | H200 has plenty VRAM |
| Grad accumulation | 4 | Effective batch = 16 |
| Steps/epoch | 250 | 4000/16 |
| LR | 1e-5 | Consistent with existing TOFU experiments |
| bf16 | True | Standard for 7B |
| Gradient checkpointing | True | Safety margin |
| LMCleaner K | 50 | Conservative, paper-aligned |
| LMCleaner hessian_mode | GGN | Stable for cross-entropy |
| LMCleaner damping | 1e-4 | Existing default |
| PDU retain_loss_eps | 0.3 | Community standard for TOFU |
| Seed | 0 (expand to 0,1,2 later) | Start with 1 seed |

## File Structure

### New files to create

```
scripts/experiments/expA/
├── config.sh                    # Shared config: model, paths, hyperparams
├── 01_finetune_full.sh         # Finetune on full TOFU (1 epoch, with TrainingLogger)
├── 02_retrain_retain.sh        # Retrain on retain90 only (1 epoch, gold standard)
├── 03_lmcleaner.sh             # LMCleaner batch-level unlearning
├── 04_baselines.sh             # GradDiff, NPO, PDU, UNDIAL unlearning
├── 05_eval_basic.sh            # Standard TOFU eval (all methods)
├── 06_eval_mia.sh              # MIA + privacy eval (all methods)
├── 07_eval_original.sh         # Eval original finetuned model (reference)
├── 08_aggregate_results.py     # Collect all results into CSV + plots
└── run_all.sh                  # Master tmux orchestration script

configs/experiment/eval/tofu/
└── mia.yaml                    # MIA evaluation config (enables MIA metrics)
```

### Existing files to reference (read-only)

- `configs/model/Llama-2-7b-hf.yaml` — model config
- `configs/trainer/LMCleanerBatch.yaml` — LMCleaner config
- `configs/trainer/GradDiff.yaml`, `NPO.yaml`, `PDU.yaml`, `UNDIAL.yaml` — baseline configs
- `configs/experiment/finetune/tofu/default.yaml` — finetune experiment config
- `configs/experiment/unlearn/tofu/default.yaml` — unlearn experiment config
- `configs/experiment/eval/tofu/default.yaml` — eval experiment config
- `configs/eval/tofu.yaml` — TOFU evaluator metrics list
- `src/trainer/utils.py` — EfficiencyTracker callback (already integrated)

### Output directory structure

```
saves/
├── finetune/llama2_7b_tofu_1epoch/          # Finetuned model + training logs
├── train_logs/llama2_7b_tofu_1epoch/        # TrainingLogger output for LMCleaner
├── finetune/llama2_7b_tofu_retrain/         # Retrain model (retain90 only)
└── unlearn/
    ├── expA_lmcleaner_s0/                   # LMCleaner output
    ├── expA_graddiff_s0/                    # GradDiff output
    ├── expA_npo_s0/                         # NPO output
    ├── expA_pdu_s0/                         # PDU output
    ├── expA_undial_s0/                      # UNDIAL output
    └── expA_retrain_s0/                     # Retrain eval output
```

---

## Task 1: Create MIA Eval Config

**Files:**
- Create: `configs/experiment/eval/tofu/mia.yaml`

This config enables MIA metrics for TOFU evaluation, required by `06_eval_mia.sh`.

- [ ] **Step 1: Create MIA eval config**

```yaml
# @package _global_

defaults:
  - override /model: Llama-3.2-1B-Instruct
  - override /eval: tofu

forget_split: forget10
holdout_split: holdout10
retain_logs_path: null

model:
  model_args:
    pretrained_model_name_or_path: ???

eval:
  tofu:
    forget_split: ${forget_split}
    holdout_split: ${holdout_split}
    retain_logs_path: ${retain_logs_path}
    overwrite: true
    metrics:
      mia_loss:
        handler: mia_loss
      mia_zlib:
        handler: mia_zlib
      mia_min_k:
        handler: mia_min_k
      mia_min_k_plus_plus:
        handler: mia_min_k_plus_plus
      privleak:
        handler: privleak

task_name: ???
```

- [ ] **Step 2: Verify config loads without errors**

Run: `python -c "from hydra import compose, initialize; initialize(config_path='../../configs'); cfg = compose(config_name='eval', overrides=['experiment=eval/tofu/mia', 'task_name=test', 'model.model_args.pretrained_model_name_or_path=test']); print('OK')" 2>&1 | tail -3`

---

## Task 2: Create Experiment Config (`config.sh`)

**Files:**
- Create: `scripts/experiments/expA/config.sh`

- [ ] **Step 1: Write config.sh**

```bash
#!/bin/bash
# Experiment A: Llama-2-7B Single-Epoch Configuration
# All shared parameters for the experiment

# ============================================
# Model Configuration
# ============================================
MODEL_NAME="Llama-2-7b-hf"
MODEL_SHORT="llama2_7b"
# Base model path (ModelScope download)
BASE_MODEL_PATH="${HOME}/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf"

# ============================================
# Path Configuration
# ============================================
FINETUNE_DIR="saves/finetune/${MODEL_SHORT}_tofu_1epoch"
RETRAIN_DIR="saves/finetune/${MODEL_SHORT}_tofu_retrain"
TRAINING_LOG_DIR="saves/train_logs/${MODEL_SHORT}_tofu_1epoch"
UNLEARN_BASE_DIR="saves/unlearn"
RESULTS_DIR="saves/results/expA"

# ============================================
# Training Configuration
# ============================================
NUM_EPOCHS=1
STEPS_PER_EPOCH=250  # 4000 samples / effective_batch_16
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE="1e-5"
WEIGHT_DECAY="0.01"

# ============================================
# Data Splits
# ============================================
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"

# ============================================
# LMCleaner Parameters
# ============================================
K=50
HESSIAN_MODE="GGN"
DAMPING="0.0001"

# ============================================
# PDU Parameters
# ============================================
PDU_RETAIN_LOSS_EPS="0.3"
PDU_DUAL_STEP_SIZE="1.0"
PDU_DUAL_WARMUP_EPOCHS="0"  # Single epoch, no warmup

# ============================================
# Baseline Methods
# ============================================
BASELINE_METHODS=("GradDiff" "NPO" "PDU" "UNDIAL")

# ============================================
# Seed
# ============================================
SEED=${SEED:-0}

# ============================================
# Environment
# ============================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1
PYTHON_CMD="python"

# ============================================
# Helper Functions
# ============================================
get_unlearn_task_name() {
    local method=$1
    local seed=${2:-$SEED}
    echo "expA_${method}_s${seed}"
}

get_unlearn_output_dir() {
    local method=$1
    local seed=${2:-$SEED}
    echo "${UNLEARN_BASE_DIR}/$(get_unlearn_task_name $method $seed)"
}

print_config() {
    echo "=============================================="
    echo "Experiment A Configuration"
    echo "=============================================="
    echo "Model: ${MODEL_NAME} (${MODEL_SHORT})"
    echo "Base model: ${BASE_MODEL_PATH}"
    echo "Finetune dir: ${FINETUNE_DIR}"
    echo "Retrain dir: ${RETRAIN_DIR}"
    echo "Training logs: ${TRAINING_LOG_DIR}"
    echo "Epochs: ${NUM_EPOCHS}"
    echo "Steps/epoch: ${STEPS_PER_EPOCH}"
    echo "Effective batch: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
    echo "Splits: ${FORGET_SPLIT}/${RETAIN_SPLIT}"
    echo "LMCleaner K: ${K}"
    echo "Seed: ${SEED}"
    echo "=============================================="
}
```

---

## Task 3: Create Finetune Script

**Files:**
- Create: `scripts/experiments/expA/01_finetune_full.sh`

This finetunes Llama-2-7B on full TOFU for 1 epoch WITH TrainingLogger enabled (needed for LMCleaner).

- [ ] **Step 1: Write finetune script**

```bash
#!/bin/bash
# Experiment A - Step 1: Finetune Llama-2-7B on full TOFU (1 epoch)
# Saves model checkpoint + training logs for LMCleaner
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 1: Finetune ${MODEL_NAME} on TOFU (1 epoch)"
echo "=============================================="
print_config

# Verify base model exists
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "ERROR: Base model not found at ${BASE_MODEL_PATH}"
    echo "Run: modelscope download --model shakechen/Llama-2-7b-hf"
    exit 1
fi

TASK_NAME="${MODEL_SHORT}_tofu_1epoch"

echo "Starting finetuning..."
echo "Task: ${TASK_NAME}"
echo "Output: ${FINETUNE_DIR}"
echo "Training logs: ${TRAINING_LOG_DIR}"

$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    model.model_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.seed=${SEED} \
    ++trainer.args.bf16=true \
    ++trainer.args.save_strategy=epoch \
    +trainer.args.training_logger.enabled=true \
    +trainer.args.training_logger.log_dir="${TRAINING_LOG_DIR}" \
    +trainer.args.training_logger.mode=batch \
    +trainer.args.training_logger.save_indices_only=true \
    +trainer.args.training_logger.save_batch_data=false \
    +trainer.args.training_logger.save_rng_state=true \
    +trainer.args.training_logger.save_interval=1 \
    trainer.args.efficiency_tracking.enabled=true

echo ""
echo "Finetuning complete!"
echo "Model: ${FINETUNE_DIR}"
echo "Training logs: ${TRAINING_LOG_DIR}"
```

---

## Task 4: Create Retrain Script

**Files:**
- Create: `scripts/experiments/expA/02_retrain_retain.sh`

Finetunes base Llama-2-7B on retain90 only (1 epoch). This is the gold-standard reference.

- [ ] **Step 1: Write retrain script**

```bash
#!/bin/bash
# Experiment A - Step 2: Retrain on retain90 only (gold standard)
# Finetunes base model on retain set, simulating perfect unlearning
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 2: Retrain ${MODEL_NAME} on retain90 only"
echo "=============================================="
print_config

# Verify base model
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "ERROR: Base model not found at ${BASE_MODEL_PATH}"
    exit 1
fi

TASK_NAME="${MODEL_SHORT}_tofu_retrain"

echo "Starting retrain (retain90 only)..."
echo "Task: ${TASK_NAME}"
echo "Output: ${RETRAIN_DIR}"

# Use the unlearn data config but train with finetune trainer on retain data only
# We override the data to use only retain split as training data
$PYTHON_CMD src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    model.model_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    'data.train.TOFU_QA_full.args.hf_args.name=${retain_split}' \
    retain_split="${RETAIN_SPLIT}" \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.seed=${SEED} \
    ++trainer.args.bf16=true \
    ++trainer.args.save_strategy=epoch \
    trainer.args.efficiency_tracking.enabled=true

echo ""
echo "Retrain complete!"
echo "Model: ${RETRAIN_DIR}"
```

---

## Task 5: Create LMCleaner Unlearn Script

**Files:**
- Create: `scripts/experiments/expA/03_lmcleaner.sh`

- [ ] **Step 1: Write LMCleaner script**

```bash
#!/bin/bash
# Experiment A - Step 3: LMCleaner Batch-Level Unlearning
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

TASK_NAME=$(get_unlearn_task_name "lmcleaner")
OUTPUT_DIR=$(get_unlearn_output_dir "lmcleaner")

echo "=============================================="
echo "ExpA Step 3: LMCleaner Unlearn"
echo "=============================================="
print_config
echo "K: ${K}, Hessian: ${HESSIAN_MODE}, Damping: ${DAMPING}"
echo "Output: ${OUTPUT_DIR}"

# Verify finetuned model exists
if [ ! -d "${FINETUNE_DIR}" ]; then
    echo "ERROR: Finetuned model not found at ${FINETUNE_DIR}"
    echo "Run 01_finetune_full.sh first."
    exit 1
fi

# Verify training logs exist
if [ ! -d "${TRAINING_LOG_DIR}" ]; then
    echo "ERROR: Training logs not found at ${TRAINING_LOG_DIR}"
    exit 1
fi

echo "Running LMCleaner Unlearn..."
$PYTHON_CMD src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name="${TASK_NAME}" \
    model="${MODEL_NAME}" \
    forget_split="${FORGET_SPLIT}" \
    retain_split="${RETAIN_SPLIT}" \
    model.model_args.pretrained_model_name_or_path="${FINETUNE_DIR}" \
    model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
    trainer.method_args.training_log_dir="${TRAINING_LOG_DIR}" \
    trainer.method_args.K=${K} \
    trainer.method_args.max_step=${STEPS_PER_EPOCH} \
    trainer.method_args.hessian_mode="${HESSIAN_MODE}" \
    trainer.method_args.damping=${DAMPING} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.seed=${SEED} \
    ++trainer.args.bf16=true \
    trainer.args.efficiency_tracking.enabled=true

echo ""
echo "LMCleaner unlearning complete!"
echo "Model: ${OUTPUT_DIR}"
```

---

## Task 6: Create Baselines Unlearn Script

**Files:**
- Create: `scripts/experiments/expA/04_baselines.sh`

Runs GradDiff, NPO, PDU, UNDIAL sequentially on the same finetuned model.

- [ ] **Step 1: Write baselines script**

```bash
#!/bin/bash
# Experiment A - Step 4: Baseline Unlearning Methods
# Runs: GradDiff, NPO, PDU, UNDIAL
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 4: Baseline Unlearning Methods"
echo "=============================================="
print_config
echo "Methods: ${BASELINE_METHODS[*]}"

# Verify finetuned model exists
if [ ! -d "${FINETUNE_DIR}" ]; then
    echo "ERROR: Finetuned model not found at ${FINETUNE_DIR}"
    echo "Run 01_finetune_full.sh first."
    exit 1
fi

for METHOD in "${BASELINE_METHODS[@]}"; do
    TASK_NAME=$(get_unlearn_task_name "${METHOD,,}")  # lowercase
    OUTPUT_DIR=$(get_unlearn_output_dir "${METHOD,,}")

    echo ""
    echo "=============================================="
    echo "Running ${METHOD}"
    echo "Task: ${TASK_NAME}"
    echo "Output: ${OUTPUT_DIR}"
    echo "=============================================="

    # Build method-specific overrides
    METHOD_OVERRIDES=""
    if [ "${METHOD}" = "PDU" ]; then
        METHOD_OVERRIDES="trainer.method_args.retain_loss_eps=${PDU_RETAIN_LOSS_EPS} \
            trainer.method_args.dual_step_size=${PDU_DUAL_STEP_SIZE} \
            trainer.method_args.dual_warmup_epochs=${PDU_DUAL_WARMUP_EPOCHS}"
    fi

    $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer="${METHOD}" \
        task_name="${TASK_NAME}" \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        retain_split="${RETAIN_SPLIT}" \
        model.model_args.pretrained_model_name_or_path="${FINETUNE_DIR}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.seed=${SEED} \
        ++trainer.args.bf16=true \
        trainer.args.efficiency_tracking.enabled=true \
        ${METHOD_OVERRIDES}

    echo "${METHOD} complete!"
done

echo ""
echo "=============================================="
echo "All baselines complete!"
echo "=============================================="
```

---

## Task 7: Create Basic Eval Script

**Files:**
- Create: `scripts/experiments/expA/05_eval_basic.sh`

Runs standard TOFU evaluation on all methods + original finetuned + retrain.

- [ ] **Step 1: Write basic eval script**

```bash
#!/bin/bash
# Experiment A - Step 5: Basic TOFU Evaluation
# Evaluates all unlearned models + original + retrain
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 5: TOFU Basic Evaluation"
echo "=============================================="
print_config

evaluate_model() {
    local model_path=$1
    local task_name=$2
    local eval_output_dir="${model_path}/evals"

    echo "----------------------------------------------"
    echo "Evaluating: ${task_name}"
    echo "Model: ${model_path}"
    echo "----------------------------------------------"

    if [ ! -d "${model_path}" ]; then
        echo "WARNING: Model not found: ${model_path}. Skipping."
        return
    fi

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_eval" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${eval_output_dir}"

    echo "Done: ${task_name}"
}

# 1. Evaluate original finetuned model (reference)
evaluate_model "${FINETUNE_DIR}" "expA_original"

# 2. Evaluate retrain model (gold standard)
evaluate_model "${RETRAIN_DIR}" "expA_retrain"

# 3. Evaluate LMCleaner
TASK_NAME=$(get_unlearn_task_name "lmcleaner")
evaluate_model "$(get_unlearn_output_dir "lmcleaner")" "${TASK_NAME}"

# 4. Evaluate all baselines
for METHOD in "${BASELINE_METHODS[@]}"; do
    TASK_NAME=$(get_unlearn_task_name "${METHOD,,}")
    evaluate_model "$(get_unlearn_output_dir "${METHOD,,}")" "${TASK_NAME}"
done

echo ""
echo "=============================================="
echo "All basic evaluations complete!"
echo "=============================================="
```

---

## Task 8: Create MIA Eval Script

**Files:**
- Create: `scripts/experiments/expA/06_eval_mia.sh`

- [ ] **Step 1: Write MIA eval script**

```bash
#!/bin/bash
# Experiment A - Step 6: MIA + Privacy Evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "ExpA Step 6: MIA + Privacy Evaluation"
echo "=============================================="

evaluate_mia() {
    local model_path=$1
    local task_name=$2
    local eval_output_dir="${model_path}/evals_mia"

    echo "----------------------------------------------"
    echo "MIA Eval: ${task_name}"
    echo "Model: ${model_path}"
    echo "----------------------------------------------"

    if [ ! -d "${model_path}" ]; then
        echo "WARNING: Model not found: ${model_path}. Skipping."
        return
    fi

    $PYTHON_CMD src/eval.py --config-name=eval.yaml \
        experiment=eval/tofu/mia \
        model="${MODEL_NAME}" \
        forget_split="${FORGET_SPLIT}" \
        holdout_split="${HOLDOUT_SPLIT}" \
        task_name="${task_name}_mia" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        model.tokenizer_args.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
        paths.output_dir="${eval_output_dir}"

    echo "Done: ${task_name}"
}

# Original finetuned
evaluate_mia "${FINETUNE_DIR}" "expA_original"

# Retrain
evaluate_mia "${RETRAIN_DIR}" "expA_retrain"

# LMCleaner
evaluate_mia "$(get_unlearn_output_dir "lmcleaner")" "$(get_unlearn_task_name "lmcleaner")"

# Baselines
for METHOD in "${BASELINE_METHODS[@]}"; do
    evaluate_mia "$(get_unlearn_output_dir "${METHOD,,}")" "$(get_unlearn_task_name "${METHOD,,}")"
done

echo ""
echo "All MIA evaluations complete!"
```

---

## Task 9: Create Results Aggregation Script

**Files:**
- Create: `scripts/experiments/expA/08_aggregate_results.py`

Collects all eval results + efficiency metrics into a single CSV table.

- [ ] **Step 1: Write aggregation script**

```python
#!/usr/bin/env python3
"""Aggregate Experiment A results into summary tables."""
import json
import csv
import os
from pathlib import Path

RESULTS_DIR = Path("saves/results/expA")
UNLEARN_DIR = Path("saves/unlearn")
FINETUNE_DIR = Path("saves/finetune/llama2_7b_tofu_1epoch")
RETRAIN_DIR = Path("saves/finetune/llama2_7b_tofu_retrain")

METHODS = {
    "Original": FINETUNE_DIR,
    "Retrain": RETRAIN_DIR,
    "LMCleaner": UNLEARN_DIR / "expA_lmcleaner_s0",
    "GradDiff": UNLEARN_DIR / "expA_graddiff_s0",
    "NPO": UNLEARN_DIR / "expA_npo_s0",
    "PDU": UNLEARN_DIR / "expA_pdu_s0",
    "UNDIAL": UNLEARN_DIR / "expA_undial_s0",
}


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def extract_basic_metrics(model_dir):
    """Extract metrics from TOFU_EVAL.json / TOFU_SUMMARY.json."""
    metrics = {}
    for subdir in ["evals", "."]:
        summary_path = model_dir / subdir / "TOFU_SUMMARY.json"
        eval_path = model_dir / subdir / "TOFU_EVAL.json"
        for p in [summary_path, eval_path]:
            if p.exists():
                data = load_json(p)
                metrics.update(data)
                break
    return metrics


def extract_mia_metrics(model_dir):
    """Extract MIA metrics."""
    metrics = {}
    for subdir in ["evals_mia", "."]:
        for fname in ["TOFU_SUMMARY.json", "TOFU_EVAL.json"]:
            p = model_dir / subdir / fname
            if p.exists():
                data = load_json(p)
                metrics.update(data)
                break
    return metrics


def extract_efficiency(model_dir):
    """Extract efficiency metrics."""
    p = model_dir / "efficiency_metrics.json"
    if p.exists():
        return load_json(p)
    return {}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for method_name, model_dir in METHODS.items():
        if not model_dir.exists():
            print(f"WARNING: {method_name} not found at {model_dir}")
            continue

        basic = extract_basic_metrics(model_dir)
        mia = extract_mia_metrics(model_dir)
        eff = extract_efficiency(model_dir)

        row = {"Method": method_name}
        row.update({f"basic_{k}": v for k, v in basic.items()})
        row.update({f"mia_{k}": v for k, v in mia.items()})
        row.update({f"eff_{k}": v for k, v in eff.items()})
        rows.append(row)

    if not rows:
        print("No results found!")
        return

    # Write full CSV
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)

    csv_path = RESULTS_DIR / "full_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method"] + [k for k in all_keys if k != "Method"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Results saved to {csv_path}")
    print(f"\nMethods found: {[r['Method'] for r in rows]}")

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT A SUMMARY")
    print("=" * 80)
    for row in rows:
        print(f"\n--- {row['Method']} ---")
        for k, v in sorted(row.items()):
            if k == "Method":
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

---

## Task 10: Create Master Run Script

**Files:**
- Create: `scripts/experiments/expA/run_all.sh`

Orchestrates all steps using tmux sessions.

- [ ] **Step 1: Write master run script**

```bash
#!/bin/bash
# Experiment A: Master Orchestration Script
# Runs all experiment steps sequentially in tmux
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
SKIP_FINETUNE=${SKIP_FINETUNE:-false}
SKIP_RETRAIN=${SKIP_RETRAIN:-false}
START_FROM=${START_FROM:-1}

cd "${PROJECT_ROOT}"

echo "=============================================="
echo "Experiment A: Full Pipeline"
echo "=============================================="
print_config
echo "Start from step: ${START_FROM}"
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
    echo "Step ${step_num}: ${description}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    bash "${SCRIPT_DIR}/${script}"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Step ${step_num} complete: ${description}"
    echo ""
}

# Step 1: Finetune
if [ "${SKIP_FINETUNE}" = "false" ]; then
    run_step 1 "01_finetune_full.sh" "Finetune on full TOFU (1 epoch)"
fi

# Step 2: Retrain
if [ "${SKIP_RETRAIN}" = "false" ]; then
    run_step 2 "02_retrain_retain.sh" "Retrain on retain90 (gold standard)"
fi

# Step 3: LMCleaner
run_step 3 "03_lmcleaner.sh" "LMCleaner Unlearning"

# Step 4: Baselines
run_step 4 "04_baselines.sh" "Baseline Unlearning (GradDiff, NPO, PDU, UNDIAL)"

# Step 5: Basic Eval
run_step 5 "05_eval_basic.sh" "TOFU Basic Evaluation"

# Step 6: MIA Eval
run_step 6 "06_eval_mia.sh" "MIA + Privacy Evaluation"

# Step 7: Aggregate
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Step 7: Aggregate Results"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
$PYTHON_CMD "${SCRIPT_DIR}/08_aggregate_results.py"

echo ""
echo "=============================================="
echo "EXPERIMENT A COMPLETE"
echo "=============================================="
echo ""
echo "Results: ${RESULTS_DIR}/"
echo "Models:"
echo "  Original:  ${FINETUNE_DIR}"
echo "  Retrain:   ${RETRAIN_DIR}"
echo "  LMCleaner: $(get_unlearn_output_dir lmcleaner)"
for METHOD in "${BASELINE_METHODS[@]}"; do
    echo "  ${METHOD}: $(get_unlearn_output_dir ${METHOD,,})"
done
```

---

## Task 11: Make All Scripts Executable and Validate

- [ ] **Step 1: chmod all scripts**

```bash
chmod +x scripts/experiments/expA/*.sh
```

- [ ] **Step 2: Validate config loads**

```bash
cd /app && source scripts/experiments/expA/config.sh && print_config
```

- [ ] **Step 3: Dry-run validate Hydra configs**

Test that all Hydra config combinations resolve without errors:
```bash
# Test finetune config
python src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    task_name=test_dry_run \
    model=Llama-2-7b-hf \
    --cfg job 2>&1 | head -20

# Test unlearn config with LMCleanerBatch
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name=test_dry_run \
    model=Llama-2-7b-hf \
    trainer.method_args.training_log_dir=/tmp \
    --cfg job 2>&1 | head -20
```

---

## Task 12: Launch Experiments in tmux

- [ ] **Step 1: Wait for model download to complete**

Check: `du -sh ~/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf`
Expected: ~13GB when complete.

- [ ] **Step 2: Create tmux session and start experiment**

```bash
tmux new-session -d -s expA "cd /app && bash scripts/experiments/expA/run_all.sh 2>&1 | tee saves/results/expA/run_all.log"
```

- [ ] **Step 3: Monitor progress**

```bash
tmux attach -t expA
# Or check log:
tail -f saves/results/expA/run_all.log
```

---

## Execution Notes

### If a step fails
Use `START_FROM=N` to resume from step N:
```bash
START_FROM=3 bash scripts/experiments/expA/run_all.sh
```

### To add more seeds later
```bash
SEED=1 bash scripts/experiments/expA/run_all.sh
SEED=2 bash scripts/experiments/expA/run_all.sh
```

### Estimated timing (H200, 7B, 1 epoch)
- Finetune: ~15-30 min (250 steps)
- Retrain: ~12-25 min (225 steps, retain90 only)
- LMCleaner: ~5-15 min (correction-based, K=50)
- GradDiff: ~30-60 min (10 epochs default)
- NPO: ~30-60 min (10 epochs default)
- PDU: ~30-60 min (10 epochs default)
- UNDIAL: ~30-60 min (10 epochs default)
- Eval (per model): ~5-10 min
- MIA (per model): ~5-10 min
- **Total estimated: 4-8 hours**
