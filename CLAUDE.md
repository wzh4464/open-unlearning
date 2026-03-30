# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Open-Unlearning is a unified framework for LLM unlearning research. It supports TOFU, MUSE, and WMDP benchmarks, 12+ unlearning methods, 10+ evaluation metrics, and 7+ model architectures. Built on Hydra for configuration and HuggingFace Transformers/Accelerate for training.

## Commands

```bash
# Install dependencies
uv sync                          # Core dependencies
uv sync --extra dev              # Include ruff, pytest

# Code quality
make quality                     # Lint + format check (ruff)
make style                       # Auto-fix lint + format

# Tests (runs CPU-only via CUDA_VISIBLE_DEVICES=)
make test                        # All tests
CUDA_VISIBLE_DEVICES= uv run pytest tests/test_eval_metrics.py -v          # Single file
CUDA_VISIBLE_DEVICES= uv run pytest tests/test_eval_metrics.py::TestClass::test_method -v  # Single test

# Training
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN

# Evaluation
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct task_name=SAMPLE_EVAL

# Distributed training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default task_name=DISTRIBUTED_TRAIN

# Data setup
python setup_data.py --eval      # Download evaluation reference logs
```

## Architecture

### Entry Points & Pipeline Flow

**Training** (`src/train.py`): Hydra config → `get_model()` → `get_data()` → `get_collators()` → `load_trainer()` → `train()` → save model

**Evaluation** (`src/eval.py`): Hydra config → `get_model()` → `get_evaluators()` → for each evaluator: `evaluate(model, tokenizer, template_args)` → save JSON results

Both use `@hydra.main()` with configs from `configs/`.

### Registry Pattern (Central to Everything)

All major components use the same pattern — a module-level `REGISTRY` dict, a `_register_*()` function, and a `load_*()` / `get_*()` factory:

| Registry | Location | Key = | Value = |
|---|---|---|---|
| `TRAINER_REGISTRY` | `src/trainer/__init__.py` | Class name (e.g., `"GradAscent"`) | Class |
| `EVALUATOR_REGISTRY` | `src/evals/__init__.py` | Class name (e.g., `"TOFUEvaluator"`) | Class |
| `DATASET_REGISTRY` | `src/data/__init__.py` | Class name (e.g., `"QADataset"`) | Class |
| `COLLATOR_REGISTRY` | `src/data/__init__.py` | Class name | Class |
| `MODEL_REGISTRY` | `src/model/__init__.py` | Class name (e.g., `"AutoModelForCausalLM"`) | Class |
| `METRICS_REGISTRY` | `src/evals/metrics/__init__.py` | Metric name | `UnlearningMetric` |

Config YAML files reference registry keys via `handler:` field. E.g., `configs/trainer/GradAscent.yaml` has `handler: GradAscent` which maps to `TRAINER_REGISTRY["GradAscent"]`.

### Trainer Hierarchy

```
HF Trainer
  └─ FinetuneTrainer (src/trainer/base.py) — adds callbacks, training logger support
       └─ UnlearnTrainer (src/trainer/unlearn/base.py) — adds ref model, DeepSpeed handling
            └─ GradAscent, GradDiff, NPO, DPO, SimNPO, RMU, UNDIAL, ...
```

Each method overrides `compute_loss(model, inputs, return_outputs)`. Method-specific hyperparams come from `trainer.method_args` in config and are passed as kwargs to the constructor.

### Data Pipeline for Unlearning

`get_data(cfg, mode="unlearn")` loads separate forget/retain datasets, then wraps them in `ForgetRetainDataset` (in `src/data/unlearn.py`). This composite dataset anchors on one split (default: forget) and randomly samples from the other, yielding `{'forget': ..., 'retain': ...}` pairs. The `anchor` parameter controls which split drives iteration length.

### Evaluation System

`Evaluator` base class (`src/evals/base.py`) loads metrics from config, runs them, caches results as JSON. Each metric is an `UnlearningMetric` (`src/evals/metrics/base.py`) wrapping a metric function. Metrics can declare dependencies (`pre_compute` metrics). Results saved to `saves/eval/{task_name}/`.

Metric configs use Hydra's `@package` directive to nest into the evaluation config tree (e.g., `@package eval.tofu.metrics.forget_Q_A_ROUGE`).

### Configuration System

Hydra configs compose hierarchically:

- **Base**: `configs/train.yaml`, `configs/unlearn.yaml`, `configs/eval.yaml`
- **Components**: `configs/trainer/*.yaml`, `configs/model/*.yaml`, `configs/data/datasets/*.yaml`
- **Experiments**: `configs/experiment/` — pre-composed setups that override all component defaults
- **Metrics**: `configs/eval/tofu_metrics/*.yaml` — use `@package` for nesting

Key override syntax:
- Dot notation: `trainer.method_args.K=1000`
- Add new keys: `++trainer.args.bf16=true`
- Override model path: `model.model_args.pretrained_model_name_or_path=path/to/model`

### Output Structure

All outputs go to `saves/{mode}/{task_name}/` where mode is `train`, `unlearn`, or `eval`.

### Adding a New Unlearning Method

1. Create `src/trainer/unlearn/mymethod.py` — class inherits `UnlearnTrainer`, override `compute_loss()`
2. Import and register in `src/trainer/__init__.py`: `_register_trainer(MyMethod)`
3. Create `configs/trainer/MyMethod.yaml` with `handler: MyMethod` and `method_args:`
4. Optionally create experiment config in `configs/experiment/unlearn/`

### Adding a New Metric

1. Create metric function in `src/evals/metrics/`
2. Register as `UnlearningMetric` in `src/evals/metrics/__init__.py`
3. Create config in `configs/eval/tofu_metrics/` with `@package` directive
4. Add to defaults list in the benchmark's eval config

## Key Technical Details

- `pytest` config sets `pythonpath = ["src"]` so imports use module names directly (e.g., `from trainer.base import FinetuneTrainer`)
- `template_args` (chat template format) flows through the entire pipeline: model config → trainer → data processing → evaluation
- Reference models for KL/DPO methods: `UnlearnTrainer` handles loading; uses ZeRO-0 when model uses ZeRO-3
- `ForgetRetainDataset` indices differ from original finetune dataset indices — relevant for LMCleaner batch reconstruction
- TrainingLogger and callbacks (EfficiencyTracker, SpectralNorm) only initialize on rank 0 in distributed training
- Transformers ≥5.0 compatibility: `load_trainer()` auto-detects whether to use `tokenizer` or `processing_class` kwarg

## Hydra Gotchas

- Trainer configs define `_target_` class and `method_args`; HuggingFace `TrainingArguments` go under `trainer.args`
- `forget_split` and `retain_split` are top-level overrides (not nested under `data.`)
- Experiment configs use `- override /model:` syntax (with leading slash) to override defaults groups

## Temp File Location

```bash
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR/claude"
```
