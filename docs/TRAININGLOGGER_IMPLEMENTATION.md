# TrainingLogger Implementation Guide

## Overview

The TrainingLogger is a flexible training log recording system that supports three storage modes for LMCleaner online unlearning. It has been integrated into the FinetuneTrainer and can be configured via Hydra configuration files.

## Architecture

### Core Components

1. **TrainingLogger** (`src/trainer/training_logger.py`)
   - Main logger class that records training steps
   - Supports configurable storage modes
   - Handles RNG state and sample indices

2. **BatchReconstructor** (`src/trainer/training_logger.py`)
   - Reconstructs batch data from indices and RNG state
   - Used during unlearning when batch data is not stored

3. **FinetuneTrainer** (`src/trainer/base.py`)
   - Extended to support TrainingLogger
   - Automatically logs each training step
   - Saves logs at the end of training

4. **LMCleanerBatchLevel & LMCleanerSampleLevel** (`src/trainer/unlearn/`)
   - Support batch reconstruction during unlearning
   - Automatically initialize BatchReconstructor when needed

## Storage Modes

### 1. Light Storage Mode (Recommended)

**Configuration:**

```yaml
trainer:
  args:
    training_logger:
      enabled: true
      log_dir: saves/train_logs/${task_name}
      mode: batch
      save_indices_only: true
      save_batch_data: false
      save_rng_state: true
      save_interval: 100
```

**Features:**

- Saves only sample indices and RNG state per step
- Minimal disk usage (~MB for thousands of steps)
- Reconstructs batches during unlearning
- Requires access to original training dataset

**Use cases:**

- Default choice for most experiments
- When disk space is limited
- When original dataset is available during unlearning

### 2. Heavy Storage Mode

**Configuration:**

```yaml
trainer:
  args:
    training_logger:
      enabled: true
      log_dir: saves/train_logs/${task_name}
      mode: batch
      save_indices_only: false
      save_batch_data: true
      save_rng_state: false
      save_interval: 50
```

**Features:**

- Saves complete batch tensors per step
- High disk usage (~GB for thousands of steps)
- No reconstruction needed during unlearning
- Most accurate HVP computation

**Use cases:**

- When exact batch reproduction is critical
- When original dataset unavailable during unlearning
- For debugging and validation

### 3. Diagonal Hessian Mode

**Configuration:**

```yaml
trainer:
  args:
    training_logger:
      enabled: true
      log_dir: saves/train_logs/${task_name}
      mode: batch
      compute_diag_h: true
      save_indices_only: true
      save_batch_data: false
      save_rng_state: true
```

**Features:**

- Pre-computes diagonal Hessian approximation
- Medium disk usage
- Fast HVP computation (O(p) instead of O(p²))
- Works with `hessian_mode=diag`

**Use cases:**

- When speed is critical
- For large models where full HVP is expensive
- When diagonal approximation is sufficient

## Usage Examples

### Training with TrainingLogger

```bash
# Light storage (recommended)
uv run python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  task_name=my_finetune_light \
  model=Llama-3.2-1B-Instruct \
  trainer.args.training_logger.enabled=true \
  trainer.args.training_logger.log_dir=saves/train_logs/my_finetune_light \
  trainer.args.training_logger.save_indices_only=true \
  trainer.args.training_logger.save_rng_state=true

# Heavy storage
uv run python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  task_name=my_finetune_heavy \
  model=Llama-3.2-1B-Instruct \
  trainer.args.training_logger.enabled=true \
  trainer.args.training_logger.log_dir=saves/train_logs/my_finetune_heavy \
  trainer.args.training_logger.save_batch_data=true

# Diagonal Hessian
uv run python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  task_name=my_finetune_diag \
  model=Llama-3.2-1B-Instruct \
  trainer.args.training_logger.enabled=true \
  trainer.args.training_logger.log_dir=saves/train_logs/my_finetune_diag \
  trainer.args.training_logger.compute_diag_h=true
```

### Unlearning with Batch Reconstruction

```bash
# Batch-level unlearning with GGN Hessian
uv run python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  task_name=my_unlearn_ggn \
  model.model_args.pretrained_model_name_or_path=saves/train/my_finetune_light \
  trainer=LMCleanerBatch \
  trainer.method_args.training_log_dir=saves/train_logs/my_finetune_light \
  trainer.method_args.K=800 \
  trainer.method_args.hessian_mode=GGN

# Batch-level unlearning with diagonal Hessian
uv run python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  task_name=my_unlearn_diag \
  model.model_args.pretrained_model_name_or_path=saves/train/my_finetune_diag \
  trainer=LMCleanerBatch \
  trainer.method_args.training_log_dir=saves/train_logs/my_finetune_diag \
  trainer.method_args.K=800 \
  trainer.method_args.hessian_mode=diag

# Sample-level unlearning
uv run python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  task_name=my_unlearn_sample \
  model.model_args.pretrained_model_name_or_path=saves/train/my_finetune_light \
  trainer=LMCleanerSample \
  trainer.method_args.training_log_dir=saves/train_logs/my_finetune_light \
  trainer.method_args.K=800 \
  trainer.method_args.hessian_mode=GGN \
  trainer.method_args.batch_size_at_training=4
```

### Using the Experiment Script

```bash
# Run with light storage mode (default)
STORAGE_MODE=light bash scripts/lmcleaner_experiments.sh

# Run with heavy storage mode
STORAGE_MODE=heavy bash scripts/lmcleaner_experiments.sh

# Run with diagonal Hessian mode
STORAGE_MODE=diag bash scripts/lmcleaner_experiments.sh
```

## Implementation Details

### Training Phase

1. **Initialization** (`src/trainer/__init__.py:load_trainer()`)
   - Reads `training_logger` config from trainer args
   - Creates TrainingLogger instance if enabled
   - Passes logger to FinetuneTrainer

2. **Per-Step Logging** (`src/trainer/base.py:training_step()`)
   - Captures model state before optimization
   - Logs after each training step:
     - Learning rate (eta)
     - Parameter update vector (u) or gradient (gbar)
     - Sample indices (if available)
     - RNG state (if enabled)
     - Batch data (if enabled)
     - Diagonal Hessian (if enabled)

3. **Saving** (`src/trainer/base.py:train()`)
   - Saves logs to disk at configured intervals
   - Final save at end of training
   - Files saved:
     - `meta.json`: Configuration and metadata
     - `batch_index.json`: Batch ID to step mapping
     - `sample_indices.json`: Sample indices per step (light mode)
     - `rng_states_*.pkl`: RNG states (light mode)
     - `step_records_*.pkl`: Step records with vectors

### Unlearning Phase

1. **Loading Logs** (`src/trainer/unlearn/lmcleaner_*.py:__init__()`)
   - Loads TrainingLogger from disk
   - Reads all saved data

2. **Batch Reconstruction** (`src/trainer/unlearn/lmcleaner_*.py:_apply_unlearning()`)
   - Creates BatchReconstructor if needed
   - Provides access to original training dataset
   - Reconstructs batches on-the-fly during HVP computation

3. **HVP Computation** (`src/trainer/unlearn/lmcleaner_core.py:hvp_apply()`)
   - Checks if batch_data is available
   - For diag mode with pre-computed diag_H: uses diag_H directly
   - Otherwise: reconstructs batch using BatchReconstructor
   - Computes Hessian-vector product

## Configuration Reference

### TrainingLogger Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable/disable logging |
| `log_dir` | str | - | Directory to save logs |
| `mode` | str | "batch" | Logging mode: "batch" or "sample" |
| `max_steps` | int | 1000 | Max steps in circular buffer |
| `save_interval` | int | 100 | Save to disk every N steps |
| `save_batch_data` | bool | false | Save full batch tensors |
| `save_indices_only` | bool | false | Save only sample indices |
| `save_rng_state` | bool | false | Save RNG state per step |
| `compute_diag_h` | bool | false | Compute diagonal Hessian |
| `batch_size_at_training` | int | None | Batch size (for reconstruction) |

### LMCleaner Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_log_dir` | str | - | Path to training logs |
| `K` | int | 800 | Truncation window size |
| `hessian_mode` | str | "GGN" | HVP mode: "GGN", "diag", "exact" |
| `damping` | float | 1e-4 | Damping coefficient λ |
| `apply_immediately` | bool | false | Apply unlearning in **init** |
| `audit_dir` | str | None | Directory for audit logs |

## Performance Considerations

### Storage Requirements

| Mode | Disk Usage (1B model, 1000 steps) |
|------|-----------------------------------|
| Light | ~50 MB |
| Heavy | ~5 GB |
| Diag | ~500 MB |

### Computation Time

| HVP Mode | Relative Speed | Accuracy |
|----------|---------------|----------|
| diag (pre-computed) | 100x | ~80% |
| GGN | 1x | ~95% |
| exact | 0.5x | 100% |

### Memory Usage

- Light mode: Minimal overhead during unlearning
- Heavy mode: No overhead (batch data pre-loaded)
- Diag mode: Minimal overhead (diag_H pre-computed)

## Troubleshooting

### Common Issues

1. **"No batch data found" error**
   - Cause: Using GGN/exact mode with light storage, but dataset unavailable
   - Solution: Provide dataset and data_collator, or use heavy storage mode

2. **"Failed to reconstruct batch" error**
   - Cause: Original dataset structure changed or indices invalid
   - Solution: Ensure dataset matches training dataset, check indices

3. **High memory usage**
   - Cause: Using heavy storage with large batches
   - Solution: Switch to light storage or reduce save_interval

4. **Slow unlearning**
   - Cause: Using exact HVP or reconstructing many batches
   - Solution: Use diag mode or GGN, increase K to reduce HVP calls

## Best Practices

1. **Choose the right mode:**
   - Start with light mode
   - Use heavy mode only if reconstruction fails
   - Use diag mode for speed-critical experiments

2. **Set appropriate intervals:**
   - save_interval=100 for light mode (frequent saves are cheap)
   - save_interval=50 for heavy mode (balance disk I/O)
   - save_interval=100 for diag mode (moderate cost)

3. **Manage disk space:**
   - Clean up old logs after experiments
   - Use light mode for exploration
   - Use heavy mode only for final runs

4. **Validate reconstruction:**
   - Test with small dataset first
   - Verify gradients match original
   - Compare light vs heavy results

## References

- LMCleaner paper: [link to be added]
- Implementation details: `docs/LMCLEANER_IMPLEMENTATION_SUMMARY.md`
- Quickstart guide: `docs/LMCLEANER_QUICKSTART.md`
