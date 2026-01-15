# LMCleaner Quick Start Guide

## What is LMCleaner?

LMCleaner is an **online unlearning** algorithm that removes data influence during training (not after). It uses second-order information (Hessian) to accurately track and reverse parameter changes caused by forget samples.

## Quick Start (3 Steps)

### Prerequisites

```bash
# Install dependencies
pip install -e .
```

### Step 1: Enable Training Logging (One-Time Setup)

**Currently**: You need to manually integrate `TrainingLogger` into your training code.

**Future**: This will be automatic with a config flag.

```python
# Add to your training script (e.g., src/train.py)
from trainer.unlearn.training_logger import TrainingLogger

# Before training loop
logger = TrainingLogger(
    log_dir=f"saves/train_logs/{task_name}",
    max_steps=1000,
    mode="batch",
    save_interval=100,
)

# In training loop, after optimizer.step()
logger.register_step(
    step_id=global_step,
    batch_id=batch_idx,
    eta=get_lr(optimizer),
    model=model,
)

# After training
logger.save_to_disk()
```

### Step 2: Run Unlearning

**Option A: Single Command (使用experiment配置)**

```bash
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name=my_lmcleaner_test \
    model=Llama-3.2-1B-Instruct \
    forget_split=forget10 \
    retain_split=retain90 \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    trainer.method_args.training_log_dir=saves/train_logs/pretrain_task_name \
    trainer.method_args.K=800 \
    trainer.method_args.hessian_mode=GGN
```

**注意**: 必须使用`experiment=unlearn/tofu/default`来设置数据配置，或者使用`+forget_split=...`添加新键。

**Option B: Batch Script**

```bash
# Edit scripts/lmcleaner_experiments.sh
# Configure models, splits, K values, etc.

bash scripts/lmcleaner_experiments.sh
```

### Step 3: Evaluate

```bash
python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=forget10 \
    model=Llama-3.2-1B-Instruct \
    task_name=my_lmcleaner_test \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/my_lmcleaner_test \
    paths.output_dir=saves/unlearn/my_lmcleaner_test/evals
```

## Key Parameters

### Essential

- **`training_log_dir`**: Path to training logs (required, no default)
- **`K`**: Truncation window (default: 800, range: 500-1000)
- **`hessian_mode`**: HVP method (default: "GGN", options: "GGN"/"diag"/"exact")

### Optional

- **`damping`**: Numerical stability (default: 1e-4)
- **`apply_immediately`**: Apply at init vs train() (default: false)
- **`audit_dir`**: Save audit logs (default: `${output_dir}/audit`)

## Configuration Templates

### Batch-Level (Recommended)

```yaml
# configs/trainer/LMCleanerBatch.yaml
handler: LMCleanerBatchLevel
method_args:
  training_log_dir: ???
  K: 800
  hessian_mode: GGN
  damping: 1e-4
```

### Sample-Level (Higher Precision)

```yaml
# configs/trainer/LMCleanerSample.yaml
handler: LMCleanerSampleLevel
method_args:
  training_log_dir: ???
  K: 800
  hessian_mode: GGN
  batch_size_at_training: 1  # Original batch size
```

## Comparison: Batch vs Sample Level

| Aspect | Batch-Level | Sample-Level |
|--------|-------------|--------------|
| **Storage** | O((N/B)*p) | O(N*p) |
| **Precision** | Good | Better |
| **Speed** | Fast | Slower |
| **Use Case** | Production | Fine-grained control |
| **Recommended** | ✅ Yes | Only if needed |

## HVP Modes Explained

| Mode | Speed | Accuracy | Memory | Use Case |
|------|-------|----------|--------|----------|
| **GGN** | Medium | High | Medium | **Default, recommended** |
| **diag** | Fast | Low | Low | Quick prototyping |
| **exact** | Slow | Highest | High | Small models, validation |
| **low_rank** | Fast | Medium | Low | TODO (not implemented) |

## Expected Results (TOFU Benchmark)

Using `Llama-3.2-1B-Instruct`, `forget10` split, `K=800`, `hessian_mode=GGN`:

| Metric | Before | LMCleaner | Gold Standard |
|--------|--------|-----------|---------------|
| **Forget Quality** |||
| Extraction Strength ↓ | High | Low | Lowest |
| Truth Ratio ↓ | ~1.0 | ~0.3 | ~0.0 |
| **Retain Performance** |||
| ROUGE-L ↑ | High | ~Same | ~Same |
| **Model Utility** |||
| MMLU ↑ | Baseline | ~Same | ~Same |

*(Exact numbers depend on model and split)*

## Common Issues

### "Training log directory not found"

→ You haven't run Step 1 (training with logging). Run pretraining with `TrainingLogger` first.

### "Step X not found in log"

→ Increase `max_steps` in TrainingLogger or reduce `K` value.

### HVP is too slow

→ Use `hessian_mode=diag` or reduce `K` to 500.

### Out of memory

→ Reduce `K`, use `diag` mode, or process fewer forget samples at once.

## Directory Structure After Running

```
saves/
├── train_logs/
│   └── pretrain_task_name/     # Training logs (Step 1)
│       ├── meta.json
│       ├── batch_index.json
│       └── step_records_*.pkl
├── unlearn/
│   └── my_lmcleaner_test/      # Unlearned model (Step 2)
│       ├── model files...
│       ├── audit/
│       │   └── audit_records.json
│       └── evals/              # Evaluation results (Step 3)
│           └── TOFU_EVAL.json
```

## Next Steps

1. **Read Full Documentation**: `docs/lmcleaner_implementation.md`
2. **Check Algorithm Details**: `Online Unlearning 29fbced0893181e79338ee6be43bcfad.md`
3. **Explore Code**: `src/trainer/unlearn/lmcleaner_*.py`
4. **Run Experiments**: `scripts/lmcleaner_experiments.sh`

## Support

- **Issues**: Check `docs/lmcleaner_implementation.md` Troubleshooting section
- **Questions**: Review algorithm notes in `Online Unlearning 29fbced0893181e79338ee6be43bcfad.md`
- **Examples**: See `scripts/lmcleaner_experiments.sh`

## Citation

If you use LMCleaner in your research, please cite:

```bibtex
@article{lmcleaner2024,
  title={LMCleaner: Online Unlearning for Language Models},
  note={Implementation in Open-Unlearning framework}
}
```
