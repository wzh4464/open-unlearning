# Efficiency Metrics Tracking

This document describes how to track and compare computational and storage overhead across different unlearning methods.

## Overview

The efficiency tracking feature provides metrics for **unlearning operations specifically**, not initial model training:
- **Computational Overhead**: Unlearning time, GPU memory usage, throughput
- **Storage Overhead**: Model size, checkpoint storage requirements

**Important**: This tracks the cost of the unlearning operation itself (e.g., gradient ascent steps, NPO optimization), not the time to train the original model. This distinction is crucial for fair comparison of unlearning methods.

## Usage

### Enable Efficiency Tracking

Add the following to your training command:

```bash
python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer.args.efficiency_tracking.enabled=true \
  task_name=MY_EXPERIMENT
```

### Run Efficiency Benchmark

Compare multiple unlearning methods:

```bash
bash scripts/benchmark_efficiency.sh
```

This will run GradAscent, NPO, and LMCleanerSampleLevel methods and save efficiency metrics for each.

### Compare Results

View a comparison table of all experiments:

```bash
python scripts/compare_efficiency.py
```

Example output:
```
================================================================================
Unlearning Efficiency Metrics Comparison
================================================================================
Experiment                     Time(s)      Steps    GPU Mem(MB)     Tokens/s
--------------------------------------------------------------------------------
efficiency_GradAscent_forget01 245.32       100      8234.56         1234.56
efficiency_NPO_forget01        312.45       100      9123.45         987.65
efficiency_LMCleanerSample_... 289.12       100      8567.89         1098.76
================================================================================

Best Performers:
  Fastest: efficiency_GradAscent_forget01 (245.32s)
  Most Memory Efficient: efficiency_GradAscent_forget01 (8234.56 MB)
  Highest Throughput: efficiency_GradAscent_forget01 (1234.56 tokens/s)
```

## Metrics Explained

### Computational Metrics

- **unlearning_time_seconds**: Wall-clock time for the unlearning operation (NOT initial training)
- **total_steps**: Number of optimization steps during unlearning
- **peak_gpu_memory_mb**: Maximum GPU memory allocated during unlearning
- **tokens_per_second**: Unlearning throughput (tokens processed per second)

### Storage Metrics

- **model_size_mb**: Size of the model parameters in memory
- **requires_retain_set**: Whether the method needs to store the retain set (automatically detected based on trainer type)

## Output Format

Metrics are saved as JSON in the experiment output directory:

```json
{
  "unlearning_time_seconds": 245.32,
  "total_steps": 100,
  "peak_gpu_memory_mb": 8234.56,
  "tokens_per_second": 1234.56,
  "model_size_mb": 13421.23,
  "requires_retain_set": false
}
```

## Configuration

Create a custom efficiency tracking config in `configs/efficiency.yaml`:

```yaml
efficiency_tracking:
  enabled: true
```

Then reference it in your experiment:

```bash
python src/train.py \
  --config-name=unlearn.yaml \
  +efficiency=efficiency \
  ...
```

## Integration with Existing Experiments

The efficiency tracker integrates seamlessly with existing experiments. Simply add the `efficiency_tracking.enabled=true` parameter to any training command.

## Motivation

This feature addresses a limitation identified in the MUSE benchmark paper, which noted that "deployers may expect other capabilities... and may prefer unlearning algorithms that are both computationally efficient and storage-wise cheap."

By tracking these metrics, researchers can:
1. Compare efficiency-effectiveness trade-offs across methods
2. Identify methods suitable for resource-constrained environments
3. Understand computational costs for production deployment
