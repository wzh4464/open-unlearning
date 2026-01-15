# Open-Unlearning Evaluation Metrics

This document provides a comprehensive overview of all evaluation metrics available in the Open-Unlearning framework.

## Overview

The framework provides **17 core metric implementations** across 4 categories, plus integration with lm-evaluation-harness for standard NLP benchmarks.

---

## 1. Memorization Metrics

Location: `src/evals/metrics/memorization.py`

| Metric | Description |
|--------|-------------|
| **probability** | Computes probabilities by data points and reports aggregated average. Measures the model's confidence in generating specific outputs. |
| **probability_w_options** | Normalizes probabilities of correct answers against false answers for open-ended datasets. Used to compare correct vs. incorrect answer probabilities. |
| **rouge** | Calculates ROUGE metrics (text similarity) between generated and reference texts. Supports different ROUGE types (rougeL_recall, rougeL_f1, etc.). |
| **truth_ratio** | Computes the ratio of false/true answer probabilities. Supports three aggregators: `closer_to_1_better` (forget data), `true_better` (non-forget data), `prob_mean` (extent of knowledge). |
| **exact_memorization (EM)** | Measures the fraction of tokens that exactly match the target sequence. Computes token-level accuracy. |
| **extraction_strength (ES)** | Measures the longest suffix of the target sequence that the model can generate correctly. Indicates how much of a sequence can be extracted from the model. |

---

## 2. Privacy Metrics

Location: `src/evals/metrics/privacy.py`

| Metric | Description |
|--------|-------------|
| **ks_test** | Performs a 2-sample Kolmogorov-Smirnov test comparing forget and retain model distributions. Used in TOFU benchmark as "forget_quality" metric. Returns p-value. |
| **privleak** | Compares forget and retain model scores using relative comparison of MIA AUC scores. Designed for MUSE benchmark (uses 1-AUC). |
| **rel_diff** | Computes relative difference between forget and retain model scores as percentage: `(score - ref) / ref * 100`. |

---

## 3. Membership Inference Attack (MIA) Metrics

Location: `src/evals/metrics/mia/`

All MIA metrics return AUC scores for distinguishing between forget and holdout data.

| Metric | Description | Reference |
|--------|-------------|-----------|
| **mia_loss** | LOSS attack using average loss as the membership signal. | Shokri et al. 2017 |
| **mia_min_k** | Min-k% Prob Attack - uses the mean of the bottom k% of token log probabilities. Default k=0.4. | Shi et al. 2023 (arXiv:2310.16789) |
| **mia_min_k_plus_plus** | Enhanced Min-k attack with vocab-wise normalization. Normalizes scores using vocabulary distribution statistics (mean and variance). | - |
| **mia_gradnorm** | Gradient-norm attack measuring the p-norm of gradients w.r.t. model parameters. Supports p=1, 2, or inf. | Maini et al. 2024 (arXiv:2402.17012) |
| **mia_zlib** | ZLIB-normalization attack - normalizes loss by compressed text length using zlib compression. | Carlini et al. 2021 (USENIX Security) |
| **mia_reference** | Reference-based attack comparing loss differences between target and reference models. Requires a reference model path. | - |

---

## 4. Utility Metrics

Location: `src/evals/metrics/utility.py`

| Metric | Description |
|--------|-------------|
| **hm_aggregate** | Computes harmonic mean of multiple pre-computed metrics. Used to aggregate multiple utility scores (e.g., model_utility in TOFU). |
| **classifier_prob** | Uses an external classifier to score generated text. Commonly used for gibberish detection. Returns probability of specified class. |

---

## 5. LM Evaluation Harness Metrics

Location: `src/evals/lm_eval.py`

The framework integrates with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standard benchmarks:

| Metric | Description |
|--------|-------------|
| **MMLU** | Massive Multitask Language Understanding benchmark |
| **WMDP** | Weapons of Mass Destruction Proxy benchmark (bio, cyber, chem variants) |
| **GSM8K** | Grade School Math benchmark |
| *Others* | Any task supported by lm-evaluation-harness |

---

## Benchmark-Specific Configurations

### TOFU Benchmark

Config location: `configs/eval/tofu_metrics/`

**Forget Set Metrics:**

- `forget_Truth_Ratio` - Truth ratio on forget set
- `forget_quality` - KS-test based forget quality
- `forget_Q_A_Prob` - Q&A probability on forget set
- `forget_Q_A_ROUGE` - ROUGE score on forget set
- `forget_Q_A_gibberish` - Gibberish detection on forget set

**Retain Set Metrics:**

- `retain_Q_A_Prob` - Q&A probability on retain set
- `retain_Q_A_ROUGE` - ROUGE score on retain set
- `retain_Truth_Ratio` - Truth ratio on retain set

**Additional Splits:**

- `ra_Q_A_Prob` - Real authors Q&A probability
- `wf_Q_A_Prob` - World facts Q&A probability
- Paraphrased (PARA) and Perturbed (PERT) variants

**Aggregate Metrics:**

- `model_utility` - Harmonic mean of 9 utility metrics

**Privacy Metrics:**

- `privleak`, `extraction_strength`, `exact_memorization`
- All MIA variants

### MUSE Benchmark

Config location: `configs/eval/muse_metrics/`

**Knowledge Metrics:**

- `forget_knowmem_ROUGE` - Knowledge memorization on forget set
- `retain_knowmem_ROUGE` - Knowledge memorization on retain set

**Verbatim Metrics:**

- `forget_verbmem_ROUGE` - Verbatim memorization on forget set

**Privacy Metrics:**

- `privleak`, `extraction_strength`, `exact_memorization`
- All MIA variants
- `forget_gibberish` - Gibberish detection

### WMDP Benchmark

Uses lm-evaluation-harness metrics:

- `wmdp_bio` - Biological hazards accuracy
- `wmdp_cyber` - Cybersecurity hazards accuracy
- `wmdp_chem` - Chemical hazards accuracy
- `mmlu` - General utility preservation

---

## 6. Efficiency Metrics (EfficiencyTracker)

Location: `src/trainer/utils.py`

Tracks computational and storage overhead for unlearning methods:

### Computational Metrics

| Metric | Description |
|--------|-------------|
| **unlearning_time_seconds** | Wall-clock time for the unlearning operation (NOT initial training) |
| **total_steps** | Number of optimization steps during unlearning |
| **peak_gpu_memory_mb** | Maximum GPU memory allocated during unlearning |
| **tokens_per_second** | Unlearning throughput (tokens processed per second) |

### Storage Metrics

| Metric | Description |
|--------|-------------|
| **model_size_mb** | Size of the model parameters in memory |
| **requires_retain_set** | Whether the method needs to store the retain set (auto-detected) |

### Usage

```bash
# Enable efficiency tracking
uv run python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer.args.efficiency_tracking.enabled=true \
  task_name=MY_EXPERIMENT

# Run benchmark comparison
bash scripts/benchmark_efficiency.sh

# Compare results
python scripts/compare_efficiency.py
```

See `docs/EFFICIENCY_TRACKING.md` for detailed usage.

---

## 7. Training Process Metrics (TrainingLogger)

Location: `src/trainer/training_logger.py`

The TrainingLogger records training process data for LMCleaner online unlearning:

| Metric | Description |
|--------|-------------|
| **eta (η[t])** | Learning rate at each training step |
| **u[t]** | Parameter update vector: θ[t+1] - θ[t] |
| **gbar[t]** | Average gradient at each step |
| **diag_H** | Diagonal Hessian approximation (optional) |
| **sample_indices** | Sample indices per batch (light storage mode) |
| **rng_state** | Random number generator state for batch reconstruction |

### Storage Modes

| Mode | Disk Usage | Description |
|------|------------|-------------|
| **Light** | ~50 MB | Saves indices + RNG state, reconstructs batches on-demand |
| **Heavy** | ~5 GB | Saves complete batch tensors |
| **Diagonal Hessian** | ~500 MB | Pre-computes diagonal Hessian approximation |

### Performance Metrics

| Metric | Relative Speed | Accuracy |
|--------|---------------|----------|
| **diag (pre-computed)** | 100x | ~80% |
| **GGN** | 1x | ~95% |
| **exact** | 0.5x | 100% |

See `docs/TRAININGLOGGER_IMPLEMENTATION.md` for detailed usage.

---

## Usage Example

```bash
# Run evaluation with specific metrics
uv run python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  task_name=SAMPLE_EVAL
```

Metrics can be configured via Hydra config files or command-line overrides.
