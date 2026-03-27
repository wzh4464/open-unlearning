# Experiment A: Current Status and Analysis

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.2-3B-Instruct (unsloth) |
| Training | SGD, lr=1e-3, 1 epoch, batch=64 |
| Forget set | forget01 (40 samples, 1%) |
| Retain set | retain99 (3960 samples) |
| LMCleaner | Fisher HVP, use_historical_params=False, K=10-50 |
| Tag | `expA-v1.0` |

## Results

| Method | model_utility | forget_quality | forget_Q_A_Prob | retain_Q_A_Prob | forget_truth_ratio |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Original** | 0.324 | 1.000 | 0.168 | 0.123 | 0.764 |
| **Retrain** | 0.319 | -- | 0.164 | 0.119 | 0.763 |
| LMCleaner K=10 | 0.257 | 1.000 | 0.096 | 0.067 | 0.754 |
| LMCleaner K=20 | 0.270 | 1.000 | 0.107 | 0.075 | 0.759 |
| LMCleaner K=50 | 0.271 | 1.000 | 0.111 | 0.076 | 0.760 |
| GradDiff | 0.329 | 0.014 | 0.026 | 0.116 | 0.764 |
| NPO | 0.268 | 0.097 | 0.025 | 0.069 | 0.746 |
| PDU | 0.246 | 0.266 | 0.023 | 0.056 | 0.795 |
| UNDIAL | 0.311 | 1.000 | 0.084 | 0.111 | 0.765 |

## Problem: forget_quality Insensitive to LMCleaner

### Root Cause

`forget_quality` = KS test p-value on `1/truth_ratio`, comparing model vs retrain.

`truth_ratio` = P(correct answer) / P(incorrect answer). This is a **ratio metric**.

LMCleaner's correction **uniformly scales down all probabilities** (both correct and
incorrect answers), so the ratio stays constant:

```
Original:  truth_ratio = P_correct / P_incorrect = 0.84
LMCleaner: truth_ratio = (0.57 * P_correct) / (0.57 * P_incorrect) = 0.84  (unchanged!)
```

Evidence:
- forget_Q_A_Prob: -43% (correct answer probability drops)
- retain_Q_A_Prob: -46% (retain drops equally -- collateral damage)
- truth_ratio: 0.764 -> 0.754 (only -1.2%, within noise)
- KS test: statistic=0.075, p=0.9999 (cannot distinguish from retrain)

In contrast, GradDiff **selectively** reduces correct-answer probability while leaving
incorrect-answer probability intact, changing the ratio:
- truth_ratio: 0.764 -> 1.195 (+56%)
- KS test: statistic=0.350, p=0.014 (clearly distinguished)

### Why LMCleaner's Correction is Non-Selective

`use_historical_params=False` computes HVP at current parameters theta[tau] instead of
historical theta[s]. This makes the correction vector v a "global perturbation" rather
than a targeted removal of the forget batch's influence.

The correction damages retain knowledge as much as forget knowledge because it
doesn't account for how the model parameters evolved through training.

## Proposed Solutions

### Solution 1: Alternative Metrics (immediate)

Use metrics sensitive to absolute probability changes, not just ratios:
- `forget_Q_A_Prob` directly (already shows -43% drop)
- MIA metrics (membership inference)
- `privleak` (already shows change: 0.48 -> 5.15)

### Solution 2: use_historical_params=True (requires I/O optimization)

The theoretically correct fix. Needs LazyUProvider to recompute u[t] on-the-fly
(0.6s/step) instead of loading from disk (38s/step). Current blocker: OOM on
GGN HVP with 3B model (needs 130GB+ for 2nd order gradients).

Options:
- Fisher HVP with historical params (less memory, less accurate)
- Smaller model (1B) where GGN fits in memory
- Gradient checkpointing + micro-batching for GGN

### Solution 3: Targeted Correction via Gradient Projection

Instead of applying v directly, project it onto the subspace that affects forget
samples more than retain samples. This can be done post-hoc without retraining.

## Revised Evaluation with Alternative Metrics

| Method | forget_quality | KS(Prob) p | Selectivity | privleak |
|--------|:---:|:---:|:---:|:---:|
| Original | 1.000 | 1.000 | -- | 0.48 |
| Retrain | -- | -- | -- | 4.38 |
| LMC K=10 | 1.000 | **0.001** | 0.94 | 5.15 |
| LMC K=50 | 1.000 | 0.054 | 0.90 | -0.96 |
| GradDiff | 0.014 | **0.000** | 13.84 | 88.02 |
| NPO | 0.097 | **0.000** | 1.94 | 88.86 |

### Key Findings

1. **KS test on absolute Prob detects LMCleaner** (p=0.001 for K=10), while
   truth_ratio-based forget_quality is blind to it. The metric choice matters.

2. **Selectivity is 0.94** -- LMCleaner damages forget and retain equally.
   GradDiff achieves 13.84x selectivity. Root cause: `use_historical_params=False`.

3. **privleak confirms LMCleaner works**: 0.48 -> 5.15 (closer to Retrain's 4.38).

4. **LMC overshoots**: forget_prob drops to 0.096 (40% below Retrain's 0.164).
   Correction magnitude needs scaling down.

### Next Steps

1. Enable `use_historical_params=True` with LazyUProvider for selective correction
2. Add correction scaling factor to control overshoot
3. Report KS(Prob) alongside forget_quality in paper
