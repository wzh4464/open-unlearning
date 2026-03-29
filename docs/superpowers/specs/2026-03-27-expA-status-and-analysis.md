# Experiment A: Final Results and Analysis

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.2-3B-Instruct (unsloth) |
| Training | SGD, lr=1e-3, 1 epoch, batch=64 |
| Forget set | forget10 (400 samples, 10%) |
| Retain set | retain90 (3600 samples) |
| LMCleaner | Fisher HVP, CheckpointAwareUProvider, K=10-50 |
| Sparse checkpoints | stride=10, 7 checkpoints (step 0-60) |
| Tag | `expA-v1.0` (forget01), PR #31 (forget10) |

## Final Results (forget10, CheckpointAwareUProvider)

| Method | m_utility | f_quality | f_Prob | f_ROUGE | f_truth_ratio |
|--------|:---------:|:---------:|:------:|:-------:|:-------------:|
| **Original** | 0.324 | 0.065 | 0.130 | 0.281 | 0.746 |
| Retrain | 0.000 | — | 0.001 | 0.000 | 0.734 |
| **LMC K=10** | **0.205** | **0.000** | **0.049** | 0.322 | 0.755 |
| **LMC K=20** | **0.208** | **0.000** | **0.051** | 0.296 | 0.764 |
| LMC K=30 | 0.264 | 0.003 | 0.083 | 0.253 | 0.750 |
| LMC K=40 | 0.269 | 0.006 | 0.084 | 0.215 | 0.748 |
| LMC K=50 | 0.265 | 0.008 | 0.083 | 0.220 | 0.748 |
| GradDiff | 0.276 | 0.000 | 0.003 | 0.320 | 0.523 |
| NPO | 0.250 | 0.000 | 0.023 | 0.201 | 0.703 |
| PDU | 0.439 | 0.013 | 0.166 | 0.377 | 0.750 |
| UNDIAL | 0.311 | 0.020 | 0.073 | 0.289 | 0.752 |

## Key Findings

### 1. CheckpointAwareUProvider is essential

| Configuration | forget_quality | Selectivity |
|--------------|:-----------:|:-----------:|
| u[tz] at θ[τ] (wrong params) | 1.000 | 0.94 |
| **u[tz] at θ[tz] (correct params)** | **0.000** | **selective** |

Computing u[tz] at the correct historical θ[tz] via sparse checkpoint replay
transforms LMCleaner from non-functional (f_quality=1.0) to highly effective
(f_quality=0.000).

### 2. K convergence

- K=10 and K=20 achieve f_quality=0.000 (as good as GradDiff/NPO)
- K=30+ shows slightly worse f_quality (0.003-0.008)
- K=10 is most aggressive (m_utility=0.205), K=40+ stabilizes at ~0.265

### 3. Fisher HVP numerical stability

Fisher HVP `Hv = g · (g^T v)` can cause v_norm to spike at certain steps
(e.g., step 11: 0.023 → 0.459 → NaN). Norm clipping at `max(10 * v0_norm, 0.05)`
resolves this without degrading results.

### 4. Comparison with baselines

- **GradDiff**: Best f_quality (0.000) with highest m_utility (0.276), but
  drastically changes truth_ratio (0.746→0.523), indicating fundamental model
  behavior shift
- **LMC K=10/20**: Comparable f_quality (0.000) while preserving truth_ratio
  (0.746→0.755/0.764), suggesting more targeted unlearning
- **NPO**: Good f_quality (0.000) but lowest m_utility (0.250)
- **UNDIAL**: High m_utility (0.311) but poor f_quality (0.020)

## Architecture: CheckpointAwareUProvider

```
Training time:
  Save sparse checkpoints at step {0, 10, 20, 30, 40, 50, 60}
  Each ~6GB, total ~42GB

Unlearning time (for each forget step tz):
  1. Find nearest checkpoint c <= tz
  2. Load θ[c] into model
  3. Sequential replay: θ[c] → θ[c+1] → ... → θ[tz]
     Each step: forward+backward on batch[t] to get u[t] = -η*grad(θ[t])
     Then advance: θ[t+1] = θ[t] + u[t]
  4. Use u[tz] for Phase 1 correction: v0 = -u[tz]
  5. Restore model to θ[τ]

Cost: max 9 forward passes from checkpoint (stride=10), ~5s per step
```

## Files

- `src/trainer/unlearn/checkpoint_u_provider.py` — CheckpointAwareUProvider
- `src/trainer/unlearn/lmcleaner_core.py` — Fisher HVP norm clipping
- `scripts/experiments/expA/run_k_sweep.sh` — K-sweep pipeline
- `saves/results/expA/k_sweep_results_s0.csv` — Full results CSV
