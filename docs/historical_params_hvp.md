# Historical Parameter HVP: θ[s] vs θ[τ]

## Background

LMCleaner's Algorithm 1 defines the truncated influence propagation as:

```
v[s+1] = (I - η_s · H[s]) · v[s]
```

where `H[s] := ∇²L(θ[s])` is the Hessian evaluated at the **historical** parameters `θ[s]` — i.e., the parameters the model had at training step `s`, not the current (post-training) parameters `θ[τ]`.

The paper explicitly requires this in:
- **Algorithm 1, Line 6**: `COMPUTEHVP(θ[s], v)`
- **Section 5**: stores parameter snapshots; HVP computed via autograd on demand
- **Appendix B**: `P[s] := I - η_s H_s(θ[s])` — propagator bound to historical params
- **Appendix C.1**: `Ṕ[t] := I - η_t Ĥ_t(θ[t])`

## The Tradeoff

| | `use_historical_params=True` (default) | `use_historical_params=False` |
|---|---|---|
| **Correctness** | Matches paper Algorithm 1 exactly | Approximation: uses θ[τ] for all steps |
| **Extra GPU memory** | ~1x model params (flat `theta_current` vector) | None |
| **Extra IO** | Reads u[t] vectors from chunk files for propagation window | None |
| **When it matters** | Large K, large learning rate, long training | Small K, small lr, short training (θ[s] ≈ θ[τ]) |

### Memory example (Llama-3.2-1B, float32)

- Model params: ~1.2B → ~4.8 GB
- `theta_current` flat vector: +4.8 GB GPU
- u[t] vectors: loaded one-at-a-time from disk, not all stored simultaneously
- Total extra GPU: **~4.8 GB**

For bfloat16: ~2.4 GB extra.

## How It Works

When `use_historical_params=True` and u[t] vectors are available in the training log:

1. Save current parameters as `θ[τ]`
2. Reconstruct `θ[start]` = `θ[τ] - Σ_{t=start}^{τ-1} u[t]`
3. For each propagation step `s`:
   - Model is at `θ[s]` → compute HVP here
   - Advance: `θ[s+1] = θ[s] + u[s]`
4. Restore `θ[τ]` after propagation

When u[t] vectors are not available (e.g., training log only saved indices), it automatically falls back to using `θ[τ]`.

## Configuration

### Via Hydra config

```yaml
# configs/trainer/LMCleanerBatch.yaml
method_args:
  use_historical_params: true   # default: paper-aligned
  # use_historical_params: false  # memory-constrained environments
```

### Via command line

```bash
python src/train.py --config-name=unlearn.yaml \
  trainer=LMCleanerBatch \
  trainer.method_args.use_historical_params=false
```

### Via Python

```python
trainer = LMCleanerBatchLevel(
    training_log_dir="path/to/logs",
    use_historical_params=False,  # disable for tight GPU budget
    ...
)
```

## When to Disable

Consider setting `use_historical_params=False` if:

- **GPU memory is tight**: you cannot spare an extra ~1x model params of VRAM
- **K is small** (e.g., K < 64): the approximation error from using θ[τ] is small because θ[s] ≈ θ[τ] when the propagation window is short
- **Learning rate is small**: smaller updates mean θ[s] stays close to θ[τ]
- **u[t] vectors were not saved**: the training log only has sample indices and learning rates (light storage mode)

## Approximation Error Analysis

The error from using θ[τ] instead of θ[s] is bounded by the curvature change along the training trajectory. Informally:

```
||H(θ[τ]) - H(θ[s])|| ≤ L₃ · ||θ[τ] - θ[s]||
```

where L₃ is the Lipschitz constant of the Hessian. Since `||θ[τ] - θ[s]|| = ||Σ u[t]||`, the error grows with the number of training steps in the window and the step sizes.

The paper's Proposition 1 and the truncation sensitivity bound (Appendix C) are derived assuming exact θ[s]. Using θ[τ] adds an additional uncontrolled error term that is **not covered** by the paper's certified unlearning guarantees.

## Disk Storage Impact

**Zero**. This feature does not change what is saved during training. It only reads existing u[t] vectors from the training log at unlearning time.
