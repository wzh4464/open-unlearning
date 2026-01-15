# LMCleaner Online Unlearning Implementation

## Overview

LMCleaner is an **online unlearning** framework that removes the influence of target data during training, rather than post-hoc. This implementation provides two variants:

1. **Batch-Level LMCleaner** (`LMCleanerBatchLevel`): More efficient, O((N/B)*p) storage
2. **Sample-Level LMCleaner** (`LMCleanerSampleLevel`): More precise, O(N*p) storage

## Core Algorithm

### Mathematical Foundation

The algorithm is based on tracking how forget samples influence parameters through training:

1. **Initial Bias**: When a sample/batch appears at step `tz`, it creates a parameter perturbation
   - Batch-level: `δ[tz+1] = -η_tz * gbar[tz]`
   - Sample-level: `δ[tz+1] = -(η_tz/B) * ∇θ`(zj; θ[tz])`

2. **Forward Propagation**: The influence propagates through subsequent optimization steps
   - `v[s+1] = (I - η_s H[s]) v[s]`
   - Uses Hessian-Vector Product (HVP) for second-order information

3. **Parameter Correction**: Apply the accumulated correction
   - `θ̂[τ] = θ[τ] + v`

4. **K-Truncation**: Only propagate for K recent steps (typically 500-1000) for computational efficiency

### Key Components

#### 1. Core Module (`lmcleaner_core.py`)

Contains the fundamental building blocks:

- **`HVPConfig`**: Configuration for Hessian-Vector Product computation
- **`StepRecord`**: Records training step information
- **`StepLog`**: Ring buffer for storing recent K steps
- **`AuditRecord`**: Audit trail for unlearning operations
- **`hvp_apply()`**: Computes H @ v using various methods:
  - `GGN`: Generalized Gauss-Newton (recommended for cross-entropy)
  - `diag`: Diagonal approximation (fast but rough)
  - `exact`: Full second-order autograd (expensive)
  - `low_rank`: Low-rank approximation (TODO)
- **`compute_correction()`**: Forward K-step propagation algorithm
- **`apply_correction()`**: Applies correction vector to parameters

#### 2. Training Logger (`training_logger.py`)

Records training trajectory for later unlearning:

- **`TrainingLogger`**: Main logger class
  - Records `{u[t], η[t], batch_id, θ[t]_ref}` at each step
  - Supports both batch-level and sample-level modes
  - Ring buffer management (keeps recent K steps)
  - Disk serialization/deserialization

- **`TrainingLoggerCallback`**: Integration with training loop

#### 3. Batch-Level Implementation (`lmcleaner_batch.py`)

- **`LMCleanerBatchLevel`**: Main trainer class
  - Loads training logs
  - Identifies forget batches
  - Computes and applies corrections
  - Optional fine-tuning on retain data
  - Generates audit logs

#### 4. Sample-Level Implementation (`lmcleaner_sample.py`)

- **`LMCleanerSampleLevel`**: Sample-level variant
  - More granular than batch-level
  - Higher storage overhead
  - Suitable for fine-grained control

## Usage Guide

### Step 1: Pretraining with Logging

**Note**: This step requires integration of `TrainingLogger` into the training loop. This is not yet implemented in the main training script.

```python
# Example integration (to be added to training code)
from trainer.unlearn.training_logger import TrainingLogger

# Initialize logger
logger = TrainingLogger(
    log_dir="saves/train_logs/model_name",
    max_steps=1000,
    mode="batch",  # or "sample"
    save_interval=100,
)

# In training loop, after optimizer.step()
logger.register_step(
    step_id=global_step,
    batch_id=batch_idx,
    eta=current_lr,
    model=model,
    batch_data=batch,  # optional, for HVP
)
```

### Step 2: Run Unlearning

#### Option A: Using Configuration Files

```bash
# Batch-level unlearning
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LMCleanerBatch \
    task_name=lmcleaner_test \
    model=Llama-3.2-1B-Instruct \
    forget_split=forget10 \
    retain_split=retain90 \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    trainer.method_args.training_log_dir=saves/train_logs/Llama-3.2-1B-Instruct_full \
    trainer.method_args.K=800 \
    trainer.method_args.hessian_mode=GGN \
    trainer.method_args.damping=1e-4
```

**重要**: 必须包含`experiment=unlearn/tofu/default`来正确设置数据加载配置。

#### Option B: Using Experiment Script

```bash
# Edit scripts/lmcleaner_experiments.sh to configure:
# - models
# - splits
# - K values
# - HVP modes

bash scripts/lmcleaner_experiments.sh
```

#### Option C: Programmatic API

```python
from trainer.unlearn.lmcleaner_batch import run_lmcleaner_batch_unlearning
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("model_path")

# Run unlearning
unlearned_model = run_lmcleaner_batch_unlearning(
    model=model,
    training_log_dir="saves/train_logs/model_name",
    forget_batch_ids=[0, 1, 2, ...],
    K=800,
    hessian_mode="GGN",
    damping=1e-4,
    output_dir="saves/unlearn/lmcleaner_output",
)
```

### Step 3: Evaluate

```bash
python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model=Llama-3.2-1B-Instruct \
    task_name=lmcleaner_test \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_test \
    paths.output_dir=saves/unlearn/lmcleaner_test/evals
```

## Configuration Parameters

### Trainer Configuration

#### `LMCleanerBatch.yaml`

```yaml
handler: LMCleanerBatchLevel

method_args:
  training_log_dir: ???          # Required: path to training logs
  K: 800                         # Truncation window size
  hessian_mode: GGN              # HVP mode: GGN/diag/exact
  damping: 1e-4                  # Damping coefficient λ
  apply_immediately: false       # Apply unlearning at init or train()
  audit_dir: ${paths.output_dir}/audit  # Audit log directory

args:
  num_train_epochs: 0            # 0 = only parameter correction, no fine-tuning
  # ... standard training args
```

#### `LMCleanerSample.yaml`

```yaml
handler: LMCleanerSampleLevel

method_args:
  training_log_dir: ???
  K: 800
  hessian_mode: GGN
  damping: 1e-4
  batch_size_at_training: 1      # Original batch size (for sample-level bias)
  apply_immediately: false
  audit_dir: ${paths.output_dir}/audit

args:
  num_train_epochs: 0
  # ... standard training args
```

### Key Parameters Explained

- **`K` (Truncation Window)**:
  - Controls how many steps to propagate corrections
  - Larger K = more accurate but slower
  - Typical range: 500-1000
  - Rule of thumb: `K ≥ log((1-γ)ε/(cP||δ||)) / log(γ)` where γ is contraction rate

- **`hessian_mode`** (HVP Method):
  - `GGN`: Generalized Gauss-Newton, recommended for CE loss, stable
  - `diag`: Diagonal approximation, fastest but least accurate
  - `exact`: Full Hessian, most accurate but very slow
  - `low_rank`: Low-rank approximation (not yet implemented)

- **`damping`** (Numerical Stability):
  - Adds λI to Hessian: `H ← H + λI`
  - Improves stability when `|I - ηH| ≥ 1`
  - Typical range: 1e-5 to 1e-3

## Implementation Details

### HVP Computation

The framework supports multiple HVP methods:

1. **GGN (Generalized Gauss-Newton)**:
   ```
   H ≈ J^T H_loss J ≈ J^T J  (for cross-entropy)
   ```
   - Most stable for CE loss
   - Recommended default

2. **Diagonal Approximation**:
   ```
   Hv ≈ diag(H) * v
   ```
   - Fast but loses off-diagonal information
   - Good for quick prototyping

3. **Exact**:
   ```
   Hv = ∇²L v  (via autograd)
   ```
   - Ground truth but expensive
   - Only for small models/testing

### Storage Requirements

| Component | Batch-Level | Sample-Level |
|-----------|-------------|--------------|
| Per step | O(p) | O(p) |
| Total steps | K | K |
| Samples | N/B batches | N samples |
| **Total** | **O(K*p)** | **O(K*p)** |

Where:
- N = total samples
- B = batch size
- p = number of parameters
- K = truncation window

**Note**: Both methods store K steps, but batch-level has fewer "forget units" (N/B vs N).

### Time Complexity

For each forget unit:
- Initial bias: O(p)
- K-step propagation: O(K * p * h)
  - h = HVP cost per step
  - GGN: h ≈ 2 (two backward passes)
  - Exact: h ≈ 3 (three backward passes)
- Total: **O(K * p)**

For M forget units: **O(M * K * p)**

## Comparison with TIMParameterRollback

| Aspect | TIMParameterRollback | LMCleaner |
|--------|---------------------|-----------|
| **Order** | First-order (gradients only) | Second-order (gradients + Hessian) |
| **Propagation** | None (direct rollback) | K-step forward propagation |
| **Accuracy** | Lower (ignores curvature) | Higher (considers curvature) |
| **Complexity** | O(M*p) | O(M*K*p) |
| **Storage** | O(N*p) | O((N/B)*p) batch, O(N*p) sample |
| **Theory** | Ad-hoc | Principled (influence functions) |

## Audit Logs

Every unlearning operation generates an `AuditRecord`:

```json
{
  "tz": 1234,           // Forget step
  "tau": 5678,          // Current step
  "K_used": 800,        // Actual propagation steps
  "v_norm": 0.0123,     // Correction vector norm
  "hvp_calls": 800,     // Number of HVP computations
  "mode": "GGN",        // HVP mode used
  "damping": 0.0001     // Damping coefficient
}
```

Saved to: `${output_dir}/audit/audit_records.json`

## Troubleshooting

### Issue: "Training log directory not found"

**Solution**: You need to run pretraining with `TrainingLogger` first. This requires integrating the logger into the training loop (see Step 1 above).

### Issue: "Step X not found in training log"

**Cause**: Ring buffer has overwritten old steps (only keeps recent K steps).

**Solutions**:
1. Increase `max_steps` in `TrainingLogger`
2. Ensure K ≤ max_steps
3. Save logs more frequently with `save_interval`

### Issue: HVP computation is slow

**Solutions**:
1. Use `hessian_mode=diag` for faster (less accurate) results
2. Reduce K to smaller value (e.g., 500)
3. Use smaller model for testing
4. Enable gradient checkpointing to reduce memory

### Issue: Out of memory during unlearning

**Solutions**:
1. Process forget units in smaller batches
2. Use `hessian_mode=diag` (lower memory)
3. Reduce K
4. Don't save `batch_data` in logger (`save_batch_data=False`)

### Issue: Numerical instability (NaN in corrections)

**Solutions**:
1. Increase `damping` (try 1e-3 or 1e-2)
2. Reduce learning rate during pretraining
3. Check for `|I - ηH| ≥ 1` (too large step size)
4. Use `hessian_mode=GGN` instead of `exact`

## Limitations and Future Work

### Current Limitations

1. **Training Logger Integration**: Not yet integrated into main training loop
   - Requires manual integration
   - Need to add hooks to `src/train.py`

2. **HVP Approximations**:
   - Low-rank HVP not implemented
   - Could reduce computation cost significantly

3. **Batch Data Storage**:
   - Storing batch data for HVP is memory-intensive
   - Need better compression or approximation strategies

4. **Optimizer State**:
   - Currently doesn't account for momentum/Adam state
   - Could improve accuracy by tracking optimizer states

5. **Multi-Epoch Training**:
   - Designed for single-pass large-scale setting
   - Multi-epoch requires extended batch indexing

### Future Enhancements

1. **Adaptive K Selection**:
   - Automatically determine K based on convergence criteria
   - Early stopping when `||v||` drops below threshold

2. **Distributed Unlearning**:
   - Support for ZeRO/TP parameter sharding
   - Parallel HVP computation

3. **Online Mode**:
   - Real-time unlearning during training
   - Immediate response to delete requests

4. **Privacy Extensions**:
   - Differential privacy guarantees
   - Add Gaussian noise: `θ_ε = θ̂ + N(0, σ²I)`

5. **Low-Rank Approximations**:
   - Lanczos/Sketch methods for HVP
   - Reduce O(Kp) to O(Kr) where r << p

## References

1. **LMCleaner Paper**: "Online Unlearning of Language Models" (see PDF in docs)
2. **Influence Functions**: Koh & Liang, "Understanding Black-box Predictions via Influence Functions"
3. **GGN Approximation**: Martens, "New Insights and Perspectives on the Natural Gradient Method"

## Code Structure

```
src/trainer/unlearn/
├── lmcleaner_core.py          # Core algorithms (HVP, correction, etc.)
├── lmcleaner_batch.py         # Batch-level implementation
├── lmcleaner_sample.py        # Sample-level implementation
├── training_logger.py         # Training trajectory logging
└── tim_rollback.py           # Baseline method (for comparison)

configs/trainer/
├── LMCleanerBatch.yaml        # Batch-level config
└── LMCleanerSample.yaml       # Sample-level config

scripts/
└── lmcleaner_experiments.sh   # Full experiment pipeline

docs/
└── lmcleaner_implementation.md  # This document
```

## Examples

See `scripts/lmcleaner_experiments.sh` for complete examples of:
- Running batch-level unlearning on TOFU benchmark
- Sweeping over K values and HVP modes
- Comparing with baseline methods
- Evaluating forget quality and retain performance
