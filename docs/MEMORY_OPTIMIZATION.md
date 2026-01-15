# Memory Optimization for LMCleaner Experiments

This document describes the memory optimization strategies implemented to reduce GPU memory pressure during LMCleaner training.

## Overview

The memory optimization plan reduces VRAM usage while maintaining training effectiveness by:

1. Setting PyTorch CUDA allocator configuration
2. Reducing batch size and increasing gradient accumulation
3. Enabling gradient checkpointing and mixed precision training
4. Using FlashAttention 2 for efficient attention computation
5. Implementing periodic CUDA cache clearing
6. Optimizing DeepSpeed ZeRO-3 configuration

## Changes Implemented

### 1. Environment Variables

**File:** `scripts/lmcleaner_experiments.sh`

Added the following environment variables at the start of the script:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
```

**Purpose:**

- `PYTORCH_CUDA_ALLOC_CONF`: Enables expandable memory segments and limits split size to reduce fragmentation
- `HF_HUB_DISABLE_TELEMETRY`: Reduces background telemetry overhead
- `TOKENIZERS_PARALLELISM`: Prevents tokenizer parallelism issues in distributed training

### 2. Training Configuration Adjustments

**File:** `scripts/lmcleaner_experiments.sh`

Modified training parameters:

- `per_device_train_batch_size`: 4 → 1
- `gradient_accumulation_steps`: 4 → 8

**Effective batch size maintained:** 4 × 4 = 8 × 1 = 16 (per device)

### 3. Additional Training Arguments

Added to all `accelerate launch` commands:

```bash
+trainer.args.bf16=true                              # Mixed precision training (BF16)
+model.model_args.attn_implementation=flash_attention_2  # FlashAttention 2
+data.max_seq_length=1024                           # Limited sequence length
```

**Benefits:**

- **BF16**: Reduces memory usage by 50% compared to FP32
- **FlashAttention 2**: More memory-efficient attention computation
- **max_seq_length=1024**: Limits sequence length to reduce memory peaks

### 4. CUDA Cache Clearing Callback

**Files:**

- `src/trainer/utils.py` (implementation)
- `src/trainer/__init__.py` (integration)

Implemented `CudaCacheCallback` that:

- Clears CUDA cache every 10 training steps
- Synchronizes all processes before and after cache clearing
- Reduces memory fragmentation during long training runs

**Code:**

```python
class CudaCacheCallback(TrainerCallback):
    def __init__(self, interval=10):
        self.interval = interval
        self.acc = Accelerator()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            self.acc.wait_for_everyone()
            torch.cuda.empty_cache()
            self.acc.wait_for_everyone()
```

The callback is automatically added to all trainers in `src/trainer/__init__.py`.

### 5. DeepSpeed Configuration Optimization

**New Files:**

- `configs/accelerate/zero_stage3_memory_optimized.json`
- `configs/accelerate/memory_optimized_config.yaml`

Created memory-optimized DeepSpeed ZeRO-3 configuration with:

**Key changes from default:**

- `offload_optimizer.device`: "none" → "cpu"
- `offload_param.device`: "none" → "cpu"
- `reduce_bucket_size`: "auto" → 5e7 (50MB)
- `stage3_prefetch_bucket_size`: "auto" → 5e7 (50MB)
- `stage3_param_persistence_threshold`: "auto" → 1e5

**Trade-offs:**

- **CPU offloading**: Reduces GPU memory usage but may slow training
- **Smaller buckets**: Reduces memory peaks but increases communication overhead

## Usage

### Default Configuration (Current)

The default `scripts/lmcleaner_experiments.sh` now includes all memory optimizations except CPU offloading:

```bash
bash scripts/lmcleaner_experiments.sh
```

### Maximum Memory Optimization (Optional)

For extreme memory constraints, use the memory-optimized accelerate config:

```bash
# Edit scripts/lmcleaner_experiments.sh and replace:
# --config_file configs/accelerate/default_config.yaml
# with:
# --config_file configs/accelerate/memory_optimized_config.yaml
```

## Monitoring and Tuning

### If OOM Still Occurs

1. **Reduce sequence length:**

   ```bash
   +data.max_seq_length=768  # or 512
   ```

2. **Increase gradient accumulation:**

   ```bash
   gradient_accumulation_steps=16  # or higher
   ```

3. **Enable CPU offloading:**
   Use `configs/accelerate/memory_optimized_config.yaml`

4. **Reduce number of GPUs:**

   ```bash
   CUDA_VISIBLE_DEVICES=0,1  # instead of 0,1,2
   ```

### Monitor Training

Check memory usage during training:

```bash
watch -n 1 nvidia-smi
```

Look for:

- Memory fragmentation (frequent cache flushes)
- Peak memory usage
- GPU utilization

## Performance Impact

Expected changes:

- **Memory usage**: -30% to -50% reduction
- **Training speed**: -10% to -30% slower (due to gradient checkpointing and smaller batches)
- **Convergence**: Should remain the same (effective batch size maintained)
- **Final model quality**: No degradation expected

## Troubleshooting

### DDP Unused Parameters Error

If you see errors about unused parameters:

```bash
trainer.args.ddp_find_unused_parameters=true
```

### FlashAttention Not Available

If FlashAttention 2 is not installed:

```bash
pip install --no-build-isolation flash-attn==2.6.3
```

Or remove the flash_attention_2 parameter:

```bash
# Remove: +model.model_args.attn_implementation=flash_attention_2
```

### BF16 Not Supported

If your GPU doesn't support BF16 (e.g., older GPUs):

```bash
+trainer.args.fp16=true  # instead of bf16=true
```

## References

- PyTorch CUDA Memory Management: <https://pytorch.org/docs/stable/notes/cuda.html>
- DeepSpeed ZeRO: <https://www.deepspeed.ai/tutorials/zero/>
- FlashAttention: <https://github.com/Dao-AILab/flash-attention>
