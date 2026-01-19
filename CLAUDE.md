# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Open-Unlearning is a unified framework for LLM unlearning research that supports multiple benchmarks (TOFU, MUSE, WMDP), 12+ unlearning methods, 10+ evaluation metrics, and 7+ model architectures. The framework uses Hydra for configuration management and supports both single-GPU and distributed training.

## Common Development Commands

### Environment Setup

This project can use either `uv` or standard Python/pip for environment management.

```bash
# Option 1: Using uv (recommended for reproducible builds)
uv sync                          # Install all dependencies from lock file
uv sync --extra lm_eval          # Include lm-evaluation-harness
uv sync --extra dev              # Include development tools (ruff, pre-commit)

# Option 2: Using pip directly
pip install -e .                 # Install in editable mode
pip install -e ".[lm_eval]"      # Include lm-evaluation-harness
pip install -e ".[dev]"          # Include development tools

# Run Python scripts (use python directly, not uv run)
python setup_data.py --eval      # Download evaluation data
python src/train.py ...          # Run training scripts
```

### Code Quality and Testing

```bash
# Check code quality (linting and formatting)
make quality

# Apply code formatting fixes
make style

# Run tests
make test
# Or explicitly: CUDA_VISIBLE_DEVICES= pytest tests/
```

### Training and Evaluation

```bash
# Run unlearning training
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN

# Run evaluation
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct task_name=SAMPLE_EVAL

# Run distributed training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default task_name=DISTRIBUTED_TRAIN

# Run baseline experiments
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```

## Architecture and Structure

### Core Components

- **src/train.py**: Main training entry point using Hydra config management
- **src/eval.py**: Evaluation entry point for running benchmarks
- **src/trainer/**: Contains unlearning method implementations (NPO, DPO, GradAscent, etc.)
- **src/evals/**: Benchmark implementations (TOFU, MUSE, WMDP) and evaluation metrics
- **src/data/**: Data preprocessing and dataset handling
- **src/model/**: Model loading and configuration utilities

### Configuration System
The framework uses Hydra for configuration management with these key config types:

- **configs/train.yaml**: Base training configuration
- **configs/unlearn.yaml**: Unlearning-specific training configuration  
- **configs/eval.yaml**: Evaluation configuration
- **configs/experiment/**: Predefined experiment configurations for common use cases
- **configs/trainer/**: Method-specific trainer configurations (GradAscent.yaml, NPO.yaml, etc.)
- **configs/model/**: Model-specific configurations (Llama-2-7b-hf.yaml, etc.)

### Output Directory Structure
Experiments generate outputs in `./saves/${mode}/${task_name}` where:

- `mode`: train/eval/unlearn
- `task_name`: User-provided experiment identifier

### Key Patterns

1. **Hydra Integration**: All entry points use `@hydra.main()` decorator with config management
2. **Component Factory Pattern**: Components are loaded via factory functions (`get_model()`, `get_data()`, `get_evaluators()`)
3. **Configuration Overrides**: Command-line arguments can override any config parameter using dot notation
4. **Distributed Training**: Uses Accelerate/DeepSpeed for multi-GPU training with automatic device management

### Benchmarks and Methods

- **Benchmarks**: TOFU (fictitious unlearning), MUSE (six-way evaluation), WMDP (hazardous knowledge)
- **Unlearning Methods**: GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, AltPO, SatImp, WGA, CE-U, PDU
- **Evaluation Metrics**: Verbatim metrics, ROUGE scores, MIA attacks, utility metrics, memorization tests

### Development Guidelines

**Core Principles**:

1. **Reuse Over Reinvention**: Always reuse existing interfaces and components
   - Search the codebase before implementing new functionality
   - Use existing factory patterns, callbacks, and configuration systems
   - Extend existing classes rather than creating parallel implementations

2. **Verify Before Implementing**: Never blindly guess implementation details
   - Search the repository to understand existing patterns
   - Search online documentation when uncertain about APIs
   - Check existing similar implementations for reference
   - Ask questions when requirements are unclear

3. **Comprehensive Testing**: All new features must include tests
   - Write unit tests that cover multiple scenarios
   - Test edge cases and error conditions
   - Test integration with existing components
   - Run `make test` before committing

4. **Code Quality**: Maintain high code quality standards
   - Code quality enforced via `ruff` for linting and formatting
   - Use `make quality` before committing changes
   - Follow existing code style and patterns
   - Add docstrings for new classes and functions

5. **Configuration Management**: Follow Hydra configuration patterns
   - All new components should be registered in appropriate config files
   - Use the factory pattern for component loading
   - Support configuration overrides via command-line

6. **Documentation**: Document all new features
   - Update relevant documentation in `docs/`
   - Add usage examples
   - Explain design decisions and rationale
   - Use descriptive task names for output organization

### Temp File Location

```bash
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR/claude"
```

This ensures temporary files created during training and evaluation are stored in a dedicated directory.

## Lessons Learned

### Hydra Configuration Patterns

1. **Config Override Syntax**:
   - Use dot notation for nested configs: `trainer.method_args.K=1000`
   - Use `++` prefix to add new keys: `++trainer.args.bf16=true`
   - Nested dict syntax in CLI: `model.model_args.pretrained_model_name_or_path=path/to/model`

2. **Trainer Configuration**:
   - Trainer configs define `_target_` class and `method_args`
   - `method_args` are passed to trainer constructor
   - HuggingFace Trainer args go in `trainer.args`

3. **Dataset Configuration**:
   - `forget_split` and `retain_split` control data splits
   - ForgetRetainDataset wraps forget/retain datasets with different indexing
   - Original finetune dataset must be loaded separately for batch reconstruction

### Memory Optimization for Large-Scale Training

1. **Lazy Loading Pattern**:
   - Never load all training records into memory at once
   - Use `LazyRecordLoader` to load records on-demand
   - Delete tensors immediately after use: `del tensor; gc.collect()`

2. **Chunk File Organization**:
   - Store each step's record in separate `.pkl` file
   - Build index from filenames (no need to read file contents)
   - Use parallel loading with `ProcessPoolExecutor` for metadata extraction

3. **Eta (Learning Rate) Caching**:
   - Store eta values in separate JSON cache file
   - Load eta cache at startup, update incrementally
   - Avoids loading 2.4GB chunk files just for a single float

4. **GPU Memory Management**:
   - Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64`
   - Enable `gradient_checkpointing=true` for large models
   - Move data to GPU only when needed, release immediately after

### LMCleaner Implementation Details

1. **Batch Reconstruction Priority**:
   - In lazy loading mode, check `sample_indices` first (not `step_record`)
   - `sample_indices` is loaded as small JSON, `step_record` requires full chunk

2. **HVP Computation**:
   - Requires batch data reconstruction from original dataset
   - ForgetRetainDataset indices differ from finetune dataset indices
   - Must use original finetune dataset (e.g., `locuslab/TOFU/full`) for batch reconstruction

3. **Forward Propagation**:
   - For step tz, propagate through steps tz+1 to min(tz+K, tau-1)
   - Each propagation step requires: load eta, reconstruct batch, compute HVP
   - Total HVP calls per forget step can be up to K (e.g., 1000)

4. **Dataset Mismatch Issue**:
   - Training uses TOFU full dataset (4000 samples, indices 0-3999)
   - Unlearning uses ForgetRetainDataset with different indexing
   - Solution: Load original finetune dataset with `SimpleQADataset` wrapper

### Parallel Processing Best Practices

1. **Multi-GPU Unlearning**:
   - Use `CUDA_VISIBLE_DEVICES` to assign each epoch to different GPU
   - Run epochs in parallel with independent processes
   - Shared eta cache is safe (read-only after initial build)

2. **Parallel Cache Building**:
   - Use `ProcessPoolExecutor` with many workers (e.g., 32)
   - Load pickle files in parallel, extract metadata only
   - Incremental save to avoid losing progress on failure

3. **Background Process Management**:
   - Use `nohup bash script.sh > log 2>&1 &` for long-running jobs
   - Monitor with `tail -f` and process count checks
   - Use `pkill -9 -f pattern` to kill stuck processes

### Common Debugging Patterns

1. **Check GPU Utilization**:
   ```bash
   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
   ```
   - 0% utilization often means computation is CPU-bound or I/O-bound

2. **Identify Bottlenecks**:
   - `K_used=0, hvp_calls=0` in logs means batch reconstruction failed
   - Check `sample_indices_per_step` is loaded in lazy loading mode
   - Verify dataset indexing matches training-time indexing

3. **Log Analysis**:
   ```bash
   grep "Applied correction" log.file | tail -5  # Check HVP execution
   grep "Processing forget step" log.file | tail -1  # Check progress
   ```

4. **Memory Monitoring**:
   ```bash
   free -h  # System memory
   ps aux | grep python | awk '{sum += $6} END {print sum/1024/1024, "GB"}'
   ```
