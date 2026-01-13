# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Open-Unlearning is a unified framework for LLM unlearning research that supports multiple benchmarks (TOFU, MUSE, WMDP), 12+ unlearning methods, 10+ evaluation metrics, and 7+ model architectures. The framework uses Hydra for configuration management and supports both single-GPU and distributed training.

## Common Development Commands

### Environment Setup

**IMPORTANT**: This project uses `uv` for environment management. Always use `uv` commands instead of pip/conda.

```bash
# Install dependencies with uv
uv sync                          # Install all dependencies from lock file
uv sync --extra lm_eval          # Include lm-evaluation-harness
uv sync --extra dev              # Include development tools (ruff, pre-commit)
uv sync --extra linux-cuda       # For Linux with CUDA support
uv sync --extra macos-mps        # For macOS with MPS support

# Add new dependencies
uv add package-name              # Add a new package
uv add --dev package-name        # Add a development dependency

# Run Python scripts with uv
uv run python setup_data.py --eval    # Download evaluation data
uv run python src/train.py ...        # Run training scripts
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
# Run unlearning training (use uv run)
uv run python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN

# Run evaluation
uv run python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct task_name=SAMPLE_EVAL

# Run distributed training
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
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
