# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Open-Unlearning is a unified framework for LLM unlearning research that supports multiple benchmarks (TOFU, MUSE, WMDP), 12+ unlearning methods, 10+ evaluation metrics, and 7+ model architectures. The framework uses Hydra for configuration management and supports both single-GPU and distributed training.

## Common Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n unlearning python=3.11
conda activate unlearning

# Install package with optional dependencies
pip install .[lm_eval]  # For lm-evaluation-harness support
pip install .[dev]      # For development tools (ruff, pre-commit)
pip install --no-build-isolation flash-attn==2.6.3

# Download evaluation data and logs
python setup_data.py --eval
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
- Code quality enforced via `ruff` for linting and formatting
- All new components should follow the factory pattern and be registered in appropriate config files
- Experiments should use descriptive task names for output organization
- Use `make quality` before committing changes
- Reference documentation in `docs/` for detailed component implementation guides