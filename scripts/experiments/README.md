# Experiment Scripts

Structured experiment scripts for LMCleaner and baseline unlearning experiments on TOFU benchmark.

## Quick Start

1. **Modify model configuration** in `config.sh`:
   ```bash
   # Edit these lines to change model
   MODEL_NAME="Llama-3.2-1B-Instruct"
   MODEL_SHORT="llama32_1b"
   ```

2. **Run all experiments** with automatic GPU scheduling:
   ```bash
   python run_all.py --gpus 0,1,2,3
   ```

3. **Or run individual steps** manually:
   ```bash
   ./01_finetune.sh 0          # Finetune on GPU 0
   ./02_lmcleaner_epoch1.sh 1  # LMCleaner epoch 1 on GPU 1
   ./04_eval_tofu.sh 0         # Evaluate all models on GPU 0
   ```

## Directory Structure

```
scripts/experiments/
├── README.md                    # This file
├── config.sh                    # Shared configuration
├── 01_finetune.sh              # Step 1: Finetune with TrainingLogger
├── 02_lmcleaner_epoch{1-5}.sh  # Step 2: LMCleaner unlearn (5 scripts)
├── 03_baseline_epoch{1-5}.sh   # Step 3: Baseline methods (GradDiff, NPO)
├── 04_eval_tofu.sh             # Step 4: TOFU basic evaluation
├── 05_eval_tofu_mia.sh         # Step 5: TOFU MIA evaluation
└── run_all.py                  # GPU scheduler
```

## Scripts Overview

| Script | Purpose | Dependencies | GPU Requirement |
|--------|---------|--------------|-----------------|
| `01_finetune.sh` | Finetune model on TOFU with TrainingLogger | None | 1-2 GPUs |
| `02_lmcleaner_epoch{N}.sh` | LMCleaner unlearn at epoch N | Finetune | 1 GPU |
| `03_baseline_epoch{N}.sh` | Run GradDiff + NPO baselines at epoch N | Finetune | 1 GPU |
| `04_eval_tofu.sh` | Basic TOFU evaluation | Unlearn models | 1 GPU |
| `05_eval_tofu_mia.sh` | MIA privacy evaluation | Unlearn models | 1 GPU |
| `run_all.py` | Automatic GPU scheduling | None | Multiple GPUs |

## Configuration

### config.sh

All shared settings are in `config.sh`. Key settings to modify:

```bash
# Model configuration
MODEL_NAME="Llama-3.2-1B-Instruct"  # Full model name
MODEL_SHORT="llama32_1b"             # Short name for paths

# Paths (auto-derived from MODEL_SHORT)
FINETUNE_DIR="saves/finetune/${MODEL_SHORT}_tofu_safe"
TRAINING_LOG_DIR="saves/train_logs/${MODEL_SHORT}_tofu_safe"

# Experiment settings
EPOCHS=(1 2 3 4 5)               # Epochs to evaluate
CHECKPOINTS=(250 500 750 1000 1250)  # Steps per epoch

# Data splits
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"

# LMCleaner parameters
K=1000                # Truncation window size
HESSIAN_MODE="GGN"    # HVP computation mode
DAMPING=0.0001        # Damping factor

# Baseline methods
BASELINE_METHODS=("GradDiff" "NPO")
```

### Changing Models

To run experiments on a different model:

1. Edit `config.sh`:
   ```bash
   MODEL_NAME="Llama-3.1-8B-Instruct"
   MODEL_SHORT="llama31_8b"
   ```

2. Run all experiments:
   ```bash
   python run_all.py --gpus 0,1,2,3
   ```

## GPU Scheduler

The `run_all.py` script automatically schedules experiments on available GPUs.

### Basic Usage

```bash
# Run all experiments on GPUs 0-3
python run_all.py --gpus 0,1,2,3

# Dry run (preview without executing)
python run_all.py --dry-run

# Run only specific epochs
python run_all.py --epochs 1,3,5
```

### Advanced Options

```bash
# Skip certain steps
python run_all.py --skip-finetune      # Skip finetuning
python run_all.py --skip-lmcleaner     # Skip LMCleaner experiments
python run_all.py --skip-baselines     # Skip baseline methods
python run_all.py --skip-eval          # Skip basic evaluation
python run_all.py --skip-mia           # Skip MIA evaluation

# Combine options
python run_all.py --gpus 0,1 --skip-finetune --epochs 1,2,3
```

### Task Dependencies

The scheduler respects these dependencies:
```
finetune
    ├── lmcleaner_epoch{1-5} ──┐
    └── baseline_epoch{1-5} ──┼── eval_tofu ── eval_tofu_mia
                              └────────────────────┘
```

In detail:
- `finetune`: No dependencies (runs first)
- `lmcleaner_epoch{1-5}`: Depends on `finetune`
- `baseline_epoch{1-5}`: Depends on `finetune`
- `eval_tofu`: Depends on ALL `lmcleaner_epoch{1-5}` AND `baseline_epoch{1-5}`
- `eval_tofu_mia`: Depends on `eval_tofu` AND ALL unlearning tasks

## Manual Execution

Each script can be run independently with a GPU ID argument:

```bash
# Step 1: Finetune (run once)
./01_finetune.sh 0

# Step 2: Run LMCleaner for each epoch (can run in parallel)
./02_lmcleaner_epoch1.sh 0 &
./02_lmcleaner_epoch2.sh 1 &
./02_lmcleaner_epoch3.sh 2 &
wait

# Step 3: Run baselines for each epoch (can run in parallel)
./03_baseline_epoch1.sh 0 &
./03_baseline_epoch2.sh 1 &
wait

# Step 4: Evaluate all models
./04_eval_tofu.sh 0

# Step 5: MIA evaluation
./05_eval_tofu_mia.sh 0
```

### Evaluate a Specific Model

```bash
./04_eval_tofu.sh 0 saves/unlearn/lmcleaner_llama32_1b_epoch3_K1000
./05_eval_tofu_mia.sh 0 saves/unlearn/baseline_llama32_1b_epoch3_GradDiff
```

## Output Structure

```
saves/
├── finetune/
│   └── llama32_1b_tofu_safe/
│       ├── checkpoint-250/   # Epoch 1
│       ├── checkpoint-500/   # Epoch 2
│       ├── checkpoint-750/   # Epoch 3
│       ├── checkpoint-1000/  # Epoch 4
│       └── checkpoint-1250/  # Epoch 5
├── train_logs/
│   └── llama32_1b_tofu_safe/
│       └── chunk_*.pkl       # Training logs for LMCleaner
├── unlearn/
│   ├── lmcleaner_llama32_1b_epoch1_K1000/
│   │   ├── model files...
│   │   ├── evals/           # Basic evaluation results
│   │   └── evals_mia/       # MIA evaluation results
│   ├── baseline_llama32_1b_epoch1_GradDiff/
│   └── baseline_llama32_1b_epoch1_NPO/
└── scheduler_logs/
    ├── finetune.log
    ├── lmcleaner_epoch1.log
    └── results.json          # Summary of all runs
```

## Key Metrics

After evaluation, check these metrics in `TOFU_EVAL.json`:

### Forget Quality
- **extraction_strength**: How much the model can extract forgotten data
- **truth_ratio**: Truthfulness on forgotten data

### Retain Performance
- **ROUGE scores**: Performance on retain set
- **perplexity**: Language modeling quality

### Privacy (MIA)
- **attack_success_rate**: Lower is better for privacy
- **ROC_AUC**: Area under ROC curve for membership inference

## Troubleshooting

### Checkpoint not found
Ensure finetuning completed successfully:
```bash
ls saves/finetune/llama32_1b_tofu_safe/checkpoint-*
```

### Training logs not found
Check that TrainingLogger was enabled during finetuning:
```bash
ls saves/train_logs/llama32_1b_tofu_safe/
```

### Out of memory
Reduce batch size in `config.sh`:
```bash
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
```

### View scheduler logs
```bash
tail -f saves/scheduler_logs/lmcleaner_epoch1.log
cat saves/scheduler_logs/results.json
```
