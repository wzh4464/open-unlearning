# Plan: Organize Experiment Scripts for Easy Reuse

## Objective
创建一个结构化的实验脚本系统，让 Claude 能够:
1. 理解每个脚本的用途
2. 方便修改模型配置后重跑所有实验
3. 像使用 tool 一样调度 GPU 运行实验

## Directory Structure
```
scripts/experiments/
├── README.md                    # 文档说明每个脚本用途
├── config.sh                    # 共享配置 (模型、路径等)
├── 01_finetune.sh              # Step 1: 微调
├── 02_lmcleaner_epoch{1-5}.sh  # Step 2: LMCleaner unlearn (5个)
├── 03_baseline_epoch{1-5}.sh   # Step 3: Baselines (GradDiff, NPO)
├── 04_eval_tofu.sh             # Step 4: TOFU 基础评估
├── 05_eval_tofu_mia.sh         # Step 5: TOFU MIA 评估
└── run_all.py                  # GPU 调度器
```

## Key Design

### 1. config.sh - 共享配置
```bash
# 模型配置 - 修改这里即可切换模型
MODEL_NAME="Llama-3.2-1B-Instruct"
MODEL_SHORT="llama32_1b"
FINETUNE_DIR="saves/finetune/${MODEL_SHORT}_tofu_safe"
TRAINING_LOG_DIR="saves/train_logs/${MODEL_SHORT}_tofu_safe"

# 实验配置
EPOCHS=(1 2 3 4 5)
CHECKPOINTS=(250 500 750 1000 1250)
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
```

### 2. 脚本模板 (每个脚本)
- 接受 GPU 作为参数: `./02_lmcleaner_epoch1.sh 0`
- source config.sh 获取配置
- 输出清晰的进度日志
- 返回成功/失败状态

### 3. README.md - Claude 可读文档
```markdown
# Experiment Scripts

## Quick Start
1. 修改 config.sh 中的模型配置
2. 运行 run_all.py 自动调度

## Scripts
| Script | 用途 | GPU 需求 | 预计时间 |
|--------|------|----------|----------|
| 01_finetune.sh | 微调模型 | 1 GPU | ~2h |
| 02_lmcleaner_epoch1.sh | LMCleaner epoch 1 | 1 GPU | ~1h |
...
```

### 4. run_all.py - GPU 调度器
- 监控 GPU 使用情况
- 自动分配空闲 GPU 运行下一个任务
- 支持依赖关系 (finetune 完成后才跑 unlearn)
- 进度追踪和日志

## Files to Create

1. `scripts/experiments/config.sh` - 配置文件
2. `scripts/experiments/01_finetune.sh` - 微调脚本
3. `scripts/experiments/02_lmcleaner_epoch{1-5}.sh` - 5个 LMCleaner 脚本
4. `scripts/experiments/03_baseline_epoch{1-5}.sh` - 5个 baseline 脚本
5. `scripts/experiments/04_eval_tofu.sh` - 基础评估
6. `scripts/experiments/05_eval_tofu_mia.sh` - MIA 评估
7. `scripts/experiments/run_all.py` - GPU 调度器
8. `scripts/experiments/README.md` - 文档

## Verification
1. 运行 `./01_finetune.sh 0` 测试单个脚本
2. 运行 `python run_all.py --dry-run` 测试调度器
3. 确认 README.md 包含完整说明
