# Open-Unlearning 使用指南

## 概述

本文档记录了在 Open-Unlearning 框架下使用 GradAscent 方法对 Llama-3.2-1B-Instruct 模型进行去学习（Unlearning）的完整流程。

## 实验配置

- **模型**: Llama-3.2-1B-Instruct (1B 参数)
- **基础模型**: `open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch5`
- **去学习方法**: GradAscent（梯度上升法）
- **数据集**: TOFU Benchmark
- **遗忘数据集**: forget10（10% 数据需要遗忘）
- **保留数据集**: retain90（90% 数据需要保留）
- **学习率**: 1e-5
- **训练轮数**: 10 epochs
- **权重衰减**: 0.01

## 核心命令

### 1. 执行去学习训练

```bash
# 在 GPU 3 上运行去学习训练
CUDA_VISIBLE_DEVICES=3 uv run python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  trainer=GradAscent \
  task_name=SAMPLE_UNLEARN
```

#### 命令参数说明

- `CUDA_VISIBLE_DEVICES=3`: 指定使用 GPU 3
- `--config-name=unlearn.yaml`: 使用 unlearn 配置文件
- `experiment=unlearn/tofu/default`: 指定实验配置路径
- `forget_split=forget10`: 遗忘数据集使用 forget10 分割（10% 需要遗忘的数据）
- `retain_split=retain90`: 保留数据集使用 retain90 分割（90% 需要保留的数据）
- `trainer=GradAscent`: 使用 GradAscent 训练器（梯度上升法）
- `task_name=SAMPLE_UNLEARN`: 实验任务名称，用于创建输出目录

#### 训练过程输出

训练过程中会显示负的 loss 值（这是正常的，因为 GradAscent 通过负损失实现反学习）：

```
{'loss': -0.4326, 'grad_norm': 11.875, 'learning_rate': 3.33e-06, 'epoch': 0.4}
{'loss': -0.4812, 'grad_norm': 12.5625, 'learning_rate': 7.50e-06, 'epoch': 0.8}
...
{'loss': -94.0869, 'grad_norm': 340.0, 'learning_rate': 8.47e-08, 'epoch': 10.0}
```

**解释**：loss 绝对值从 -0.4326 降到 -94.0869，表明模型在 forget 数据上的表现持续变差，遗忘效果在增强。

#### 模型保存位置

训练完成后，模型保存在：

```
./saves/unlearn/SAMPLE_UNLEARN/
```

---

### 2. 评估去学习效果

```bash
# 在 GPU 3 上运行评估
CUDA_VISIBLE_DEVICES=3 uv run python src/eval.py \
  experiment=eval/tofu/default \
  forget_split=forget10 \
  holdout_split=holdout10 \
  model.model_args.pretrained_model_name_or_path=./saves/unlearn/SAMPLE_UNLEARN \
  model.tokenizer_args.pretrained_model_name_or_path=./saves/unlearn/SAMPLE_UNLEARN \
  task_name=SAMPLE_UNLEARN_EVAL
```

#### 命令参数说明

- `CUDA_VISIBLE_DEVICES=3`: 指定使用 GPU 3
- `experiment=eval/tofu/default`: 使用 TOFU 评估配置
- `forget_split=forget10`: 评估时使用 forget10 数据集
- `holdout_split=holdout10`: 评估时使用 holdout10 数据集
- `model.model_args.pretrained_model_name_or_path`: 指定已训练的模型路径
- `model.tokenizer_args.pretrained_model_name_or_path`: 指定 tokenizer 路径（通常与模型路径相同）
- `task_name=SAMPLE_UNLEARN_EVAL`: 评估任务名称

#### 评估结果输出

评估结果保存在：

```
./saves/eval/SAMPLE_UNLEARN_EVAL/TOFU_EVAL.json     # 详细结果
./saves/eval/SAMPLE_UNLEARN_EVAL/TOFU_SUMMARY.json  # 总结结果
```

#### 评估指标说明

| 指标 | 值 | 含义 | 期望 |
|------|-----|------|------|
| **forget_Q_A_Prob** | 1.10e-39 | Forget 数据集上的问题回答准确率 | 越低越好（表明成功遗忘） |
| **forget_Q_A_ROUGE** | 0.0 | Forget 数据集上的 ROUGE 分数 | 越低越好（0.0 表示完全无法匹配参考答案） |
| **extraction_strength** | 0.033 | 记忆提取强度 | 越低越好（低于 0.05 通常表示成功遗忘） |
| **model_utility** | 0.0 | 模型整体效用 | 可能表示模型在某些任务上性能下降 |
| **privleak** | -6.83 | 隐私泄露指标 | 越低越好（负数表示隐私保护良好） |

## 评估结果分析

### 成功指标

1. **forget_Q_A_Prob = 1.10e-39**：极低的概率，说明模型已经无法在 forget 数据集上回答正确
2. **forget_Q_A_ROUGE = 0.0**：完全无法匹配参考答案，遗忘效果显著
3. **extraction_strength = 0.033**：记忆提取强度低，表明遗忘成功
4. **privleak = -6.83**：隐私泄露程度低

### 注意事项

- **model_utility = 0.0**：这表明模型在保留任务上可能也受到了影响
- 在应用去学习时，需要平衡遗忘效果和模型保留能力
- 可以考虑使用更复杂的去学习方法（如 GradDiff、NPO 等），它们同时优化 forget 和 retain 损失

## 快速开始

### 步骤 1：训练去学习模型

```bash
CUDA_VISIBLE_DEVICES=3 uv run python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  trainer=GradAscent \
  task_name=MY_UNLEARN
```

### 步骤 2：评估去学习效果

```bash
CUDA_VISIBLE_DEVICES=3 uv run python src/eval.py \
  experiment=eval/tofu/default \
  forget_split=forget10 \
  holdout_split=holdout10 \
  model.model_args.pretrained_model_name_or_path=./saves/unlearn/MY_UNLEARN \
  model.tokenizer_args.pretrained_model_name_or_path=./saves/unlearn/MY_UNLEARN \
  task_name=MY_UNLEARN_EVAL
```

### 步骤 3：查看评估结果

```bash
# 查看简要总结
cat saves/eval/MY_UNLEARN_EVAL/TOFU_SUMMARY.json

# 查看详细结果
cat saves/eval/MY_UNLEARN_EVAL/TOFU_EVAL.json
```

## 代码修复记录

在使用过程中遇到的 BFloat16 兼容性问题已经修复：

1. **文件**: `src/evals/metrics/utils.py` (第 98-99 行)
   - 修复：`avg_losses.cpu().float().numpy()`

2. **文件**: `src/evals/metrics/mia/min_k.py` (第 20 行)
   - 修复：`lp.cpu().float().numpy()`

3. **文件**: `src/evals/metrics/mia/min_k_plus_plus.py` (第 33-35 行)
   - 修复：添加 `.float()` 转换解决 BFloat16 → numpy 类型错误

## 相关资源

- Open-Unlearning 项目：[GitHub](https://github.com/locuslab/open-unlearning)
- 论文：[arXiv](https://arxiv.org/abs/2506.12618)
- TOFU Benchmark：[论文](https://arxiv.org/abs/2401.06121)
- HuggingFace 模型：`open-unlearning/tofu-models`

## 参考

本次实验使用的配置基于以下文件：

- 训练配置：`configs/experiment/unlearn/tofu/default.yaml`
- 评估配置：`configs/experiment/eval/tofu/default.yaml`
- 模型配置：`configs/model/Llama-3.2-1B-Instruct.yaml`
