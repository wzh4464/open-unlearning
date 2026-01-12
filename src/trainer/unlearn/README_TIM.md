# TIM Parameter Rollback

基于训练轨迹进行参数直接回退的 Unlearning 方法。

## 概述

`analysing repercussions of training trajectory diversity on samples` (TIM) 方法通过记录训练过程中每个样本对模型参数的贡献，然后在 unlearning 时反向应用这些更新来实现遗忘。

## 工作原理

1. **训练阶段**：在训练过程中记录每个 forget 样本产生的梯度更新
2. **Unlearning 阶段**：
   - 加载训练轨迹记录
   - 识别 forget 样本在训练中的所有更新步
   - 反向应用这些更新（即从参数中"减去"forget 样本的贡献）

## 使用方法

### 1. 配置文件

创建或修改配置文件 `configs/trainer/TIMParameterRollback.yaml`：

```yaml
handler: TIMParameterRollback
method_args:
  tim_output_dir: "/path/to/training/trajectories"  # 训练轨迹数据目录
  rollback_strength: 1.0  # 回退强度，1.0=完全回退
  trajectory_format: json  # 数据格式：'json' 或 'pickle'
args:
  # ... 训练参数
```

### 2. 训练轨迹格式

训练轨迹数据应存储在 `tim_output_dir` 目录中，包含以下文件：

```
tim_output_dir/
├── global_info.json          # 全局训练信息
├── epoch_0.json              # 第 0 个 epoch 的轨迹数据
├── epoch_1.json              # 第 1 个 epoch 的轨迹数据
└── ...
```

#### global_info.json

```json
{
  "num_epoch": 10,
  "num_steps": 1000,
  "learning_rate": 1e-5
}
```

#### epoch_N.json

```json
{
  "sample_gradients": {
    "0": {
      "model.layers.0.weight": [[...梯度数据...]],
      "model.layers.1.weight": [[...梯度数据...]]
    },
    "1": {
      "model.layers.0.weight": [[...梯度数据...]]
    }
  }
}
```

### 3. 记录训练轨迹

在训练过程中，使用提供的辅助函数记录梯度：

```python
from trainer.unlearn.tim_rollback import save_gradient_trajectory

# 在每个训练步骤中
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 记录梯度（仅对 forget 样本）
        for idx, sample_idx in enumerate(batch['forget_indices']):
            for name, param in model.named_parameters():
                if param.grad is not None:
                    save_gradient_trajectory(
                        output_dir=Path(tim_output_dir),
                        epoch=epoch,
                        step=step,
                        sample_idx=sample_idx,
                        param_name=name,
                        gradient=param.grad[idx],  # 样本级梯度
                        global_info={'num_epoch': num_epochs}
                    )
        
        optimizer.step()
```

### 4. 运行 Unlearning

在配置文件中指定 trainer 为 `TIMParameterRollback` 并运行：

```bash
python src/train.py trainer=TIMParameterRollback \
    data.unlearn=... \
    trainer.method_args.tim_output_dir=/path/to/trajectories
```

## 注意事项

1. **存储开销**：记录每个样本的梯度会占用大量存储空间，仅对需要 unlearn 的样本记录
2. **学习率**：当前实现简化了学习率处理，实际应用中需要记录并应用原始学习率
3. **优化器状态**：当前实现不考虑优化器状态（如 Adam 的动量），可能需要额外处理
4. **批量训练**：如果训练使用批量梯度，需要记录每个样本的个体贡献

## 参数说明

- `tim_output_dir`：训练轨迹数据存储目录
- `rollback_strength`：回退强度（0.0-1.0），控制回退的程度
- `trajectory_format`：数据格式（'json' 或 'pickle'）

## 相关论文

TIM (Training Impact Measurement) 相关的论文和方法。

