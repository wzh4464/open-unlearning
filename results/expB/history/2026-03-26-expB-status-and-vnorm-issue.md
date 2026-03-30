# ExpB 实验状态与 v_norm=0 问题分析

> Date: 2026-03-26
> Status: **Blocked** — u[t] 全零导致 unlearning 无效

---

## 1. 当前实验进展

### 训练数据（已完成）

| B | steps | indices/step | 大小 | u[t] 状态 |
|---|:-:|:-:|:-:|:-:|
| 8 | 500 | 8 | 1.2 TB | **全零** |
| 16 | 250 | 16 | 588 GB | 正常 |
| 32 | 125 | 32 | 291 GB | **全零** |
| 64 | 62 | 64 | 148 GB | **全零** |
| 128 | 31 | 128 | 74 GB | **全零** |
| 256 | 15 | 256 | 37 GB | **全零** |

### Unlearning Sweep（运行中但无效）
- B=8 K=10: 已跑 31/37 forget steps，**每步 v_norm=0.000000**
- 每步耗时 ~13-58 分钟（主要花在加载 2.3GB pkl 文件上）
- 所有 correction 都是零向量 → unlearning 实际没有做任何修改

---

## 2. v_norm=0 根因分析

### 直接原因
u[t] = θ[t+1] - θ[t] 在所有新训练的日志中为全零向量。

由于 LMCleaner 的初始校正 `v0 = -u[tz]`，u 为零意味着 v0 = 0，后续 HVP 传播也保持为零。

### 为什么 B=16（旧训练）正常但新训练全零？

B=16 使用的是**旧版 `training_step()` 代码**（PR #24 之前），u[t] 计算逻辑是：

```python
# 旧版 (在 training_step 内部直接 register)
def training_step(self, model, inputs, ...):
    # 1. 克隆当前参数 (prev_params)
    if self.training_logger.prev_params is None:
        self.training_logger.prev_params = clone_parameters(model)

    # 2. 执行训练步 (参数会被更新)
    loss = super().training_step(model, inputs)

    # 3. 在这里 register → register_step 计算 u = current - prev
    self.training_logger.register_step(...)  # u[t] = θ_new - θ_old ≠ 0
```

PR #24 改为 **StepEndCallback 方案**后：

```python
# 新版 (callback 在 training_step 外部 register)
def training_step(self, model, inputs, ...):
    # 1. 只克隆一次 prev_params (第一步之前)
    if self.training_logger.prev_params is None:
        self.training_logger.prev_params = clone_parameters(model)

    # 2. 累积 indices
    self._accumulated_sample_indices.extend(...)

    # 3. 执行训练步
    loss = super().training_step(model, inputs)
    return loss  # ← 不再在这里 register

class StepEndCallback:
    def on_step_end(self, ...):
        # 4. 在 callback 中 register
        trainer.training_logger.register_step(...)
        # register_step 内部: u = clone_parameters(model) - prev_params
        # 但 prev_params 只在步骤 1 克隆了一次，之后再也没更新过！
```

### 核心 Bug

`prev_params` 只在 `training_step()` 中克隆一次（条件 `is None`），之后再也不更新。`register_step()` 内部计算 `u = current - prev` 后，会在 `prev_params` 上设置新值：

```python
# register_step 内部 (training_logger.py)
def register_step(self, ..., model=None, ...):
    if u is None and model is not None and self.prev_params is not None:
        current_params = clone_parameters(model)
        u = current_params - self.prev_params  # ← 计算 u
    # ... 但没有 self.prev_params = current_params 的更新！
```

实际上检查 `register_step` 的代码后发现：**`prev_params` 的更新在哪里？**

可能性：
1. `register_step` 计算完 u 后没有更新 `prev_params` → 后续步的 u 都是 `θ_current - θ_step0`
2. 或者 `prev_params` 被更新了但在 callback 的时序下有问题

### 已确认根因

`register_step()` 内部（line 621-623）确实会更新 `prev_params`，这不是问题。

**真正的根因**：HF Trainer 的 `CallbackHandler.on_step_end()` **不传 `model` 参数**：

```python
# transformers/trainer_callback.py
def on_step_end(self, args, state, control):
    return self.call_event("on_step_end", args, state, control)
    # ← 没有 model 参数！
```

所以 `StepEndCallback.on_step_end(model=None)` → `register_step(model=None)` → u 计算被跳过（`model is not None` 条件不满足）→ u 为 None → pkl 中 u 全零。

### 修复方案

在 `StepEndCallback` 中通过 `trainer._trainer` 引用（已有）获取 model：

```python
def on_step_end(self, args, state, control, **kwargs):
    trainer = self._trainer
    model = trainer.model  # ← 直接从 trainer 取 model
    ...
    trainer.training_logger.register_step(..., model=model, ...)
```

---

## 3. 确认步骤

### Step 1: 确认 register_step 中 prev_params 的处理
```bash
grep -A20 "def register_step" src/trainer/training_logger.py | grep "prev_params"
```
检查是否有 `self.prev_params = current_params` 或类似的更新逻辑。

### Step 2: 确认 callback 时序
在 StepEndCallback.on_step_end 中，`model` 参数的状态是什么？
- 是否是 optimizer.step() 后的新参数？
- 还是和 training_step() 中克隆的 prev_params 相同？

### Step 3: 对比验证
用 B=16 旧日志的非零 u[t] 做 sanity check：
```python
# 如果 register_step 不更新 prev_params:
# u[5] = θ[5] - θ[0], u[10] = θ[10] - θ[0]  (不是相邻差)
# 如果正确更新:
# u[5] = θ[5] - θ[4], u[10] = θ[10] - θ[9]  (相邻差)
```

### Step 4: 最小复现
```bash
# 跑 5 步训练，检查 u[t] 是否全零
python src/train.py ... trainer.args.num_train_epochs=1 \
    trainer.args.max_steps=5 \
    +trainer.args.training_logger.enabled=true ...
```

---

## 4. 可能的修复方案

### 方案 A: 在 StepEndCallback 中手动更新 prev_params
```python
class StepEndCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        ...
        trainer.training_logger.register_step(...)
        # 更新 prev_params 为当前模型参数
        from trainer.unlearn.lmcleaner_core import clone_parameters
        trainer.training_logger.prev_params = clone_parameters(model)
```

### 方案 B: 回退到 training_step 内 register，但用 micro_batch_counter + 尾部 flush
保留原有在 `training_step()` 内直接 register 的逻辑（已验证 u[t] 正确），加上一个 epoch_end callback 处理最后一个 partial batch。

### 方案 C: 修复 register_step 本身
确保 `register_step()` 在计算 u 后更新 `self.prev_params = current_params`。

### 推荐
先做 **Step 1** 确认根因，再决定方案。如果确认是 prev_params 未更新，方案 C 是最小改动。

---

## 5. 时间影响

### 当前运行（B=8 K=10）
- 已跑 ~18 小时，31/37 步
- 即使跑完结果也无效（v_norm 全零）
- **应该 kill 掉**

### 修复后需要重训
- 所有 B={8,32,64,128,256} 的训练数据需要用修复后的代码重新生成
- 只有 B=16 的旧数据可以保留（用旧版正确代码训练的）
- 重训耗时估计 ~2 小时（和之前一样）
- 重训存储 ~1.7 TB（和之前一样，不含 B=16）

---

## 6. 性能问题

即使修复 v_norm=0，当前的运行速度也是一个问题：

| B | 每步加载 pkl | 预估每 forget step 耗时 |
|---|:-:|:-:|
| 8 | 2.3 GB × K 次 | K=10: ~13 min, K=50: ~65 min |
| 16 | 2.3 GB × K 次 | K=10: ~13 min |
| 32 | 2.3 GB × K 次 | K=10: ~13 min |

B=8 有 37 个 forget steps × K=50 × ~65 min/step = **~40 小时单个 (K,B) 组合**。

完整 sweep（~20 个有效组合）可能需要 **数周**。需要考虑：
1. 不存 u[t] 到磁盘，改为在 unlearning 时从 checkpoint + batch 重算
2. 用更高效的 IO（如 mmap 或 safetensors 代替 pickle）
3. 减少 B=8 的 K 范围或直接去掉 B=8
