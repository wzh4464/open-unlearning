<!-- e724c33f-845c-40c9-807a-7ef6329a7233 35c4cd7b-78ff-4aff-9d33-91a21ed78114 -->
# LMCleaner 在线遗忘算法实现计划

## 1. 代码逻辑检查与修复

### 1.1 检查 tim_rollback.py 现有问题

- ❌ 当前实现只是简单梯度回退，缺少Hessian传播
- ❌ 未正确应用学习率（`_apply_gradient_to_params`中直接减去梯度）
- ❌ 缺少HVP（Hessian-Vector Product）计算
- ❌ 未实现前向K步传播机制
- ✅ 数据加载逻辑基本正确

### 1.2 修复方向

- 修复学习率应用：`param.data -= rollback_strength * learning_rate * gradient`
- 补充HVP计算接口（GGN近似优先）
- 实现参数校正向量计算

## 2. 实现LMCleaner两种变体

### 2.1 样本级LMCleaner (LMCleanerSampleLevel)

**文件**: `src/trainer/unlearn/lmcleaner_sample.py`

- 初始偏差：`δ[tz+1] = -(η_tzj/|Stzj|) ∇θ`(zj; θ[tzj])`（对每个forget样本）
- 需要存储每个样本的梯度轨迹
- 实现`compute_correction()`进行前向K步传播
- 支持HVP计算（GGN/diag/exact模式）

### 2.2 批次级LMCleaner (LMCleanerBatchLevel)  

**文件**: `src/trainer/unlearn/lmcleaner_batch.py`

- 初始偏差：`δ[tz+1] = -η_tz g¯[tz]`（或直接使用记录的u[t]）
- 存储每个batch的更新向量u[t] = θ[t+1] - θ[t]
- 存储复杂度从O(N p)降为O((N/B) p)
- 同样实现前向K步传播

### 2.3 共享核心组件

**文件**: `src/trainer/unlearn/lmcleaner_core.py`

- `hvp_apply()`: HVP计算（支持GGN、diag、low_rank、exact）
- `compute_correction()`: 参数校正向量计算（K截断传播）
- `apply_correction()`: 参数校正应用
- `StepLog`: 环形缓冲区存储最近K+Δ步
- `AuditRecord`: 审计信息记录

## 3. 训练日志记录模块

### 3.1 训练时记录

**文件**: `src/trainer/unlearn/training_logger.py`

- `TrainingLogger`: 记录每步的{u[t], η_t, batch_id, θ[t]引用}
- 支持样本级和批次级两种模式
- 环形缓冲区管理（仅保存最近K+Δ步）
- 自动序列化/反序列化到磁盘

### 3.2 与训练循环集成

- 在`trainer.train()`开始时初始化logger
- 在每个训练步骤后记录更新信息
- 支持resume训练时恢复日志

## 4. 配置文件

### 4.1 LMCleaner配置

- `configs/trainer/LMCleanerSample.yaml`: 样本级配置
- `configs/trainer/LMCleanerBatch.yaml`: 批次级配置
- 参数：
- `K`: 截断窗口大小（默认800）
- `hessian_mode`: "GGN"/"diag"/"low_rank"/"exact"
- `damping`: 阻尼系数λ（默认1e-4）
- `log_dir`: 训练日志存储目录

## 5. 实验bash脚本

### 5.1 完整实验流程脚本

**文件**: `scripts/lmcleaner_experiments.sh`

实验流程：

1. **预训练模型评估**（baseline）

- 评估forget集、retain集、holdout集性能

2. **训练日志生成**（可选，如果使用在线训练）

- 在预训练过程中记录LMCleaner所需日志
- 或使用已有预训练模型+模拟训练过程

3. **Unlearning执行**

- 运行样本级LMCleaner
- 运行批次级LMCleaner
- 对比baseline TIMParameterRollback

4. **评估阶段**

- Extraction Strength
- Truth Ratio  
- MIA攻击（min_k, loss, gradnorm等）
- Retain性能（MMLU等）
- Forget质量指标

5. **结果汇总**

- 生成对比表格
- 可视化结果

### 5.2 实验配置

- 模型：Llama-3.2-1B/3B-Instruct
- Forget splits：forget01, forget05, forget10
- K值范围：500, 800, 1000
- HVP模式：GGN（默认），diag（快速测试）

## 6. 实现细节

### 6.1 HVP实现

- **GGN**: 使用广义Gauss-Newton近似（对交叉熵损失稳定）
- **diag**: 对角Hessian近似（快速但粗糙）
- **exact**: 完整二阶autograd（仅小模型/测试用）
- **low_rank**: Lanczos/Sketch低秩近似

### 6.2 数值稳定性

- 自动检测收缩性：`|I - η_s H[s]| >= 1`时增加阻尼
- 自适应截断：当`||v||`衰减到阈值以下时提前停止

### 6.3 并行支持

- 支持ZeRO/TP分片参数对齐
- HVP计算时多向量并行
- 确保barrier同步

## 7. 测试与验证

### 7.1 单元测试

- 二次模型验证（误差应≈O(γ^K)）
- 小型MLP对比从头重训的参数差
- 退化测试（K=0, λ=0）

### 7.2 集成测试

- 与现有训练流程集成
- 与评估脚本集成