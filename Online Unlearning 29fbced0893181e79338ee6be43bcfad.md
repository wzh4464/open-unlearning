# Online Unlearning

[Online%20Unlearning%2029fbced0893181e79338ee6be43bcfad/LMCleaner1023.pdf](Online%20Unlearning%2029fbced0893181e79338ee6be43bcfad/LMCleaner1023.pdf)

### 算法方法笔记

- 概述
    - 主题：在线遗忘（Online Unlearning）框架 LMCleaner，在训练过程中即时移除目标数据的影响，而非训练后再处理。
    - 动机：训练后遗忘存在三大问题——不合规或不及时、无法阻止训练期间的攻击传播、额外高昂开销。
    - 核心思路：将“移除某样本/批次”视为在其出现时刻引入的参数偏差，并沿后续优化轨迹传播；通过近似地“反向抵消”这条传播来得到未见该数据时的对照模型。
- 形式化设定
    - 数据与训练：单遍大规模设置；每个样本只出现一次于某个 mini-batch St，批大小 B；优化以 mini-batch SGD（可扩展至动量/自适应优化器）。
    - 记号：参数 θ[t] ∈ R^p；损失 `(z; θ)`；平均梯度 g¯[t] 与平均 Hessian H[t]；学习率 ηt。
    - 目标：在步骤 τ 收到删除请求 Du 后，给出参数 θ̂[τ]，尽量接近“从头在 DDu 上训练至 τ”所得参数。
- 方法总览（LMCleaner 三段式）
    
    1) 样本级（理论基石）：
    
    - 若要忘记样本 zj（在 tzj 被用过），初始偏差 δ[tzj+1] = −(η_tzj/|Stzj|) ∇θ`(zj; θ[tzj])。
    - 后续演化近似线性：δ[t+1] ≈ (I − ηt H[t]) δ[t]。
    - 到 τ 的累计影响为对初始梯度经“传播子”P = ∏(I − ηs H[s]) 的线性变换。
    
    2) 批次级（存储可行）：
    
    - 将一个训练步的真实更新 u[t] = −ηt g¯[t] 作为原子日志，避免逐样本存储（内含了动量/自适应等优化器效应）。
    - 忘记批次 Stzj 的初始偏差简化为 δ[tzj+1] = −η_tzj g¯[tzj]，随后同样经传播子前向传播。
    - 存储从 O(N p) 降为 O((N/B) p)。
    
    3) 前向 K 截断（计算可扩展）：
    
    - 基于算法稳定性，早期步的影响随时间指数衰减，故仅保留最近 K 步传播即可，误差有上界。
    - 复杂度从 O(T p) 降为 O(K p)，通常 K∈[500, 1000] 即可。
- 伪代码（截断的批次级在线遗忘）
    
    ```
    Input: 当前参数 θ[τ]；需遗忘的批次出现步 tz；截断窗口 K；训练日志 {(S_s, η_s, θ[s])}_{s=0}^{τ-1}
    # 1) 初始偏差（等价于去掉该步更新）
    v ← η_tz · g¯[tz]   # 或由已记录的 u[tz] 反推
    # 2) 传播窗口
    K* ← min{K, τ − tz − 1}
    for s = tz+1, …, tz+K*:
        hvp ← (1/|S_s|) ∑_{i∈S_s} ∇^2_θ ` (z_i; θ[s]) · v   # Hessian-vector product
        v ← v − η_s · hvp        # v ← (I − η_s H[s]) v
    # 3) 参数校正
    θ̂[τ] ← θ[τ] + v
    return θ̂[τ]
    ```
    
- 复杂度与资源
    - 时间：精确传播 O((τ−tz) p)；K-截断 O(K p)。
    - 空间：逐样本 O(N p) → 批级 O((N/B) p)。
    - 实务提示：记录每步的实际更新向量 u[t] 足以重构初始偏差，且与优化器无关。
- 理论性质（摘要）
    - 截断误差：在“收缩性”假设下，误差项 ≤ C · γ^K · ||δ[tz+1]||，呈指数衰减。
    - 实用选择 K：为给定容差 ε，取 K ≥ log((1−γ)ε/(cP||δ||))/log(γ)。
    - 隐私（可选扩展）：在样本级差分隐私目标下，可按窗口敏感度 Δ_K 对校正后的参数加高斯噪声，得到 (ε, δ)-unlearning。
- 与训练后遗忘对比
    - 训练中即时移除污染或敏感数据影响，缩短“污染窗口”。
    - 无需在收敛后再做额外微调或编辑，端到端时延更低。
    - 对比影响函数、负偏好优化、模型编辑、输出过滤：LMCleaner 减少过度或不足遗忘的风险，更贴近“对照重训”的反事实。
- 实现细节与工程要点
    - 日志：按步记录 {η_t, u[t]} 或 {η_t, g¯[t]}；需要可取用的 HVP 接口（高效近似如 GGN、对角近似、随机投影可选）。
    - 优化器：批级做法用 u[t] 可自动包含动量/自适应效应；如需更严格一致性，可在 AdamW 下补偿状态变量 m_t, v_t 的影响（论文附录有讨论）。
    - 数值稳定：衰减不充分时可引入阻尼 H ← H + λI；截断窗口内适配学习率上限以确保收缩性。
    - 批次副作用：批内良性样本被一并移除影响，可将其回补进未用池以继续学习。
- 实验要点（基于 TOFU 与预训练/微调设置）
    - 忘却质量：Extraction Strength、Truth Ratio 显著下降，接近“保留集重训”的金标准；
    - 保留性能：retain 集与通用评测（如 MMLU）基本持平；
    - 效率：总时长接近常规训练，较“训练后+额外遗忘”显著节省；
    - 鲁棒性与安全：对再学习、量化等压力测试下表现稳健；在危险知识移除（WMDP）中降低有害输出率。
- 局限与展望
    - 一阶近似在高曲率或大偏差下可能有误差；可考虑二阶修正或自适应补偿。
    - 收缩性假设在早期非凸阶段未必满足；可用平均收缩或局部稳定性替代分析。
    - K 线性规模的 HVP 成本在超大模型上仍然可观；可探索低秩/对角/学习到的传播近似。
- 速查清单（落地建议）
    
    1) 开启每步 u[t] 或 g¯[t] 的轻量日志；
    
    2) 实现高效 HVP（向量多点并行、近似 Hessian）；
    
    3) 发生删除请求：定位 tz，计算 v=η_tz g¯[tz]，前向 K 步传播，θ ← θ+v；
    
    4) 选 K≈500–1000 并验证误差-开销折中；
    
    5) 重要场景可结合差分隐私噪声发布与再训练回补策略；
    
    6) 维护基线对照：与“从头重训(去忘集合)”对齐度定期评估。
    
- 关键公式摘录
    - 初始偏差（批级）：δ[tz+1] = −η_tz g¯[tz]
    - 传播更新：v ← (∏_{s=tz+1}^{tz+K*} (I − η_s H[s])) · (η_tz g¯[tz])
    - 参数校正：θ̂[τ] = θ[τ] + v

---

### Coding memo（面向 Coding Agent，可直接落地实现）

<aside>
🧩

目标：实现可插拔的在线遗忘模块 LMCleaner，满足“训练中删除请求即可生效、逻辑正确、细节完整、易于审计与回滚”。

</aside>

### 1. 模块边界与依赖

- 必备能力
    - 训练循环可插入 Hook（per-step）：BeforeStep、AfterStep、OnDeleteRequest。
    - 能获取每步的学习率 η_t、batch 标识 S_t、当前参数快照引用 θ[t]（仅引用不复制）、以及平均梯度 g¯[t] 或实际更新向量 u[t]。
    - Hessian-Vector Product（HVP）接口：给定 v、数据批次 S_s 与 θ[s]，返回 H[s] v 的近似。
- 依赖最小化
    - 若优化器为 SGD：直接使用 g¯[t] 与 η_t；
    - 若优化器为 AdamW：优先记录 u[t] = θ[t+1] − θ[t]，以隐式包含一阶、二阶动量影响；必要时提供状态补偿扩展（见 6.3）。

### 2. 数据结构与日志（不可省略）

- StepLog（环形缓冲，保存最近 K+Δ 个步骤）
    - step_id: int
    - eta: float
    - batch_id: Hashable（可由数据加载器生成）
    - u: Tensor or None（优先）
    - gbar: Tensor or None（备选）
    - theta_ref: weakref to parameters（仅用于 HVP 的 θ[s] 上下文）
- BatchIndex
    - map: batch_id -> step_id（单遍训练假设）
- DeleteQueue（线程安全）
    - 存放删除请求：{batch_id, request_time, policy}
- Config
    - K: int（截断窗口）
    - hessian_mode: {"GGN", "diag", "low_rank", "exact"}
    - damping: float λ ≥ 0（阻尼）
    - device / dtype 策略、并行度（HVP 批内向量并行）

### 3. 关键 API 设计

- register_step(step_id, batch_id, eta, u=None, gbar=None, theta_ref=None)
- request_delete(batch_id, policy={"apply_now": true, "dp": null})
- apply_unlearn(current_step_id)
    - 扫描 DeleteQueue 所有待处理批次，按时间顺序处理。
- compute_correction(tz, τ, K, mode)
    - 返回 v（参数校正向量），并提供审计信息 AuditRecord。
- apply_correction(v)
    - 原子加：θ ← θ + v（支持 ZeRO/TP 并行的分片加法）。

### 4. 在线遗忘算法（批级 + 前向 K 截断）

- 初始偏差构造
    - 优先：若 StepLog[tz].u 可用，设 v0 ← −StepLog[tz].u（注意 u = θ[t+1] − θ[t] = −η_t g¯[t]，因此 v0 = η_t g¯[t]）。
    - 备选：若仅有 g¯[tz] 与 η_tz，则 v0 ← η_tz g¯[tz]。
- 传播窗口
    - s ∈ [tz+1, tz+K*]，K* = min{K, τ − tz − 1}。
    - 每步执行 v ← (I − η_s (H[s] + λI)) v = v − η_s (H[s]v) − η_s λ v。
- 参数校正
    - θ̂[τ] ← θ[τ] + v，原地更新。
- 审计输出
    - 返回 {tz, τ, K*, v_norm, est_error_bound = C·γ^K*·||v0||, hvp_calls = K*, damping=λ, hessian_mode}。

伪代码（PyTorch 风格）：

```python
import torch
from typing import Iterable, Tuple

class AuditRecord(dict):
    pass

@[torch.no](http://torch.no)_grad()
def hvp_apply(v: torch.Tensor, step_rec, cfg) -> torch.Tensor:
    # 示例：GGN/diag/low_rank 等模式在此分派
    mode = cfg.hessian_mode
    if mode == "diag":
        # 需要事先缓存/估计一个对角 Hessian 向量 diag_H
        return step_rec.diag_H * v
    elif mode == "GGN":
        # 典型实现略，实际应在 θ[s] 与批次 S_s 上下文评估
        return generalized_gauss_newton_vec(step_rec.theta_ref(), step_rec.batch_id, v)
    elif mode == "low_rank":
        return low_rank_hvp(step_rec.theta_ref(), step_rec.batch_id, v, rank=cfg.rank)
    elif mode == "exact":
        return exact_hvp(step_rec.theta_ref(), step_rec.batch_id, v)
    else:
        raise ValueError(f"Unknown hessian_mode: {mode}")

@[torch.no](http://torch.no)_grad()
def compute_correction(tz: int, tau: int, K: int, step_log, cfg) -> Tuple[torch.Tensor, AuditRecord]:
    rec = step_log[tz]
    assert rec is not None and (rec.u is not None or (rec.gbar is not None and rec.eta is not None)), "missing u or (gbar, eta)"
    # 初始偏差 v0
    v = (rec.eta * rec.gbar) if rec.u is None else (-rec.u).clone()
    start = tz + 1
    end = min(tz + K, tau - 1)
    hvp_calls = 0
    for s in range(start, end + 1):
        srec = step_log[s]
        hvp = hvp_apply(v, srec, cfg)   # H[s] @ v 的近似
        v.add_(hvp, alpha=-srec.eta)    # v ← v − η_s * hvp
        if getattr(cfg, "damping", 0.0) and cfg.damping > 0:
            v.add_(v, alpha=-srec.eta * cfg.damping)
        hvp_calls += 1
    audit = AuditRecord(tz=tz, tau=tau, K_used=(end - start + 1 if end >= start else 0),
                        v_norm=float(v.norm().item()), hvp_calls=hvp_calls,
                        mode=cfg.hessian_mode, damping=getattr(cfg, "damping", 0.0))
    return v, audit

@[torch.no](http://torch.no)_grad()
def apply_correction(v: torch.Tensor, params: Iterable[torch.nn.Parameter]):
    # 将扁平向量 v 加回到参数
    # 要求外部先将参数按相同次序展平得到 v
    offset = 0
    for p in params:
        n = p.numel()
        p.add_(v[offset:offset+n].view_as(p))
        offset += n
    assert offset == v.numel(), "v size mismatch"
```

### 5. HVP 实现选项（逻辑正确且可替换）

- exact：二阶 autograd，成本高，仅小模型或单测。
- GGN：对交叉熵等常见损失稳定，推荐默认。
- diag：对角近似，快但粗。
- low_rank：随机化低秩近似（Lanczos/Sketch）。

### 6. 关键边界与正确性细节

- 单遍训练假设：若多 epoch，BatchIndex 需扩展为 (epoch, step)。
- 收缩性与数值稳定：若检测 |I − η_s H[s]| ≥ 1，自动降低 η_s 或增大 λ。
- AdamW 状态补偿：如需严格一致性，提供可选的 m_t/v_t 逆向修正；默认关闭以控复杂度。
- 批内良性样本回补：把受影响的良性样本回灌到未用池。
- 并行一致性：ZeRO/TP 下 v 与参数分片需对齐；更改前后 barrier。
- 时序与覆盖：若 tz 不在 StepLog 窗口内，拒绝并返回错误码，需要更大窗口或回退点。
- 审计与回滚：记录 {tz, τ, hash(v), v_norm, hvp_calls, cfg}；支持与 checkpoint 的差分对账。

### 7. DP 扩展（可选）

- 依据 Δ_K 设 σ ≥ Δ_K·sqrt(2 ln(1.25/δ))/ε，采样 ξ∼N(0,σ^2 I)，发布 θ_e=θ̂+ξ。

### 8. 单测与金标

- 二次模型：验证误差 ~ O(γ^K)。
- 小型 MLP：对比从头重训(去忘)的参数差与验证损失差。
- 退化测试：K=0、λ=0 行为合理，无请求时系统零副作用。

### 9. 性能建议

- HVP 合并与多向量并行；K 自适应提前截断；u[t] 量化存储并保留关键步全精度。

### 10. 最小可运行循环（示意）

```python
cfg = Config(K=800, hessian_mode="GGN", damping=1e-4)
for step, (batch_id, data) in enumerate(loader):
    loss = model(data)
    loss.backward(); opt.step(); [opt.zero](http://opt.zero)_grad()
    eta = get_lr(opt)
    u = capture_update_vector(model)  # θ[t+1] − θ[t]
    step_log.add(step_id=step, batch_id=batch_id, eta=eta, u=u, theta_ref=weakref(model))

    while delete_queue:
        req = delete_queue.pop()
        tz = batch_index[req.batch_id]
        v, audit = compute_correction(tz, tau=step+1, K=cfg.K, step_log=step_log, cfg=cfg)
        apply_correction(v, model.parameters())
        audit_logger.write(audit)
```

### 11. 常见错误

- tz 取错；
- 多卡未同步；
- HVP 在错误的 θ[s] 上下文；
- 极强动量下被快速拉回未生效，需要短窗冻结或衰减动量。