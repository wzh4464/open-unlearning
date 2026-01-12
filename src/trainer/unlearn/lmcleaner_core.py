"""
LMCleaner Core Components: 在线遗忘的核心算法组件

包含:
- HVP (Hessian-Vector Product) 计算
- 参数校正向量计算 (前向K步传播)
- 参数校正应用
- 训练日志数据结构
- 审计记录
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import weakref

import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord(dict):
    """审计记录，用于记录遗忘操作的详细信息"""

    tz: int = 0  # 遗忘批次的步骤
    tau: int = 0  # 当前步骤
    K_used: int = 0  # 实际使用的截断窗口大小
    v_norm: float = 0.0  # 校正向量的范数
    hvp_calls: int = 0  # HVP调用次数
    mode: str = "GGN"  # Hessian模式
    damping: float = 0.0  # 阻尼系数

    def __post_init__(self):
        # 使其可以像字典一样使用
        dict.__init__(self,
                     tz=self.tz, tau=self.tau, K_used=self.K_used,
                     v_norm=self.v_norm, hvp_calls=self.hvp_calls,
                     mode=self.mode, damping=self.damping)


@dataclass
class StepRecord:
    """单步训练记录"""

    step_id: int
    eta: float  # 学习率
    batch_id: Any  # 批次标识符
    u: Optional[torch.Tensor] = None  # 参数更新向量 θ[t+1] - θ[t]
    gbar: Optional[torch.Tensor] = None  # 平均梯度
    theta_ref: Optional[weakref.ref] = None  # 参数弱引用
    batch_data: Optional[Any] = None  # 批次数据(用于HVP计算)
    diag_H: Optional[torch.Tensor] = None  # 对角Hessian近似(可选)

    def __repr__(self):
        return f"StepRecord(step={self.step_id}, eta={self.eta}, batch_id={self.batch_id})"


class StepLog:
    """环形缓冲区，存储最近K+Δ个训练步骤的记录"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.step_map = {}  # step_id -> index in buffer

    def add(self, record: StepRecord):
        """添加新的步骤记录"""
        # 如果缓冲区满了,移除最旧的记录
        if len(self.buffer) == self.max_size:
            oldest = self.buffer[0]
            if oldest.step_id in self.step_map:
                del self.step_map[oldest.step_id]

        # 添加新记录
        self.buffer.append(record)
        self.step_map[record.step_id] = len(self.buffer) - 1

    def get(self, step_id: int) -> Optional[StepRecord]:
        """获取指定步骤的记录"""
        if step_id not in self.step_map:
            return None
        idx = self.step_map[step_id]
        # 检查索引是否仍然有效(环形缓冲区可能已覆盖)
        if idx < len(self.buffer) and self.buffer[idx].step_id == step_id:
            return self.buffer[idx]
        return None

    def __getitem__(self, step_id: int) -> Optional[StepRecord]:
        return self.get(step_id)

    def has_range(self, start: int, end: int) -> bool:
        """检查是否包含指定范围的所有步骤"""
        return all(self.get(s) is not None for s in range(start, end + 1))

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.step_map.clear()


class HVPConfig:
    """HVP计算配置"""

    def __init__(
        self,
        mode: str = "GGN",
        damping: float = 1e-4,
        rank: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.mode = mode  # "GGN", "diag", "low_rank", "exact"
        self.damping = damping
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.hessian_mode = mode  # 兼容性


def hvp_exact(
    model: nn.Module,
    loss_fn: Callable,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    精确HVP计算(使用二阶autograd)

    Args:
        model: 模型
        loss_fn: 损失函数
        batch_data: 批次数据
        v: 向量
        params: 参数列表(如果为None,使用model.parameters())

    Returns:
        Hessian-vector product
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # 前向传播计算损失
    model.zero_grad()
    outputs = model(**batch_data)
    loss = loss_fn(outputs) if callable(loss_fn) else outputs.loss

    # 计算一阶梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # 计算梯度与v的点积
    dot = sum((g * v_part).sum() for g, v_part in zip(grads, _unflatten_like(v, params)))

    # 计算二阶导数
    hvp = torch.autograd.grad(dot, params)

    # 展平并拼接
    return _flatten(hvp)


def hvp_ggn(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    广义Gauss-Newton近似的HVP
    对于交叉熵损失,这是稳定且高效的近似

    GGN: H ≈ J^T H_loss J
    其中 J 是输出关于参数的Jacobian, H_loss 是损失关于输出的Hessian

    对于交叉熵: H_loss ≈ I (忽略二阶项)
    因此: Hv ≈ J^T (J v)
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # 分解v为参数形状
    v_list = _unflatten_like(v, params)

    # 1. 计算 Jv (Jacobian-vector product)
    model.zero_grad()

    # 使用torch.func.jvp计算Jacobian-vector product
    # 首先前向传播
    outputs = model(**batch_data)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

    # 计算 ∇params(output) @ v
    # 使用反向传播: 如果设置grad_outputs=v, 得到的是 J^T v
    # 我们需要先计算 Jv, 再计算 J^T(Jv)

    # 使用有限差分近似或者双重反向传播
    # 简化: 使用 GGN ≈ J^T J
    # 实际上对于交叉熵,可以用Fisher信息矩阵近似

    # 计算损失关于logits的梯度
    if hasattr(outputs, 'loss'):
        loss = outputs.loss
    else:
        # 计算交叉熵损失
        labels = batch_data.get('labels', None)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='mean'
            )
        else:
            raise ValueError("No loss or labels found")

    # 计算梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # 计算 g^T v
    gv = sum((g * v_part).sum() for g, v_part in zip(grads, v_list))

    # 计算 ∇(g^T v) = GGN @ v (近似)
    hvp = torch.autograd.grad(gv, params)

    return _flatten(hvp)


def hvp_diagonal(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    diag_H: Optional[torch.Tensor] = None,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    对角Hessian近似

    Args:
        model: 模型
        batch_data: 批次数据
        v: 向量
        diag_H: 预计算的对角Hessian(如果为None,会计算)
        params: 参数列表

    Returns:
        对角近似的 Hv
    """
    if diag_H is not None:
        return diag_H * v

    # 如果没有预计算,需要计算对角Hessian
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    outputs = model(**batch_data)
    loss = outputs.loss if hasattr(outputs, 'loss') else outputs

    # 计算梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # 计算每个参数的对角Hessian
    diag_hessian = []
    for g in grads:
        # 对每个梯度分量计算二阶导数
        diag_h = []
        g_flat = g.view(-1)
        for i in range(g_flat.numel()):
            # 这个实现非常慢,仅用于演示
            # 实际应该使用更高效的方法
            h_ii = torch.autograd.grad(g_flat[i], params, retain_graph=True)[0]
            diag_h.append(h_ii.view(-1)[i])
        diag_hessian.append(torch.stack(diag_h))

    diag_H_full = torch.cat(diag_hessian)
    return diag_H_full * v


def hvp_apply(
    v: torch.Tensor,
    step_rec: StepRecord,
    cfg: HVPConfig,
    model: nn.Module,
    loss_fn: Optional[Callable] = None,
    batch_reconstructor: Optional[Any] = None,
) -> torch.Tensor:
    """
    应用HVP计算: 返回 H[s] @ v

    Args:
        v: 输入向量
        step_rec: 步骤记录
        cfg: HVP配置
        model: 模型
        loss_fn: 损失函数
        batch_reconstructor: 批次重建器(当batch_data为None时使用)

    Returns:
        Hessian-vector product
    """
    mode = cfg.hessian_mode

    # 获取批次数据
    batch_data = step_rec.batch_data

    # 如果没有batch_data,尝试重建
    if batch_data is None:
        # 对于diag模式,如果有预计算的diag_H,可以直接使用
        if mode == "diag" and step_rec.diag_H is not None:
            logger.debug(f"Using precomputed diag_H for step {step_rec.step_id}")
            return step_rec.diag_H * v

        # 否则需要重建批次数据
        if batch_reconstructor is None:
            raise ValueError(
                f"No batch data found for step {step_rec.step_id} and no batch_reconstructor provided"
            )

        logger.debug(f"Reconstructing batch data for step {step_rec.step_id}")
        batch_data = batch_reconstructor.get_batch_for_step(step_rec.step_id)

        if batch_data is None:
            raise ValueError(f"Failed to reconstruct batch data for step {step_rec.step_id}")

    # 确保数据在正确的设备上
    batch_data = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch_data.items()}

    if mode == "diag":
        return hvp_diagonal(model, batch_data, v, step_rec.diag_H)
    elif mode == "GGN":
        return hvp_ggn(model, batch_data, v)
    elif mode == "exact":
        return hvp_exact(model, loss_fn, batch_data, v)
    elif mode == "low_rank":
        # TODO: 实现低秩近似
        logger.warning("Low-rank HVP not implemented, falling back to GGN")
        return hvp_ggn(model, batch_data, v)
    else:
        raise ValueError(f"Unknown hessian_mode: {mode}")


@torch.no_grad()
def compute_correction(
    tz: int,
    tau: int,
    K: int,
    step_log: StepLog,
    cfg: HVPConfig,
    model: nn.Module,
    loss_fn: Optional[Callable] = None,
    batch_reconstructor: Optional[Any] = None,
) -> Tuple[torch.Tensor, AuditRecord]:
    """
    计算参数校正向量(前向K步传播)

    算法:
    1. 初始偏差: v0 = -u[tz] 或 v0 = η[tz] * gbar[tz]
    2. 前向传播: v[s+1] = v[s] - η[s] * (H[s] @ v[s]) - η[s] * λ * v[s]
    3. 返回最终的校正向量v

    Args:
        tz: 遗忘批次的步骤
        tau: 当前步骤
        K: 截断窗口大小
        step_log: 步骤日志
        cfg: HVP配置
        model: 模型
        loss_fn: 损失函数
        batch_reconstructor: 批次重建器(可选)

    Returns:
        (v, audit_record): 校正向量和审计记录
    """
    rec = step_log[tz]
    if rec is None:
        raise ValueError(f"Step {tz} not found in step log")

    if rec.u is None and (rec.gbar is None or rec.eta is None):
        raise ValueError(f"Step {tz} missing update vector u or (gbar, eta)")

    # 1. 初始偏差 v0
    if rec.u is not None:
        # v0 = -u[tz] = -(θ[tz+1] - θ[tz]) = θ[tz] - θ[tz+1]
        # 因为 u[tz] = -η[tz] * gbar[tz], 所以 -u[tz] = η[tz] * gbar[tz]
        v = -rec.u.clone()
    else:
        # v0 = η[tz] * gbar[tz]
        v = rec.eta * rec.gbar.clone()

    # 2. 前向传播窗口
    start = tz + 1
    end = min(tz + K, tau - 1)
    K_used = max(0, end - start + 1)

    hvp_calls = 0

    # 检查是否有足够的步骤记录
    if not step_log.has_range(start, end):
        logger.warning(f"Missing some steps in range [{start}, {end}]")

    for s in range(start, end + 1):
        srec = step_log[s]
        if srec is None:
            logger.warning(f"Step {s} not found, skipping")
            continue

        # 计算 H[s] @ v
        with torch.enable_grad():
            hvp = hvp_apply(v, srec, cfg, model, loss_fn, batch_reconstructor)

        # v ← v - η[s] * hvp
        v = v - srec.eta * hvp

        # 添加阻尼: v ← v - η[s] * λ * v
        if cfg.damping > 0:
            v = v - srec.eta * cfg.damping * v

        hvp_calls += 1

    # 3. 创建审计记录
    audit = AuditRecord(
        tz=tz,
        tau=tau,
        K_used=K_used,
        v_norm=float(v.norm().item()),
        hvp_calls=hvp_calls,
        mode=cfg.hessian_mode,
        damping=cfg.damping,
    )

    return v, audit


@torch.no_grad()
def apply_correction(
    v: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> None:
    """
    将校正向量v应用到参数: θ ← θ + v

    Args:
        v: 校正向量(展平的)
        params: 参数列表
    """
    offset = 0
    for p in params:
        n = p.numel()
        v_part = v[offset:offset+n].view_as(p)
        p.add_(v_part)
        offset += n

    if offset != v.numel():
        raise ValueError(f"Vector size mismatch: {offset} vs {v.numel()}")


# 辅助函数

def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    """将张量列表展平为单个向量"""
    return torch.cat([t.view(-1) for t in tensors])


def _unflatten_like(
    flat: torch.Tensor,
    target: List[torch.Tensor],
) -> List[torch.Tensor]:
    """将展平的向量还原为目标张量的形状列表"""
    result = []
    offset = 0
    for t in target:
        n = t.numel()
        result.append(flat[offset:offset+n].view_as(t))
        offset += n
    return result


def compute_param_update_vector(
    old_params: List[torch.Tensor],
    new_params: List[torch.Tensor],
) -> torch.Tensor:
    """
    计算参数更新向量: u = new_params - old_params

    Args:
        old_params: 旧参数列表
        new_params: 新参数列表

    Returns:
        展平的更新向量
    """
    updates = [new - old for old, new in zip(old_params, new_params)]
    return _flatten(updates)


def clone_parameters(model: nn.Module) -> List[torch.Tensor]:
    """克隆模型参数"""
    return [p.clone().detach() for p in model.parameters() if p.requires_grad]
