"""
LMCleaner Core Components: 在线遗忘的核心算法组件

包含:
- HVP (Hessian-Vector Product) 计算
- 参数校正向量计算 (前向K步传播)
- 参数校正应用
- 隐私噪声注入 (Phase 4 from paper)
- 训练日志数据结构
- 审计记录

论文参考: LMCleaner: Efficient and Certified Online Unlearning via Truncated Influence Propagation

实现与论文对齐说明:
- Fisher HVP: 论文 Algorithm 1 使用 Hv = g · (g^T v) 近似
- GGN HVP: 使用 Hv = ∇(g^T v) 近似 (更精确但计算量更大)
- 隐私噪声: 论文 Phase 4 注入 N(0, σ²I) 实现 (ε,δ)-certified unlearning
- 阻尼项: 实现添加的数值稳定化技术，论文理论分析未包含
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any, Protocol
import weakref
import gc

import torch
import torch.nn as nn

# Keys whose first dimension is the batch dimension in standard HF batch_data dicts.
# Used by _get_batch_size and micro-batch slicing in hvp_apply.
_BATCH_DIM_KEYS = ("input_ids", "labels", "attention_mask")

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord(dict):
    """审计记录，用于记录遗忘操作的详细信息

    包含论文 Algorithm 1 各阶段的追踪信息:
    - Phase 1: 初始偏差计算
    - Phase 2: 截断影响传播
    - Phase 3: 影响移除
    - Phase 4: 隐私保护 (噪声注入)
    """

    tz: int = 0  # 遗忘批次的步骤
    tau: int = 0  # 当前步骤
    K_used: int = 0  # 实际使用的截断窗口大小
    v_norm: float = 0.0  # 校正向量的范数 (Phase 3 前)
    hvp_calls: int = 0  # HVP调用次数
    mode: str = "GGN"  # Hessian模式: "fisher", "GGN", "diag", "exact"
    damping: float = 0.0  # 阻尼系数 (实现添加，论文未包含)
    # Phase 4: 隐私噪声相关
    noise_sigma: float = 0.0  # Legacy isotropic σ (kept for backward compat)
    noise_injected: bool = False  # 是否注入了隐私噪声
    epsilon: float = 0.0  # (ε,δ)-certified unlearning 的 ε
    delta: float = 0.0  # (ε,δ)-certified unlearning 的 δ
    # Subspace-aware noise (Eq. 13-15 of paper)
    noise_mode: str = "none"  # "none" | "isotropic" | "subspace"
    sigma_parallel: float = 0.0  # σ_∥ for top-k subspace
    sigma_perp: float = 0.0  # σ_⊥ for orthogonal complement
    beta: float = 1.0  # concentration factor β ∈ (0,1]
    projector_rank: int = 0  # k in Π_k
    delta_cert: float = 0.0  # Δ̄_cert(K) public sensitivity bound
    correction_norm: float = 0.0  # ||μ_K||₂
    noise_norm: float = 0.0  # ||ξ||₂
    # 历史参数重建
    used_historical_params: bool = False  # 是否实际使用了历史 θ[s]
    # 效率追踪
    wall_time_ms: float = 0.0  # 该 forget step 的处理耗时(毫秒)

    def __post_init__(self):
        # 使其可以像字典一样使用
        dict.__init__(self, **{f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()})


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
        return (
            f"StepRecord(step={self.step_id}, eta={self.eta}, batch_id={self.batch_id})"
        )


class LazyLoaderProtocol(Protocol):
    """Protocol for lazy record loaders that support on-demand loading."""

    def load_single_step(
        self, step_id: int, include_tensors: bool = True
    ) -> Optional[Dict]:
        """Load a single step record on demand."""
        ...

    def get_etas_for_steps(self, step_ids: List[int]) -> Dict[int, float]:
        """Get eta (learning rate) values for multiple steps."""
        ...

    @property
    def sample_indices(self) -> Dict[int, List[int]]:
        """Get sample indices mapping."""
        ...


class StepLog:
    """环形缓冲区，存储最近K+Δ个训练步骤的记录"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.step_map = {}  # step_id -> StepRecord (direct reference)

    def add(self, record: StepRecord):
        """添加新的步骤记录"""
        # 如果缓冲区满了,移除最旧的记录
        if len(self.buffer) == self.max_size:
            oldest = self.buffer[0]
            if oldest.step_id in self.step_map:
                del self.step_map[oldest.step_id]

        # 添加新记录
        self.buffer.append(record)
        self.step_map[record.step_id] = record  # Store record directly

    def get(self, step_id: int) -> Optional[StepRecord]:
        """获取指定步骤的记录"""
        return self.step_map.get(step_id)

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
    """HVP计算配置

    支持的 Hessian 近似模式:
    - "fisher": 论文 Algorithm 1 使用的 Fisher 信息矩阵近似
              Hv = g · (g^T v), 其中 g 是梯度向量
              这是 rank-1 近似，计算高效但精度较低
    - "GGN": 广义 Gauss-Newton 近似
            Hv = ∇(g^T v) = ∇²L · v
            使用二阶自动微分，更精确但计算量更大
    - "diag": 对角 Hessian 近似
    - "exact": 精确 Hessian-vector product (非常慢)

    阻尼项说明 (damping):
        论文原始算法不包含阻尼项。实现添加阻尼项 λ 用于数值稳定化:
        v ← v - η[s] * (H[s] @ v) - η[s] * λ * v
        等价于在 Hessian 上添加正则化: (H + λI) @ v
        当 damping=0 时与论文算法一致。
    """

    def __init__(
        self,
        mode: str = "fisher",  # 默认使用论文的 Fisher 近似
        damping: float = 0.0,  # 默认不添加阻尼，与论文一致
        rank: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        hvp_micro_batch_size: int = 0,  # 0 = use full batch; >0 = micro-batch HVP
    ):
        valid_modes = {"fisher", "GGN", "diag", "exact", "low_rank"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")
        self.mode = mode
        self.damping = damping
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.hessian_mode = mode  # 兼容性
        self.hvp_micro_batch_size = hvp_micro_batch_size


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
    dot = sum(
        (g * v_part).sum() for g, v_part in zip(grads, _unflatten_like(v, params))
    )

    # 计算二阶导数
    hvp = torch.autograd.grad(dot, params)

    # 展平并拼接
    return _flatten(hvp)


def hvp_fisher(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
    loss_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Fisher 信息矩阵近似的 HVP (论文 Algorithm 1 使用的方法)

    论文公式: Hv ← g · (g^T v)
    其中 g 是损失关于参数的梯度向量

    这是 rank-1 近似:
    - H ≈ g g^T (外积)
    - Hv = g (g^T v) = g * scalar

    特点:
    - 计算高效: 只需要一次梯度计算 + 向量点积
    - 精度较低: rank-1 近似可能丢失 Hessian 的高阶信息
    - 与论文一致: 这是论文 Algorithm 1 Line 6 描述的方法

    Args:
        model: 模型
        batch_data: 批次数据
        v: 向量
        params: 参数列表 (如果为 None, 使用 model.parameters())
        loss_fn: 自定义损失函数 (优先使用)

    Returns:
        Fisher 近似的 Hessian-vector product
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # 前向传播计算损失
    model.zero_grad()
    outputs = model(**batch_data)

    # 优先使用显式提供的 loss_fn，与其他 HVP 路径保持一致
    if loss_fn is not None:
        loss = loss_fn(outputs, batch_data)
    elif hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    else:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = batch_data.get("labels", None)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
            )
        else:
            raise ValueError("No loss_fn, outputs.loss, or labels found")

    # 计算梯度 g = ∇L
    grads = torch.autograd.grad(loss, params)

    # 展平梯度为向量
    g = _flatten([grad.detach() for grad in grads])

    # 计算 g^T v (标量)
    g_dot_v = (g * v).sum()  # torch.dot fails for vectors > 2.1B elements (int32 limit)

    # 返回 g * (g^T v) = Hv (Fisher 近似)
    return g * g_dot_v


def hvp_ggn(
    model: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    广义Gauss-Newton近似的HVP

    注意: 这与论文 Algorithm 1 的 Fisher 近似不同!
    论文使用: Hv = g · (g^T v) (rank-1 外积)
    GGN 使用: Hv = ∇(g^T v) (通过二阶自动微分)

    GGN: H ≈ J^T H_loss J
    其中 J 是输出关于参数的Jacobian, H_loss 是损失关于输出的Hessian

    对于交叉熵: H_loss ≈ I (忽略二阶项)
    因此: Hv ≈ J^T (J v)

    特点:
    - 比 Fisher 更精确
    - 需要二阶自动微分,计算量更大
    - 适用于需要更高精度的场景
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend

    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # Ensure v matches both device and dtype of model parameters (A40 GPU fix)
    ref = params[0] if params else v
    if v.device != ref.device or v.dtype != ref.dtype:
        v = v.to(ref)

    # Ensure all batch_data tensors are on the same device as the model
    if isinstance(batch_data, dict):
        for k, val in batch_data.items():
            if isinstance(val, torch.Tensor) and val.device != ref.device:
                batch_data[k] = val.to(ref.device)

    # 分解v为参数形状
    v_list = _unflatten_like(v, params)

    # 1. 计算 Jv (Jacobian-vector product)
    model.zero_grad()

    # 使用torch.func.jvp计算Jacobian-vector product
    # 首先前向传播
    # NOTE: Use MATH backend for SDPA to support 2nd order gradients (A40 GPU)
    # Flash/Memory-efficient attention don't support 2nd order derivatives
    with sdpa_kernel(SDPBackend.MATH):
        outputs = model(**batch_data)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    # 计算 ∇params(output) @ v
    # 使用反向传播: 如果设置grad_outputs=v, 得到的是 J^T v
    # 我们需要先计算 Jv, 再计算 J^T(Jv)

    # 使用有限差分近似或者双重反向传播
    # 简化: 使用 GGN ≈ J^T J
    # 实际上对于交叉熵,可以用Fisher信息矩阵近似

    # 计算损失关于logits的梯度
    if hasattr(outputs, "loss"):
        loss = outputs.loss
    else:
        # 计算交叉熵损失
        labels = batch_data.get("labels", None)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
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
    from torch.nn.attention import sdpa_kernel, SDPBackend

    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    # NOTE: Use MATH backend for SDPA to support 2nd order gradients (A40 GPU)
    with sdpa_kernel(SDPBackend.MATH):
        outputs = model(**batch_data)
    loss = outputs.loss if hasattr(outputs, "loss") else outputs

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


def _load_u_for_step(
    t: int,
    step_log: Optional["StepLog"] = None,
    lazy_loader: Optional["LazyLoaderProtocol"] = None,
    use_lazy_loading: bool = False,
    u_provider: Optional["UProviderProtocol"] = None,
) -> Optional[torch.Tensor]:
    """Shared helper: load u[t] from pkl → step_log → u_provider (recompute)."""
    # Try pkl
    if use_lazy_loading and lazy_loader is not None:
        rec_dict = lazy_loader.load_single_step(t, include_tensors=True)
        if rec_dict is not None:
            u_val = rec_dict.get("u")
            del rec_dict
            if u_val is not None and isinstance(u_val, torch.Tensor):
                return u_val

    # Try step_log (in-memory)
    if step_log is not None:
        srec_tmp = step_log[t]
        if srec_tmp is not None and srec_tmp.u is not None:
            return srec_tmp.u

    # Fall back to LazyUProvider (recompute u[t] = -η * grad)
    if u_provider is not None:
        return u_provider.get_u(t)

    return None


class HistoricalParamContext:
    """Context manager that temporarily sets model to historical θ[s] during propagation.

    Reconstructs θ[s] = θ[τ] - Σ_{t=s}^{τ-1} u[t] using stored parameter update
    vectors, and advances through θ[s] → θ[s+1] → ... during the propagation loop.
    Restores θ[τ] on exit (including exceptions).

    When disabled or when u[t] vectors are unavailable, acts as a no-op.

    Usage:
        with HistoricalParamContext(model, ...) as ctx:
            for s in range(start, end + 1):
                # model is now at θ[s]
                hvp = hvp_apply(v, srec, cfg, model, ...)
                ctx.advance(s)
        # model is back to θ[τ]
    """

    def __init__(
        self,
        model: nn.Module,
        start: int,
        end: int,
        tau: int,
        enabled: bool,
        step_log: Optional["StepLog"],
        lazy_loader: Optional["LazyLoaderProtocol"],
        use_lazy_loading: bool,
        u_provider=None,
    ):
        self.model = model
        self.start = start
        self.end = end
        self.tau = tau
        self._model_device = next(model.parameters()).device
        self._u_provider = u_provider

        self.active = False  # whether historical params are actually in use
        self._theta_tau: Optional[torch.Tensor] = None
        self._theta_current: Optional[torch.Tensor] = None
        self._u_vectors: Dict[int, torch.Tensor] = {}

        if not enabled or (end < start):
            if not enabled:
                logger.debug("Historical parameter reconstruction disabled by user")
            return

        self._try_load(step_log, lazy_loader, use_lazy_loading)

    def _load_u(
        self,
        t: int,
        step_log: Optional["StepLog"],
        lazy_loader: Optional["LazyLoaderProtocol"],
        use_lazy_loading: bool,
    ) -> Optional[torch.Tensor]:
        return _load_u_for_step(
            t,
            step_log=step_log,
            lazy_loader=lazy_loader,
            use_lazy_loading=use_lazy_loading,
            u_provider=self._u_provider,
        )

    def _try_load(
        self,
        step_log: Optional["StepLog"],
        lazy_loader: Optional["LazyLoaderProtocol"],
        use_lazy_loading: bool,
    ) -> None:
        """Try to load u vectors; set self.active = True on success."""
        # Phase A: load u vectors for propagation window [start, end]
        for t in range(self.start, self.end + 1):
            u_t = self._load_u(t, step_log, lazy_loader, use_lazy_loading)
            if u_t is None:
                self._u_vectors.clear()
                logger.warning(
                    "Cannot reconstruct historical parameters: u[t] not available "
                    f"for all steps in [{self.start}, {self.end}]. "
                    "Falling back to θ[τ]."
                )
                return
            self._u_vectors[t] = u_t.to(self._model_device)

        # Phase B: stream-accumulate tail [end+1, tau-1] without storing
        tail_sum: Optional[torch.Tensor] = None
        for t in range(self.end + 1, self.tau):
            u_t = self._load_u(t, step_log, lazy_loader, use_lazy_loading)
            if u_t is None:
                self._u_vectors.clear()
                logger.warning(
                    "Cannot reconstruct historical parameters: u[t] not available "
                    f"for tail steps in [{self.end + 1}, {self.tau - 1}]. "
                    "Falling back to θ[τ]."
                )
                return
            u_t = u_t.to(self._model_device)
            if tail_sum is None:
                tail_sum = u_t.clone()
            else:
                tail_sum += u_t
            del u_t

        # Compute θ[start] = θ[τ] - Σ u[t]
        self._theta_tau = _get_flat_params(self.model).clone()
        self._theta_current = self._theta_tau.clone()
        for t in range(self.start, self.end + 1):
            self._theta_current -= self._u_vectors[t]
        if tail_sum is not None:
            self._theta_current -= tail_sum

        self.active = True
        logger.debug(
            f"Historical parameter reconstruction enabled for steps "
            f"[{self.start}, {self.end}] ({len(self._u_vectors)} u vectors)"
        )

    def __enter__(self) -> "HistoricalParamContext":
        if self.active:
            _set_flat_params(self.model, self._theta_current)
        return self

    def __exit__(self, *exc) -> None:
        if self.active and self._theta_tau is not None:
            _set_flat_params(self.model, self._theta_tau)
            del self._theta_tau, self._theta_current
            self._u_vectors.clear()
            # Note: keep self.active = True for audit purposes

    def advance(self, s: int) -> None:
        """Advance model from θ[s] to θ[s+1]."""
        if self.active and self._theta_current is not None and s in self._u_vectors:
            self._theta_current += self._u_vectors[s]
            _set_flat_params(self.model, self._theta_current)


@torch.no_grad()
def _set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set model parameters from a flattened vector (in-place)."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            p.copy_(flat_params[offset : offset + n].view_as(p))
            offset += n
    if offset != flat_params.numel():
        raise ValueError(
            f"Flat params size mismatch: consumed {offset}, got {flat_params.numel()}"
        )


@torch.no_grad()
def _get_flat_params(model: nn.Module) -> torch.Tensor:
    """Get model parameters as a flattened, detached vector."""
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]
    )


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

    根据论文 Algorithm 1, HVP 应在历史参数 θ[s] 处计算。
    调用方 (compute_correction) 负责在调用前将模型参数设置为 θ[s],
    调用后恢复为 θ[τ]。

    Args:
        v: 输入向量
        step_rec: 步骤记录
        cfg: HVP配置
        model: 模型 (调用方应确保参数已设置为历史 θ[s])
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
            raise ValueError(
                f"Failed to reconstruct batch data for step {step_rec.step_id}"
            )

    # 确保数据在正确的设备上
    batch_data = {
        k: val.to(cfg.device) if isinstance(val, torch.Tensor) else val
        for k, val in batch_data.items()
    }

    def _hvp_single(bd):
        """Run HVP on a single (micro-)batch."""
        if mode == "fisher":
            return hvp_fisher(model, bd, v, loss_fn=loss_fn)
        elif mode == "diag":
            return hvp_diagonal(model, bd, v, step_rec.diag_H)
        elif mode == "GGN":
            return hvp_ggn(model, bd, v)
        elif mode == "exact":
            return hvp_exact(model, loss_fn, bd, v)
        elif mode == "low_rank":
            logger.warning("Low-rank HVP not implemented, falling back to fisher")
            return hvp_fisher(model, bd, v, loss_fn=loss_fn)
        else:
            raise ValueError(f"Unknown hessian_mode: {mode}")

    # Micro-batch HVP to avoid OOM on large batches
    mbs = cfg.hvp_micro_batch_size
    batch_size = _get_batch_size(batch_data)
    if batch_size == 0:
        logger.warning("Could not infer batch size from batch_data, falling back to full-batch HVP")
        return _hvp_single(batch_data)
    if mbs <= 0 or batch_size <= mbs:
        return _hvp_single(batch_data)

    # Guard: micro-batch averaging is only correct under mean reduction.
    # HF Trainer and all built-in HVP modes (fisher, GGN) use mean loss.
    # If a custom loss_fn with sum/none reduction is used, fall back to full batch.
    if loss_fn is not None and hasattr(loss_fn, "reduction") and loss_fn.reduction != "mean":
        logger.warning(
            f"HVP micro-batching requires mean-reduced loss, got {loss_fn.reduction}; "
            "falling back to full-batch HVP"
        )
        return _hvp_single(batch_data)

    # Split batch into micro-batches, compute HVP on each, weighted average.
    # Each chunk's HVP is weighted by chunk_size / total_samples to preserve
    # equivalence with full-batch mean-loss HVP.
    slice_keys = _get_sliceable_keys(batch_data, batch_size)
    hvp_acc = torch.zeros_like(v)
    total_samples = 0
    for start_idx in range(0, batch_size, mbs):
        end_idx = min(start_idx + mbs, batch_size)
        chunk_size = end_idx - start_idx
        micro = {
            k: val[start_idx:end_idx] if k in slice_keys else val
            for k, val in batch_data.items()
        }
        chunk_hvp = _hvp_single(micro)
        hvp_acc += chunk_hvp * chunk_size
        total_samples += chunk_size
        del micro, chunk_hvp

    return hvp_acc / total_samples


def compute_noise_sigma(
    delta_det: float,
    epsilon: float,
    delta: float,
) -> float:
    """
    Isotropic fallback: σ_iso = (Δ_det / ε) * sqrt(2 log(1.25/δ))

    Kept for backward compatibility. Use compute_subspace_noise_params for
    the paper's subspace-aware calibration (Eq. 14-15).
    """
    if delta_det < 0:
        raise ValueError(f"delta_det must be >= 0, got {delta_det}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    sigma = (delta_det / epsilon) * math.sqrt(2 * math.log(1.25 / delta))
    return sigma


def compute_subspace_noise_params(
    delta_cert: float,
    epsilon: float,
    delta: float,
    beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Paper Eq. 14-15: Subspace-aware noise calibration.

    Splits privacy budget (ε/2, δ/2) across parallel and orthogonal components:
        σ_∥ = (2 Δ̄_cert(K) / ε) * sqrt(2 log(2.5/δ))      (Eq. 14)
        σ_⊥ = (2β Δ̄_cert(K) / ε) * sqrt(2 log(2.5/δ))     (Eq. 15)

    Args:
        delta_cert: Public sensitivity upper bound Δ̄_cert(K)
        epsilon: Privacy parameter ε > 0
        delta: Privacy parameter δ ∈ (0, 1)
        beta: Concentration factor β ∈ (0, 1], controls noise reduction
              outside top-k subspace. β=1 makes σ_⊥ = σ_∥ (isotropic).

    Returns:
        (sigma_parallel, sigma_perp)
    """
    if delta_cert < 0:
        raise ValueError(f"delta_cert must be >= 0, got {delta_cert}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if beta <= 0 or beta > 1:
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    # The 2 in numerator comes from budget splitting: each block gets (ε/2, δ/2)
    # 2.5/δ instead of 1.25/δ because δ → δ/2
    multiplier = math.sqrt(2 * math.log(2.5 / delta))
    sigma_parallel = (2 * delta_cert / epsilon) * multiplier
    sigma_perp = (2 * beta * delta_cert / epsilon) * multiplier

    return sigma_parallel, sigma_perp


def build_public_projector(
    d: int,
    k: int,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a fixed public random projector Π_k (d×k orthonormal matrix).

    Paper Appendix C.5: "from public randomness" — a fixed random
    k-dimensional subspace independent of the training data.

    The projector Π_k satisfies: Π_k @ Π_k^T is the rank-k projection matrix.
    We store Q (d×k) such that Π_k = Q @ Q^T.

    Args:
        d: Parameter dimension
        k: Subspace rank (k << d)
        seed: Fixed seed for reproducibility (public randomness)
        device: Device for the projector

    Returns:
        Q: Orthonormal basis matrix of shape (d, k)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # Generate random matrix and orthogonalize via QR
    # For memory efficiency with large d, use chunked random generation
    A = torch.randn(d, k, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    return Q.to(device)


def inject_privacy_noise(
    v: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Isotropic noise injection: v + N(0, σ²I). Kept for backward compatibility.
    """
    if sigma <= 0:
        return v
    noise = torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=generator) * sigma
    return v + noise


def inject_subspace_noise(
    v: torch.Tensor,
    sigma_parallel: float,
    sigma_perp: float,
    projector_Q: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Paper Eq. 13: Subspace-aware noise injection.

        ξ = ξ_∥ + ξ_⊥
        ξ_∥ ~ N(0, σ²_∥ Π_k)       [noise in top-k subspace]
        ξ_⊥ ~ N(0, σ²_⊥ (I - Π_k)) [noise in orthogonal complement]

    Implementation:
        z ~ N(0, I_d)
        z_∥ = Q @ (Q^T @ z)    [project onto subspace]
        z_⊥ = z - z_∥           [orthogonal component]
        ξ = σ_∥ * z_∥ + σ_⊥ * z_⊥

    Args:
        v: Correction vector to add noise to
        sigma_parallel: σ_∥
        sigma_perp: σ_⊥
        projector_Q: Orthonormal basis Q of shape (d, k)
        generator: Random generator for reproducibility

    Returns:
        (v + ξ, ||ξ||₂)
    """
    if sigma_parallel <= 0 and sigma_perp <= 0:
        return v, 0.0

    z = torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=generator)

    # Project z onto subspace: z_∥ = Q @ (Q^T @ z)
    Q = projector_Q.to(v.device, v.dtype)
    coeffs = Q.T @ z  # (k,)
    z_parallel = Q @ coeffs  # (d,)
    z_perp = z - z_parallel

    noise = sigma_parallel * z_parallel + sigma_perp * z_perp
    noise_norm = float(noise.norm().item())

    return v + noise, noise_norm


class HistoricalParamProvider:
    """
    Provides historical parameters theta[s] for HVP computation.

    Reconstruction strategy (prioritized):
    1. Sparse checkpoint: theta[s] = theta[c] + sum_{t=c}^{s-1} u[t]  (c = nearest checkpoint)
    2. Current model: theta[s] = theta[tau] - sum_{t=s}^{tau-1} u[t]  (fallback)

    Supports incremental advancement within a propagation window:
    theta[s+1] = theta[s] + u[s]

    Args:
        model: The model whose parameters will be temporarily swapped
        log_dir: Training log directory (for loading sparse checkpoints)
        lazy_loader: LazyRecordLoader for loading u[t] vectors
        step_log: StepLog (in-memory, for non-lazy mode)
        max_cache_entries: LRU cache size for reconstructed params
    """

    def __init__(
        self,
        model: nn.Module,
        log_dir: Optional[str] = None,
        lazy_loader: Optional[LazyLoaderProtocol] = None,
        step_log: Optional[StepLog] = None,
        max_cache_entries: int = 4,
    ):
        self.model = model
        self.log_dir = Path(log_dir) if log_dir else None
        self.lazy_loader = lazy_loader
        self.step_log = step_log
        self.max_cache_entries = max_cache_entries

        # Load checkpoint index
        self._checkpoint_index: Dict[int, str] = {}
        self._checkpoint_stride: int = 25
        if self.log_dir:
            ckpt_index_file = self.log_dir / "checkpoint_index.json"
            if ckpt_index_file.exists():
                import json

                with open(ckpt_index_file) as f:
                    ckpt_data = json.load(f)
                self._checkpoint_index = {
                    int(k): v for k, v in ckpt_data.get("checkpoints", {}).items()
                }
                self._checkpoint_stride = ckpt_data.get("stride", 25)
                logger.info(
                    f"HistoricalParamProvider: loaded {len(self._checkpoint_index)} "
                    f"sparse checkpoints (stride={self._checkpoint_stride})"
                )

        # theta[tau] backup (saved once, restored after all HVP calls)
        self._theta_tau: Optional[torch.Tensor] = None
        # Current reconstructed theta[s] as flat vector
        self._theta_current: Optional[torch.Tensor] = None
        self._current_step: Optional[int] = None

        # LRU cache: step_id -> flat param tensor (on CPU)
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_order: List[int] = []

        # Device of model parameters
        self._model_device = next(model.parameters()).device

        # Pre-loaded u vectors for propagation window
        self._u_vectors: Dict[int, torch.Tensor] = {}
        self._active = False  # Whether we're in an active historical session

    def _load_u(self, t: int) -> Optional[torch.Tensor]:
        """Load u[t] from lazy_loader or step_log."""
        if t in self._u_vectors:
            return self._u_vectors[t]
        if self.lazy_loader is not None:
            rec_dict = self.lazy_loader.load_single_step(t, include_tensors=True)
            if rec_dict is not None:
                u_val = rec_dict.get("u")
                del rec_dict
                return u_val
        elif self.step_log is not None:
            srec = self.step_log[t]
            if srec is not None:
                return srec.u
        return None

    def _find_nearest_checkpoint(self, step_id: int) -> Optional[int]:
        """Find the nearest checkpoint at or before step_id."""
        if not self._checkpoint_index:
            return None
        candidates = [c for c in self._checkpoint_index.keys() if c <= step_id]
        return max(candidates) if candidates else None

    def _load_checkpoint(self, checkpoint_step: int) -> torch.Tensor:
        """Load a sparse checkpoint and return as flat parameter tensor."""
        if self.log_dir is None:
            raise ValueError("log_dir required for checkpoint loading")
        rel_path = self._checkpoint_index[checkpoint_step]
        ckpt_path = self.log_dir / rel_path
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Convert to flat vector matching model parameter order
        flat_parts = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in state_dict:
                    flat_parts.append(state_dict[name].reshape(-1).float())
                else:
                    logger.warning(
                        f"Checkpoint at step {checkpoint_step} missing key '{name}', "
                        "using current model parameter (may skew reconstruction)"
                    )
                    flat_parts.append(param.detach().cpu().reshape(-1).float())
        return torch.cat(flat_parts)

    def _reconstruct_from_checkpoint(self, target_step: int) -> Optional[torch.Tensor]:
        """Reconstruct theta[target_step] from nearest sparse checkpoint + u vectors."""
        ckpt_step = self._find_nearest_checkpoint(target_step)
        if ckpt_step is None:
            return None

        # Load checkpoint
        theta = self._load_checkpoint(ckpt_step)

        # Accumulate u vectors: theta[target] = theta[ckpt] + sum_{t=ckpt}^{target-1} u[t]
        for t in range(ckpt_step, target_step):
            u_t = self._load_u(t)
            if u_t is None:
                logger.warning(
                    f"Missing u[{t}] for checkpoint reconstruction "
                    f"(ckpt={ckpt_step} -> target={target_step}). Aborting."
                )
                return None
            theta = theta + u_t.cpu().float()

        return theta

    def _reconstruct_from_current(
        self, target_step: int, tau: int
    ) -> Optional[torch.Tensor]:
        """Reconstruct theta[target_step] from theta[tau] - sum u[t]."""
        if self._theta_tau is None:
            self._theta_tau = _get_flat_params(self.model).clone()

        theta = self._theta_tau.clone()
        for t in range(target_step, tau):
            u_t = self._load_u(t)
            if u_t is None:
                return None
            theta = theta - u_t.to(theta.device)

        return theta

    def _cache_put(self, step_id: int, theta: torch.Tensor):
        """Put a reconstructed theta into LRU cache."""
        if step_id in self._cache:
            self._cache_order.remove(step_id)
        elif len(self._cache) >= self.max_cache_entries:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        self._cache[step_id] = theta.cpu()
        self._cache_order.append(step_id)

    def get_params_for_step(self, step_id: int, tau: int) -> Optional[torch.Tensor]:
        """
        Get theta[step_id] as a flat tensor.

        Priority:
        1. LRU cache
        2. Incremental from current position
        3. Sparse checkpoint + u accumulation
        4. Current model fallback (theta[tau] - sum u[t])
        """
        # Check cache
        if step_id in self._cache:
            return self._cache[step_id].to(self._model_device)

        # Incremental from current position
        if (
            self._theta_current is not None
            and self._current_step is not None
            and self._current_step <= step_id
        ):
            theta = self._theta_current.clone()
            for t in range(self._current_step, step_id):
                u_t = self._load_u(t)
                if u_t is None:
                    break
                theta = theta + u_t.to(theta.device)
            else:
                self._cache_put(step_id, theta)
                return theta.to(self._model_device)

        # Try sparse checkpoint path
        theta = self._reconstruct_from_checkpoint(step_id)
        if theta is not None:
            self._cache_put(step_id, theta)
            return theta.to(self._model_device)

        # Fallback: reconstruct from current model
        theta = self._reconstruct_from_current(step_id, tau)
        if theta is not None:
            self._cache_put(step_id, theta)
            return theta.to(self._model_device)

        return None

    def prepare_window(self, start: int, end: int, tau: int) -> bool:
        """
        Pre-load u vectors for a propagation window [start, end] and
        reconstruct theta[start]. Returns True if successful.
        """
        # Save theta[tau] if not already saved
        if self._theta_tau is None:
            self._theta_tau = _get_flat_params(self.model).clone()

        # Pre-load u vectors for the window
        self._u_vectors.clear()
        for t in range(start, end + 1):
            u_t = self._load_u(t)
            if u_t is not None:
                self._u_vectors[t] = u_t.to(self._model_device)
            else:
                logger.warning(f"Missing u[{t}] in window [{start}, {end}]")
                self._u_vectors.clear()
                return False

        # Reconstruct theta[start]
        theta_start = self.get_params_for_step(start, tau)
        if theta_start is None:
            self._u_vectors.clear()
            return False

        self._theta_current = theta_start
        self._current_step = start
        self._active = True

        # Set model to theta[start]
        _set_flat_params(self.model, self._theta_current)

        return True

    def advance_to_next_step(self, current_s: int):
        """
        Advance theta from current_s to current_s + 1: theta[s+1] = theta[s] + u[s].
        Updates the model parameters in-place.
        """
        if not self._active or self._theta_current is None:
            return

        u_s = self._u_vectors.get(current_s)
        if u_s is not None:
            self._theta_current = self._theta_current + u_s
            self._current_step = current_s + 1
            _set_flat_params(self.model, self._theta_current)

    def restore_model(self):
        """Restore model parameters to theta[tau]."""
        if self._theta_tau is not None:
            _set_flat_params(self.model, self._theta_tau)
        self._active = False
        self._theta_current = None
        self._current_step = None
        self._u_vectors.clear()

    def cleanup(self):
        """Release all resources."""
        self.restore_model()
        self._theta_tau = None
        self._cache.clear()
        self._cache_order.clear()


@torch.no_grad()
def compute_correction(
    tz: int,
    tau: int,
    K: int,
    step_log: Optional[StepLog],
    cfg: HVPConfig,
    model: nn.Module,
    loss_fn: Optional[Callable] = None,
    batch_reconstructor: Optional[Any] = None,
    # Phase 4: 隐私噪声参数
    epsilon: float = 0.0,
    delta: float = 1e-5,
    delta_det: Optional[float] = None,
    # Subspace-aware noise (paper Eq. 13-15)
    noise_mode: str = "subspace",  # "subspace" | "isotropic" | "none"
    beta: float = 0.1,  # concentration factor β ∈ (0,1]
    projector_rank: int = 100,  # k for Π_k
    projector_seed: int = 42,  # seed for public random projector
    # On-demand loading support for large K values
    lazy_loader: Optional[LazyLoaderProtocol] = None,
    initial_record: Optional[StepRecord] = None,
    # Historical parameter reconstruction
    use_historical_params: bool = True,
    historical_param_provider: Optional[HistoricalParamProvider] = None,
    # Lazy u[t] recomputation (for steps without saved pkl)
    u_provider: Optional["UProviderProtocol"] = None,
) -> Tuple[torch.Tensor, AuditRecord]:
    """
    计算参数校正向量(前向K步传播) - 论文 Algorithm 1 完整实现

    算法:
    Phase 1: 计算初始偏差 v0 = -u[tz] 或 v0 = η[tz] * gbar[tz]
    Phase 2: 前向传播 v[s+1] = v[s] - η[s] * (H[s] @ v[s]) - η[s] * λ * v[s]
             其中 H[s] 在历史参数 θ[s] 处计算 (论文 Algorithm 1 Line 6)
    Phase 3: 返回校正向量 v (将在外部应用: θ ← θ + v)
    Phase 4: 可选的隐私噪声注入

    历史参数重建 (论文 Section 5 - Parameter Snapshots):
        θ[s] = θ[τ] - Σ_{t=s}^{τ-1} u[t]
        其中 u[t] = θ[t+1] - θ[t] 是每步的参数更新向量。
        当 use_historical_params=True 且 u[t] 向量可用时，在每步传播前
        将模型参数临时设为 θ[s], 计算 HVP 后恢复为 θ[τ]。
        当 u[t] 不可用或 use_historical_params=False 时，使用 θ[τ]。

    ! 注意 (内存开销):
    !   启用 use_historical_params 会额外占用约 1x 模型参数量的 GPU 显存
    !   (用于维护 theta_current flat 向量) 以及读取窗口内 u[t] 的 IO 开销。
    !   对于显存紧张的环境，可设置 use_historical_params=False 退化为旧行为。
    !   详见 docs/historical_params_hvp.md

    注意 (与论文的差异):
    - 阻尼项 λ 是实现添加的数值稳定化技术，论文理论分析未包含
    - 设置 cfg.damping=0 以与论文算法完全一致

    Args:
        tz: 遗忘批次的步骤
        tau: 当前步骤
        K: 截断窗口大小
        step_log: 步骤日志 (如果 lazy_loader 提供，可以为 None)
        cfg: HVP配置
        model: 模型
        loss_fn: 损失函数
        batch_reconstructor: 批次重建器(可选)
        epsilon: 隐私参数 ε (>0 时启用噪声注入)
        delta: 隐私参数 δ (默认 1e-5)
        delta_det: 近似误差上界 (如果为 None, 使用 ||v||₂ 作为估计)
        lazy_loader: 懒加载器，用于大 K 值时按需加载记录 (避免 OOM)
        initial_record: 初始步骤记录 (tz 对应的记录，用于 Phase 1)
        use_historical_params: 是否在历史参数 θ[s] 处计算 HVP (默认 True)
            True: 论文 Algorithm 1 的精确实现，额外 ~1x 模型参数量显存
            False: 在 θ[τ] 处计算 HVP，节省显存但引入近似误差

    Returns:
        (v, audit_record): 校正向量和审计记录
    """
    # 确定是否使用 lazy loading 模式
    use_lazy_loading = lazy_loader is not None

    # 获取初始记录
    if initial_record is not None:
        rec = initial_record
    elif step_log is not None:
        rec = step_log[tz]
    else:
        raise ValueError("Either step_log or initial_record must be provided")

    if rec is None:
        raise ValueError(f"Step {tz} not found in step log")

    if rec.u is None and (rec.gbar is None or rec.eta is None):
        raise ValueError(f"Step {tz} missing update vector u or (gbar, eta)")

    # Phase 1: 初始偏差 v0
    if rec.u is not None:
        # v0 = -u[tz] = -(θ[tz+1] - θ[tz]) = θ[tz] - θ[tz+1]
        # 因为 u[tz] = -η[tz] * gbar[tz], 所以 -u[tz] = η[tz] * gbar[tz]
        v = -rec.u.clone()
    else:
        # v0 = η[tz] * gbar[tz]
        v = rec.eta * rec.gbar.clone()

    # 确保 v 在正确的设备上 (从训练日志加载的 u/gbar 可能在 CPU 上)
    v = v.to(cfg.device)

    # 清理初始记录
    if use_lazy_loading and initial_record is not None:
        del rec
        gc.collect()

    # Phase 2: 前向传播窗口
    start = tz + 1
    end = min(tz + K, tau - 1)
    K_used = max(0, end - start + 1)

    hvp_calls = 0

    # 检查是否有足够的步骤记录 (only for non-lazy mode)
    if not use_lazy_loading and step_log is not None:
        if not step_log.has_range(start, end):
            logger.warning(f"Missing some steps in range [{start}, {end}]")

    # 预先获取所有需要的 eta 值 (轻量操作，可以批量获取)
    if use_lazy_loading and lazy_loader is not None:
        needed_steps = list(range(start, end + 1))
        eta_map = lazy_loader.get_etas_for_steps(needed_steps)
    else:
        eta_map = None

    # --- Historical parameter setup ---
    _do_historical = False
    _external_provider = historical_param_provider is not None

    # 获取模型参数所在的 device (与 cfg.device 可能不同)
    model_device = next(model.parameters()).device

    # Legacy inline state (only used when no external provider)
    u_vectors: Dict[int, torch.Tensor] = {}
    theta_tau: Optional[torch.Tensor] = None
    theta_current: Optional[torch.Tensor] = None

    if K_used > 0 and use_historical_params:
        if _external_provider:
            # Use the provided HistoricalParamProvider
            _do_historical = historical_param_provider.prepare_window(start, end, tau)
            if not _do_historical:
                logger.warning(
                    "HistoricalParamProvider failed to prepare window "
                    f"[{start}, {end}]. Falling back to θ[τ]."
                )
        else:
            # Legacy inline reconstruction (for backward compatibility)
            def _load_u(t: int) -> Optional[torch.Tensor]:
                return _load_u_for_step(
                    t,
                    step_log=step_log,
                    lazy_loader=lazy_loader,
                    use_lazy_loading=use_lazy_loading,
                    u_provider=u_provider,
                )

            # Phase A: 收集传播窗口 [start, end] 内的 u 向量
            _missing_u = False
            for t in range(start, end + 1):
                u_t = _load_u(t)
                if u_t is not None:
                    u_vectors[t] = u_t.to(model_device)
                else:
                    _missing_u = True
                    break

            if _missing_u:
                u_vectors.clear()
                logger.warning(
                    "Cannot reconstruct historical parameters: u[t] vectors not available "
                    f"for all steps in [{start}, {end}]. "
                    "Falling back to current parameters θ[τ] for HVP."
                )
            else:
                # Phase B: Compute θ[start] = θ[τ] - Σ_{t=start}^{τ-1} u[t]
                # Load tail u vectors [end+1, tau-1] in a single pass:
                # verify availability and accumulate directly into theta_current.
                _do_historical = True
                logger.debug(
                    f"Historical parameter reconstruction enabled for steps [{start}, {end}] "
                    f"(loaded {len(u_vectors)} u vectors for propagation)"
                )
    elif K_used > 0 and not use_historical_params:
        logger.debug("Historical parameter reconstruction disabled by user")

    # Build θ[start] for legacy path: single-pass tail accumulation
    if _do_historical and not _external_provider:
        theta_tau = _get_flat_params(model).clone()
        theta_current = theta_tau.clone()
        # Subtract window u vectors (already in memory)
        for t in range(start, end + 1):
            theta_current -= u_vectors[t]
        # Subtract tail u vectors in a single streaming pass (no double-load)
        for t in range(end + 1, tau):
            u_t = _load_u(t)
            if u_t is None:
                # Tail incomplete — abort historical reconstruction
                _do_historical = False
                u_vectors.clear()
                _set_flat_params(model, theta_tau)
                del theta_tau, theta_current
                logger.warning(
                    "Cannot reconstruct historical parameters: u[t] not available "
                    f"for tail step {t} in [{end + 1}, {tau - 1}]. "
                    "Falling back to current parameters θ[τ] for HVP."
                )
                break
            theta_current -= u_t.to(model_device)
            del u_t
        else:
            # All tail u vectors loaded successfully
            _set_flat_params(model, theta_current)

    # Helper: advance θ[s] -> θ[s+1] via theta_current += u[s] (legacy path)
    def _advance_theta(s: int) -> None:
        nonlocal theta_current
        if (
            _do_historical
            and not _external_provider
            and theta_current is not None
            and s in u_vectors
        ):
            theta_current += u_vectors[s]
            _set_flat_params(model, theta_current)

    # Helper: advance historical params for current step
    def _do_advance(s: int) -> None:
        if _do_historical and _external_provider:
            historical_param_provider.advance_to_next_step(s)
        elif _do_historical:
            _advance_theta(s)  # legacy path

    for s in range(start, end + 1):
        # 获取步骤记录
        if use_lazy_loading and lazy_loader is not None:
            # 优化: 当 batch_reconstructor 可用时，只需要 eta，无需加载完整 tensor
            # 这避免了每次 HVP 调用都加载 2.4GB 的 pickle 文件
            if batch_reconstructor is not None and cfg.hessian_mode in (
                "GGN",
                "fisher",
            ):
                # 使用预加载的 eta_map，无需加载完整记录
                if s not in eta_map:
                    logger.warning(f"Step {s} eta not found, skipping")
                    _do_advance(s)
                    continue
                # 创建轻量级 StepRecord，只包含 HVP 计算所需的最小信息
                srec = StepRecord(
                    step_id=s,
                    eta=eta_map[s],
                    batch_id=s,  # batch_id will be used by batch_reconstructor
                    u=None,
                    gbar=None,
                    diag_H=None,
                    batch_data=None,  # Will be reconstructed by batch_reconstructor
                )
            else:
                # 需要加载完整记录 (如 diag 模式需要 diag_H)
                rec_dict = lazy_loader.load_single_step(s, include_tensors=True)
                if rec_dict is None:
                    logger.warning(f"Step {s} not found, skipping")
                    _do_advance(s)
                    continue
                srec = StepRecord(
                    step_id=rec_dict["step_id"],
                    eta=rec_dict["eta"],
                    batch_id=rec_dict["batch_id"],
                    u=rec_dict.get("u"),
                    gbar=rec_dict.get("gbar"),
                    diag_H=rec_dict.get("diag_H"),
                    batch_data=rec_dict.get("batch_data"),
                )
                del rec_dict
        else:
            srec = step_log[s] if step_log else None
            if srec is None:
                logger.warning(f"Step {s} not found, skipping")
                _do_advance(s)
                continue

        # 计算 H[s] @ v (模型参数已是 θ[s] 或 θ[τ])
        with torch.enable_grad():
            hvp = hvp_apply(v, srec, cfg, model, loss_fn, batch_reconstructor)

        eta = srec.eta
        v = v - eta * hvp

        # 添加阻尼: v ← v - η[s] * λ * v (实现添加，论文未包含)
        if cfg.damping > 0:
            v = v - eta * cfg.damping * v

        hvp_calls += 1

        # 推进到 θ[s+1]
        _do_advance(s)

        # 清理当前记录 (lazy loading 模式)
        if use_lazy_loading:
            del srec, hvp
            if hvp_calls % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # 恢复模型参数为 θ[τ]
    if _do_historical:
        if _external_provider:
            historical_param_provider.restore_model()
        elif theta_tau is not None:
            _set_flat_params(model, theta_tau)
            del theta_tau, theta_current
            u_vectors.clear()

    # 记录 Phase 3 前的 v_norm
    v_norm_before_noise = float(v.norm().item())

    # Phase 4: 隐私噪声注入 (论文 Algorithm 1, Eq. 13-15)
    _noise_sigma = 0.0  # legacy isotropic sigma
    _sigma_par = 0.0
    _sigma_perp = 0.0
    _noise_norm = 0.0
    _noise_injected = False
    _actual_noise_mode = "none"

    if epsilon > 0 and noise_mode != "none":
        # Determine sensitivity bound Δ̄_cert(K)
        if delta_det is None:
            logger.warning(
                "delta_det not provided, using v_norm as estimate. "
                "For guaranteed (ε,δ)-certified unlearning, provide a proven Δ̄_cert(K) bound."
            )
            delta_det = v_norm_before_noise

        if noise_mode == "subspace" and projector_rank > 0:
            # Paper Eq. 14-15: subspace-aware noise
            _sigma_par, _sigma_perp = compute_subspace_noise_params(
                delta_cert=delta_det, epsilon=epsilon, delta=delta, beta=beta,
            )
            d = v.numel()
            k = min(projector_rank, d)
            Q = build_public_projector(d, k, seed=projector_seed, device=v.device)
            v, _noise_norm = inject_subspace_noise(v, _sigma_par, _sigma_perp, Q)
            _noise_injected = True
            _actual_noise_mode = "subspace"
            _noise_sigma = _sigma_par  # legacy field: use σ_∥ for compat
            del Q

            logger.info(
                f"Phase 4: Subspace noise σ_∥={_sigma_par:.6f}, σ_⊥={_sigma_perp:.6f} "
                f"(β={beta}, k={k}, ε={epsilon}, δ={delta}), ||ξ||={_noise_norm:.4f}"
            )
        else:
            # Isotropic fallback (β=1, Π_k=I)
            _noise_sigma = compute_noise_sigma(delta_det, epsilon, delta)
            v = inject_privacy_noise(v, _noise_sigma)
            _noise_norm = 0.0  # not tracked for isotropic
            _noise_injected = True
            _actual_noise_mode = "isotropic"
            _sigma_par = _noise_sigma
            _sigma_perp = _noise_sigma

            logger.info(
                f"Phase 4: Isotropic noise σ={_noise_sigma:.6f} "
                f"(ε={epsilon}, δ={delta})"
            )

    # 创建审计记录
    audit = AuditRecord(
        tz=tz,
        tau=tau,
        K_used=K_used,
        v_norm=v_norm_before_noise,
        hvp_calls=hvp_calls,
        mode=cfg.hessian_mode,
        damping=cfg.damping,
        noise_sigma=_noise_sigma,
        noise_injected=_noise_injected,
        epsilon=epsilon if _noise_injected else 0.0,
        delta=delta if _noise_injected else 0.0,
        noise_mode=_actual_noise_mode,
        sigma_parallel=_sigma_par,
        sigma_perp=_sigma_perp,
        beta=beta,
        projector_rank=projector_rank if _actual_noise_mode == "subspace" else 0,
        delta_cert=delta_det if delta_det is not None else 0.0,
        correction_norm=v_norm_before_noise,
        noise_norm=_noise_norm,
        used_historical_params=_do_historical,
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
        v_part = v[offset : offset + n].view_as(p)
        p.add_(v_part)
        offset += n

    if offset != v.numel():
        raise ValueError(f"Vector size mismatch: {offset} vs {v.numel()}")


# 辅助函数


def _get_batch_size(batch_data: Dict[str, torch.Tensor]) -> int:
    """Infer batch size from batch_data dict.

    Uses _BATCH_DIM_KEYS to identify tensors whose dim-0 is the batch dimension.
    Returns 0 if batch size cannot be determined.
    """
    sizes = set()
    for key in _BATCH_DIM_KEYS:
        if key in batch_data and isinstance(batch_data[key], torch.Tensor) and batch_data[key].dim() > 0:
            sizes.add(batch_data[key].size(0))
    if len(sizes) == 1:
        return sizes.pop()
    if len(sizes) > 1:
        logger.warning(f"Inconsistent batch sizes across keys: {sizes}, using min")
        return min(sizes)
    return 0


def _get_sliceable_keys(batch_data: Dict[str, torch.Tensor], batch_size: int) -> set:
    """Identify keys in batch_data whose dim-0 matches batch_size (safe to slice)."""
    keys = set()
    for k, v in batch_data.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
            keys.add(k)
    return keys


def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    """将张量列表展平为单个向量"""
    if not tensors:
        return torch.empty(0)  # Use empty() for consistent dtype (float32)
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
        result.append(flat[offset : offset + n].view_as(t))
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
    """
    克隆模型参数并移到CPU以减少GPU显存压力 (Fix #3)

    Returns:
        List of cloned parameters on CPU
    """
    return [p.clone().detach().cpu() for p in model.parameters() if p.requires_grad]
