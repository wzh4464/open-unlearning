"""
LMCleaner Sample-Level Implementation: 样本级在线遗忘

论文: LMCleaner: Efficient and Certified Online Unlearning via Truncated Influence Propagation

算法特点 (论文 Eq. 3 样本级公式):
- 初始偏差: δ[t_{z_j}+1] = -(η_{t_{z_j}} / |S_{t_{z_j}}|) * ∇θℓ(z_j; θ[t_{z_j}])
- 存储复杂度: O(N * p) (N为样本数)
- 前向K步传播: v[s+1] = (I - η_s H[s]) v[s]
- 参数校正: θ̂[τ] = θ[τ] + v
- 隐私保护: θ̃[τ] = θ̂[τ] + N(0, σ²I) (Phase 4, 可选)

相比批次级:
- 更精确(样本级梯度)
- 存储开销更大
- 适用于需要精细控制的场景

用法:
1. 预训练时使用TrainingLogger记录每个样本的梯度
2. 加载预训练模型和训练日志
3. 运行遗忘(自动应用参数校正)
4. 评估遗忘效果

实现说明:
- hessian_mode="fisher": 论文 Algorithm 1 使用的方法 Hv = g·(g^T v)
- hessian_mode="GGN": 使用二阶自动微分，更精确但计算量更大
- epsilon > 0: 启用 Phase 4 隐私噪声注入
- damping=0: 与论文一致 (damping>0 是额外的数值稳定化)
"""

import logging
from pathlib import Path
from typing import List, Optional, Any

import torch

from trainer.unlearn.base import UnlearnTrainer
from .lmcleaner_core import (
    HVPConfig,
    AuditRecord,
    apply_correction,
    compute_noise_sigma,
    inject_privacy_noise,
)
from ..training_logger import TrainingLogger, BatchReconstructor

logger = logging.getLogger(__name__)


class LMCleanerSampleLevel(UnlearnTrainer):
    """
    样本级LMCleaner在线遗忘

    实现论文 Algorithm 1 的完整四阶段算法 (样本级版本):
    - Phase 1: 样本级初始偏差计算 (论文 Eq. 3)
    - Phase 2: 截断影响传播 (K步)
    - Phase 3: 影响移除
    - Phase 4: 隐私噪声注入 (可选, 实现 certified unlearning)

    Args:
        training_log_dir: 训练日志目录(TrainingLogger输出)
        K: 截断窗口大小
        hessian_mode: HVP模式
            - "fisher": 论文 Algorithm 1 使用的 Fisher 近似 Hv = g·(g^T v)
            - "GGN": 广义 Gauss-Newton 近似 (更精确但计算量更大)
            - "diag": 对角 Hessian 近似
            - "exact": 精确 HVP (非常慢)
        damping: 阻尼系数λ (论文未包含，默认0以与论文一致)
        batch_size_at_training: 训练时的批次大小(用于计算样本级初始偏差)
        apply_immediately: 是否在初始化时立即应用遗忘
        audit_dir: 审计日志输出目录
        epsilon: 隐私参数 ε (>0 时启用 Phase 4 噪声注入)
        delta: 隐私参数 δ (默认 1e-5)
    """

    def __init__(
        self,
        training_log_dir: str,
        K: int = 800,
        hessian_mode: str = "fisher",  # 默认使用论文的 Fisher 近似
        damping: float = 0.0,  # 默认不添加阻尼，与论文一致
        batch_size_at_training: int = 1,
        apply_immediately: bool = False,
        audit_dir: Optional[str] = None,
        # Phase 4: 隐私噪声参数
        epsilon: float = 0.0,  # 默认不注入噪声
        delta: float = 1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.training_log_dir = Path(training_log_dir)
        self.K = K
        self.hessian_mode = hessian_mode
        self.damping = damping
        self.batch_size_at_training = batch_size_at_training
        self.apply_immediately = apply_immediately

        # Phase 4: 隐私参数
        self.epsilon = epsilon
        self.delta = delta

        # HVP配置
        self.hvp_config = HVPConfig(
            mode=hessian_mode,
            damping=damping,
            device=str(self.model.device) if hasattr(self.model, "device") else "cuda",
            dtype=next(self.model.parameters()).dtype,
        )

        # 审计日志
        self.audit_dir = Path(audit_dir) if audit_dir else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        # 加载训练日志 (需要加载tensor数据用于遗忘计算)
        self.training_logger = TrainingLogger(
            log_dir=str(self.training_log_dir), mode="sample"
        )
        self.training_logger.load_from_disk(load_tensors=True)

        # 创建批次重建器(用于轻存储模式)
        self.batch_reconstructor = None

        # 审计记录列表
        self.audit_records: List[AuditRecord] = []

        # 是否已应用遗忘
        self.unlearning_applied = False

        logger.info(
            f"LMCleanerSampleLevel initialized: K={K}, hessian_mode={hessian_mode}, "
            f"damping={damping}, batch_size={batch_size_at_training}, "
            f"epsilon={epsilon}, delta={delta}"
        )

        # 立即应用遗忘(如果设置)
        if self.apply_immediately:
            self._apply_unlearning()

    def _get_forget_sample_ids(self) -> List[Any]:
        """
        获取需要遗忘的样本ID列表

        从forget_dataset中提取样本ID
        """
        forget_sample_ids = []

        if hasattr(self.train_dataset, "forget_dataset"):
            forget_ds = self.train_dataset.forget_dataset

            # 如果数据集有sample_id属性
            if hasattr(forget_ds, "sample_ids"):
                forget_sample_ids = forget_ds.sample_ids
            # 否则使用索引作为sample_id
            elif hasattr(forget_ds, "original_indices"):
                forget_sample_ids = forget_ds.original_indices
            else:
                forget_sample_ids = list(range(len(forget_ds)))

        logger.info(f"Found {len(forget_sample_ids)} forget samples")
        return forget_sample_ids

    def _get_forget_steps(self, forget_sample_ids: List[Any]) -> List[int]:
        """
        获取forget样本对应的训练步骤

        Args:
            forget_sample_ids: 样本ID列表

        Returns:
            步骤ID列表
        """
        forget_steps = []

        for sample_id in forget_sample_ids:
            # 样本级记录中,sample_id被用作batch_id
            step_id = self.training_logger.get_batch_step(sample_id)
            if step_id is not None:
                forget_steps.append(step_id)
            else:
                logger.warning(f"Sample {sample_id} not found in training log")

        logger.info(f"Found {len(forget_steps)} forget steps")
        return forget_steps

    def _compute_sample_correction(
        self,
        tz: int,
        tau: int,
    ) -> tuple[torch.Tensor, AuditRecord]:
        """
        计算单个样本的参数校正向量 (论文 Eq. 3)

        样本级的初始偏差 (Phase 1):
        δ[t_{z_j}+1] = -(η_{t_{z_j}} / |S_{t_{z_j}}|) * ∇θℓ(z_j; θ[t_{z_j}])

        其中 |S_{t_{z_j}}| 是训练时的批次大小

        Args:
            tz: 样本出现的步骤
            tau: 当前步骤

        Returns:
            (v, audit): 校正向量和审计记录
        """
        # 获取步骤记录
        rec = self.training_logger.step_log[tz]
        if rec is None:
            raise ValueError(f"Step {tz} not found in training log")

        # 样本级的gbar实际上是单个样本的梯度
        if rec.gbar is None:
            raise ValueError(f"Step {tz} missing gradient")

        # Phase 1: 初始偏差 v0 = -(η_tz / B) * ∇θℓ(zj; θ[tz])
        # 由于我们记录的gbar已经是单个样本的梯度,需要除以批次大小
        v = -(rec.eta / self.batch_size_at_training) * rec.gbar.clone()

        # Phase 2: 前向传播
        start = tz + 1
        end = min(tz + self.K, tau - 1)
        K_used = max(0, end - start + 1)

        hvp_calls = 0

        for s in range(start, end + 1):
            srec = self.training_logger.step_log[s]
            if srec is None:
                logger.warning(f"Step {s} not found, skipping")
                continue

            # 计算 H[s] @ v using hvp_apply
            # 根据 hessian_mode 自动选择 Fisher 或 GGN 近似
            from .lmcleaner_core import hvp_apply

            with torch.enable_grad():
                hvp = hvp_apply(
                    v, srec, self.hvp_config, self.model, None, self.batch_reconstructor
                )

            # v ← v - η[s] * hvp
            v = v - srec.eta * hvp

            # 添加阻尼: v ← v - η[s] * λ * v (实现添加，论文未包含)
            if self.damping > 0:
                v = v - srec.eta * self.damping * v

            hvp_calls += 1

        # 记录 Phase 3 前的 v_norm
        v_norm_before_noise = float(v.norm().item())

        # Phase 4: 隐私噪声注入 (论文 Algorithm 1)
        noise_sigma = 0.0
        noise_injected = False

        if self.epsilon > 0:
            # 使用 v_norm 作为 delta_det 的保守估计
            delta_det = v_norm_before_noise
            noise_sigma = compute_noise_sigma(delta_det, self.epsilon, self.delta)
            v = inject_privacy_noise(v, noise_sigma)
            noise_injected = True
            logger.debug(
                f"  Phase 4: Injected privacy noise σ={noise_sigma:.6f} "
                f"(ε={self.epsilon}, δ={self.delta})"
            )

        # 创建审计记录
        audit = AuditRecord(
            tz=tz,
            tau=tau,
            K_used=K_used,
            v_norm=v_norm_before_noise,
            hvp_calls=hvp_calls,
            mode=self.hessian_mode,
            damping=self.damping,
            noise_sigma=noise_sigma,
            noise_injected=noise_injected,
            epsilon=self.epsilon if noise_injected else 0.0,
            delta=self.delta if noise_injected else 0.0,
        )

        return v, audit

    def _apply_unlearning(self):
        """
        应用样本级在线遗忘

        对每个forget样本:
        1. 定位其训练步骤tz
        2. 计算参数校正向量v (考虑样本级初始偏差)
        3. 应用校正: θ ← θ + v
        4. 记录审计信息
        """
        if self.unlearning_applied:
            logger.warning("Unlearning already applied, skipping")
            return

        # 获取forget样本和对应的步骤
        forget_sample_ids = self._get_forget_sample_ids()
        forget_steps = self._get_forget_steps(forget_sample_ids)

        if not forget_steps:
            logger.warning("No forget steps found, skipping unlearning")
            return

        # 初始化批次重建器(如果需要)
        if (
            self.batch_reconstructor is None
            and hasattr(self, "train_dataset")
            and hasattr(self, "data_collator")
        ):
            reconstruct_dataset = self.train_dataset
            if hasattr(self.train_dataset, "full_dataset"):
                reconstruct_dataset = self.train_dataset.full_dataset

            self.batch_reconstructor = BatchReconstructor(
                training_logger=self.training_logger,
                dataset=reconstruct_dataset,
                data_collator=self.data_collator,
            )
            logger.info("Initialized BatchReconstructor for batch data reconstruction")

        # 当前步骤
        tau = self.training_logger.current_step

        # 获取模型参数列表
        params = [p for p in self.model.parameters() if p.requires_grad]

        logger.info(f"Applying sample-level unlearning for {len(forget_steps)} samples")
        logger.info(f"Current step: {tau}, K: {self.K}")

        # 对每个forget样本计算并应用校正
        for i, tz in enumerate(forget_steps):
            logger.info(
                f"Processing forget sample {i + 1}/{len(forget_steps)}: tz={tz}"
            )

            try:
                # 计算参数校正向量(样本级)
                v, audit = self._compute_sample_correction(tz, tau)

                # 应用校正
                apply_correction(v, params)

                # 记录审计信息
                self.audit_records.append(audit)

                logger.info(
                    f"Applied correction for sample at step {tz}: "
                    f"v_norm={audit['v_norm']:.6f}, "
                    f"K_used={audit['K_used']}"
                )

            except Exception as e:
                logger.error(f"Failed to apply correction for step {tz}: {e}")
                continue

        # 保存审计日志
        if self.audit_dir:
            self._save_audit_records()

        self.unlearning_applied = True
        logger.info("Sample-level unlearning applied successfully")

    def _save_audit_records(self):
        """保存审计记录到磁盘"""
        import json

        audit_file = self.audit_dir / "audit_records.json"

        # 转换为可序列化的格式
        serializable_records = [dict(record) for record in self.audit_records]

        with open(audit_file, "w") as f:
            json.dump(serializable_records, f, indent=2)

        logger.info(f"Saved {len(self.audit_records)} audit records to {audit_file}")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        重写损失计算

        对于LMCleaner,主要是通过参数校正实现遗忘,
        训练阶段可以选择性地在retain数据上微调
        """
        # 如果有retain数据,计算retain loss
        if "retain" in inputs:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_outputs = model(**retain_inputs)
            loss = retain_outputs.loss
        else:
            # 返回一个很小的损失
            loss = torch.tensor(0.0, device=next(model.parameters()).device)

        return (loss, None) if return_outputs else loss

    def train(self, resume_from_checkpoint=None):
        """
        重写训练方法

        在训练开始前应用遗忘,然后可选地在retain数据上微调

        Args:
            resume_from_checkpoint: 兼容父类接口，LMCleaner 不使用此参数
        """
        # 应用遗忘(如果尚未应用)
        if not self.unlearning_applied:
            self._apply_unlearning()

        # 如果有retain数据且需要微调,调用父类训练
        if hasattr(self.train_dataset, "retain_dataset"):
            logger.info("Fine-tuning on retain data...")
            return super().train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            logger.info("No retain data, skipping training")
            return None

    def save_model(self, output_dir: str):
        """
        保存遗忘后的模型

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 确保已应用遗忘
        if not self.unlearning_applied:
            self._apply_unlearning()

        # 保存模型
        self.model.save_pretrained(output_dir)

        # 保存tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 保存遗忘元信息
        import json

        meta = {
            "method": "LMCleanerSampleLevel",
            "K": self.K,
            "hessian_mode": self.hessian_mode,
            "damping": self.damping,
            "batch_size_at_training": self.batch_size_at_training,
            "num_forget_samples": len(self.audit_records),
            "total_corrections": len(self.audit_records),
        }

        meta_file = output_dir / "unlearning_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved unlearned model to {output_dir}")
