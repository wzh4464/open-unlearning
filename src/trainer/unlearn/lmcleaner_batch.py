"""
LMCleaner Batch-Level Implementation: 批次级在线遗忘

算法特点:
- 初始偏差: δ[tz+1] = -η_tz * gbar[tz] (或直接使用记录的u[tz])
- 存储复杂度: O((N/B) * p) vs 样本级的 O(N * p)
- 前向K步传播: v[s+1] = (I - η_s H[s]) v[s]
- 参数校正: θ̂[τ] = θ[τ] + v

用法:
1. 预训练时使用TrainingLogger记录训练轨迹
2. 加载预训练模型和训练日志
3. 运行遗忘(自动应用参数校正)
4. 评估遗忘效果
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from trainer.unlearn.base import UnlearnTrainer
from .lmcleaner_core import (
    HVPConfig,
    StepLog,
    AuditRecord,
    compute_correction,
    apply_correction,
)
from ..training_logger import TrainingLogger, BatchReconstructor

logger = logging.getLogger(__name__)


class LMCleanerBatchLevel(UnlearnTrainer):
    """
    批次级LMCleaner在线遗忘

    Args:
        training_log_dir: 训练日志目录(TrainingLogger输出)
        K: 截断窗口大小
        hessian_mode: HVP模式 ("GGN", "diag", "exact", "low_rank")
        damping: 阻尼系数λ
        apply_immediately: 是否在初始化时立即应用遗忘(否则在train()时)
        audit_dir: 审计日志输出目录
    """

    def __init__(
        self,
        training_log_dir: str,
        K: int = 800,
        hessian_mode: str = "GGN",
        damping: float = 1e-4,
        apply_immediately: bool = False,
        audit_dir: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.training_log_dir = Path(training_log_dir)
        self.K = K
        self.hessian_mode = hessian_mode
        self.damping = damping
        self.apply_immediately = apply_immediately

        # HVP配置
        self.hvp_config = HVPConfig(
            mode=hessian_mode,
            damping=damping,
            device=str(self.model.device) if hasattr(self.model, 'device') else 'cuda',
            dtype=next(self.model.parameters()).dtype,
        )

        # 审计日志
        self.audit_dir = Path(audit_dir) if audit_dir else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        # 加载训练日志
        self.training_logger = TrainingLogger(log_dir=str(self.training_log_dir))
        self.training_logger.load_from_disk()

        # 创建批次重建器(用于轻存储模式)
        # 注意: dataset和data_collator会在train()方法中设置
        self.batch_reconstructor = None

        # 审计记录列表
        self.audit_records: List[AuditRecord] = []

        # 是否已应用遗忘
        self.unlearning_applied = False

        logger.info(
            f"LMCleanerBatchLevel initialized: K={K}, hessian_mode={hessian_mode}, "
            f"damping={damping}"
        )

        # 立即应用遗忘(如果设置)
        if self.apply_immediately:
            self._apply_unlearning()

    def _get_forget_batch_ids(self) -> List[Any]:
        """
        获取需要遗忘的批次ID列表

        从forget_dataset中提取批次ID
        假设数据集有batch_id属性,或者使用数据索引作为batch_id
        """
        forget_batch_ids = []

        if hasattr(self.train_dataset, 'forget_dataset'):
            forget_ds = self.train_dataset.forget_dataset

            # 如果数据集有batch_id属性
            if hasattr(forget_ds, 'batch_ids'):
                forget_batch_ids = forget_ds.batch_ids
            # 否则使用索引作为batch_id
            else:
                # 获取forget样本的原始索引
                if hasattr(forget_ds, 'original_indices'):
                    forget_batch_ids = forget_ds.original_indices
                else:
                    # 假设batch_id就是数据集索引
                    forget_batch_ids = list(range(len(forget_ds)))

        logger.info(f"Found {len(forget_batch_ids)} forget batches")
        return forget_batch_ids

    def _get_forget_steps(self, forget_batch_ids: List[Any]) -> List[int]:
        """
        获取forget批次对应的训练步骤

        Args:
            forget_batch_ids: 批次ID列表

        Returns:
            步骤ID列表
        """
        forget_steps = []

        for batch_id in forget_batch_ids:
            step_id = self.training_logger.get_batch_step(batch_id)
            if step_id is not None:
                forget_steps.append(step_id)
            else:
                logger.warning(f"Batch {batch_id} not found in training log")

        logger.info(f"Found {len(forget_steps)} forget steps")
        return forget_steps

    def _apply_unlearning(self):
        """
        应用在线遗忘

        对每个forget批次:
        1. 定位其训练步骤tz
        2. 计算参数校正向量v (前向K步传播)
        3. 应用校正: θ ← θ + v
        4. 记录审计信息
        """
        if self.unlearning_applied:
            logger.warning("Unlearning already applied, skipping")
            return

        # 获取forget批次和对应的步骤
        forget_batch_ids = self._get_forget_batch_ids()
        forget_steps = self._get_forget_steps(forget_batch_ids)

        if not forget_steps:
            logger.warning("No forget steps found, skipping unlearning")
            return

        # 初始化批次重建器(如果需要)
        if self.batch_reconstructor is None and hasattr(self, 'train_dataset') and hasattr(self, 'data_collator'):
            # 获取原始训练数据集(用于重建)
            # 如果train_dataset是组合的forget+retain数据集,需要获取原始完整数据集
            reconstruct_dataset = self.train_dataset
            if hasattr(self.train_dataset, 'full_dataset'):
                reconstruct_dataset = self.train_dataset.full_dataset

            self.batch_reconstructor = BatchReconstructor(
                training_logger=self.training_logger,
                dataset=reconstruct_dataset,
                data_collator=self.data_collator,
            )
            logger.info("Initialized BatchReconstructor for batch data reconstruction")

        # 当前步骤(训练结束时的步骤)
        tau = self.training_logger.current_step

        # 获取模型参数列表
        params = [p for p in self.model.parameters() if p.requires_grad]

        logger.info(f"Applying unlearning for {len(forget_steps)} batches")
        logger.info(f"Current step: {tau}, K: {self.K}")

        # 对每个forget步骤计算并应用校正
        for i, tz in enumerate(forget_steps):
            logger.info(f"Processing forget step {i+1}/{len(forget_steps)}: tz={tz}")

            try:
                # 计算参数校正向量
                v, audit = compute_correction(
                    tz=tz,
                    tau=tau,
                    K=self.K,
                    step_log=self.training_logger.step_log,
                    cfg=self.hvp_config,
                    model=self.model,
                    loss_fn=None,  # 使用默认损失
                    batch_reconstructor=self.batch_reconstructor,
                )

                # 应用校正
                apply_correction(v, params)

                # 记录审计信息
                self.audit_records.append(audit)

                logger.info(
                    f"Applied correction for step {tz}: "
                    f"v_norm={audit['v_norm']:.6f}, "
                    f"K_used={audit['K_used']}, "
                    f"hvp_calls={audit['hvp_calls']}"
                )

            except Exception as e:
                logger.error(f"Failed to apply correction for step {tz}: {e}")
                continue

        # 保存审计日志
        if self.audit_dir:
            self._save_audit_records()

        self.unlearning_applied = True
        logger.info("Unlearning applied successfully")

    def _save_audit_records(self):
        """保存审计记录到磁盘"""
        import json

        audit_file = self.audit_dir / "audit_records.json"

        # 转换为可序列化的格式
        serializable_records = [dict(record) for record in self.audit_records]

        with open(audit_file, 'w') as f:
            json.dump(serializable_records, f, indent=2)

        logger.info(f"Saved {len(self.audit_records)} audit records to {audit_file}")

    def compute_loss(self, model, inputs, return_outputs=False):
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

    def train(self):
        """
        重写训练方法

        在训练开始前应用遗忘,然后可选地在retain数据上微调
        """
        # 应用遗忘(如果尚未应用)
        if not self.unlearning_applied:
            self._apply_unlearning()

        # 如果有retain数据且需要微调,调用父类训练
        if hasattr(self.train_dataset, 'retain_dataset'):
            logger.info("Fine-tuning on retain data...")
            return super().train()
        else:
            logger.info("No retain data, skipping training")
            # 返回空的训练结果
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
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 保存遗忘元信息
        import json
        meta = {
            "method": "LMCleanerBatchLevel",
            "K": self.K,
            "hessian_mode": self.hessian_mode,
            "damping": self.damping,
            "num_forget_batches": len(self.audit_records),
            "total_corrections": sum(r["hvp_calls"] for r in self.audit_records),
        }

        meta_file = output_dir / "unlearning_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved unlearned model to {output_dir}")


# 便捷函数

def run_lmcleaner_batch_unlearning(
    model: nn.Module,
    training_log_dir: str,
    forget_batch_ids: List[Any],
    K: int = 800,
    hessian_mode: str = "GGN",
    damping: float = 1e-4,
    output_dir: Optional[str] = None,
) -> nn.Module:
    """
    运行批次级LMCleaner遗忘(独立函数)

    Args:
        model: 预训练模型
        training_log_dir: 训练日志目录
        forget_batch_ids: 需要遗忘的批次ID列表
        K: 截断窗口
        hessian_mode: HVP模式
        damping: 阻尼系数
        output_dir: 输出目录(可选)

    Returns:
        遗忘后的模型
    """
    # 加载训练日志
    training_logger = TrainingLogger(log_dir=training_log_dir)
    training_logger.load_from_disk()

    # 获取forget步骤
    forget_steps = []
    for batch_id in forget_batch_ids:
        step_id = training_logger.get_batch_step(batch_id)
        if step_id is not None:
            forget_steps.append(step_id)

    # HVP配置
    hvp_config = HVPConfig(
        mode=hessian_mode,
        damping=damping,
        device=str(model.device) if hasattr(model, 'device') else 'cuda',
        dtype=next(model.parameters()).dtype,
    )

    # 当前步骤
    tau = training_logger.current_step

    # 参数列表
    params = [p for p in model.parameters() if p.requires_grad]

    # 应用遗忘
    audit_records = []

    for tz in forget_steps:
        v, audit = compute_correction(
            tz=tz,
            tau=tau,
            K=K,
            step_log=training_logger.step_log,
            cfg=hvp_config,
            model=model,
        )

        apply_correction(v, params)
        audit_records.append(audit)

        logger.info(f"Applied correction for step {tz}: v_norm={audit['v_norm']:.6f}")

    # 保存模型(如果指定输出目录)
    if output_dir:
        output_dir = Path(output_dir)
        model.save_pretrained(output_dir)
        logger.info(f"Saved unlearned model to {output_dir}")

    return model
