"""
TIMParameterRollback: 使用训练轨迹进行参数直接回退的 Unlearning 方法

工作流程：
1. 加载训练过程中的参数更新记录
2. 识别 forget 样本在训练中的所有更新步
3. 反向应用这些更新，从参数中"减去"forget 样本的贡献
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import torch
import numpy as np

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class TIMParameterRollback(UnlearnTrainer):
    """
    使用 TIM 训练轨迹进行参数直接回退的 Unlearning 方法
    
    Args:
        tim_output_dir: tim_all_epochs 输出目录，包含训练轨迹数据
        rollback_strength: 回退强度，1.0=完全回退
        trajectory_format: 训练轨迹数据格式，'json' 或 'pickle'
    """
    
    def __init__(
        self,
        tim_output_dir: str,
        rollback_strength: float = 1.0,
        trajectory_format: str = 'json',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.tim_output_dir = Path(tim_output_dir)
        self.rollback_strength = rollback_strength
        self.trajectory_format = trajectory_format
        
        # 加载训练元信息
        self.global_info = self._load_global_info()
        self.num_epochs = self.global_info.get('num_epoch', 0)
        
        # 存储每个样本的参数更新轨迹
        self.sample_updates = {}  # {sample_idx: [{epoch, step, param_name, gradient}, ...]}
        
        logger.info(f"TIM Rollback initialized: {self.num_epochs} epochs")
    
    def _load_global_info(self) -> Dict:
        """加载训练的全局信息"""
        global_info_file = self.tim_output_dir / 'global_info.json'
        
        if not global_info_file.exists():
            logger.warning(f"Global info file not found at {global_info_file}")
            return {'num_epoch': 0, 'num_steps': 0}
        
        with open(global_info_file, 'r') as f:
            return json.load(f)
    
    def _load_epoch_data(self, epoch: int) -> Dict:
        """加载指定 epoch 的训练数据"""
        if self.trajectory_format == 'json':
            epoch_file = self.tim_output_dir / f'epoch_{epoch}.json'
            if epoch_file.exists():
                with open(epoch_file, 'r') as f:
                    return json.load(f)
        elif self.trajectory_format == 'pickle':
            epoch_file = self.tim_output_dir / f'epoch_{epoch}.pkl'
            if epoch_file.exists():
                with open(epoch_file, 'rb') as f:
                    return pickle.load(f)
        
        logger.warning(f"Epoch {epoch} data not found")
        return {}
    
    def _get_sample_gradients(self, epoch_data: Dict, sample_idx: int) -> Optional[Dict]:
        """从 epoch 数据中提取指定样本的梯度信息"""
        if 'sample_gradients' not in epoch_data or sample_idx not in epoch_data['sample_gradients']:
            return None
        
        return epoch_data['sample_gradients'][sample_idx]
    
    def _collect_forget_trajectory(self, forget_indices: List[int]) -> List[Dict]:
        """
        收集 forget 样本在训练中的所有更新记录
        
        Returns:
            参数更新轨迹列表，格式: [{epoch, step, param_name, gradient}, ...]
        """
        trajectory = []
        
        logger.info(f"Collecting trajectory for {len(forget_indices)} forget samples")
        
        # 遍历所有 epoch
        for epoch in range(self.num_epochs):
            epoch_data = self._load_epoch_data(epoch)
            
            if not epoch_data:
                continue
            
            # 遍历每个 forget 样本
            for sample_idx in forget_indices:
                sample_grads = self._get_sample_gradients(epoch_data, sample_idx)
                
                if sample_grads is None:
                    continue
                
                # 提取梯度信息
                for param_name, gradient_data in sample_grads.items():
                    trajectory.append({
                        'epoch': epoch,
                        'sample_idx': sample_idx,
                        'param_name': param_name,
                        'gradient': gradient_data
                    })
        
        logger.info(f"Collected {len(trajectory)} parameter updates")
        return trajectory
    
    def _apply_gradient_to_params(self, param_name: str, gradient: torch.Tensor) -> None:
        """将梯度应用到模型参数"""
        # 获取模型参数
        param = next(
            (p for name, p in self.model.named_parameters() if name == param_name),
            None
        )
        
        if param is None:
            logger.warning(f"Parameter {param_name} not found in model")
            return
        
        # 确保梯度在正确的设备和数据类型
        gradient = torch.tensor(gradient, dtype=param.dtype, device=param.device)
        
        # 反向应用梯度（减去而不是加上）
        with torch.no_grad():
            # 计算更新：delta = -lr * gradient * rollback_strength
            # 这里假设使用学习率（需要从训练记录中获取）
            # 简化版本：直接减去梯度
            param.data -= self.rollback_strength * gradient
    
    def _apply_rollback(self, trajectory: List[Dict]) -> None:
        """
        将参数回退
        
        Args:
            trajectory: 参数更新轨迹列表
        """
        if not trajectory:
            logger.warning("No trajectory to apply")
            return
        
        logger.info(f"Applying rollback for {len(trajectory)} parameter updates")
        
        # 按 epoch 和 step 排序，确保按时间顺序回退
        trajectory_sorted = sorted(trajectory, key=lambda x: (x['epoch'], x.get('step', 0)), reverse=True)
        
        for update in trajectory_sorted:
            param_name = update['param_name']
            gradient = update['gradient']
            
            self._apply_gradient_to_params(param_name, gradient)
        
        logger.info("Parameter rollback completed")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写损失计算
        对于 rollback 方法，我们不需要额外的训练损失
        只需要返回一个小的损失以保持训练流程
        """
        # 如果有 retain 数据，计算 retain loss
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
        在训练开始前执行参数回退
        """
        # 从训练数据中提取 forget indices
        forget_indices = self._get_forget_indices()
        
        if not forget_indices:
            logger.warning("No forget indices found, skipping rollback")
        else:
            # Step 1: 收集 forget 样本的所有训练轨迹
            forget_trajectory = self._collect_forget_trajectory(forget_indices)
            
            # Step 2: 反向应用参数更新
            self._apply_rollback(forget_trajectory)
        
        # 调用父类的训练方法
        return super().train()
    
    def _get_forget_indices(self) -> List[int]:
        """
        从训练数据集中提取 forget 样本的索引
        """
        forget_indices = []
        
        if hasattr(self.train_dataset, 'forget_dataset'):
            # 如果是 ForgetRetainDataset，获取 forget 部分的索引
            forget_ds = self.train_dataset.forget_dataset
            
            # 如果有原始索引信息，使用原始索引，否则使用连续索引
            forget_indices = list(range(len(forget_ds))) if not hasattr(forget_ds, 'original_indices') else forget_ds.original_indices
        
        logger.info(f"Found {len(forget_indices)} forget samples")
        return forget_indices


# 辅助函数：用于记录训练过程中的梯度
def save_gradient_trajectory(
    output_dir: Path,
    epoch: int,
    step: int,
    sample_idx: int,
    param_name: str,
    gradient: torch.Tensor,
    global_info: Optional[Dict] = None
):
    """
    保存单个样本在某个步骤的梯度信息
    
    Args:
        output_dir: 输出目录
        epoch: epoch 编号
        step: step 编号
        sample_idx: 样本索引
        param_name: 参数名称
        gradient: 梯度张量
        global_info: 全局信息（可选）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存全局信息
    if global_info is not None:
        global_info_file = output_dir / 'global_info.json'
        with open(global_info_file, 'w') as f:
            json.dump(global_info, f)
    
    # 加载或创建 epoch 数据
    epoch_file = output_dir / f'epoch_{epoch}.json'
    epoch_data = json.load(epoch_file.open('r')) if epoch_file.exists() else {'sample_gradients': {}}
    
    # 添加梯度信息
    if 'sample_gradients' not in epoch_data:
        epoch_data['sample_gradients'] = {}
    
    if sample_idx not in epoch_data['sample_gradients']:
        epoch_data['sample_gradients'][sample_idx] = {}
    
    # 将梯度转换为 numpy 数组保存
    gradient_np = gradient.detach().cpu().numpy().tolist()
    epoch_data['sample_gradients'][sample_idx][param_name] = gradient_np
    epoch_data['sample_gradients'][sample_idx][f'{param_name}_step'] = step
    
    # 保存 epoch 数据
    with open(epoch_file, 'w') as f:
        json.dump(epoch_data, f)
