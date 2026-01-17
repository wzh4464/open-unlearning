"""
TrainingLogger: 训练过程日志记录模块

用于LMCleaner在线遗忘算法,记录训练过程中的关键信息:
- 每步的学习率η[t]
- 参数更新向量u[t] = θ[t+1] - θ[t]
- 平均梯度gbar[t]
- 批次数据(用于HVP计算)

支持两种模式:
1. 样本级(sample-level): 记录每个样本的梯度
2. 批次级(batch-level): 记录每个批次的平均梯度/更新向量

支持三种存储模式:
1. 轻存储: 只保存索引+随机种子(save_indices_only=True, save_rng_state=True)
2. 重存储: 保存完整batch张量(save_batch_data=True)
3. 对角Hessian: 保存对角Hessian近似(compute_diag_h=True)

支持epoch感知的检查点保存:
- steps_per_epoch: 每个epoch的步数
- checkpoints_per_epoch: 每个epoch保存的中间检查点数量
- save_at_epoch_end: 是否在epoch结束时保存模型参数
"""

import gc
import json
import logging
import pickle
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, Any, List
import weakref

import torch
import torch.nn as nn

from .unlearn.lmcleaner_core import StepLog, StepRecord, clone_parameters

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    训练日志记录器

    记录训练过程中的参数更新轨迹,用于后续的在线遗忘

    Args:
        log_dir: 日志保存目录
        max_steps: 最大保留步数(环形缓冲区大小)
        mode: "batch" 或 "sample"
        save_interval: 保存到磁盘的间隔步数(0表示不保存,会被epoch感知参数覆盖)
        save_batch_data: 是否保存批次数据(用于HVP,会占用较大空间)
        save_indices_only: 是否只保存样本索引(轻存储模式)
        save_rng_state: 是否保存随机数生成器状态(用于批次重建)
        compute_diag_h: 是否计算并保存对角Hessian近似
        batch_size_at_training: 训练时的批次大小(用于重建批次)
        steps_per_epoch: 每个epoch的步数(用于epoch感知保存)
        checkpoints_per_epoch: 每个epoch保存的中间检查点数量(不包括epoch结束时的检查点)
        save_at_epoch_end: 是否在epoch结束时保存模型参数检查点
    """

    def __init__(
        self,
        log_dir: str,
        max_steps: int = 1000,
        mode: str = "batch",
        save_interval: int = 100,
        save_batch_data: bool = False,
        save_indices_only: bool = False,
        save_rng_state: bool = False,
        compute_diag_h: bool = False,
        batch_size_at_training: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        checkpoints_per_epoch: int = 0,
        save_at_epoch_end: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = max_steps
        self.mode = mode
        self.save_batch_data = save_batch_data
        self.save_indices_only = save_indices_only
        self.save_rng_state = save_rng_state
        self.compute_diag_h = compute_diag_h
        self.batch_size_at_training = batch_size_at_training

        # Epoch-aware checkpoint parameters
        self.steps_per_epoch = steps_per_epoch
        self.checkpoints_per_epoch = checkpoints_per_epoch
        self.save_at_epoch_end = save_at_epoch_end

        # Calculate save_interval based on epoch-aware parameters
        if steps_per_epoch is not None and checkpoints_per_epoch > 0:
            # Ensure equal spacing of checkpoints within each epoch
            self.save_interval = steps_per_epoch // checkpoints_per_epoch
            if self.save_interval == 0:
                self.save_interval = 1
            logger.info(
                f"Epoch-aware saving: {checkpoints_per_epoch} checkpoints per epoch, "
                f"save_interval={self.save_interval} (steps_per_epoch={steps_per_epoch})"
            )
        else:
            self.save_interval = save_interval

        # Epoch tracking
        self.current_epoch = 0
        self.epoch_end_steps: List[int] = []  # Steps at which each epoch ends
        self.model_checkpoints_dir = self.log_dir / "model_checkpoints"
        if save_at_epoch_end:
            self.model_checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # 创建步骤日志
        self.step_log = StepLog(max_size=max_steps)

        # 批次索引: batch_id -> step_id
        self.batch_index = {}

        # 样本索引: step_id -> sample_indices (轻存储模式)
        self.sample_indices_per_step: Dict[int, List[int]] = {}

        # RNG状态: step_id -> rng_state (用于重建批次)
        self.rng_states_per_step: Dict[int, Dict] = {}

        # 当前步数
        self.current_step = 0

        # 上一步的参数(用于计算u[t])
        self.prev_params: Optional[List[torch.Tensor]] = None

        # 增量保存: 跟踪已保存的最后一个 step_id
        self._last_saved_step_id = -1

        # 异步写入
        self._write_queue: Queue = Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        self._async_write = True  # 默认启用异步写入

        logger.info(
            f"TrainingLogger initialized: mode={mode}, max_steps={max_steps}, "
            f"save_indices_only={save_indices_only}, save_rng_state={save_rng_state}, "
            f"compute_diag_h={compute_diag_h}, steps_per_epoch={steps_per_epoch}, "
            f"checkpoints_per_epoch={checkpoints_per_epoch}, save_at_epoch_end={save_at_epoch_end}"
        )

    def _writer_loop(self):
        """后台写入线程的主循环"""
        while not self._stop_writer.is_set():
            try:
                task = self._write_queue.get(timeout=0.5)
                if task is None:  # 停止信号
                    break
                file_path, data = task
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
                self._write_queue.task_done()
                logger.debug(f"Async write completed: {file_path}")
            except Exception:
                continue  # 超时或其他错误，继续检查停止标志

    def start_async_writer(self):
        """启动后台写入线程"""
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._stop_writer.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True
            )
            self._writer_thread.start()
            logger.debug("Async writer thread started")

    def stop_async_writer(self, wait: bool = True):
        """停止后台写入线程"""
        self._stop_writer.set()
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._write_queue.put(None)  # 发送停止信号
            if wait:
                self._writer_thread.join(timeout=30)
            logger.debug("Async writer thread stopped")

    def _prune_old_entries(self):
        """
        Prune old entries from dictionaries to prevent unbounded memory growth (Fix #2)

        Keep only the most recent entries up to save_interval to prevent memory accumulation
        between saves. Since we clear these dicts after each save, we only need to limit
        growth within a save interval.
        """
        if self.save_interval <= 0:
            return

        # Limit to 2x save_interval as a safety buffer
        max_entries = max(self.save_interval * 2, 100)

        # Prune sample_indices_per_step
        if len(self.sample_indices_per_step) > max_entries:
            # Keep only the most recent entries
            sorted_keys = sorted(self.sample_indices_per_step.keys())
            keys_to_remove = sorted_keys[:-max_entries]
            for key in keys_to_remove:
                del self.sample_indices_per_step[key]
            logger.debug(
                f"Pruned {len(keys_to_remove)} old entries from sample_indices_per_step"
            )

        # Prune rng_states_per_step
        if len(self.rng_states_per_step) > max_entries:
            # Keep only the most recent entries
            sorted_keys = sorted(self.rng_states_per_step.keys())
            keys_to_remove = sorted_keys[:-max_entries]
            for key in keys_to_remove:
                del self.rng_states_per_step[key]
            logger.debug(
                f"Pruned {len(keys_to_remove)} old entries from rng_states_per_step"
            )

    def register_step(
        self,
        step_id: int,
        batch_id: Any,
        eta: float,
        model: Optional[nn.Module] = None,
        u: Optional[torch.Tensor] = None,
        gbar: Optional[torch.Tensor] = None,
        batch_data: Optional[Dict[str, torch.Tensor]] = None,
        diag_H: Optional[torch.Tensor] = None,
        sample_indices: Optional[List[int]] = None,
    ):
        """
        注册一个训练步骤

        Args:
            step_id: 步骤ID
            batch_id: 批次ID
            eta: 学习率
            model: 模型(用于获取当前参数)
            u: 参数更新向量(如果提供,优先使用)
            gbar: 平均梯度(备选)
            batch_data: 批次数据(可选,用于HVP)
            diag_H: 对角Hessian(可选)
            sample_indices: 样本索引列表(轻存储模式)
        """
        # 计算参数更新向量u[t] = θ[t+1] - θ[t]
        # Note: Skip parameter cloning with DeepSpeed ZeRO-3 as parameters are sharded
        if u is None and model is not None and self.prev_params is not None:
            try:
                current_params = clone_parameters(model)
                # Check if parameters are valid (not empty due to ZeRO-3 sharding)
                if current_params and len(current_params) == len(self.prev_params):
                    u = torch.cat(
                        [
                            (new - old).view(-1)
                            for new, old in zip(current_params, self.prev_params)
                            if new.numel() > 0 and old.numel() > 0
                        ]
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to compute parameter update vector: {e}. Skipping u computation."
                )
                u = None

        # 保存样本索引(轻存储模式)
        if self.save_indices_only and sample_indices is not None:
            self.sample_indices_per_step[step_id] = sample_indices

        # 保存RNG状态(用于批次重建)
        if self.save_rng_state:
            self.rng_states_per_step[step_id] = {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            }

        # Prune old entries to prevent unbounded growth (Fix #2)
        self._prune_old_entries()

        # 保存批次数据(可选)
        if batch_data is not None and self.save_batch_data:
            # 深拷贝批次数据,避免被后续修改
            batch_data = {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }
        else:
            batch_data = None

        # 创建步骤记录
        record = StepRecord(
            step_id=step_id,
            eta=eta,
            batch_id=batch_id,
            u=u.detach() if u is not None else None,
            gbar=gbar.detach() if gbar is not None else None,
            theta_ref=weakref.ref(model) if model is not None else None,
            batch_data=batch_data,
            diag_H=diag_H,
        )

        # 添加到日志
        self.step_log.add(record)

        # 更新批次索引
        self.batch_index[batch_id] = step_id

        # 更新当前步数
        self.current_step = step_id

        # 保存当前参数用于下一步
        # Note: Skip with DeepSpeed ZeRO-3 as parameters are sharded
        if model is not None:
            try:
                self.prev_params = clone_parameters(model)
            except Exception as e:
                logger.debug(
                    f"Failed to clone parameters: {e}. This is expected with DeepSpeed ZeRO-3."
                )

        # 定期保存到磁盘
        if self.save_interval > 0 and step_id % self.save_interval == 0:
            self.save_to_disk(step_id)

        logger.debug(f"Registered step {step_id}, batch_id={batch_id}, eta={eta}")

    def register_batch_gradient(
        self,
        step_id: int,
        batch_id: Any,
        eta: float,
        gradients: List[torch.Tensor],
    ):
        """
        注册批次级的平均梯度

        Args:
            step_id: 步骤ID
            batch_id: 批次ID
            eta: 学习率
            gradients: 梯度列表
        """
        # 展平梯度
        gbar = torch.cat([g.view(-1) for g in gradients])

        self.register_step(
            step_id=step_id,
            batch_id=batch_id,
            eta=eta,
            gbar=gbar,
        )

    def register_sample_gradient(
        self,
        step_id: int,
        sample_id: Any,
        eta: float,
        gradients: List[torch.Tensor],
    ):
        """
        注册样本级的梯度

        Args:
            step_id: 步骤ID
            sample_id: 样本ID
            eta: 学习率
            gradients: 梯度列表
        """
        # 样本级记录使用sample_id作为batch_id
        self.register_batch_gradient(
            step_id=step_id,
            batch_id=sample_id,
            eta=eta,
            gradients=gradients,
        )

    def get_step_record(self, step_id: int) -> Optional[StepRecord]:
        """获取步骤记录"""
        return self.step_log.get(step_id)

    def get_batch_step(self, batch_id: Any) -> Optional[int]:
        """获取批次对应的步骤ID"""
        return self.batch_index.get(batch_id)

    def save_model_checkpoint(self, model: nn.Module, epoch: int, step_id: int):
        """
        保存模型参数检查点

        Args:
            model: 模型
            epoch: 当前epoch
            step_id: 当前步骤ID
        """
        checkpoint_dir = self.model_checkpoints_dir / f"epoch_{epoch}_step_{step_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        state_dict = model.state_dict()
        checkpoint_file = checkpoint_dir / "model_state_dict.pt"
        torch.save(state_dict, checkpoint_file)

        # Save checkpoint metadata
        checkpoint_meta = {
            "epoch": epoch,
            "step_id": step_id,
            "is_epoch_end": step_id in self.epoch_end_steps,
        }
        meta_file = checkpoint_dir / "checkpoint_meta.json"
        with open(meta_file, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)

        logger.info(
            f"Saved model checkpoint at epoch {epoch}, step {step_id} to {checkpoint_dir}"
        )

    def on_epoch_end(self, epoch: int, step_id: int, model: Optional[nn.Module] = None):
        """
        Epoch结束时调用

        Args:
            epoch: 当前epoch (0-indexed)
            step_id: 当前步骤ID
            model: 模型(用于保存检查点)
        """
        self.current_epoch = epoch + 1  # Update to next epoch
        self.epoch_end_steps.append(step_id)

        # Save training log at epoch end
        self.save_to_disk(step_id)

        # Save model checkpoint if enabled
        if self.save_at_epoch_end and model is not None:
            self.save_model_checkpoint(model, epoch, step_id)

        logger.info(f"Epoch {epoch} ended at step {step_id}")

    def get_epoch_checkpoints(self) -> Dict[int, List[int]]:
        """
        获取每个epoch的检查点步骤列表

        Returns:
            Dict[epoch_id, List[step_id]]: 每个epoch的检查点步骤
        """
        if self.steps_per_epoch is None:
            return {}

        checkpoints_by_epoch: Dict[int, List[int]] = {}
        for step_id in range(1, self.current_step + 1):
            epoch = (step_id - 1) // self.steps_per_epoch
            if epoch not in checkpoints_by_epoch:
                checkpoints_by_epoch[epoch] = []

            # Check if this step is a checkpoint
            is_intermediate = (
                self.save_interval > 0 and step_id % self.save_interval == 0
            )
            is_epoch_end = step_id in self.epoch_end_steps

            if is_intermediate or is_epoch_end:
                checkpoints_by_epoch[epoch].append(step_id)

        return checkpoints_by_epoch

    def save_to_disk(self, step_id: Optional[int] = None):
        """
        保存日志到磁盘（增量保存 + 异步写入）

        Args:
            step_id: 当前步骤ID(用于文件名)
        """
        if step_id is None:
            step_id = self.current_step

        # 保存元信息（包含增量保存标记）
        meta = {
            "max_steps": self.max_steps,
            "mode": self.mode,
            "current_step": self.current_step,
            "num_records": len(self.step_log.buffer),
            "save_batch_data": self.save_batch_data,
            "save_indices_only": self.save_indices_only,
            "save_rng_state": self.save_rng_state,
            "compute_diag_h": self.compute_diag_h,
            "batch_size_at_training": self.batch_size_at_training,
            "steps_per_epoch": self.steps_per_epoch,
            "checkpoints_per_epoch": self.checkpoints_per_epoch,
            "save_at_epoch_end": self.save_at_epoch_end,
            "current_epoch": self.current_epoch,
            "epoch_end_steps": self.epoch_end_steps,
            "incremental_save": True,  # 标记使用增量保存格式
        }

        meta_file = self.log_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        # 保存批次索引
        index_file = self.log_dir / "batch_index.json"
        with open(index_file, "w") as f:
            json.dump(self.batch_index, f, indent=2)

        # 保存样本索引(轻存储模式)
        if self.save_indices_only and self.sample_indices_per_step:
            indices_file = self.log_dir / "sample_indices.json"
            # Convert int keys to strings for JSON serialization
            serializable_indices = {
                str(k): v for k, v in self.sample_indices_per_step.items()
            }
            with open(indices_file, "w") as f:
                json.dump(serializable_indices, f, indent=2)

        # 保存RNG状态(用于批次重建)
        if self.save_rng_state and self.rng_states_per_step:
            rng_file = self.log_dir / f"rng_states_{step_id}.pkl"
            with open(rng_file, "wb") as f:
                pickle.dump(self.rng_states_per_step, f)

        # 增量保存: 只保存新增的记录
        new_records = [
            rec
            for rec in self.step_log.buffer
            if rec.step_id > self._last_saved_step_id
        ]

        if not new_records:
            logger.debug(f"No new records to save at step {step_id}")
            return

        # 准备可序列化的记录
        serializable_records = []
        for rec in new_records:
            rec_dict = {
                "step_id": rec.step_id,
                "eta": rec.eta,
                "batch_id": rec.batch_id,
                "u": rec.u.cpu() if rec.u is not None else None,
                "gbar": rec.gbar.cpu() if rec.gbar is not None else None,
                "diag_H": rec.diag_H.cpu() if rec.diag_H is not None else None,
            }
            serializable_records.append(rec_dict)

        # 使用 chunk 文件名（增量保存）
        chunk_file = self.log_dir / f"step_records_chunk_{step_id}.pkl"

        # 异步写入或同步写入
        if self._async_write:
            self.start_async_writer()
            self._write_queue.put((chunk_file, serializable_records))
            logger.info(
                f"Queued {len(new_records)} records for async write at step {step_id}"
            )
        else:
            with open(chunk_file, "wb") as f:
                pickle.dump(serializable_records, f)
            logger.info(
                f"Saved {len(new_records)} records at step {step_id} to {self.log_dir}"
            )

        # 更新已保存的最后一个 step_id
        self._last_saved_step_id = new_records[-1].step_id

        # 保存后清除已保存记录的 tensor 以释放内存
        for rec in new_records:
            rec.u = None
            rec.gbar = None
            rec.diag_H = None

        # Clear dictionaries after saving to free memory (Fix #1)
        if self.save_indices_only and self.sample_indices_per_step:
            self.sample_indices_per_step.clear()
            logger.debug("Cleared sample_indices_per_step after saving")

        if self.save_rng_state and self.rng_states_per_step:
            self.rng_states_per_step.clear()
            logger.debug("Cleared rng_states_per_step after saving")

        # Explicitly clear CUDA cache to reduce fragmentation (Fix #4)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache after saving")

        # Clear prev_params to release CPU memory (~2GB for 1B model)
        # It will be automatically re-cloned on the next register_step call
        self.prev_params = None
        logger.debug("Cleared prev_params after saving")

        # Force Python garbage collection to reclaim memory
        gc.collect()
        logger.debug("Forced garbage collection after saving")

    def load_from_disk(self, step_id: Optional[int] = None):
        """
        从磁盘加载日志

        Args:
            step_id: 要加载的步骤ID(如果为None,加载最新的)
        """
        # 加载元信息
        meta_file = self.log_dir / "meta.json"
        if not meta_file.exists():
            logger.warning(f"Meta file not found: {meta_file}")
            return

        with open(meta_file, "r") as f:
            meta = json.load(f)

        # 更新配置
        self.save_batch_data = meta.get("save_batch_data", False)
        self.save_indices_only = meta.get("save_indices_only", False)
        self.save_rng_state = meta.get("save_rng_state", False)
        self.compute_diag_h = meta.get("compute_diag_h", False)
        self.batch_size_at_training = meta.get("batch_size_at_training")
        self.steps_per_epoch = meta.get("steps_per_epoch")
        self.checkpoints_per_epoch = meta.get("checkpoints_per_epoch", 0)
        self.save_at_epoch_end = meta.get("save_at_epoch_end", False)
        self.current_epoch = meta.get("current_epoch", 0)
        self.epoch_end_steps = meta.get("epoch_end_steps", [])

        # 加载批次索引
        index_file = self.log_dir / "batch_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                self.batch_index = json.load(f)

        # 加载样本索引(轻存储模式)
        indices_file = self.log_dir / "sample_indices.json"
        if indices_file.exists():
            with open(indices_file, "r") as f:
                serializable_indices = json.load(f)
                # Convert string keys back to int
                self.sample_indices_per_step = {
                    int(k): v for k, v in serializable_indices.items()
                }

        # 加载RNG状态
        if step_id is None:
            step_id = meta.get("current_step", 0)

        rng_file = self.log_dir / f"rng_states_{step_id}.pkl"
        if rng_file.exists():
            with open(rng_file, "rb") as f:
                self.rng_states_per_step = pickle.load(f)

        # 检查是否使用增量保存格式
        is_incremental = meta.get("incremental_save", False)

        self.step_log.clear()

        if is_incremental:
            # 增量保存格式: 加载所有 chunk 文件
            chunk_files = sorted(self.log_dir.glob("step_records_chunk_*.pkl"))
            if chunk_files:
                for chunk_file in chunk_files:
                    with open(chunk_file, "rb") as f:
                        chunk_records = pickle.load(f)
                    for rec_dict in chunk_records:
                        record = StepRecord(
                            step_id=rec_dict["step_id"],
                            eta=rec_dict["eta"],
                            batch_id=rec_dict["batch_id"],
                            u=rec_dict["u"],
                            gbar=rec_dict["gbar"],
                            diag_H=rec_dict.get("diag_H"),
                        )
                        self.step_log.add(record)
                logger.info(
                    f"Loaded {len(chunk_files)} chunk files from {self.log_dir}"
                )
            else:
                logger.warning(f"No chunk files found in {self.log_dir}")
        else:
            # 旧格式: 加载单个 step_records 文件
            records_file = self.log_dir / f"step_records_{step_id}.pkl"
            if not records_file.exists():
                logger.warning(f"Records file not found: {records_file}")
                return

            with open(records_file, "rb") as f:
                serializable_records = pickle.load(f)

            for rec_dict in serializable_records:
                record = StepRecord(
                    step_id=rec_dict["step_id"],
                    eta=rec_dict["eta"],
                    batch_id=rec_dict["batch_id"],
                    u=rec_dict["u"],
                    gbar=rec_dict["gbar"],
                    diag_H=rec_dict.get("diag_H"),
                )
                self.step_log.add(record)

        self.current_step = meta.get("current_step", 0)

        logger.info(
            f"Loaded training log from {self.log_dir}, {len(self.step_log.buffer)} records"
        )

    def clear(self):
        """清空日志"""
        self.stop_async_writer()
        self.step_log.clear()
        self.batch_index.clear()
        self.sample_indices_per_step.clear()
        self.rng_states_per_step.clear()
        self.current_step = 0
        self.current_epoch = 0
        self.epoch_end_steps.clear()
        self.prev_params = None
        self._last_saved_step_id = -1


def reconstruct_batch_from_indices(
    dataset,
    sample_indices: List[int],
    data_collator,
    rng_state: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """
    从样本索引重建批次数据

    Args:
        dataset: 原始数据集
        sample_indices: 样本索引列表
        data_collator: 数据collator
        rng_state: RNG状态(可选,用于恢复随机状态)

    Returns:
        重建的批次数据
    """
    # 恢复RNG状态(如果提供)
    if rng_state is not None:
        if "torch" in rng_state and rng_state["torch"] is not None:
            torch.set_rng_state(rng_state["torch"])
        if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    # 从数据集获取样本
    samples = [dataset[idx] for idx in sample_indices]

    # 使用data_collator重建批次
    batch = data_collator(samples)

    return batch


class BatchReconstructor:
    """
    批次重建器

    用于从训练日志中重建批次数据,支持:
    1. 直接使用保存的batch_data
    2. 从sample_indices + rng_state重建
    3. 从sample_indices重建(不恢复随机状态)
    """

    def __init__(
        self,
        training_logger: TrainingLogger,
        dataset=None,
        data_collator=None,
    ):
        self.training_logger = training_logger
        self.dataset = dataset
        self.data_collator = data_collator

    def get_batch_for_step(self, step_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取指定步骤的批次数据

        Args:
            step_id: 步骤ID

        Returns:
            批次数据,如果无法获取则返回None
        """
        # 获取步骤记录
        record = self.training_logger.get_step_record(step_id)
        if record is None:
            logger.warning(f"Step {step_id} not found in training log")
            return None

        # 1. 如果有保存的batch_data,直接使用
        if record.batch_data is not None:
            return record.batch_data

        # 2. 如果有sample_indices,尝试重建
        if step_id in self.training_logger.sample_indices_per_step:
            if self.dataset is None or self.data_collator is None:
                logger.warning(
                    f"Cannot reconstruct batch for step {step_id}: "
                    f"dataset or data_collator not provided"
                )
                return None

            sample_indices = self.training_logger.sample_indices_per_step[step_id]
            rng_state = self.training_logger.rng_states_per_step.get(step_id, None)

            logger.debug(
                f"Reconstructing batch for step {step_id} from {len(sample_indices)} samples"
            )

            return reconstruct_batch_from_indices(
                self.dataset,
                sample_indices,
                self.data_collator,
                rng_state,
            )

        # 3. 无法重建
        logger.warning(
            f"Cannot reconstruct batch for step {step_id}: "
            f"no batch_data or sample_indices available"
        )
        return None


class TrainingLoggerCallback:
    """
    训练回调,用于在训练过程中自动记录日志

    可以集成到Trainer中,在每个训练步骤后自动记录
    """

    def __init__(self, logger: TrainingLogger):
        self.logger = logger

    def on_step_end(
        self,
        step: int,
        batch_id: Any,
        eta: float,
        model: nn.Module,
        batch_data: Optional[Dict] = None,
    ):
        """
        训练步骤结束时调用

        Args:
            step: 当前步骤
            batch_id: 批次ID
            eta: 学习率
            model: 模型
            batch_data: 批次数据(可选)
        """
        self.logger.register_step(
            step_id=step,
            batch_id=batch_id,
            eta=eta,
            model=model,
            batch_data=batch_data,
        )

    def on_training_end(self):
        """训练结束时调用"""
        self.logger.save_to_disk()
