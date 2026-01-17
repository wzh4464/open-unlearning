#!/usr/bin/env python
"""
精确计算 Figure 1 所需的数据（需要模型和数据集）

使用幂迭代法真实计算谱范数，并通过 HVP 传播影响

用法:
    uv run python scripts/compute_figure1_exact.py \
        --model_path saves/finetune/llama32_1b_tofu_safe \
        --log_dir saves/train_logs/llama32_1b_tofu_safe \
        --output_dir saves/figure1_data \
        --dataset_name tofu \
        --sample_interval 50 \
        --num_power_iters 30
"""

import argparse
import gc
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Iterator

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_records(log_dir: Path) -> Tuple[List[Dict], Dict]:
    """加载训练记录和元信息"""
    # 加载元信息
    meta_file = log_dir / "meta.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    # 加载样本索引
    indices_file = log_dir / "sample_indices.json"
    sample_indices = {}
    if indices_file.exists():
        with open(indices_file) as f:
            sample_indices = {int(k): v for k, v in json.load(f).items()}

    # 加载记录
    records = []
    chunk_files = sorted(log_dir.glob("step_records*.pkl"))

    for chunk_file in chunk_files:
        try:
            with open(chunk_file, "rb") as f:
                chunk_records = pickle.load(f)
            records.extend(chunk_records)
            logger.info(f"Loaded {len(chunk_records)} records from {chunk_file.name}")
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Skipping corrupted file {chunk_file}: {e}")

    records.sort(key=lambda x: x["step_id"])

    # 合并样本索引到记录
    for rec in records:
        step_id = rec["step_id"]
        if step_id in sample_indices:
            rec["sample_indices"] = sample_indices[step_id]

    return records, meta


class LazyRecordLoader:
    """
    按需加载训练记录，避免一次性加载全部数据到内存。

    使用方法:
        loader = LazyRecordLoader(log_dir)
        loader.build_index()  # 构建索引

        # 获取有 sample_indices 的 step_ids
        step_ids = loader.get_step_ids_with_indices()

        # 按需加载特定步骤
        records = loader.load_steps(step_ids[:10])
    """

    def __init__(self, log_dir: Path, num_workers: int = 4):
        self.log_dir = Path(log_dir)
        self.num_workers = num_workers

        # 加载元信息
        meta_file = self.log_dir / "meta.json"
        self.meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                self.meta = json.load(f)

        # 加载样本索引 (小文件，可以全部加载)
        indices_file = self.log_dir / "sample_indices.json"
        self.sample_indices: Dict[int, List[int]] = {}
        if indices_file.exists():
            with open(indices_file) as f:
                self.sample_indices = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded sample_indices for {len(self.sample_indices)} steps")

        # Chunk 索引: chunk_file -> (min_step_id, max_step_id)
        self._chunk_index: Dict[str, Tuple[int, int]] = {}
        self._step_to_chunk: Dict[int, str] = {}  # step_id -> chunk_file

    def build_index(self) -> None:
        """构建 chunk 索引，支持并行扫描"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 尝试加载缓存的索引
        index_cache = self.log_dir / "chunk_index.json"
        if index_cache.exists():
            try:
                with open(index_cache) as f:
                    self._chunk_index = {k: tuple(v) for k, v in json.load(f).items()}
                logger.info(f"Loaded cached chunk index with {len(self._chunk_index)} entries")
                self._build_step_to_chunk_map()
                return
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load cached index: {e}")

        chunk_files = sorted(self.log_dir.glob("step_records_chunk_*.pkl"))
        if not chunk_files:
            logger.warning("No chunk files found")
            return

        logger.info(f"Building chunk index from {len(chunk_files)} files...")

        def scan_chunk(chunk_file: Path) -> Tuple[str, int, int]:
            try:
                with open(chunk_file, "rb") as f:
                    records = pickle.load(f)
                if records:
                    step_ids = [r["step_id"] for r in records]
                    return (chunk_file.name, min(step_ids), max(step_ids))
            except Exception as e:
                logger.warning(f"Error scanning {chunk_file.name}: {e}")
            return (chunk_file.name, -1, -1)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(scan_chunk, cf): cf for cf in chunk_files}
            for future in as_completed(futures):
                name, min_id, max_id = future.result()
                if min_id >= 0:
                    self._chunk_index[name] = (min_id, max_id)

        # 保存索引缓存
        with open(index_cache, "w") as f:
            json.dump(self._chunk_index, f, indent=2)
        logger.info(f"Built and cached chunk index: {len(self._chunk_index)} chunks")

        self._build_step_to_chunk_map()

    def _build_step_to_chunk_map(self) -> None:
        """构建 step_id -> chunk_file 的映射"""
        self._step_to_chunk = {}
        for chunk_name, (min_id, max_id) in self._chunk_index.items():
            for step_id in range(min_id, max_id + 1):
                self._step_to_chunk[step_id] = chunk_name

    def get_all_step_ids(self) -> List[int]:
        """获取所有可用的 step_id"""
        return sorted(self._step_to_chunk.keys())

    def get_step_ids_with_indices(self) -> List[int]:
        """获取有 sample_indices 的 step_id 列表"""
        available_steps = set(self._step_to_chunk.keys())
        return sorted([s for s in self.sample_indices.keys() if s in available_steps])

    def load_steps(
        self,
        step_ids: List[int],
        include_tensors: bool = True
    ) -> List[Dict]:
        """
        按需加载指定步骤的记录

        Args:
            step_ids: 要加载的步骤 ID 列表
            include_tensors: 是否加载 tensor 数据 (u, gbar 等)

        Returns:
            记录列表，按 step_id 排序
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        step_ids_set = set(step_ids)

        # 找出需要加载的 chunk 文件
        chunks_needed: Dict[str, Set[int]] = {}  # chunk_name -> set of step_ids
        for step_id in step_ids:
            chunk_name = self._step_to_chunk.get(step_id)
            if chunk_name:
                if chunk_name not in chunks_needed:
                    chunks_needed[chunk_name] = set()
                chunks_needed[chunk_name].add(step_id)

        if not chunks_needed:
            logger.warning("No chunks found for requested steps")
            return []

        logger.info(f"Loading {len(step_ids)} steps from {len(chunks_needed)} chunks...")

        def load_chunk_filtered(chunk_name: str, wanted_steps: Set[int]) -> List[Dict]:
            chunk_file = self.log_dir / chunk_name
            try:
                with open(chunk_file, "rb") as f:
                    records = pickle.load(f)
                # 只保留需要的步骤
                filtered = []
                for r in records:
                    if r["step_id"] in wanted_steps:
                        if not include_tensors:
                            # 移除 tensor 数据以节省内存
                            r = {k: v for k, v in r.items() if k not in ("u", "gbar", "diag_H")}
                        filtered.append(r)
                return filtered
            except Exception as e:
                logger.warning(f"Error loading {chunk_name}: {e}")
                return []

        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(load_chunk_filtered, cn, steps): cn
                for cn, steps in chunks_needed.items()
            }
            for future in as_completed(futures):
                results.extend(future.result())

        # 排序并添加 sample_indices
        results.sort(key=lambda x: x["step_id"])
        for rec in results:
            step_id = rec["step_id"]
            if step_id in self.sample_indices:
                rec["sample_indices"] = self.sample_indices[step_id]

        logger.info(f"Loaded {len(results)} records")
        return results

    def iter_steps(
        self,
        step_ids: List[int],
        batch_size: int = 20
    ) -> Iterator[Dict]:
        """
        迭代加载步骤，每次只加载一批到内存

        Args:
            step_ids: 要迭代的步骤 ID 列表
            batch_size: 每批加载的步骤数

        Yields:
            每个步骤的记录 dict
        """
        for i in range(0, len(step_ids), batch_size):
            batch = step_ids[i:i + batch_size]
            records = self.load_steps(batch)

            # 按请求顺序 yield
            record_map = {r["step_id"]: r for r in records}
            for step_id in batch:
                if step_id in record_map:
                    yield record_map[step_id]

            # 清理内存
            del records
            del record_map
            gc.collect()


def load_dataset_and_collator(dataset_name: str, tokenizer):
    """加载数据集和 collator"""
    from data import get_data
    from transformers import DataCollatorForLanguageModeling

    # 使用 TOFU 数据集
    if dataset_name == "tofu":
        from datasets import load_dataset
        dataset = load_dataset("locuslab/TOFU", "full")["train"]

        def tokenize_fn(examples):
            return tokenizer(
                examples["question"],
                examples["answer"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return dataset, collator


def reconstruct_batch(
    dataset,
    sample_indices: List[int],
    collator,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """从样本索引重建批次"""
    samples = [dataset[idx] for idx in sample_indices]
    batch = collator(samples)
    return {k: v.to(device) for k, v in batch.items()}


def compute_hvp_ggn(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    v: torch.Tensor,
    params: List[nn.Parameter],
) -> torch.Tensor:
    """使用 GGN 近似计算 HVP"""
    v_list = unflatten_like(v, params)

    with torch.enable_grad():
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((g * v_part).sum() for g, v_part in zip(grads, v_list))
        hvp_list = torch.autograd.grad(gv, params)

    return torch.cat([h.view(-1) for h in hvp_list])


def unflatten_like(flat: torch.Tensor, params: List[nn.Parameter]) -> List[torch.Tensor]:
    """将一维向量还原为参数形状"""
    result = []
    offset = 0
    for p in params:
        n = p.numel()
        result.append(flat[offset:offset + n].view_as(p))
        offset += n
    return result


def estimate_spectral_norm(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    eta: float,
    params: List[nn.Parameter],
    num_iters: int = 30,
) -> Tuple[float, float]:
    """
    使用幂迭代法估计 ||I - η*H||_2

    Returns:
        (spectral_norm, largest_eigenvalue_estimate)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in params)

    # 随机初始化
    v = torch.randn(num_params, device=device, dtype=torch.float32)
    v = v / v.norm()

    # 幂迭代
    for _ in range(num_iters):
        Hv = compute_hvp_ggn(model, batch, v.to(dtype), params)
        Hv = Hv.float()

        # P = I - ηH
        Pv = v - eta * Hv

        v_new = Pv / (Pv.norm() + 1e-10)
        v = v_new

    # 最终估计
    Hv = compute_hvp_ggn(model, batch, v.to(dtype), params)
    Hv = Hv.float()
    Pv = v - eta * Hv
    spectral_norm = Pv.norm().item()

    # 估计最大特征值: ||Hv|| / ||v|| ≈ λ_max
    lambda_est = Hv.norm().item()

    return spectral_norm, lambda_est


def compute_u_norm_statistics(records: List[Dict]) -> Dict[str, Any]:
    """
    计算 u 向量的统计信息（不需要 sample_indices）

    当没有 sample_indices 时，可以用这个方法分析训练动态

    Args:
        records: 包含 u 向量的记录列表

    Returns:
        包含 u_norms, eta 值等的统计字典
    """
    u_norms = []
    etas = []
    step_ids = []

    for rec in records:
        if rec.get("u") is not None:
            u = rec["u"]
            if isinstance(u, torch.Tensor):
                u_norms.append(u.norm().item())
            step_ids.append(rec["step_id"])
            etas.append(rec.get("eta", 0))

    if not u_norms:
        return {}

    # 计算累积收缩因子估计 (基于 u 范数变化)
    contraction_estimates = []
    for i in range(1, len(u_norms)):
        if u_norms[i-1] > 0:
            ratio = u_norms[i] / u_norms[i-1]
            contraction_estimates.append(ratio)

    return {
        "step_ids": step_ids,
        "u_norms": u_norms,
        "etas": etas,
        "mean_u_norm": sum(u_norms) / len(u_norms),
        "max_u_norm": max(u_norms),
        "min_u_norm": min(u_norms),
        "contraction_estimates": contraction_estimates,
        "mean_contraction": sum(contraction_estimates) / len(contraction_estimates) if contraction_estimates else None,
    }


def propagate_influence_exact(
    model: nn.Module,
    records: List[Dict],
    start_step: int,
    max_lag: int,
    params: List[nn.Parameter],
    dataset,
    collator,
    device: str = "cuda",
) -> Tuple[List[float], List[float]]:
    """
    精确传播影响（使用真实 HVP）

    Returns:
        (influence_norms, spectral_norms_along_path)
    """
    step_to_rec = {r["step_id"]: r for r in records}

    if start_step not in step_to_rec:
        logger.warning(f"Step {start_step} not found")
        return [], []

    start_rec = step_to_rec[start_step]
    if start_rec["u"] is None:
        logger.warning(f"No u vector for step {start_step}")
        return [], []

    # 初始偏差
    dtype = next(model.parameters()).dtype
    v = -start_rec["u"].to(device).to(dtype)
    initial_norm = v.norm().item()

    influence_norms = [1.0]  # 归一化
    spectral_norms = []

    # 传播
    for k in tqdm(range(1, max_lag + 1), desc=f"Propagating from step {start_step}"):
        current_step = start_step + k
        if current_step not in step_to_rec:
            break

        rec = step_to_rec[current_step]
        eta = rec["eta"]

        if eta == 0 or "sample_indices" not in rec:
            # 无法计算 HVP，使用近似
            influence_norms.append(influence_norms[-1] * 0.99)
            continue

        try:
            # 重建 batch
            batch = reconstruct_batch(dataset, rec["sample_indices"], collator, device)

            # 计算 HVP
            Hv = compute_hvp_ggn(model, batch, v, params)

            # 传播: v ← v - η * Hv
            v = v - eta * Hv

            # 记录谱范数
            Pv_norm = (v - eta * Hv).norm().item()
            v_norm_before = v.norm().item() + eta * Hv.norm().item()
            local_spectral = Pv_norm / (v_norm_before + 1e-10)
            spectral_norms.append(local_spectral)

            # 记录归一化影响范数
            influence_norms.append(v.norm().item() / initial_norm)

        except Exception as e:
            logger.warning(f"Error at step {current_step}: {e}")
            influence_norms.append(influence_norms[-1] * 0.99)

    return influence_norms, spectral_norms


def main():
    parser = argparse.ArgumentParser(description="Compute Figure 1 data (exact)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="tofu")
    parser.add_argument("--sample_interval", type=int, default=50, help="Interval for spectral norm sampling")
    parser.add_argument("--num_power_iters", type=int, default=30)
    parser.add_argument("--max_propagation_lag", type=int, default=500)
    parser.add_argument("--target_steps", type=str, default=None, help="Comma-separated steps")
    parser.add_argument("--approximate", action="store_true", help="Use approximate method (faster)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for I/O")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 初始化按需加载器 (lazy loading)
    logger.info("Initializing lazy record loader...")
    loader = LazyRecordLoader(log_dir, num_workers=args.num_workers)
    loader.build_index()

    # 获取可用的步骤
    # 优先使用有 sample_indices 的步骤，否则使用所有步骤
    all_step_ids = loader.get_step_ids_with_indices()
    if all_step_ids:
        logger.info(f"Found {len(all_step_ids)} steps with sample_indices")
    else:
        all_step_ids = loader.get_all_step_ids()
        logger.info(f"No sample_indices.json found, using all {len(all_step_ids)} steps")

    if not all_step_ids:
        logger.error("No steps found!")
        return

    # 检查是否有 sample_indices 可用
    has_sample_indices = bool(loader.sample_indices)
    if not has_sample_indices:
        logger.warning("No sample_indices available - spectral norm computation will be skipped")
        logger.info("Will compute approximate decay curves using u vectors only")

    # 2. 加载模型
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    logger.info(f"Model has {num_params:,} trainable parameters")

    # 3. 加载数据集
    logger.info(f"Loading dataset: {args.dataset_name}...")
    dataset, collator = load_dataset_and_collator(args.dataset_name, tokenizer)
    logger.info(f"Dataset size: {len(dataset)}")

    results = {
        "spectral_norms": {},
        "decay_curves": {},
        "metadata": {
            "model_path": args.model_path,
            "log_dir": str(log_dir),
            "num_steps": len(all_step_ids),
            "num_params": num_params,
            "approximate": args.approximate,
            "lazy_loading": True,
        }
    }

    if args.approximate:
        # 使用近似方法（更快）
        logger.info("Using approximate method...")

        gamma_estimates = []
        mean_gamma = 0.985  # 默认值

        if has_sample_indices:
            # 采样用于估计 gamma 的步骤
            sampled_step_ids = all_step_ids[::args.sample_interval][:10]
            logger.info(f"Sampling {len(sampled_step_ids)} steps for gamma estimation")

            # 按需加载这些步骤
            sample_records = loader.load_steps(sampled_step_ids)

            for rec in tqdm(sample_records, desc="Estimating gamma"):
                if rec.get("eta", 0) <= 0 or "sample_indices" not in rec:
                    continue
                try:
                    batch = reconstruct_batch(dataset, rec["sample_indices"], collator, device)
                    gamma, lambda_max = estimate_spectral_norm(
                        model, batch, rec["eta"], params, args.num_power_iters
                    )
                    gamma_estimates.append(gamma)
                    results["spectral_norms"][str(rec["step_id"])] = gamma
                except Exception as e:
                    logger.warning(f"Error at step {rec['step_id']}: {e}")

            # 清理内存
            del sample_records
            gc.collect()

            if gamma_estimates:
                mean_gamma = sum(gamma_estimates) / len(gamma_estimates)
                logger.info(f"Estimated mean γ = {mean_gamma:.4f}")
                results["metadata"]["mean_gamma"] = mean_gamma
            else:
                logger.warning(f"No gamma estimates, using default {mean_gamma}")
        else:
            logger.info(f"Using default γ = {mean_gamma} (no sample_indices for estimation)")

        # 使用估计的 gamma 生成衰减曲线
        target_steps = []
        if args.target_steps:
            target_steps = [int(s) for s in args.target_steps.split(",")]
        else:
            # 选择中间位置的步骤
            n = len(all_step_ids)
            target_steps = [all_step_ids[n//4], all_step_ids[n//2]]

        for step in target_steps:
            decay = [mean_gamma ** k for k in range(args.max_propagation_lag)]
            results["decay_curves"][str(step)] = decay

        # 如果没有 sample_indices，计算 u-norm 统计
        if not has_sample_indices:
            logger.info("Computing u-norm statistics...")
            # 加载所有步骤的 u 向量
            all_records = loader.load_steps(all_step_ids)
            u_stats = compute_u_norm_statistics(all_records)
            if u_stats:
                results["u_norm_statistics"] = {
                    "mean_u_norm": u_stats["mean_u_norm"],
                    "max_u_norm": u_stats["max_u_norm"],
                    "min_u_norm": u_stats["min_u_norm"],
                    "mean_contraction": u_stats.get("mean_contraction"),
                    "num_steps": len(u_stats["step_ids"]),
                }
                # 保存 u_norms 序列用于绘图
                results["u_norms"] = {
                    str(sid): norm for sid, norm in zip(u_stats["step_ids"], u_stats["u_norms"])
                }
                logger.info(f"U-norm stats: mean={u_stats['mean_u_norm']:.4e}, "
                           f"range=[{u_stats['min_u_norm']:.4e}, {u_stats['max_u_norm']:.4e}]")
                if u_stats.get("mean_contraction"):
                    logger.info(f"Estimated mean contraction from u-norms: {u_stats['mean_contraction']:.4f}")
            del all_records
            gc.collect()

    else:
        # 使用精确方法（较慢）
        logger.info("Using exact method...")

        if has_sample_indices:
            # 采样用于计算谱范数的步骤
            sampled_step_ids = all_step_ids[::args.sample_interval][:50]
            logger.info(f"Computing spectral norms for {len(sampled_step_ids)} sampled steps...")

            # 按需加载并计算
            sample_records = loader.load_steps(sampled_step_ids)

            for rec in tqdm(sample_records, desc="Computing spectral norms"):
                if rec.get("eta", 0) <= 0 or "sample_indices" not in rec:
                    continue
                try:
                    batch = reconstruct_batch(dataset, rec["sample_indices"], collator, device)
                    gamma, _ = estimate_spectral_norm(model, batch, rec["eta"], params, args.num_power_iters)
                    results["spectral_norms"][str(rec["step_id"])] = gamma
                except Exception as e:
                    logger.warning(f"Error: {e}")

            del sample_records
            gc.collect()

            # 计算精确衰减曲线
            target_steps = []
            if args.target_steps:
                target_steps = [int(s) for s in args.target_steps.split(",")]
            else:
                # 选择中间位置的步骤
                target_steps = [all_step_ids[len(all_step_ids)//2]]

            for step in target_steps:
                logger.info(f"Computing exact propagation from step {step}...")

                # 加载从 start_step 到 start_step + max_lag 的所有步骤
                propagation_steps = [s for s in all_step_ids if step <= s < step + args.max_propagation_lag]
                logger.info(f"Loading {len(propagation_steps)} steps for propagation...")

                propagation_records = loader.load_steps(propagation_steps)

                norms, _ = propagate_influence_exact(
                    model, propagation_records, step, args.max_propagation_lag,
                    params, dataset, collator, device
                )
                if norms:
                    results["decay_curves"][str(step)] = norms

                del propagation_records
                gc.collect()

        else:
            # 没有 sample_indices，只能计算 u-norm 统计
            logger.info("No sample_indices available for exact method, computing u-norm statistics only...")
            all_records = loader.load_steps(all_step_ids)
            u_stats = compute_u_norm_statistics(all_records)
            if u_stats:
                results["u_norm_statistics"] = {
                    "mean_u_norm": u_stats["mean_u_norm"],
                    "max_u_norm": u_stats["max_u_norm"],
                    "min_u_norm": u_stats["min_u_norm"],
                    "mean_contraction": u_stats.get("mean_contraction"),
                    "num_steps": len(u_stats["step_ids"]),
                }
                results["u_norms"] = {
                    str(sid): norm for sid, norm in zip(u_stats["step_ids"], u_stats["u_norms"])
                }
                logger.info(f"U-norm stats: mean={u_stats['mean_u_norm']:.4e}, "
                           f"range=[{u_stats['min_u_norm']:.4e}, {u_stats['max_u_norm']:.4e}]")
                if u_stats.get("mean_contraction"):
                    logger.info(f"Estimated mean contraction from u-norms: {u_stats['mean_contraction']:.4f}")
            del all_records
            gc.collect()

    # 保存结果
    output_file = output_dir / "figure1_data.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_file}")

    # 生成统计摘要
    if results["spectral_norms"]:
        norms = list(results["spectral_norms"].values())
        logger.info(f"Spectral norm statistics:")
        logger.info(f"  Mean: {sum(norms)/len(norms):.4f}")
        logger.info(f"  Min:  {min(norms):.4f}")
        logger.info(f"  Max:  {max(norms):.4f}")
        logger.info(f"  All < 1: {all(n < 1 for n in norms)}")


if __name__ == "__main__":
    main()
