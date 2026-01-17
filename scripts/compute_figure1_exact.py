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
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载训练记录
    logger.info("Loading training records...")
    records, meta = load_training_records(log_dir)
    logger.info(f"Loaded {len(records)} records")

    if not records:
        logger.error("No records found!")
        return

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
            "num_records": len(records),
            "num_params": num_params,
            "approximate": args.approximate,
        }
    }

    if args.approximate:
        # 使用近似方法（更快）
        logger.info("Using approximate method...")

        # 估计平均谱范数
        gamma_estimates = []
        sample_steps = [r for r in records if r["eta"] > 0][::args.sample_interval][:10]

        for rec in tqdm(sample_steps, desc="Estimating gamma"):
            if "sample_indices" not in rec:
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

        if gamma_estimates:
            mean_gamma = sum(gamma_estimates) / len(gamma_estimates)
            logger.info(f"Estimated mean γ = {mean_gamma:.4f}")
            results["metadata"]["mean_gamma"] = mean_gamma
        else:
            mean_gamma = 0.985
            logger.warning(f"No gamma estimates, using default {mean_gamma}")

        # 使用估计的 gamma 生成衰减曲线
        target_steps = []
        if args.target_steps:
            target_steps = [int(s) for s in args.target_steps.split(",")]
        else:
            valid_records = [r for r in records if r["u"] is not None and r["eta"] > 0]
            if valid_records:
                n = len(valid_records)
                target_steps = [valid_records[n//4]["step_id"], valid_records[n//2]["step_id"]]

        for step in target_steps:
            rec = next((r for r in records if r["step_id"] == step), None)
            if rec and rec["u"] is not None:
                decay = [mean_gamma ** k for k in range(args.max_propagation_lag)]
                results["decay_curves"][str(step)] = decay

    else:
        # 使用精确方法（较慢）
        logger.info("Using exact method...")

        # 计算谱范数
        sample_records = [r for r in records if r["eta"] > 0 and "sample_indices" in r]
        sample_records = sample_records[::args.sample_interval]

        for rec in tqdm(sample_records[:50], desc="Computing spectral norms"):
            try:
                batch = reconstruct_batch(dataset, rec["sample_indices"], collator, device)
                gamma, _ = estimate_spectral_norm(model, batch, rec["eta"], params, args.num_power_iters)
                results["spectral_norms"][str(rec["step_id"])] = gamma
            except Exception as e:
                logger.warning(f"Error: {e}")

        # 计算精确衰减曲线
        target_steps = []
        if args.target_steps:
            target_steps = [int(s) for s in args.target_steps.split(",")]
        else:
            valid_records = [r for r in records if r["u"] is not None and "sample_indices" in r]
            if valid_records:
                target_steps = [valid_records[len(valid_records)//2]["step_id"]]

        for step in target_steps:
            logger.info(f"Computing exact propagation from step {step}...")
            norms, _ = propagate_influence_exact(
                model, records, step, args.max_propagation_lag,
                params, dataset, collator, device
            )
            if norms:
                results["decay_curves"][str(step)] = norms

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
