#!/usr/bin/env python3
"""
Compute spectral norms from saved model checkpoints and RNG states.

Uses the saved RNG states to reproduce exact batches from training,
then computes ||I - η*H||_2 at each checkpoint.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import pickle
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


def restore_rng_state(rng_state: Dict, device: str = "cuda:0"):
    """Restore PyTorch RNG state from saved state."""
    torch.set_rng_state(rng_state["torch"])
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        # rng_state["torch_cuda"] is a list of states for all CUDA devices
        cuda_states = rng_state["torch_cuda"]
        if cuda_states:
            # Get device index
            if device.startswith("cuda:"):
                device_idx = int(device.split(":")[1])
            else:
                device_idx = 0
            if device_idx < len(cuda_states):
                torch.cuda.set_rng_state(cuda_states[device_idx], device=device_idx)


def load_rng_state(log_dir: Path, step: int) -> Optional[Dict]:
    """Load RNG state for a specific step."""
    rng_file = log_dir / f"rng_states_{step}.pkl"
    if not rng_file.exists():
        print(f"RNG state file not found: {rng_file}")
        return None

    with open(rng_file, "rb") as f:
        rng_data = pickle.load(f)

    # rng_data is {step: {torch: ..., torch_cuda: ...}}
    if step in rng_data:
        return rng_data[step]
    return None


def compute_loss(model, batch, params_list=None):
    """Compute cross-entropy loss."""
    outputs = model(**{k: v for k, v in batch.items() if k not in ["index", "labels"]},
                    labels=batch.get("labels"))
    return outputs.loss


def compute_hvp_for_batch(
    model: torch.nn.Module,
    batch: Dict,
    v: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> torch.Tensor:
    """Compute Hessian-vector product using double backward (R-operator)."""
    model.zero_grad()

    # Forward pass
    loss = compute_loss(model, batch)

    # First backward - compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

    # Flatten gradients
    grad_flat = torch.cat([
        g.view(-1) if g is not None else torch.zeros(p.numel(), device=v.device, dtype=v.dtype)
        for g, p in zip(grads, params)
    ])

    # Second backward - compute Hv = d(grad . v)/dparams
    grad_v = torch.dot(grad_flat, v)

    hvp_grads = torch.autograd.grad(grad_v, params, allow_unused=True)

    hvp = torch.cat([
        g.view(-1) if g is not None else torch.zeros(p.numel(), device=v.device, dtype=v.dtype)
        for g, p in zip(hvp_grads, params)
    ])

    model.zero_grad()
    return hvp


def estimate_spectral_norm(
    model: torch.nn.Module,
    batch: Dict,
    eta: float,
    params: List[torch.nn.Parameter],
    num_iters: int = 20,
    device: str = "cuda",
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate ||I - η*H||_2 using power iteration."""
    dtype = next(model.parameters()).dtype
    num_params = sum(p.numel() for p in params)

    # Initialize random vector
    v = torch.randn(num_params, device=device, dtype=dtype)
    v = v / v.norm()

    model.eval()

    try:
        for _ in range(num_iters):
            Hv = compute_hvp_for_batch(model, batch, v, params)
            Pv = v - eta * Hv
            Pv_norm = Pv.norm()
            if Pv_norm > 1e-10:
                v = Pv / Pv_norm

        # Final estimate
        Hv = compute_hvp_for_batch(model, batch, v, params)
        Pv = v - eta * Hv
        spectral_norm = Pv.norm().item()
        lambda_est = Hv.norm().item()

        return spectral_norm, lambda_est

    except Exception as e:
        print(f"Error in spectral norm estimation: {e}")
        return None, None

    finally:
        model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="saves/train_logs/llama32_1b_tofu_safe")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_power_iters", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    device = args.device

    # Find checkpoints
    checkpoint_dir = log_dir / "model_checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("epoch_*_step_*"))

    if not checkpoints:
        print("No checkpoints found")
        return

    print(f"Found {len(checkpoints)} checkpoints")

    # Load tokenizer from the finetune output (contains tokenizer files)
    # The model_checkpoints only contain model weights, not tokenizer
    finetune_dir = log_dir.parent.parent / "finetune" / log_dir.name
    if not finetune_dir.exists():
        # Try to find it from checkpoint parent
        finetune_dir = log_dir  # Fallback
    tokenizer = AutoTokenizer.from_pretrained(finetune_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset directly
    from datasets import load_dataset

    # Load TOFU dataset
    dataset = load_dataset("locuslab/TOFU", "full")["train"]

    # Simple preprocessing without template (just Q+A concatenation)
    def preprocess(examples):
        texts = []
        for q, a in zip(examples["question"], examples["answer"]):
            # Use simple format
            text = f"Question: {q}\n\nAnswer: {a}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )

        # Set up labels (same as input_ids for causal LM)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]

        return tokenized

    train_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    train_dataset.set_format("torch")

    results = {
        "spectral_norms": {},
        "lambda_maxs": {},
        "etas": {},
        "metadata": {
            "learning_rate": args.learning_rate,
            "num_power_iters": args.num_power_iters,
            "batch_size": args.batch_size,
            "checkpoints": [str(c) for c in checkpoints],
        }
    }

    for ckpt_path in tqdm(checkpoints, desc="Processing checkpoints"):
        # Extract step from checkpoint name
        step = int(ckpt_path.name.split("_step_")[1])

        print(f"\n=== Processing step {step} ===")

        # Load RNG state for this step (or closest available)
        rng_state = load_rng_state(log_dir, step)
        if rng_state is None:
            # Try step - 1 (e.g., for step 1250, use 1249)
            rng_state = load_rng_state(log_dir, step - 1)
            if rng_state is not None:
                print(f"Using RNG state from step {step - 1}")
            else:
                print(f"Skipping step {step}: no RNG state")
                continue

        # Restore RNG state
        restore_rng_state(rng_state, device)

        # Load model from finetune dir and apply checkpoint weights
        # Disable SDPA to allow double backward for HVP
        model = AutoModelForCausalLM.from_pretrained(
            finetune_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",  # Use standard attention for HVP
        )

        # Load the checkpoint state dict
        ckpt_file = ckpt_path / "model_state_dict.pt"
        if ckpt_file.exists():
            state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

        # Create dataloader with restored RNG
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Will use restored RNG
            num_workers=0,
        )

        # Get one batch
        batch = next(iter(dataloader))
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get trainable params
        params = [p for p in model.parameters() if p.requires_grad]

        # Compute spectral norm
        spectral_norm, lambda_max = estimate_spectral_norm(
            model, batch, args.learning_rate, params,
            num_iters=args.num_power_iters, device=device
        )

        if spectral_norm is not None:
            results["spectral_norms"][str(step)] = spectral_norm
            results["lambda_maxs"][str(step)] = lambda_max
            results["etas"][str(step)] = args.learning_rate

            print(f"Step {step}: ||P||_2 = {spectral_norm:.6f}, λ_max ≈ {lambda_max:.4f}")

            if spectral_norm >= 1.0:
                print(f"  WARNING: Non-contractive!")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Compute statistics
    norms = list(results["spectral_norms"].values())
    if norms:
        results["statistics"] = {
            "mean": sum(norms) / len(norms),
            "min": min(norms),
            "max": max(norms),
            "all_contractive": all(n < 1.0 for n in norms),
            "contraction_rate": sum(1 for n in norms if n < 1.0) / len(norms),
        }

    # Save results
    output_file = log_dir / "spectral_norms_posthoc.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results saved to {output_file} ===")
    if "statistics" in results:
        stats = results["statistics"]
        print(f"Mean ||P||_2: {stats['mean']:.4f}")
        print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"All contractive: {stats['all_contractive']}")


if __name__ == "__main__":
    main()
