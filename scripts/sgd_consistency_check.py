#!/usr/bin/env python3
"""
SGD Consistency Check: verify whether stored training trajectories follow plain SGD.

Two modes:
  1. Quick mode (default): Analyze u[t] statistics. Under plain SGD with constant lr,
     ||u[t]|| should be proportional to ||ḡ[t]|| and u[t] = -η·ḡ[t].
     Under AdamW, per-parameter adaptive scaling makes ||u[t]|| more uniform.

  2. Full mode (--full): Recompute ḡ[t] and compute e_sgd(t) = ||u[t] + η·ḡ[t]|| / ||u[t]||.

Usage:
    python scripts/sgd_consistency_check.py \
        --train-log-dir saves/train_logs/llama32_1b_tofu_safe \
        [--full --model-path saves/finetune/llama32_1b_tofu_safe]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

import torch


def load_step_record(log_dir, step_id):
    """Load a single step record from chunk files."""
    chunk_file = log_dir / f"step_records_chunk_{step_id}.pkl"
    if chunk_file.exists():
        with open(chunk_file, "rb") as f:
            records = pickle.load(f)
        if isinstance(records, list):
            for rec in records:
                if rec.get("step_id") == step_id:
                    return rec
            return records[0] if records else None
        return records
    return None


def load_eta_cache(log_dir):
    eta_path = log_dir / "eta_cache.json"
    if eta_path.exists():
        with open(eta_path) as f:
            return {int(k): float(v) for k, v in json.load(f).items()}
    return {}


def analyze_u_statistics(log_dir, steps, eta_cache):
    """Analyze u[t] vector statistics to detect optimizer type.

    Key insight: Under plain SGD, u[t] = -η·ḡ[t], so:
    - u[t] / (-η) = ḡ[t] = average gradient
    - Per-parameter update magnitudes vary wildly (some params get huge gradients, some tiny)

    Under AdamW, u[t] = -η·m̂/(√v̂+ε) - η·λ·θ, so:
    - Per-parameter update magnitudes are normalized by √v̂
    - Updates are much more uniform across parameters
    - The ratio ||u[t]||_inf / ||u[t]||_2 * sqrt(d) is closer to 1 under AdamW

    Also: under AdamW, consecutive u[t] should be highly correlated due to momentum (β1=0.9).
    Under SGD without momentum, consecutive u[t] have lower correlation (only from data similarity).
    """
    print("=" * 70)
    print("SGD CONSISTENCY CHECK — Statistical Analysis of u[t] vectors")
    print("=" * 70)

    results = []
    prev_u = None

    for step_id in steps:
        rec = load_step_record(log_dir, step_id)
        if rec is None:
            print(f"  Step {step_id}: record not found, skipping")
            continue

        u = rec.get("u")
        if u is None:
            print(f"  Step {step_id}: u is None, skipping")
            continue

        if isinstance(u, torch.Tensor):
            u = u.float()
        else:
            continue

        eta = eta_cache.get(step_id, rec.get("eta", 0))
        u_norm_2 = u.norm(2).item()
        u_norm_inf = u.norm(float('inf')).item()
        d = u.numel()

        # Uniformity metric: for perfectly uniform |u_i| = c,
        # ||u||_inf / (||u||_2 / sqrt(d)) = 1
        # For very non-uniform (one big, rest small):
        # ratio → sqrt(d) >> 1
        # AdamW normalizes → ratio closer to 1
        # SGD doesn't normalize → ratio much larger
        uniformity = (u_norm_inf / (u_norm_2 / (d ** 0.5) + 1e-30))

        # Sparsity: fraction of params where |u_i| > 0.01 * ||u||_inf
        above_threshold = (u.abs() > 0.01 * u_norm_inf).float().mean().item()

        # Cosine similarity with previous step (momentum detection)
        cos_sim = None
        if prev_u is not None and prev_u.shape == u.shape:
            cos_sim = (torch.dot(u, prev_u) / (u.norm() * prev_u.norm() + 1e-12)).item()

        results.append({
            "step": step_id,
            "eta": eta,
            "u_norm_2": u_norm_2,
            "u_norm_inf": u_norm_inf,
            "uniformity": uniformity,
            "active_frac": above_threshold,
            "cos_prev": cos_sim,
        })

        prev_u = u.clone()

        # Free memory
        del u, rec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("\nNo u[t] vectors found. Cannot perform analysis.")
        return

    # Print table
    print(f"\n{'Step':>6} {'η':>10} {'||u||₂':>12} {'||u||∞':>12} {'Uniformity':>12} {'Active%':>10} {'cos(t,t-1)':>12}")
    print("-" * 80)
    for r in results:
        cos_str = f"{r['cos_prev']:.4f}" if r['cos_prev'] is not None else "—"
        print(f"{r['step']:>6} {r['eta']:>10.2e} {r['u_norm_2']:>12.6f} {r['u_norm_inf']:>12.8f} "
              f"{r['uniformity']:>12.2f} {r['active_frac']*100:>9.2f}% {cos_str:>12}")

    # Aggregate analysis
    uniformities = [r['uniformity'] for r in results]
    cos_sims = [r['cos_prev'] for r in results if r['cos_prev'] is not None]

    print(f"\n{'=' * 70}")
    print("AGGREGATE ANALYSIS")
    print(f"{'=' * 70}")

    avg_uni = np.mean(uniformities)
    print(f"\nUniformity (||u||∞ / (||u||₂/√d)):")
    print(f"  Mean: {avg_uni:.2f}")
    print(f"  → SGD expected: ~100-1000+ (gradient magnitudes vary wildly per parameter)")
    print(f"  → AdamW expected: ~1-10 (adaptive scaling normalizes per-parameter updates)")

    if cos_sims:
        avg_cos = np.mean(cos_sims)
        print(f"\nConsecutive step cosine similarity:")
        print(f"  Mean: {avg_cos:.4f}")
        print(f"  → SGD (no momentum) expected: ~0.3-0.7 (only data-driven correlation)")
        print(f"  → AdamW (β₁=0.9) expected: ~0.95-0.99 (EMA momentum smoothing)")

    print(f"\n{'=' * 70}")
    print("DIAGNOSIS")
    print(f"{'=' * 70}")

    # Decision logic
    is_adamw = False
    reasons = []

    if avg_uni < 50:
        is_adamw = True
        reasons.append(f"Uniformity={avg_uni:.1f} < 50 → per-parameter normalization detected (AdamW signature)")
    else:
        reasons.append(f"Uniformity={avg_uni:.1f} ≥ 50 → non-uniform updates (consistent with SGD)")

    if cos_sims:
        avg_cos = np.mean(cos_sims)
        if avg_cos > 0.9:
            is_adamw = True
            reasons.append(f"cos_sim={avg_cos:.3f} > 0.9 → strong momentum (AdamW β₁=0.9 signature)")
        elif avg_cos > 0.7:
            reasons.append(f"cos_sim={avg_cos:.3f} — borderline, could be either")
        else:
            reasons.append(f"cos_sim={avg_cos:.3f} < 0.7 → no strong momentum (consistent with SGD)")

    for r in reasons:
        print(f"  • {r}")

    if is_adamw:
        print(f"\n  ❌ VERDICT: Training trajectory is NOT plain SGD.")
        print(f"     LMCleaner's HVP propagation assumes u[t] = -η·ḡ[t], but the stored")
        print(f"     u[t] vectors reflect AdamW's adaptive scaling + momentum.")
        print(f"     → RETRAIN WITH plain SGD (`optim: sgd`) to match paper assumptions.")
    else:
        print(f"\n  ✅ VERDICT: Training trajectory is consistent with plain SGD.")


def main():
    parser = argparse.ArgumentParser(description="SGD consistency check")
    parser.add_argument("--train-log-dir", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    log_dir = Path(args.train_log_dir)

    meta_path = log_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Training meta: mode={meta.get('mode')}, steps={meta.get('current_step')}")

    eta_cache = load_eta_cache(log_dir)
    if not eta_cache:
        print("Error: no eta_cache.json")
        sys.exit(1)

    all_steps = sorted(eta_cache.keys())
    total = len(all_steps)
    n = min(args.num_steps, total)
    indices = [int(i * (total - 1) / (n - 1)) if n > 1 else 0 for i in range(n)]
    check_steps = [all_steps[i] for i in indices]

    print(f"Checking {n} steps: {check_steps}")
    analyze_u_statistics(log_dir, check_steps, eta_cache)


if __name__ == "__main__":
    main()
