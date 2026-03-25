#!/usr/bin/env python3
"""
RebuttalB: Analyze forget sample distribution across training mini-batches.

For each batch size B, computes:
  - Number of affected batches (containing ≥1 forget sample)
  - Per-batch forget count distribution (histogram, mean, median, max, variance)
  - Collateral ratio per batch: 1 - (#forget_in_batch / B)
  - Forget sample coverage across training steps
  - Dispersion metrics (Gini, entropy, Herfindahl)

Usage:
    python scripts/rebuttalB_forget_distribution.py \
        --train-log-dir saves/train_logs/rebuttalB_B256_seed0 \
        --forget-indices-file data/tofu_forget01_indices.json \
        --effective-batch-size 256 \
        --output-dir results/rebuttalB/stats
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_sample_indices(log_dir):
    """Load sample_indices.json mapping step_id -> list of sample indices."""
    path = Path(log_dir) / "sample_indices.json"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_forget_indices(path):
    """Load forget sample indices."""
    if path and Path(path).exists():
        with open(path) as f:
            return set(json.load(f))
    return None


def get_tofu_forget_indices(forget_split="forget01"):
    """Get TOFU forget sample indices from the dataset."""
    try:
        from datasets import load_dataset

        full = load_dataset("locuslab/TOFU", "full", split="train")
        forget = load_dataset("locuslab/TOFU", forget_split, split="train")

        # Match by question field
        forget_questions = set(forget["question"])
        forget_indices = set()
        for i, item in enumerate(full):
            if item["question"] in forget_questions:
                forget_indices.add(i)

        print(f"Found {len(forget_indices)} forget samples in full dataset")
        return forget_indices
    except Exception as e:
        print(f"Failed to load TOFU dataset: {e}")
        return None


def gini_coefficient(values):
    """Compute Gini coefficient of a distribution."""
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n


def herfindahl_index(values):
    """Compute Herfindahl-Hirschman Index (HHI) for concentration."""
    total = sum(values)
    if total == 0:
        return 0.0
    shares = [v / total for v in values]
    return sum(s ** 2 for s in shares)


def analyze_distribution(sample_indices_map, forget_indices, effective_b, output_dir=None):
    """Analyze forget sample distribution across training batches."""
    results = {}

    # 1. Find affected batches
    affected_batches = {}  # step_id -> list of forget indices in this batch
    all_steps = sorted(sample_indices_map.keys())

    for step_id in all_steps:
        batch_indices = set(sample_indices_map[step_id])
        forget_in_batch = batch_indices & forget_indices
        if forget_in_batch:
            affected_batches[step_id] = sorted(forget_in_batch)

    num_affected = len(affected_batches)
    total_steps = len(all_steps)
    results["num_affected_batches"] = num_affected
    results["total_steps"] = total_steps
    results["affected_ratio"] = num_affected / total_steps if total_steps > 0 else 0
    results["effective_batch_size"] = effective_b

    # 2. Per-batch forget count
    forget_counts = [len(v) for v in affected_batches.values()]
    if forget_counts:
        results["forget_per_batch_mean"] = float(np.mean(forget_counts))
        results["forget_per_batch_median"] = float(np.median(forget_counts))
        results["forget_per_batch_max"] = int(np.max(forget_counts))
        results["forget_per_batch_min"] = int(np.min(forget_counts))
        results["forget_per_batch_std"] = float(np.std(forget_counts))
        results["forget_per_batch_p90"] = float(np.percentile(forget_counts, 90))
        results["forget_per_batch_p95"] = float(np.percentile(forget_counts, 95))
    else:
        results["forget_per_batch_mean"] = 0
        results["forget_per_batch_median"] = 0
        results["forget_per_batch_max"] = 0

    # 3. Collateral ratio: 1 - (#forget_in_batch / B)
    actual_batch_sizes = [len(sample_indices_map[s]) for s in affected_batches.keys()]
    collateral_ratios = []
    for step_id, forget_list in affected_batches.items():
        batch_size = len(sample_indices_map[step_id])
        collateral_ratios.append(1.0 - len(forget_list) / batch_size)

    if collateral_ratios:
        results["collateral_ratio_mean"] = float(np.mean(collateral_ratios))
        results["collateral_ratio_median"] = float(np.median(collateral_ratios))
        results["collateral_ratio_min"] = float(np.min(collateral_ratios))
    else:
        results["collateral_ratio_mean"] = 0

    # 4. Dispersion metrics
    if forget_counts:
        results["dispersion_gini"] = float(gini_coefficient(np.array(forget_counts)))
        results["dispersion_hhi"] = float(herfindahl_index(forget_counts))
        total_forget_in_batches = sum(forget_counts)
        if total_forget_in_batches > 0:
            probs = [c / total_forget_in_batches for c in forget_counts]
            results["dispersion_entropy"] = float(-sum(p * math.log2(p) for p in probs if p > 0))
        else:
            results["dispersion_entropy"] = 0

    # 5. Coverage: forget samples appearing across training timeline
    affected_steps = sorted(affected_batches.keys())
    if affected_steps:
        results["earliest_affected_step"] = affected_steps[0]
        results["latest_affected_step"] = affected_steps[-1]
        results["step_span"] = affected_steps[-1] - affected_steps[0]
        # Quartile distribution of affected steps
        q = np.percentile(affected_steps, [25, 50, 75])
        results["affected_step_q25"] = float(q[0])
        results["affected_step_q50"] = float(q[1])
        results["affected_step_q75"] = float(q[2])

    # 6. Forget count histogram (for plotting)
    count_hist = dict(Counter(forget_counts))
    results["forget_count_histogram"] = {str(k): v for k, v in sorted(count_hist.items())}

    # 7. Step-level detail (for plotting step distribution)
    results["affected_steps_detail"] = {
        str(step_id): len(forget_list)
        for step_id, forget_list in affected_batches.items()
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Forget Distribution Analysis (B={effective_b})")
    print(f"{'=' * 60}")
    print(f"Total training steps:        {total_steps}")
    print(f"Affected batches:            {num_affected} ({results['affected_ratio']*100:.1f}%)")
    print(f"Forget samples per batch:")
    print(f"  Mean:    {results.get('forget_per_batch_mean', 0):.2f}")
    print(f"  Median:  {results.get('forget_per_batch_median', 0):.1f}")
    print(f"  Max:     {results.get('forget_per_batch_max', 0)}")
    print(f"  Std:     {results.get('forget_per_batch_std', 0):.2f}")
    print(f"Collateral ratio (mean):     {results.get('collateral_ratio_mean', 0):.4f}")
    print(f"Dispersion (Gini):           {results.get('dispersion_gini', 0):.4f}")
    print(f"Dispersion (HHI):            {results.get('dispersion_hhi', 0):.4f}")
    print(f"Dispersion (Entropy):        {results.get('dispersion_entropy', 0):.4f}")
    if affected_steps:
        print(f"Affected step range:         [{affected_steps[0]}, {affected_steps[-1]}]")
    print(f"Forget count histogram:      {count_hist}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"batch_distribution_B{effective_b}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze forget distribution in training batches")
    parser.add_argument("--train-log-dir", type=str, required=True)
    parser.add_argument("--forget-indices-file", type=str, default=None,
                        help="JSON file with list of forget sample indices")
    parser.add_argument("--forget-split", type=str, default="forget01",
                        help="TOFU forget split name (used if --forget-indices-file not given)")
    parser.add_argument("--effective-batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Load sample indices
    sample_indices_map = load_sample_indices(args.train_log_dir)
    print(f"Loaded sample indices for {len(sample_indices_map)} steps")

    # Check indices per step
    step_sizes = [len(v) for v in sample_indices_map.values()]
    print(f"Indices per step: min={min(step_sizes)}, max={max(step_sizes)}, "
          f"mean={sum(step_sizes)/len(step_sizes):.1f}")

    # Load forget indices
    if args.forget_indices_file:
        forget_indices = load_forget_indices(args.forget_indices_file)
    else:
        forget_indices = get_tofu_forget_indices(args.forget_split)

    if forget_indices is None:
        print("Error: could not determine forget indices")
        sys.exit(1)

    print(f"Forget set size: {len(forget_indices)}")

    # Analyze
    analyze_distribution(
        sample_indices_map,
        forget_indices,
        args.effective_batch_size,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
