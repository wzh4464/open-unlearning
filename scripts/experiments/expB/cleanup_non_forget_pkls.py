#!/usr/bin/env python3
"""Delete step_records pkl files for non-forget steps to save storage.

Only forget steps need u[t] (for Phase 1: v0 = -u[tz]).
Phase 2 HVP only needs eta + batch reconstruction, not u[t].

Usage:
    python scripts/experiments/expB/cleanup_non_forget_pkls.py \
        --train-log-dir /workspace/saves/train_logs/rebuttalB_B32_seed0 \
        --forget-split forget01 [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path


def get_tofu_forget_indices(forget_split="forget01"):
    from datasets import load_dataset
    full = load_dataset("locuslab/TOFU", "full", split="train")
    forget = load_dataset("locuslab/TOFU", forget_split, split="train")
    forget_questions = set(forget["question"])
    return {i for i, item in enumerate(full) if item["question"] in forget_questions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log-dir", type=str, required=True)
    parser.add_argument("--forget-split", type=str, default="forget01")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.train_log_dir)
    si_path = log_dir / "sample_indices.json"
    if not si_path.exists():
        print(f"Error: {si_path} not found")
        sys.exit(1)

    # Load sample indices and forget set
    sample_indices = {int(k): v for k, v in json.load(open(si_path)).items()}
    forget_indices = get_tofu_forget_indices(args.forget_split)
    print(f"Forget set: {len(forget_indices)} samples")
    print(f"Total steps: {len(sample_indices)}")

    # Find forget steps
    forget_steps = set()
    for step_id, indices in sample_indices.items():
        if set(indices) & forget_indices:
            forget_steps.add(step_id)
    print(f"Forget steps (contain ≥1 forget sample): {len(forget_steps)}")

    # Find pkls to delete
    all_pkls = sorted(log_dir.glob("step_records_chunk_*.pkl"))
    keep = []
    delete = []
    for pkl in all_pkls:
        step_id = int(pkl.stem.split("_")[-1])
        if step_id in forget_steps:
            keep.append(pkl)
        else:
            delete.append(pkl)

    keep_size = sum(p.stat().st_size for p in keep) / 1e9
    delete_size = sum(p.stat().st_size for p in delete) / 1e9

    print(f"\nKeep: {len(keep)} pkls ({keep_size:.1f} GB) — forget steps")
    print(f"Delete: {len(delete)} pkls ({delete_size:.1f} GB) — non-forget steps")
    print(f"Savings: {delete_size:.1f} GB ({delete_size/(keep_size+delete_size)*100:.0f}%)")

    if args.dry_run:
        print("\n[dry-run] No files deleted")
        return

    for pkl in delete:
        pkl.unlink()
    print(f"\nDeleted {len(delete)} files")


if __name__ == "__main__":
    main()
