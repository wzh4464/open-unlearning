#!/usr/bin/env python3
"""
RebuttalB Step 4: Aggregate all (K,B) results into a single CSV.

Collects:
  - TOFU eval metrics (from TOFU_SUMMARY.json)
  - Efficiency metrics (from efficiency_metrics.json)
  - Forget distribution stats (from batch_distribution_B*.json)

Usage:
    python scripts/experiments/expB/04_aggregate_results.py \
        --saves-dir /workspace/saves \
        --output results/expB_results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# B values and K values matching config.sh
B_VALUES = [8, 16, 32, 64, 128, 256]
K_VALUES = [10, 20, 30, 40, 50]
STEPS_PER_EPOCH = {8: 500, 16: 250, 32: 125, 64: 62, 128: 31, 256: 15}


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saves-dir", default="/workspace/saves")
    parser.add_argument("--output", default="/workspace/saves/experiments/expB/expB_results.csv")
    args = parser.parse_args()

    saves = Path(args.saves_dir)
    stats_dir = saves / "experiments" / "expB" / "stats"

    # Load distribution stats
    dist_stats = {}
    for B in B_VALUES:
        p = stats_dir / f"batch_distribution_B{B}.json"
        dist_stats[B] = load_json(p)

    # Collect rows
    rows = []
    for B in B_VALUES:
        max_k = STEPS_PER_EPOCH[B]
        for K in K_VALUES:
            if K > max_k:
                continue

            task = f"expB_B{B}_K{K}"
            row = {"B": B, "K": K, "max_K": max_k}

            # TOFU eval
            eval_path = saves / "eval" / task / "TOFU_SUMMARY.json"
            tofu = load_json(eval_path)
            if tofu:
                row["model_utility"] = tofu.get("model_utility")
                row["forget_Q_A_ROUGE"] = tofu.get("forget_Q_A_ROUGE")
                row["forget_Q_A_Prob"] = tofu.get("forget_Q_A_Prob")
                row["forget_truth_ratio"] = tofu.get("forget_truth_ratio")
                row["extraction_strength"] = tofu.get("extraction_strength")
                row["privleak"] = tofu.get("privleak")
                row["mia_min_k"] = tofu.get("mia_min_k")
                row["status"] = "evaluated"
            else:
                row["status"] = "no_eval"

            # Efficiency: try efficiency_metrics.json first, fallback to audit_records
            eff_path = saves / "unlearn" / task / "efficiency_metrics.json"
            audit_path = saves / "unlearn" / task / "audit" / "audit_records.json"
            meta_path = saves / "unlearn" / task / "unlearning_meta.json"
            eff = load_json(eff_path)
            audit = load_json(audit_path)
            umeta = load_json(meta_path)

            if eff:
                row["unlearn_time_s"] = eff.get("lmcleaner_unlearning_total_time_seconds",
                                                  eff.get("unlearning_time_seconds"))
                row["peak_gpu_mb"] = eff.get("lmcleaner_unlearning_peak_gpu_mb",
                                              eff.get("peak_gpu_memory_mb"))
                row["total_hvp_calls"] = eff.get("lmcleaner_total_hvp_calls")
                row["K_used_mean"] = eff.get("lmcleaner_K_used_mean")
                row["K_used_max"] = eff.get("lmcleaner_K_used_max")
                row["correction_norm_mean"] = eff.get("lmcleaner_correction_norm_mean")
                row["per_step_time_mean_ms"] = eff.get("lmcleaner_per_forget_step_time_mean_ms")
                row["num_forget_steps"] = eff.get("lmcleaner_num_forget_steps")
                row["num_affected_batches"] = eff.get("lmcleaner_num_affected_batches")
                row["train_logs_gb"] = eff.get("lmcleaner_train_logs_size_gb")
            elif audit and isinstance(audit, list) and len(audit) > 0:
                # Derive efficiency from audit records
                import statistics
                times = [r.get("wall_time_ms", 0) for r in audit]
                hvps = [r.get("hvp_calls", 0) for r in audit]
                k_used = [r.get("K_used", 0) for r in audit]
                vnorms = [r.get("v_norm", 0) for r in audit]
                row["unlearn_time_s"] = sum(times) / 1000.0
                row["total_hvp_calls"] = sum(hvps)
                row["K_used_mean"] = statistics.mean(k_used) if k_used else 0
                row["K_used_max"] = max(k_used) if k_used else 0
                row["correction_norm_mean"] = statistics.mean(vnorms) if vnorms else 0
                row["per_step_time_mean_ms"] = statistics.mean(times) if times else 0
                row["num_forget_steps"] = len(audit)

            if umeta:
                row["num_affected_batches"] = umeta.get("num_forget_batches",
                                                         row.get("num_affected_batches"))

            if row.get("unlearn_time_s") is not None:
                if row["status"] == "no_eval":
                    row["status"] = "unlearned"
            else:
                if row["status"] == "no_eval":
                    row["status"] = "not_run"

            # Distribution stats
            ds = dist_stats.get(B)
            if ds:
                row["dist_affected_batches"] = ds.get("num_affected_batches")
                row["dist_forget_per_batch_mean"] = ds.get("forget_per_batch_mean")
                row["dist_forget_per_batch_max"] = ds.get("forget_per_batch_max")
                row["dist_collateral_ratio"] = ds.get("collateral_ratio_mean")
                row["dist_gini"] = ds.get("dispersion_gini")
                row["dist_entropy"] = ds.get("dispersion_entropy")

            rows.append(row)

    # Write CSV
    if not rows:
        print("No results found")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = dict.fromkeys(k for row in rows for k in row.keys())
    fieldnames = list(all_keys)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Print summary table
    print(f"\n{'B':>4} {'K':>4} {'status':>12} {'utility':>10} {'forget_ROUGE':>13} {'time_s':>8} {'hvp':>6} {'gpu_mb':>8}")
    print("-" * 75)
    def fmt(val, spec=".4f"):
        return f"{val:{spec}}" if val is not None else "—"

    for r in rows:
        u = fmt(r.get("model_utility"))
        fr = fmt(r.get("forget_Q_A_ROUGE"))
        t = fmt(r.get("unlearn_time_s"), ".1f")
        h = fmt(r.get("total_hvp_calls"), "d") if isinstance(r.get("total_hvp_calls"), int) else "—"
        g = fmt(r.get("peak_gpu_mb"), ".0f")
        print(f"{r['B']:>4} {r['K']:>4} {r['status']:>12} {u:>10} {fr:>13} {t:>8} {h:>6} {g:>8}")


if __name__ == "__main__":
    main()
