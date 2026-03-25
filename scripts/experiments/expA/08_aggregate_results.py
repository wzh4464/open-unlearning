#!/usr/bin/env python3
"""Aggregate Experiment A results into summary tables."""
import json
import csv
import os
import sys
from pathlib import Path


def load_json(path):
    """Load JSON file, return empty dict if not found or on parse error."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Failed to load {path}: {e}")
    return {}


def find_methods(seed=0):
    """Build method name -> model directory mapping."""
    unlearn_dir = Path("saves/unlearn")
    finetune_dir = Path("saves/finetune/llama2_7b_tofu_1epoch")
    retrain_dir = Path("saves/finetune/llama2_7b_tofu_retrain")

    methods = {}
    methods["Original"] = finetune_dir
    methods["Retrain"] = retrain_dir

    # Scan for all expA methods
    method_names = ["lmcleaner", "graddiff", "npo", "pdu", "undial"]
    for m in method_names:
        d = unlearn_dir / f"expA_{m}_s{seed}"
        if d.exists():
            methods[m.upper() if m != "lmcleaner" else "LMCleaner"] = d

    return methods


def extract_metrics(model_dir):
    """Extract all metrics from a model directory."""
    result = {}

    # Basic eval - read both SUMMARY and EVAL files (they contain different data)
    for subdir in ["evals", "."]:
        found = False
        for fname in ["TOFU_SUMMARY.json", "TOFU_EVAL.json"]:
            p = model_dir / subdir / fname
            if p.exists():
                found = True
                data = load_json(p)
                if fname == "TOFU_SUMMARY.json":
                    result.update(data)
                elif fname == "TOFU_EVAL.json":
                    for metric_name, metric_data in data.items():
                        if isinstance(metric_data, dict) and "agg_value" in metric_data:
                            result[metric_name] = metric_data["agg_value"]
        if found:
            break  # Found files in this subdir, skip next subdir

    # MIA eval - read both SUMMARY and EVAL files
    for subdir in ["evals_mia", "."]:
        found = False
        for fname in ["TOFU_SUMMARY.json", "TOFU_EVAL.json"]:
            p = model_dir / subdir / fname
            if p.exists():
                found = True
                data = load_json(p)
                if fname == "TOFU_SUMMARY.json":
                    for k, v in data.items():
                        result[f"mia_{k}"] = v
                elif fname == "TOFU_EVAL.json":
                    for metric_name, metric_data in data.items():
                        if isinstance(metric_data, dict) and "agg_value" in metric_data:
                            result[f"mia_{metric_name}"] = metric_data["agg_value"]
        if found:
            break  # Found files in this subdir, skip next subdir

    # Efficiency
    eff_path = model_dir / "efficiency_metrics.json"
    if eff_path.exists():
        eff = load_json(eff_path)
        result["time_seconds"] = eff.get("unlearning_time_seconds", None)
        result["peak_gpu_mb"] = eff.get("peak_gpu_memory_mb", None)
        result["tokens_per_sec"] = eff.get("tokens_per_second", None)
        result["forward_passes"] = eff.get("forward_passes_total", None)
        result["backward_passes"] = eff.get("backward_passes_total", None)
        result["flops_estimate"] = eff.get("flops_estimate", None)

    return result


def print_table(rows, key_metrics=None):
    """Print a formatted comparison table."""
    if not rows:
        print("No data to display.")
        return

    if key_metrics is None:
        # Auto-detect common metrics
        all_keys = set()
        for r in rows:
            all_keys.update(k for k in r.keys() if k != "Method")
        key_metrics = sorted(all_keys)

    # Header
    col_width = 16
    header = f"{'Method':<20}"
    for k in key_metrics:
        header += f"{k:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for row in rows:
        line = f"{row['Method']:<20}"
        for k in key_metrics:
            v = row.get(k, "N/A")
            if isinstance(v, float):
                line += f"{v:>{col_width}.4f}"
            elif v is None:
                line += f"{'N/A':>{col_width}}"
            else:
                line += f"{str(v):>{col_width}}"
        print(line)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    results_dir = Path("saves/results/expA")
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = find_methods(seed)
    if not methods:
        print("No experiment results found!")
        return

    rows = []
    for method_name, model_dir in methods.items():
        if not model_dir.exists():
            print(f"WARNING: {method_name} not found at {model_dir}")
            continue

        metrics = extract_metrics(model_dir)
        row = {"Method": method_name}
        row.update(metrics)
        rows.append(row)

    # Write full CSV
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)

    csv_path = results_dir / f"full_results_s{seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Method"] + [k for k in all_keys if k != "Method"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Results saved to {csv_path}")
    print(f"Methods found: {[r['Method'] for r in rows]}")

    # Print summary tables
    print("\n" + "=" * 80)
    print(f"EXPERIMENT A SUMMARY (seed={seed})")
    print("=" * 80)

    # Try to print key metrics table
    key_metrics = [
        "model_utility", "forget_quality", "forget_Truth_Ratio",
        "extraction_strength", "time_seconds", "peak_gpu_mb"
    ]
    available_keys = [k for k in key_metrics if any(k in r for r in rows)]
    if available_keys:
        print("\n--- Key Metrics ---")
        print_table(rows, available_keys)

    # Print all metrics per method
    print("\n--- Full Details ---")
    for row in rows:
        print(f"\n=== {row['Method']} ===")
        for k, v in sorted(row.items()):
            if k == "Method":
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
