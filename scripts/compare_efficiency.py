#!/usr/bin/env python3
"""
Compare efficiency metrics across different unlearning methods.
Usage: python scripts/compare_efficiency.py [results_dir]
"""

import json
import sys
from pathlib import Path
from typing import Dict


def load_efficiency_metrics(exp_dir: Path) -> Dict:
    """Load efficiency metrics from an experiment directory"""
    metrics_file = exp_dir / "efficiency_metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Add experiment name
    metrics['experiment'] = exp_dir.name
    return metrics


def compare_metrics(results_dir: Path = None):
    """Compare efficiency metrics across experiments"""
    if results_dir is None:
        results_dir = Path("saves/unlearn")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Find all experiments with efficiency metrics
    experiments = []
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "efficiency_metrics.json").exists():
            metrics = load_efficiency_metrics(exp_dir)
            if metrics:
                experiments.append(metrics)

    if not experiments:
        print(f"No efficiency metrics found in {results_dir}")
        return

    # Sort by experiment name
    experiments.sort(key=lambda x: x['experiment'])

    # Print comparison table
    print("\n" + "="*80)
    print("Unlearning Efficiency Metrics Comparison")
    print("="*80)
    print(f"{'Experiment':<30} {'Time(s)':<12} {'Steps':<8} {'GPU Mem(MB)':<15} {'Tokens/s':<12}")
    print("-"*80)

    for exp in experiments:
        # Defensive handling of potentially missing fields
        time_s = exp.get('unlearning_time_seconds', 0.0)
        steps = exp.get('total_steps', 0)
        memory_mb = exp.get('peak_gpu_memory_mb', 0.0)
        tokens_s = exp.get('tokens_per_second', 0.0)

        print(f"{exp['experiment']:<30} "
              f"{time_s:<12.2f} "
              f"{steps:<8} "
              f"{memory_mb:<15.2f} "
              f"{tokens_s:<12.2f}")

    print("="*80)

    # Find best performers
    if len(experiments) > 1:
        fastest = min(experiments, key=lambda x: x.get('unlearning_time_seconds', float('inf')))
        most_memory_efficient = min(experiments, key=lambda x: x.get('peak_gpu_memory_mb', float('inf')))

        # Check if throughput metrics are meaningful
        throughput_values = [exp.get('tokens_per_second', 0) for exp in experiments]
        has_positive_throughput = any(v > 0 for v in throughput_values)

        print("\nBest Performers:")
        print(f"  Fastest: {fastest['experiment']} ({fastest.get('unlearning_time_seconds', 0):.2f}s)")
        print(f"  Most Memory Efficient: {most_memory_efficient['experiment']} ({most_memory_efficient.get('peak_gpu_memory_mb', 0):.2f} MB)")

        if has_positive_throughput:
            highest_throughput = max(experiments, key=lambda x: x.get('tokens_per_second', 0))
            print(f"  Highest Throughput: {highest_throughput['experiment']} ({highest_throughput.get('tokens_per_second', 0):.2f} tokens/s)")
        else:
            print("  Highest Throughput: N/A (tokens_per_second unavailable or zero for all experiments)")
        print()


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("saves/unlearn")
    compare_metrics(results_dir)
