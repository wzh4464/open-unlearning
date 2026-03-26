"""
Analyze distribution of forget samples per batch across training experiments.

For each experiment (different batch sizes), count how many forget01 samples
(indices 3960-3999) appear in each training batch, then plot:
1. Empirical frequency histogram
2. Theoretical hypergeometric distribution density
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from pathlib import Path
from collections import Counter

# TOFU forget01 indices
FORGET_INDICES = set(range(3960, 4000))  # 40 samples
N_TOTAL = 4000  # total dataset size
N_FORGET = len(FORGET_INDICES)  # 40

TRAIN_LOGS_DIR = Path("saves/train_logs")

EXPERIMENTS = {
    "llama32_1b_tofu_safe": "B=16",
    "rebuttalB_B8_seed0": "B=8",
    "rebuttalB_B32_seed0": "B=32",
    "rebuttalB_B64_seed0": "B=64",
    "rebuttalB_B128_seed0": "B=128",
    "rebuttalB_B256_seed0": "B=256",
}


def load_forget_counts(exp_name):
    """Load sample_indices and count forget samples per batch."""
    si_path = TRAIN_LOGS_DIR / exp_name / "sample_indices.json"
    with open(si_path) as f:
        sample_indices = json.load(f)

    counts = []
    batch_sizes = []
    for step_id, indices in sorted(sample_indices.items(), key=lambda x: int(x[0])):
        n_forget = sum(1 for idx in indices if idx in FORGET_INDICES)
        counts.append(n_forget)
        batch_sizes.append(len(indices))

    return counts, batch_sizes


def main():
    # Collect data
    results = {}
    for exp_name, label in EXPERIMENTS.items():
        try:
            counts, batch_sizes = load_forget_counts(exp_name)
            B = int(np.median(batch_sizes))
            results[label] = {
                "counts": counts,
                "batch_size": B,
                "n_steps": len(counts),
            }
            print(
                f"{label}: {len(counts)} steps, batch_size={B}, "
                f"forget_per_batch: mean={np.mean(counts):.3f}, "
                f"max={max(counts)}, zero_batches={counts.count(0)}/{len(counts)}"
            )
        except Exception as e:
            print(f"{label}: FAILED - {e}")

    # Sort by batch size
    sorted_labels = sorted(results.keys(), key=lambda x: results[x]["batch_size"])

    # Plot
    n_plots = len(sorted_labels)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, label in enumerate(sorted_labels):
        ax = axes[i]
        data = results[label]
        counts = data["counts"]
        B = data["batch_size"]
        n_steps = data["n_steps"]

        # Empirical histogram
        max_count = max(counts)
        bins = np.arange(-0.5, max_count + 1.5, 1)
        hist_values, _, bars = ax.hist(
            counts,
            bins=bins,
            density=True,
            alpha=0.7,
            color="#4C72B0",
            edgecolor="white",
            label="Empirical",
        )

        # Theoretical hypergeometric distribution
        # Drawing B samples from N=4000, K=40 forget items
        x_range = np.arange(0, min(B, N_FORGET) + 1)
        # Only plot up to where pmf is meaningful
        pmf = hypergeom.pmf(x_range, N_TOTAL, N_FORGET, B)
        # Filter to visible range
        mask = pmf > 1e-6
        x_plot = x_range[mask]
        pmf_plot = pmf[mask]

        ax.plot(
            x_plot,
            pmf_plot,
            "ro-",
            markersize=6,
            linewidth=2,
            label=f"Hypergeometric\n(N={N_TOTAL}, K={N_FORGET}, n={B})",
        )

        ax.set_xlabel("# forget samples in batch", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(f"{label}  ({n_steps} steps/epoch)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.5, max(max_count, x_plot[-1]) + 0.5)

        # Add expected value line
        E_x = B * N_FORGET / N_TOTAL
        ax.axvline(E_x, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label=f"E[X]={E_x:.2f}")
        ax.legend(fontsize=9)

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribution of Forget Samples per Batch (TOFU forget01, 1 epoch)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("saves/forget_batch_distribution.png", dpi=150, bbox_inches="tight")
    print("\nSaved: saves/forget_batch_distribution.png")

    # Also print summary table
    print("\n=== Summary Table ===")
    print(f"{'Batch Size':>10} | {'Steps':>6} | {'E[X]':>6} | {'Mean':>6} | {'Std':>6} | {'Max':>4} | {'P(X=0)':>8} | {'Theo P(X=0)':>11}")
    print("-" * 80)
    for label in sorted_labels:
        data = results[label]
        counts = data["counts"]
        B = data["batch_size"]
        E_x = B * N_FORGET / N_TOTAL
        theo_p0 = hypergeom.pmf(0, N_TOTAL, N_FORGET, B)
        emp_p0 = counts.count(0) / len(counts)
        print(
            f"{B:>10} | {data['n_steps']:>6} | {E_x:>6.3f} | {np.mean(counts):>6.3f} | "
            f"{np.std(counts):>6.3f} | {max(counts):>4} | {emp_p0:>8.3f} | {theo_p0:>11.3f}"
        )


if __name__ == "__main__":
    main()
