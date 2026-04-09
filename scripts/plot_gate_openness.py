#!/usr/bin/env python3
"""Plot fraction of open transforms vs input L2 norm from gate stats JSON.

Reads an initial_gate_stats.json (or similar) produced by experiment 09 and
creates a bar chart showing % transforms open at each input norm.

Usage:
    uv run python scripts/plot_gate_openness.py experiments/09_gemma_default_open

    # Custom input/output paths:
    uv run python scripts/plot_gate_openness.py experiments/09_gemma_default_open \
        --input results/initial_gate_stats.json \
        --output figures/gate_openness_vs_norm.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_gate_stats(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_gate_openness(
    stats: list[dict],
    save_path: Path,
    title: str = "Gate Openness vs Input L2 Norm",
) -> None:
    """Bar chart of % transforms open at each input norm."""
    norms = []
    fractions = []
    labels = []
    for entry in stats:
        norm = entry["input_norm_mean"]
        frac = entry["l0"] / entry["total_transforms"] * 100
        norms.append(norm)
        fractions.append(frac)
        label = entry["label"].replace("_", " ").replace("synthetic ", "")
        if "real" in entry["label"]:
            label = f"Real Gemma (‖x‖≈{norm:.0f})"
        else:
            label = f"‖x‖={norm:.0f}"
        labels.append(label)

    colors = ["#2563eb" if "real" not in s["label"] else "#dc2626" for s in stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(norms)), fractions, color=colors, edgecolor="white", width=0.7)

    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Transforms Open (%)", fontsize=11)
    ax.set_xlabel("Input L2 Norm", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, frac in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{frac:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot gate openness vs input norm")
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory")
    parser.add_argument("--input", type=Path, default=Path("results/initial_gate_stats.json"),
                        help="Relative path to gate stats JSON (default: results/initial_gate_stats.json)")
    parser.add_argument("--output", type=Path, default=Path("figures/gate_openness_vs_norm.png"),
                        help="Relative path for output PNG (default: figures/gate_openness_vs_norm.png)")
    parser.add_argument("--title", type=str, default="Gate Openness vs Input L2 Norm")
    args = parser.parse_args()

    stats_path = args.experiment_dir / args.input
    save_path = args.experiment_dir / args.output

    stats = load_gate_stats(stats_path)
    plot_gate_openness(stats, save_path, title=args.title)


if __name__ == "__main__":
    main()
