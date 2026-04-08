"""Shared plotting utilities for MOLT experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curves(
    history: list[dict],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot 4-panel training curves: NMSE, L0, Sparsity Loss, MSE.

    Args:
        history: list of metric dicts from training (must have step, nmse, l0, sparsity_loss, mse)
        title: figure title
        save_path: path to save the PNG
    """
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    panels = [
        (axes[0, 0], "nmse", "NMSE", "Normalized MSE", "#2563eb"),
        (axes[0, 1], "l0", "L0 (active transforms)", "L0 Sparsity", "#dc2626"),
        (axes[1, 0], "sparsity_loss", "Sparsity Loss", "Sparsity Loss", "#16a34a"),
        (axes[1, 1], "mse", "MSE", "Raw MSE", "#9333ea"),
    ]

    for ax, key, ylabel, ax_title, color in panels:
        if key in history[0]:
            vals = [h[key] for h in history]
            ax.plot(steps, vals, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(ax_title)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # If threshold is tracked, replace bottom-right panel
    if "threshold" in history[0]:
        axes[1, 1].clear()
        axes[1, 1].plot(steps, [h["threshold"] for h in history], color="#9333ea", linewidth=1.5)
        axes[1, 1].set_ylabel("Threshold (theta)")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_title("Learned Threshold")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def plot_multi_run_curves(
    runs: dict[str, list[dict]],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot overlaid training curves for multiple runs on shared 4-panel axes.

    Args:
        runs: {run_name: history_list} mapping
        title: figure title
        save_path: path to save the PNG
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    for i, (name, history) in enumerate(runs.items()):
        c = colors[i % len(colors)]
        steps = [h["step"] for h in history]

        axes[0, 0].plot(steps, [h["l0"] for h in history], lw=1.5, color=c, label=name)
        axes[0, 1].plot(steps, [h["nmse"] for h in history], lw=1.5, color=c, label=name)
        axes[1, 0].plot(steps, [h["mse"] for h in history], lw=1.5, color=c, label=name)

        if "threshold" in history[0]:
            axes[1, 1].plot(steps, [h["threshold"] for h in history], lw=1.5, color=c, label=name)

    axes[0, 0].set_ylabel("L0")
    axes[0, 0].set_title("Active Transforms (L0)")
    axes[0, 1].set_ylabel("NMSE")
    axes[0, 1].set_title("Normalized MSE")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_title("Raw MSE")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_ylabel("Threshold")
    axes[1, 1].set_title("Learned Threshold")
    axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("Step")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def plot_l0_vs_nmse(
    results: list[dict],
    save_path: str | Path,
    transcoder_results: list[dict] | None = None,
) -> None:
    """Plot L0 vs NMSE scatter with Pareto frontier.

    Args:
        results: list of dicts with 'l0' and 'nmse' keys
        save_path: path to save the PNG
        transcoder_results: optional baseline comparison points
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    l0s = [r["l0"] for r in results]
    nmses = [r["nmse"] for r in results]
    ax.scatter(nmses, l0s, label="MOLT", marker="o", s=60)

    # Pareto frontier
    sorted_pts = sorted(zip(nmses, l0s), key=lambda p: p[0])
    pareto_nmse, pareto_l0 = [], []
    min_l0 = float("inf")
    for nmse, l0 in sorted_pts:
        if l0 < min_l0:
            pareto_nmse.append(nmse)
            pareto_l0.append(l0)
            min_l0 = l0
    ax.plot(pareto_nmse, pareto_l0, "--", alpha=0.5, color="tab:blue")

    if transcoder_results:
        tc_l0s = [r["l0"] for r in transcoder_results]
        tc_nmses = [r["nmse"] for r in transcoder_results]
        labels = [r.get("label", "Transcoder") for r in transcoder_results]
        for l0, nmse, label in zip(tc_l0s, tc_nmses, labels):
            ax.scatter([nmse], [l0], marker="x", s=100, label=label)

    ax.set_xlabel("Normalized MSE")
    ax.set_ylabel("L0 (Active Transforms / Features)")
    ax.set_title("Normalized MSE vs. L0")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved Pareto plot to {save_path}")
