#!/usr/bin/env python3
"""Generate training-curve and L0-vs-NMSE figures for experiment 13.

Loads each checkpoint in `out/checkpoints/`, reads the embedded `history`
list and `eval` dict, and writes:

  figures/train_<name>.png       — 4-panel curves (NMSE, L0, sparsity, MSE)
  figures/l0_vs_nmse.png         — final eval points overlaid on exp-11 sweep

Run:
    uv run python experiments/13_retrain_4x_molt_transcoder/plot_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = EXPERIMENT_DIR / "out" / "checkpoints"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
EXP11_RESULTS_DIR = EXPERIMENT_DIR.parent / "11_transcoder_comparison" / "results"

SLICES = [
    "molt_1x_lam1e-03",
    "molt_4x_lam1e-03",
    "tc_1x_lam1e+00",
    "tc_4x_lam3e+00",
]


def plot_single_run(name: str, history: list[dict]) -> None:
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "nmse", "NMSE", "#2563eb"),
        (axes[0, 1], "l0", "L0", "#dc2626"),
        (axes[1, 0], "sparsity_loss", "Sparsity Loss", "#16a34a"),
        (axes[1, 1], "mse", "MSE", "#9333ea"),
    ]
    for ax, key, ylabel, color in panels:
        if key in history[0]:
            ax.plot(steps, [h[key] for h in history], color=color, lw=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / f"train_{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out.relative_to(EXPERIMENT_DIR)}")


def load_exp11_points() -> tuple[list[dict], list[dict]]:
    """Load exp-11 sweep result_*.json files for the comparison scatter."""
    molt, tc = [], []
    if not EXP11_RESULTS_DIR.exists():
        return molt, tc
    for f in sorted(EXP11_RESULTS_DIR.glob("result_*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r["method"] == "molt":
            molt.append(r)
        elif r["method"] == "transcoder":
            tc.append(r)
    return molt, tc


def plot_l0_vs_nmse(points: list[dict]) -> None:
    """Scatter exp-13's 4 retrained points over exp-11's full sweep."""
    exp11_molt, exp11_tc = load_exp11_points()
    fig, ax = plt.subplots(figsize=(10, 7))
    scale_colors = {"1x": "tab:blue", "2x": "tab:orange", "4x": "tab:green"}

    # exp-11 background sweep (faded)
    for scale, color in scale_colors.items():
        m = [r for r in exp11_molt if r["scale"] == scale]
        t = [r for r in exp11_tc if r["scale"] == scale]
        if m:
            ax.scatter(
                [r["nmse"] for r in m], [r["l0"] for r in m],
                marker="o", s=40, color=color, alpha=0.25,
                label=f"exp 11 MOLT {scale}",
            )
        if t:
            ax.scatter(
                [r["nmse"] for r in t], [r["l0"] for r in t],
                marker="x", s=50, color=color, alpha=0.25, linewidths=1.5,
                label=f"exp 11 TC {scale}",
            )

    # exp-13's 4 retrained points (bold)
    for p in points:
        scale = p["scale"]
        color = scale_colors.get(scale, "black")
        marker = "o" if p["method"] == "molt" else "x"
        ax.scatter(
            [p["nmse"]], [p["l0"]],
            marker=marker, s=200, color=color,
            edgecolors="black", linewidths=1.5,
            label=f"exp 13 {p['name']}",
        )

    ax.set_xlabel("Normalized MSE", fontsize=12)
    ax.set_ylabel("L0 (active features per token)", fontsize=12)
    ax.set_title(
        "Exp 13 retrained checkpoints vs exp 11 sweep\n(GPT-2 layer 6)",
        fontsize=13,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "l0_vs_nmse.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(EXPERIMENT_DIR)}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    points = []
    for name in SLICES:
        ckpt = CHECKPOINTS_DIR / f"{name}.pt"
        if not ckpt.exists():
            print(f"  [skip] {name} — checkpoint missing")
            continue
        d = torch.load(ckpt, weights_only=False, map_location="cpu")
        plot_single_run(name, d["history"])
        points.append({
            "name": name,
            "method": d["method"] if d["method"] != "tc" else "transcoder",
            "scale": d["scale"],
            "l0": d["eval"]["l0"],
            "nmse": d["eval"]["nmse"],
        })
    plot_l0_vs_nmse(points)


if __name__ == "__main__":
    main()
