"""Plot individual 4-panel training curves for each lambda value."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

history_files = sorted(
    Path("checkpoints").rglob("history_*.json"),
    key=lambda p: float(re.search(r"lam(.+)\.json", p.name).group(1)),
)

for f in history_files:
    lam = float(re.search(r"lam(.+)\.json", f.name).group(1))
    lam_str = re.search(r"lam(.+)\.json", f.name).group(1)

    with open(f) as fh:
        history = json.load(fh)

    steps = [h["step"] for h in history]
    nmse = [h["nmse"] for h in history]
    l0 = [h["l0"] for h in history]
    sparsity = [h["sparsity_loss"] for h in history]
    mse = [h["mse"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"MOLT Training  (N=1, λ={lam_str})"
    if lam == 0.0:
        title = "MOLT Training  (N=1, λ=0 — no sparsity)"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    panels = [
        (axes[0, 0], nmse, "NMSE", "#2563eb"),
        (axes[0, 1], l0, "L0 (active transforms)", "#dc2626"),
        (axes[1, 0], sparsity, "Sparsity Loss", "#16a34a"),
        (axes[1, 1], mse, "Raw MSE", "#9333ea"),
    ]

    for ax, vals, ylabel, color in panels:
        ax.plot(steps, vals, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    out = f"results/training_curve_lam{lam_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out}")

print("\nDone!")
