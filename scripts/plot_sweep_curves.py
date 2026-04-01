"""Plot training curves for all lambda values on shared axes."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

# Collect all history files
history_files = sorted(
    Path("checkpoints").rglob("history_*.json"),
    key=lambda p: float(re.search(r"lam(.+)\.json", p.name).group(1)),
)

# Load data keyed by lambda
runs = {}
for f in history_files:
    lam = float(re.search(r"lam(.+)\.json", f.name).group(1))
    with open(f) as fh:
        runs[lam] = json.load(fh)

# Color map: log-spaced lambdas mapped to a colormap
lambdas = sorted(runs.keys())
colors = cm.viridis(np.linspace(0, 0.9, len(lambdas)))

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("MOLT Training Curves Across λ Values  (N=1, Gemma-3-1B Layer 13)", fontsize=15, fontweight="bold")

metrics = [
    ("nmse", "NMSE", axes[0, 0]),
    ("l0", "L0 (active transforms)", axes[0, 1]),
    ("sparsity_loss", "Sparsity Loss", axes[1, 0]),
    ("mse", "Raw MSE", axes[1, 1]),
]

for metric_key, ylabel, ax in metrics:
    for lam, color in zip(lambdas, colors):
        history = runs[lam]
        steps = [h["step"] for h in history]
        vals = [h[metric_key] for h in history]
        label = f"λ={lam}" if lam > 0 else "λ=0 (no sparsity)"
        ax.plot(steps, vals, color=color, linewidth=1.2, alpha=0.85, label=label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Set y-limits to useful ranges (clip outlier spikes)
axes[0, 0].set_ylim(0, 1.5)    # NMSE
axes[0, 1].set_ylim(0, 6)      # L0
axes[1, 0].set_ylim(0, 0.8)    # Sparsity Loss
axes[1, 1].set_ylim(0, 0.01)   # Raw MSE

# Single shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.01, 0.5), fontsize=10, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.savefig("results/sweep_training_curves.png", dpi=150, bbox_inches="tight")
print("Saved to results/sweep_training_curves.png")
