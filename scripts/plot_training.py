"""Plot training curves from the first training run (N=1, λ=0.001)."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("checkpoints/full_N1/history_N1_lam0.001.json") as f:
    history = json.load(f)

steps = [h["step"] for h in history]
nmse = [h["nmse"] for h in history]
l0 = [h["l0"] for h in history]
sparsity = [h["sparsity_loss"] for h in history]
mse = [h["mse"] for h in history]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MOLT Training Run  (N=1, λ=0.001, 10M tokens)", fontsize=15, fontweight="bold")

# NMSE
ax = axes[0, 0]
ax.plot(steps, nmse, color="#2563eb", linewidth=1.5)
ax.set_ylabel("NMSE")
ax.set_xlabel("Step")
ax.set_title("Normalized MSE")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# L0
ax = axes[0, 1]
ax.plot(steps, l0, color="#dc2626", linewidth=1.5)
ax.set_ylabel("L0 (active transforms)")
ax.set_xlabel("Step")
ax.set_title("L0 Sparsity")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Sparsity Loss
ax = axes[1, 0]
ax.plot(steps, sparsity, color="#16a34a", linewidth=1.5)
ax.set_ylabel("Sparsity Loss")
ax.set_xlabel("Step")
ax.set_title("Sparsity Loss (tanh-weighted L2)")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# MSE
ax = axes[1, 1]
ax.plot(steps, mse, color="#9333ea", linewidth=1.5)
ax.set_ylabel("MSE")
ax.set_xlabel("Step")
ax.set_title("Raw MSE")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig("results/training_curves_N1.png", dpi=150, bbox_inches="tight")
print("Saved to results/training_curves_N1.png")
