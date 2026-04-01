#!/usr/bin/env python3
"""Train MOLT with λ=0 (no sparsity) to diagnose L0 collapse."""

import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from molt.config import MOLTConfig
from molt.data import stream_fineweb_tokens, collect_activations
from molt.train import train_molt
from molt.eval import compute_l0, compute_nmse


def main():
    config = MOLTConfig(
        num_tokens=2_000_000,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
        save_dir="checkpoints/no_sparsity",
        log_every=100,
        sparsity_coeff=0.0,
        sparsity_warmup_frac=0.0,
    )

    print(f"Config: N={config.rank_multiplier}, λ={config.sparsity_coeff}, transforms={config.total_transforms}")

    # Collect activations
    print("Streaming tokens...")
    token_chunks = stream_fineweb_tokens(config)
    print(f"Got {len(token_chunks)} chunks")
    mlp_inputs, mlp_outputs = collect_activations(config, token_chunks, cache_path=None)
    del token_chunks
    print(f"Activations: {mlp_inputs.shape}")

    # Train
    model, history = train_molt(config, mlp_inputs, mlp_outputs)

    # Eval
    eval_in = mlp_inputs[-10000:]
    eval_out = mlp_outputs[-10000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)
    print(f"\nL0: {l0:.2f}, NMSE: {nmse:.4f}")

    # Per-transform activity
    with torch.no_grad():
        x = eval_in[:512].cuda()
        _, aux = model(x)
        all_gates = torch.cat(aux["gate_acts"], dim=1)
        active = (all_gates > 0).float()
        freq = active.mean(dim=0)

        print(f"\nPer-transform activation frequency:")
        cumulative = 0
        for count, rank in config.rank_distribution:
            for j in range(count):
                f = freq[cumulative + j].item()
                if f > 0.001:
                    print(f"  T{cumulative+j:2d} (rank={rank:3d}): {f*100:.1f}%")
                cumulative += 1

    # Plot training curves
    steps = [h["step"] for h in history]
    l0s = [h["l0"] for h in history]
    nmses = [h["nmse"] for h in history]
    mses = [h["mse"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MOLT Training with λ=0 (No Sparsity Penalty)", fontsize=14)

    axes[0].plot(steps, l0s, linewidth=1.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("L0")
    axes[0].set_title("Active Transforms (L0)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, nmses, linewidth=1.5, color="tab:orange")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("NMSE")
    axes[1].set_title("Normalized MSE")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, mses, linewidth=1.5, color="tab:green")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Raw MSE")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_no_sparsity.png", dpi=150)
    print("\nSaved plot to results/training_no_sparsity.png")

    # Save history
    with open("results/history_no_sparsity.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
