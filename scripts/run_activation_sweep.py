#!/usr/bin/env python3
"""Sweep activation/sparsity setups with λ=0 to compare gating dynamics.

Runs 3 configurations:
  1. Tanh + ReLU
  2. L1 + ReLU
  3. L1 + JumpReLU

All with sparsity_coeff=0.0 so sparsity type is irrelevant for training,
but infrastructure is set up for future λ>0 sweeps.
"""

import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from molt.config import MOLTConfig
from molt.data import stream_fineweb_tokens, collect_activations
from molt.train import train_molt
from molt.eval import compute_l0, compute_nmse


SETUPS = [
    {"name": "tanh_relu",     "sparsity_type": "tanh", "activation": "relu"},
    {"name": "l1_relu",       "sparsity_type": "l1",   "activation": "relu"},
    {"name": "l1_jumprelu",   "sparsity_type": "l1",   "activation": "jumprelu"},
]


def main():
    # Collect activations once (shared across all runs)
    base_config = MOLTConfig(
        num_tokens=2_000_000,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
    )

    cache_path = "data/activations_2M.pt"
    Path("data").mkdir(parents=True, exist_ok=True)

    print("Streaming tokens...")
    token_chunks = stream_fineweb_tokens(base_config)
    print(f"Got {len(token_chunks)} chunks")
    mlp_inputs, mlp_outputs = collect_activations(base_config, token_chunks, cache_path=cache_path)
    del token_chunks
    print(f"Activations: {mlp_inputs.shape}")

    eval_in = mlp_inputs[-10000:]
    eval_out = mlp_outputs[-10000:]

    results_dir = Path("results/activation_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for setup in SETUPS:
        name = setup["name"]
        print(f"\n{'='*60}")
        print(f"Running: {name} (sparsity_type={setup['sparsity_type']}, activation={setup['activation']})")
        print(f"{'='*60}")

        config = MOLTConfig(
            num_tokens=2_000_000,
            batch_size=64,
            device="cuda",
            wandb_enabled=False,
            save_dir=f"checkpoints/activation_sweep/{name}",
            log_every=100,
            sparsity_coeff=0.0,
            sparsity_warmup_frac=0.0,
            activation=setup["activation"],
            sparsity_type=setup["sparsity_type"],
        )

        print(f"Config: activation={config.activation}, sparsity_type={config.sparsity_type}, "
              f"λ={config.sparsity_coeff}, transforms={config.total_transforms}")

        model, history = train_molt(config, mlp_inputs, mlp_outputs)

        # Eval
        l0 = compute_l0(model, eval_in)
        nmse = compute_nmse(model, eval_in, eval_out)
        print(f"  L0: {l0:.2f}, NMSE: {nmse:.4f}")

        # Per-transform activity
        with torch.no_grad():
            x = eval_in[:512].cuda()
            _, aux = model(x)
            all_gates = torch.cat(aux["gate_acts"], dim=1)
            active = (all_gates > 0).float()
            freq = active.mean(dim=0)

            active_transforms = []
            cumulative = 0
            for count, rank in config.rank_distribution:
                for j in range(count):
                    f = freq[cumulative + j].item()
                    if f > 0.001:
                        active_transforms.append({
                            "transform": cumulative + j,
                            "rank": rank,
                            "frequency": round(f * 100, 1),
                        })
                    cumulative += 1

        result = {
            "name": name,
            "sparsity_type": setup["sparsity_type"],
            "activation": setup["activation"],
            "sparsity_coeff": 0.0,
            "l0": round(l0, 2),
            "nmse": round(nmse, 4),
            "active_transforms": active_transforms,
            "history": history,
        }
        all_results.append(result)

        # Save per-run history
        with open(results_dir / f"history_{name}.json", "w") as f:
            json.dump(history, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Save combined results (without full history for readability)
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {results_dir / 'sweep_results.json'}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Activation/Sparsity Setup Sweep (λ=0)", fontsize=14)

    colors = {"tanh_relu": "tab:blue", "l1_relu": "tab:orange", "l1_jumprelu": "tab:green"}

    for result in all_results:
        name = result["name"]
        history = result["history"]
        steps = [h["step"] for h in history]
        c = colors[name]
        label = f"{result['sparsity_type']}+{result['activation']}"

        axes[0].plot(steps, [h["l0"] for h in history], linewidth=1.5, color=c, label=label)
        axes[1].plot(steps, [h["nmse"] for h in history], linewidth=1.5, color=c, label=label)
        axes[2].plot(steps, [h["mse"] for h in history], linewidth=1.5, color=c, label=label)

    axes[0].set_xlabel("Step"); axes[0].set_ylabel("L0"); axes[0].set_title("Active Transforms (L0)")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].set_xlabel("Step"); axes[1].set_ylabel("NMSE"); axes[1].set_title("Normalized MSE")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3); axes[1].legend()

    axes[2].set_xlabel("Step"); axes[2].set_ylabel("MSE"); axes[2].set_title("Raw MSE")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3); axes[2].legend()

    plt.tight_layout()
    plot_path = results_dir / "activation_sweep_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to {plot_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Setup':<20} {'L0':>6} {'NMSE':>8} {'Winner Transform(s)'}")
    print(f"{'-'*60}")
    for r in summary:
        transforms_str = ", ".join(
            f"T{t['transform']}(r{t['rank']},{t['frequency']}%)"
            for t in r["active_transforms"]
        )
        print(f"{r['name']:<20} {r['l0']:>6.2f} {r['nmse']:>8.4f} {transforms_str}")


if __name__ == "__main__":
    main()
