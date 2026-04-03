#!/usr/bin/env python3
"""Run JumpReLU experiments with learned vs fixed threshold (λ=0).

Uses cached activations from data/activations_2M.pt.

Setups:
  1. Tanh + JumpReLU (fixed θ=0.0)   — re-run baseline for fair comparison
  2. Tanh + JumpReLU (learned θ, init=0.0)
  3. L1 + JumpReLU (fixed θ=0.0)     — re-run for fair comparison
  4. L1 + JumpReLU (learned θ, init=0.0)
"""

import json
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from molt.config import MOLTConfig
from molt.train import train_molt
from molt.eval import compute_l0, compute_nmse


SETUPS = [
    {"name": "tanh_jumprelu_fixed",   "sparsity_type": "tanh", "learned_threshold": False},
    {"name": "tanh_jumprelu_learned", "sparsity_type": "tanh", "learned_threshold": True},
    {"name": "l1_jumprelu_fixed",     "sparsity_type": "l1",   "learned_threshold": False},
    {"name": "l1_jumprelu_learned",   "sparsity_type": "l1",   "learned_threshold": True},
]


def run_setup(setup: dict, mlp_inputs: torch.Tensor, mlp_outputs: torch.Tensor) -> dict:
    name = setup["name"]
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    config = MOLTConfig(
        num_tokens=2_000_000,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
        save_dir=f"checkpoints/learned_threshold/{name}",
        log_every=100,
        sparsity_coeff=0.0,
        sparsity_warmup_frac=0.0,
        activation="jumprelu",
        sparsity_type=setup["sparsity_type"],
        learned_threshold=setup["learned_threshold"],
        jumprelu_threshold=0.0,
    )

    print(f"  activation={config.activation}, sparsity_type={config.sparsity_type}, "
          f"learned_threshold={config.learned_threshold}, λ={config.sparsity_coeff}")

    model, history = train_molt(config, mlp_inputs, mlp_outputs)

    # Eval
    eval_in = mlp_inputs[-10000:]
    eval_out = mlp_outputs[-10000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)

    # Final threshold value
    final_threshold = model.threshold.item() if model.threshold is not None else 0.0

    print(f"  Final — L0: {l0:.2f}, NMSE: {nmse:.4f}, θ: {final_threshold:.4f}")

    # Per-transform activity
    active_transforms = []
    with torch.no_grad():
        x = eval_in[:512].cuda()
        _, aux = model(x)
        all_gates = torch.cat(aux["gate_acts"], dim=1)
        active = (all_gates > 0).float()
        freq = active.mean(dim=0)

        cumulative = 0
        for count, rank in config.rank_distribution:
            for j in range(count):
                f = freq[cumulative].item()
                if f > 0.001:
                    active_transforms.append({
                        "transform": cumulative,
                        "rank": rank,
                        "frequency": round(f * 100, 1),
                    })
                cumulative += 1

    result = {
        "name": name,
        "sparsity_type": setup["sparsity_type"],
        "learned_threshold": setup["learned_threshold"],
        "sparsity_coeff": 0.0,
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "final_threshold": round(final_threshold, 4),
        "active_transforms": active_transforms,
    }

    # Save per-run files
    results_dir = Path("results/learned_threshold")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"result_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    del model
    torch.cuda.empty_cache()

    return {**result, "history": history}


def main():
    # Load cached activations
    cache_path = "data/activations_2M.pt"
    print(f"Loading cached activations from {cache_path}")
    data = torch.load(cache_path, weights_only=True)
    mlp_inputs = data["mlp_inputs"]
    mlp_outputs = data["mlp_outputs"]
    del data
    print(f"Activations: {mlp_inputs.shape}")

    # Run specific setup or all
    if len(sys.argv) > 1:
        name = sys.argv[1]
        setup = next((s for s in SETUPS if s["name"] == name), None)
        if setup is None:
            print(f"Unknown setup: {name}. Options: {[s['name'] for s in SETUPS]}")
            sys.exit(1)
        run_setup(setup, mlp_inputs, mlp_outputs)
    else:
        all_results = []
        for setup in SETUPS:
            result = run_setup(setup, mlp_inputs, mlp_outputs)
            all_results.append(result)

        # Save combined summary
        results_dir = Path("results/learned_threshold")
        summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
        with open(results_dir / "sweep_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Plot comparison
        plot_comparison(all_results, results_dir)

        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Setup':<30} {'L0':>5} {'NMSE':>8} {'θ_final':>8} {'Transforms'}")
        print(f"{'-'*70}")
        for r in summary:
            transforms_str = ", ".join(
                f"T{t['transform']}(r{t['rank']},{t['frequency']}%)"
                for t in r["active_transforms"]
            )
            print(f"{r['name']:<30} {r['l0']:>5.2f} {r['nmse']:>8.4f} {r['final_threshold']:>8.4f} {transforms_str}")


def plot_comparison(all_results: list[dict], results_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learned vs Fixed JumpReLU Threshold (λ=0)", fontsize=14)

    colors = {
        "tanh_jumprelu_fixed": "tab:blue",
        "tanh_jumprelu_learned": "tab:cyan",
        "l1_jumprelu_fixed": "tab:orange",
        "l1_jumprelu_learned": "tab:red",
    }
    labels = {
        "tanh_jumprelu_fixed": "Tanh+JumpReLU (fixed θ)",
        "tanh_jumprelu_learned": "Tanh+JumpReLU (learned θ)",
        "l1_jumprelu_fixed": "L1+JumpReLU (fixed θ)",
        "l1_jumprelu_learned": "L1+JumpReLU (learned θ)",
    }

    for result in all_results:
        name = result["name"]
        history = result["history"]
        steps = [h["step"] for h in history]
        c = colors[name]
        label = labels[name]

        axes[0, 0].plot(steps, [h["l0"] for h in history], linewidth=1.5, color=c, label=label)
        axes[0, 1].plot(steps, [h["nmse"] for h in history], linewidth=1.5, color=c, label=label)
        axes[1, 0].plot(steps, [h["mse"] for h in history], linewidth=1.5, color=c, label=label)

        # Threshold trajectory (only for learned)
        if "threshold" in history[0]:
            axes[1, 1].plot(steps, [h["threshold"] for h in history], linewidth=1.5, color=c, label=label)

    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("L0")
    axes[0, 0].set_title("Active Transforms (L0)")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("NMSE")
    axes[0, 1].set_title("Normalized MSE")
    axes[0, 1].set_yscale("log"); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_title("Raw MSE")
    axes[1, 0].set_yscale("log"); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("θ")
    axes[1, 1].set_title("Learned Threshold θ")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plot_path = results_dir / "learned_threshold_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
