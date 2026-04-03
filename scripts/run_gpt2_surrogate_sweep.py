#!/usr/bin/env python3
"""GPT-2 sweep: ReLU and JumpReLU (smooth surrogate, learned θ) × {L0, Tanh}.

Uses cached activations from data/activations_openai_community_gpt2_2M.pt.
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
    {"name": "tanh_relu",              "sparsity_type": "tanh", "activation": "relu",     "learned_threshold": False},
    {"name": "tanh_jumprelu_learned",  "sparsity_type": "tanh", "activation": "jumprelu", "learned_threshold": True},
    {"name": "l0_relu",               "sparsity_type": "l0",   "activation": "relu",     "learned_threshold": False},
    {"name": "l0_jumprelu_learned",   "sparsity_type": "l0",   "activation": "jumprelu", "learned_threshold": True},
]

MODEL = "openai-community/gpt2"


def run_setup(setup, mlp_inputs, mlp_outputs):
    name = setup["name"]
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    config = MOLTConfig.from_preset(
        MODEL,
        num_tokens=2_000_000,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
        save_dir=f"checkpoints/gpt2_surrogate/{name}",
        log_every=100,
        sparsity_coeff=0.0,
        sparsity_warmup_frac=0.0,
        activation=setup["activation"],
        sparsity_type=setup["sparsity_type"],
        learned_threshold=setup["learned_threshold"],
        jumprelu_threshold=0.0,
    )

    print(f"  activation={config.activation}, sparsity_type={config.sparsity_type}, "
          f"learned_θ={config.learned_threshold}, λ=0.0")

    model, history = train_molt(config, mlp_inputs, mlp_outputs)

    eval_in = mlp_inputs[-10000:]
    eval_out = mlp_outputs[-10000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)
    final_theta = model.threshold.item() if model.threshold is not None else None

    print(f"  Final — L0: {l0:.2f}, NMSE: {nmse:.4f}" +
          (f", θ: {final_theta:.4f}" if final_theta is not None else ""))

    # Per-transform activity
    active_transforms = []
    with torch.no_grad():
        x = eval_in[:512].cuda()
        _, aux = model(x)
        all_gates = torch.cat(aux["gate_acts"], dim=1)
        freq = (all_gates > 0).float().mean(dim=0)
        cumulative = 0
        for count, rank in config.rank_distribution:
            for j in range(count):
                f = freq[cumulative].item()
                if f > 0.001:
                    active_transforms.append({"transform": cumulative, "rank": rank, "frequency": round(f * 100, 1)})
                cumulative += 1

    result = {
        "name": name,
        "activation": setup["activation"],
        "sparsity_type": setup["sparsity_type"],
        "learned_threshold": setup["learned_threshold"],
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "final_threshold": round(final_theta, 4) if final_theta is not None else None,
        "active_transforms": active_transforms,
    }

    results_dir = Path("results/gpt2_surrogate")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"result_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return {**result, "history": history}


def main():
    cache_path = "data/activations_openai_community_gpt2_2M.pt"
    print(f"Loading cached activations from {cache_path}")
    data = torch.load(cache_path, weights_only=True)
    mlp_inputs, mlp_outputs = data["mlp_inputs"], data["mlp_outputs"]
    del data
    print(f"Activations: {mlp_inputs.shape}")

    target_name = sys.argv[1] if len(sys.argv) > 1 else None
    setups = [s for s in SETUPS if s["name"] == target_name] if target_name else SETUPS
    if target_name and not setups:
        print(f"Unknown: {target_name}. Options: {[s['name'] for s in SETUPS]}")
        sys.exit(1)

    all_results = []
    for setup in setups:
        all_results.append(run_setup(setup, mlp_inputs, mlp_outputs))

    results_dir = Path("results/gpt2_surrogate")
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*75}")
    print(f"SUMMARY — GPT-2 Smooth Surrogate Sweep (λ=0)")
    print(f"{'='*75}")
    print(f"{'Setup':<30} {'L0':>6} {'NMSE':>8} {'θ':>8} {'#Active':>7}")
    print(f"{'-'*75}")
    for r in summary:
        theta_str = f"{r['final_threshold']:.4f}" if r['final_threshold'] is not None else "  fixed"
        print(f"{r['name']:<30} {r['l0']:>6.2f} {r['nmse']:>8.4f} {theta_str:>8} {len(r['active_transforms']):>7}")

    # Plot
    if len(all_results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("GPT-2: Smooth Surrogate JumpReLU Sweep (λ=0)", fontsize=14)
        colors = {"tanh_relu": "tab:blue", "tanh_jumprelu_learned": "tab:cyan",
                  "l0_relu": "tab:orange", "l0_jumprelu_learned": "tab:red"}

        for r in all_results:
            h = r["history"]; steps = [e["step"] for e in h]
            c = colors.get(r["name"], "gray"); label = r["name"]
            axes[0,0].plot(steps, [e["l0"] for e in h], lw=1.5, color=c, label=label)
            axes[0,1].plot(steps, [e["nmse"] for e in h], lw=1.5, color=c, label=label)
            axes[1,0].plot(steps, [e["mse"] for e in h], lw=1.5, color=c, label=label)
            if "threshold" in h[0]:
                axes[1,1].plot(steps, [e["threshold"] for e in h], lw=1.5, color=c, label=label)

        axes[0,0].set_ylabel("L0"); axes[0,0].set_title("Active Transforms")
        axes[0,1].set_ylabel("NMSE"); axes[0,1].set_title("Normalized MSE"); axes[0,1].set_yscale("log")
        axes[1,0].set_ylabel("MSE"); axes[1,0].set_title("Raw MSE"); axes[1,0].set_yscale("log")
        axes[1,1].set_ylabel("θ"); axes[1,1].set_title("Learned Threshold")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=7); ax.set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(results_dir / "surrogate_sweep.png", dpi=150)
        print(f"Saved plot to {results_dir / 'surrogate_sweep.png'}")


if __name__ == "__main__":
    main()
