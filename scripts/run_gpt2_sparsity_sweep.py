#!/usr/bin/env python3
"""GPT-2: 4-way sweep {ReLU, JumpReLU} × {Tanh, L0} × {λ=1e-5, λ=1e-4}.

JumpReLU uses smooth surrogate backward + learned θ.
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

MODEL = "openai-community/gpt2"

SETUPS = []
for lam in [1e-5, 1e-4]:
    for sparsity in ["tanh", "l0"]:
        for act, learned in [("relu", False), ("jumprelu", True)]:
            name = f"{sparsity}_{act}_lam{lam:.0e}"
            SETUPS.append({
                "name": name,
                "sparsity_type": sparsity,
                "activation": act,
                "learned_threshold": learned,
                "sparsity_coeff": lam,
            })


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
        save_dir=f"checkpoints/gpt2_sparsity/{name}",
        log_every=100,
        sparsity_coeff=setup["sparsity_coeff"],
        sparsity_warmup_frac=0.1,
        activation=setup["activation"],
        sparsity_type=setup["sparsity_type"],
        learned_threshold=setup["learned_threshold"],
        jumprelu_threshold=0.0,
    )

    print(f"  activation={config.activation}, sparsity={config.sparsity_type}, "
          f"learned_θ={config.learned_threshold}, λ={config.sparsity_coeff}")

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
        "sparsity_coeff": setup["sparsity_coeff"],
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "final_threshold": round(final_theta, 4) if final_theta is not None else None,
        "num_active": len(active_transforms),
        "active_transforms": active_transforms,
    }

    results_dir = Path("results/gpt2_sparsity")
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

    target = sys.argv[1] if len(sys.argv) > 1 else None
    setups = [s for s in SETUPS if s["name"] == target] if target else SETUPS
    if target and not setups:
        print(f"Unknown: {target}. Options: {[s['name'] for s in SETUPS]}")
        sys.exit(1)

    all_results = []
    for setup in setups:
        all_results.append(run_setup(setup, mlp_inputs, mlp_outputs))

    results_dir = Path("results/gpt2_sparsity")
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY — GPT-2 Sparsity Penalty Sweep")
    print(f"{'='*80}")
    print(f"{'Setup':<35} {'λ':>8} {'L0':>6} {'NMSE':>8} {'θ':>8} {'#Act':>5}")
    print(f"{'-'*80}")
    for r in summary:
        theta_str = f"{r['final_threshold']:.4f}" if r['final_threshold'] is not None else " fixed"
        print(f"{r['name']:<35} {r['sparsity_coeff']:>8.0e} {r['l0']:>6.2f} {r['nmse']:>8.4f} {theta_str:>8} {r['num_active']:>5}")

    # Plots: one per lambda
    if len(all_results) >= 4:
        for lam in [1e-5, 1e-4]:
            lam_results = [r for r in all_results if r["sparsity_coeff"] == lam]
            if not lam_results:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"GPT-2 Sparsity Sweep — λ={lam:.0e}", fontsize=14)
            colors_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

            for i, r in enumerate(lam_results):
                h = r["history"]; steps = [e["step"] for e in h]
                c = colors_cycle[i % len(colors_cycle)]
                label = f"{r['sparsity_type']}+{r['activation']}"
                axes[0,0].plot(steps, [e["l0"] for e in h], lw=1.5, color=c, label=label)
                axes[0,1].plot(steps, [e["nmse"] for e in h], lw=1.5, color=c, label=label)
                axes[1,0].plot(steps, [e["mse"] for e in h], lw=1.5, color=c, label=label)
                if "threshold" in h[0]:
                    axes[1,1].plot(steps, [e["threshold"] for e in h], lw=1.5, color=c, label=label)

            axes[0,0].set_ylabel("L0"); axes[0,0].set_title("Active Transforms (L0)")
            axes[0,1].set_ylabel("NMSE"); axes[0,1].set_title("Normalized MSE"); axes[0,1].set_yscale("log")
            axes[1,0].set_ylabel("MSE"); axes[1,0].set_title("Raw MSE"); axes[1,0].set_yscale("log")
            axes[1,1].set_ylabel("θ"); axes[1,1].set_title("Learned Threshold θ")
            axes[1,1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            for ax in axes.flat:
                ax.grid(True, alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel("Step")
            plt.tight_layout()
            plot_path = results_dir / f"sparsity_sweep_lam{lam:.0e}.png"
            plt.savefig(plot_path, dpi=150)
            print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
