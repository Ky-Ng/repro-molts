#!/usr/bin/env python3
"""Generalized activation/sparsity sweep for any model preset.

Usage:
  python scripts/run_model_sweep.py <model_name> [setup_name]

Examples:
  python scripts/run_model_sweep.py openai-community/gpt2          # all 4 setups
  python scripts/run_model_sweep.py openai-community/gpt2 tanh_relu  # single setup
  python scripts/run_model_sweep.py google/gemma-3-1b-it            # all 4 setups

Runs 4 configurations: {Tanh, L1} × {ReLU, JumpReLU} with λ=0.
Uses cached activations when available.
"""

import json
import sys
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
    {"name": "tanh_jumprelu", "sparsity_type": "tanh", "activation": "jumprelu"},
    {"name": "l1_relu",       "sparsity_type": "l1",   "activation": "relu"},
    {"name": "l1_jumprelu",   "sparsity_type": "l1",   "activation": "jumprelu"},
]


def model_slug(model_name: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return model_name.replace("/", "_").replace("-", "_")


def run_setup(
    setup: dict,
    model_name: str,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    results_dir: Path,
    num_tokens: int,
) -> dict:
    name = setup["name"]
    slug = model_slug(model_name)

    print(f"\n{'='*60}")
    print(f"Running: {name} on {model_name}")
    print(f"{'='*60}")

    config = MOLTConfig.from_preset(
        model_name,
        num_tokens=num_tokens,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
        save_dir=f"checkpoints/{slug}/{name}",
        log_every=100,
        sparsity_coeff=0.0,
        sparsity_warmup_frac=0.0,
        activation=setup["activation"],
        sparsity_type=setup["sparsity_type"],
    )

    print(f"  d_model={config.d_model}, transforms={config.total_transforms}, "
          f"activation={config.activation}, sparsity_type={config.sparsity_type}, λ=0.0")

    model, history = train_molt(config, mlp_inputs, mlp_outputs)

    # Eval
    eval_in = mlp_inputs[-10000:]
    eval_out = mlp_outputs[-10000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)
    print(f"  Final — L0: {l0:.2f}, NMSE: {nmse:.4f}")

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
        "model": model_name,
        "d_model": config.d_model,
        "sparsity_type": setup["sparsity_type"],
        "activation": setup["activation"],
        "sparsity_coeff": 0.0,
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "active_transforms": active_transforms,
    }

    with open(results_dir / f"result_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    del model
    torch.cuda.empty_cache()

    return {**result, "history": history}


def plot_comparison(all_results: list[dict], model_name: str, results_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Activation/Sparsity Sweep — {model_name} (λ=0)", fontsize=14)

    colors = {
        "tanh_relu": "tab:blue",
        "tanh_jumprelu": "tab:cyan",
        "l1_relu": "tab:orange",
        "l1_jumprelu": "tab:red",
    }

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
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Step"); axes[1].set_ylabel("NMSE"); axes[1].set_title("Normalized MSE")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Step"); axes[2].set_ylabel("MSE"); axes[2].set_title("Raw MSE")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = results_dir / "sweep_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_name> [setup_name]")
        print(f"  Models: {list(MOLTConfig.__dataclass_fields__.keys())}")
        sys.exit(1)

    model_name = sys.argv[1]
    setup_filter = sys.argv[2] if len(sys.argv) > 2 else None

    slug = model_slug(model_name)
    num_tokens = 2_000_000

    # Load or collect activations
    cache_path = f"data/activations_{slug}_2M.pt"
    base_config = MOLTConfig.from_preset(model_name, num_tokens=num_tokens, device="cuda")

    if Path(cache_path).exists():
        print(f"Loading cached activations from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        mlp_inputs = data["mlp_inputs"]
        mlp_outputs = data["mlp_outputs"]
        del data
    else:
        print(f"Collecting activations for {model_name}...")
        Path("data").mkdir(parents=True, exist_ok=True)
        token_chunks = stream_fineweb_tokens(base_config)
        print(f"Got {len(token_chunks)} chunks")
        mlp_inputs, mlp_outputs = collect_activations(base_config, token_chunks, cache_path=cache_path)
        del token_chunks
    print(f"Activations: {mlp_inputs.shape}")

    results_dir = Path(f"results/{slug}")
    results_dir.mkdir(parents=True, exist_ok=True)

    setups = SETUPS
    if setup_filter:
        setups = [s for s in SETUPS if s["name"] == setup_filter]
        if not setups:
            print(f"Unknown setup: {setup_filter}. Options: {[s['name'] for s in SETUPS]}")
            sys.exit(1)

    all_results = []
    for setup in setups:
        result = run_setup(setup, model_name, mlp_inputs, mlp_outputs, results_dir, num_tokens)
        all_results.append(result)

    # Save combined summary
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot if we ran all setups
    if len(all_results) == len(SETUPS):
        plot_comparison(all_results, model_name, results_dir)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY — {model_name}")
    print(f"{'='*70}")
    print(f"{'Setup':<20} {'L0':>5} {'NMSE':>8} {'Transforms'}")
    print(f"{'-'*70}")
    for r in summary:
        transforms_str = ", ".join(
            f"T{t['transform']}(r{t['rank']},{t['frequency']}%)"
            for t in r["active_transforms"]
        ) or "None"
        print(f"{r['name']:<20} {r['l0']:>5.2f} {r['nmse']:>8.4f} {transforms_str}")


if __name__ == "__main__":
    main()
