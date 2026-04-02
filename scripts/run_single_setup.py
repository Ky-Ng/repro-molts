#!/usr/bin/env python3
"""Train a single MOLT setup with λ=0 using cached activations."""

import json
import sys
import torch
from pathlib import Path

from molt.config import MOLTConfig
from molt.data import make_dataloader
from molt.train import train_molt
from molt.eval import compute_l0, compute_nmse


def main(name: str, sparsity_type: str, activation: str):
    cache_path = "data/activations_2M.pt"
    print(f"Loading cached activations from {cache_path}")
    data = torch.load(cache_path, weights_only=True)
    mlp_inputs = data["mlp_inputs"]
    mlp_outputs = data["mlp_outputs"]
    print(f"Activations: {mlp_inputs.shape}")

    config = MOLTConfig(
        num_tokens=2_000_000,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
        save_dir=f"checkpoints/activation_sweep/{name}",
        log_every=100,
        sparsity_coeff=0.0,
        sparsity_warmup_frac=0.0,
        activation=activation,
        sparsity_type=sparsity_type,
    )

    print(f"Config: activation={config.activation}, sparsity_type={config.sparsity_type}, "
          f"λ={config.sparsity_coeff}, transforms={config.total_transforms}")

    model, history = train_molt(config, mlp_inputs, mlp_outputs)

    # Eval
    eval_in = mlp_inputs[-10000:]
    eval_out = mlp_outputs[-10000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)
    print(f"\nFinal — L0: {l0:.2f}, NMSE: {nmse:.4f}")

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
        "sparsity_type": sparsity_type,
        "activation": activation,
        "sparsity_coeff": 0.0,
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "active_transforms": active_transforms,
    }

    results_dir = Path("results/activation_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"result_{name}.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(results_dir / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResult: {json.dumps(result, indent=2)}")
    print(f"Saved to {results_dir / f'result_{name}.json'}")


if __name__ == "__main__":
    setups = {
        "tanh_relu":   ("tanh", "relu"),
        "l1_relu":     ("l1",   "relu"),
        "l1_jumprelu": ("l1",   "jumprelu"),
    }
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name not in setups:
        print(f"Usage: {sys.argv[0]} <{'|'.join(setups.keys())}>")
        sys.exit(1)
    sparsity_type, activation = setups[name]
    main(name, sparsity_type, activation)
