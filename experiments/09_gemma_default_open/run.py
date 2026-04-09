#!/usr/bin/env python3
"""Experiment 09: Measure initial gate openness for Gemma-3-1B MOLT.

Instantiates a fresh (untrained) MOLT with Gemma-3-1B config and measures
how many transform gates are open before any training update. Uses both
real cached Gemma activations and synthetic inputs at varying norms.

Usage:
    uv run python experiments/09_gemma_default_open/run.py
"""

import json
from pathlib import Path

import torch

from molt.config import MOLTConfig
from molt.model import MOLT

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
CACHE_PATH = "data/activations_2M.pt"


def measure_initial_gates(
    model: MOLT, config: MOLTConfig, x: torch.Tensor, label: str
) -> dict:
    """Measure gate statistics for a fresh MOLT on given inputs."""
    with torch.no_grad():
        _, aux = model.forward(x)

    result = {
        "label": label,
        "num_tokens": x.shape[0],
        "input_norm_mean": x.norm(dim=1).mean().item(),
        "input_norm_std": x.norm(dim=1).std().item(),
        "l0": aux["l0"].item(),
        "total_transforms": config.total_transforms,
        "groups": [],
    }

    for i, (gate, (n_t, rank)) in enumerate(
        zip(aux["gate_acts"], config.rank_distribution)
    ):
        active_per_token = (gate > 0).float().sum(dim=1)
        pre_acts = x @ model.groups[i].encoder.T - model.groups[i].bias

        group_info = {
            "group_idx": i,
            "num_transforms": n_t,
            "rank": rank,
            "active_per_token_mean": active_per_token.mean().item(),
            "active_per_token_min": active_per_token.min().item(),
            "active_per_token_max": active_per_token.max().item(),
            "fraction_open": (gate > 0).float().mean().item(),
            "pre_act_mean": pre_acts.mean().item(),
            "pre_act_std": pre_acts.std().item(),
            "pre_act_frac_positive": (pre_acts > 0).float().mean().item(),
        }
        result["groups"].append(group_info)

    return result


def print_result(result: dict):
    label = result["label"]
    print(f"\n=== {label} (norm~{result['input_norm_mean']:.1f}) ===")
    print(f"L0: {result['l0']:.2f} / {result['total_transforms']}")
    for g in result["groups"]:
        print(
            f"  Group {g['group_idx']} ({g['num_transforms']:2d}x rank{g['rank']:3d}): "
            f"active/token={g['active_per_token_mean']:.1f}/{g['num_transforms']} "
            f"({g['fraction_open']:.4f})  "
            f"pre_act: mean={g['pre_act_mean']:.3f} std={g['pre_act_std']:.3f}"
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    config = MOLTConfig.from_preset("google/gemma-3-1b-it", sparsity_coeff=0.0)
    model = MOLT(config)
    model.eval()

    print(f"Total transforms: {config.total_transforms}")
    print(f"Rank distribution: {config.rank_distribution}")
    print(f"Activation: {config.activation}, threshold: {config.jumprelu_threshold}")
    print("Bias init: -1.0 (hardcoded in TransformGroup._init_weights)")

    all_results = []

    # --- Real Gemma activations ---
    cache = Path(CACHE_PATH)
    if cache.exists():
        data = torch.load(cache, weights_only=True, map_location="cpu", mmap=True)
        x_real = data["mlp_inputs"][:256].clone()
        del data
        result = measure_initial_gates(model, config, x_real, "real_gemma_activations")
        print_result(result)
        all_results.append(result)
    else:
        print(f"\nSkipping real activations ({CACHE_PATH} not found)")

    # --- Synthetic inputs at varying norms ---
    x_unit = torch.randn(256, config.d_model)
    x_unit = x_unit / x_unit.norm(dim=1, keepdim=True)

    for target_norm, label in [
        (5.0, "synthetic_norm5"),
        (10.0, "synthetic_norm10"),
        (15.0, "synthetic_norm15_gemma_like"),
        (34.0, "synthetic_norm34_std_normal"),
        (50.0, "synthetic_norm50"),
    ]:
        x = x_unit * target_norm
        result = measure_initial_gates(model, config, x, label)
        print_result(result)
        all_results.append(result)

    # --- Save results ---
    out_path = RESULTS_DIR / "initial_gate_stats.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Initial gate openness before first training update")
    print("=" * 60)
    print(f"{'Input':35s} {'Norm':>6s} {'L0':>8s} {'Frac open':>10s}")
    print("-" * 60)
    for r in all_results:
        frac = r["l0"] / r["total_transforms"]
        print(
            f"{r['label']:35s} {r['input_norm_mean']:6.1f} "
            f"{r['l0']:5.2f}/31  {frac:8.1%}"
        )


if __name__ == "__main__":
    main()
