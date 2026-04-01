#!/usr/bin/env python3
"""Sweep sparsity coefficients to trace the L0 vs NMSE Pareto frontier."""

import argparse
import json
from pathlib import Path

import torch

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.eval import evaluate_molt, plot_pareto
from molt.train import train_molt


DEFAULT_LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep sparsity λ for MOLT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDAS,
        help="Sparsity coefficients to sweep",
    )
    parser.add_argument("--rank-multiplier", type=int, default=1, help="N value")
    parser.add_argument("--num-tokens", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/sweep")
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml

    config_dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)

    config_dict["rank_multiplier"] = args.rank_multiplier
    config_dict["num_tokens"] = args.num_tokens
    config_dict["device"] = args.device

    base_config = MOLTConfig(**config_dict)

    # Step 1: Collect activations once
    print("Streaming FineWeb data...")
    token_chunks = stream_fineweb_tokens(base_config)
    cache_path = f"data/activations_layer{base_config.layer_idx}.pt"
    mlp_inputs, mlp_outputs = collect_activations(base_config, token_chunks, cache_path)

    # Hold out eval set
    eval_size = min(10000, len(mlp_inputs) // 10)
    eval_inputs = mlp_inputs[-eval_size:]
    eval_outputs = mlp_outputs[-eval_size:]
    train_inputs = mlp_inputs[:-eval_size]
    train_outputs = mlp_outputs[:-eval_size]

    # Step 2: Train and evaluate for each lambda
    all_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lam in args.lambdas:
        print(f"\n{'='*60}")
        print(f"Training with λ={lam}")
        print(f"{'='*60}")

        config = MOLTConfig(**{**config_dict, "sparsity_coeff": lam})
        model, history = train_molt(config, train_inputs, train_outputs)

        # Evaluate
        results = evaluate_molt(model, eval_inputs, eval_outputs)
        results["lambda"] = lam
        results["rank_multiplier"] = args.rank_multiplier
        all_results.append(results)

        print(f"λ={lam}: L0={results['l0']:.1f}, NMSE={results['nmse']:.6f}")

        del model
        torch.cuda.empty_cache()

    # Save results and plot
    results_path = output_dir / f"sweep_N{args.rank_multiplier}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved sweep results to {results_path}")

    plot_path = output_dir / f"pareto_N{args.rank_multiplier}.png"
    plot_pareto(all_results, save_path=str(plot_path))


if __name__ == "__main__":
    main()
