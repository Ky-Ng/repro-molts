#!/usr/bin/env python3
"""Train a MOLT model on Gemma MLP activations from FineWeb data."""

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.train import train_molt


def parse_args():
    parser = argparse.ArgumentParser(description="Train MOLT")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config YAML path"
    )
    # Allow overriding any config field from CLI
    for f in fields(MOLTConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base config from YAML
    config_dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)

    # Override with CLI args
    for f in fields(MOLTConfig):
        cli_val = getattr(args, f.name, None)
        if cli_val is not None:
            config_dict[f.name] = cli_val

    config = MOLTConfig(**config_dict)
    print(f"Config: N={config.rank_multiplier}, λ={config.sparsity_coeff}, "
          f"total_transforms={config.total_transforms}")

    # Step 1: Stream and tokenize data
    print("Streaming FineWeb data...")
    token_chunks = stream_fineweb_tokens(config)
    print(f"Collected {len(token_chunks)} chunks ({len(token_chunks) * config.seq_len} tokens)")

    # Step 2: Collect MLP activations
    cache_path = f"data/activations_layer{config.layer_idx}.pt"
    mlp_inputs, mlp_outputs = collect_activations(config, token_chunks, cache_path)
    print(f"Activations shape: {mlp_inputs.shape}")

    # Step 3: Train MOLT
    model, history = train_molt(config, mlp_inputs, mlp_outputs)

    final = history[-1] if history else {}
    print(f"Training complete. Final metrics: {final}")


if __name__ == "__main__":
    main()
