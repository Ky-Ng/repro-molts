#!/usr/bin/env python3
"""Evaluate a trained MOLT: Jacobian faithfulness, L0, NMSE."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from molt.eval import evaluate_molt
from molt.train import load_molt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOLT")
    parser.add_argument("--checkpoint", type=str, required=True, help="MOLT checkpoint path")
    parser.add_argument("--activations", type=str, required=True, help="Cached activations path")
    parser.add_argument("--jacobian", action="store_true", help="Compute Jacobian faithfulness")
    parser.add_argument("--jacobian-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load MOLT
    model, config = load_molt(args.checkpoint, args.device)
    print(f"Loaded MOLT: N={config.rank_multiplier}, λ={config.sparsity_coeff}")

    # Load activations
    data = torch.load(args.activations, weights_only=True)
    x = data["mlp_inputs"][: args.eval_samples]
    target = data["mlp_outputs"][: args.eval_samples]

    # Optionally build MLP function for Jacobian
    mlp_fn = None
    if args.jacobian:
        print(f"Loading {config.model_name} for Jacobian comparison...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map=args.device,
        )
        base_model.eval()
        mlp_layer = base_model.model.layers[config.layer_idx].mlp

        def mlp_fn(xi):
            return mlp_layer(xi.unsqueeze(0)).squeeze(0)

    results = evaluate_molt(
        model,
        x,
        target,
        mlp_fn=mlp_fn,
        jacobian_samples=args.jacobian_samples,
    )

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    if args.jacobian:
        del base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
