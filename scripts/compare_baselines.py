#!/usr/bin/env python3
"""Compare MOLTs against Transcoder baselines (skip and non-skip)."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from molt.config import MOLTConfig
from molt.eval import evaluate_molt, plot_pareto
from molt.train import load_molt
from molt.transcoder import evaluate_transcoder


# Gemma Scope 2 transcoder releases for 1B model
TRANSCODER_CONFIGS = [
    {
        "release": "gemma-scope-2-1b-pt-transcoder",
        "sae_id": "layer_13/width_16k",
        "label": "Transcoder 16k (non-skip)",
    },
    {
        "release": "gemma-scope-2-1b-pt-transcoder-skip",
        "sae_id": "layer_13/width_16k",
        "label": "Transcoder 16k (skip)",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MOLT vs Transcoders")
    parser.add_argument(
        "--molt-results",
        type=str,
        required=True,
        help="Path to MOLT sweep results JSON",
    )
    parser.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to cached activations",
    )
    parser.add_argument("--jacobian", action="store_true")
    parser.add_argument("--jacobian-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--layer-idx", type=int, default=13)
    parser.add_argument("--output-dir", type=str, default="results/comparison")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load cached activations
    data = torch.load(args.activations, weights_only=True)
    x = data["mlp_inputs"][: args.eval_samples]
    target = data["mlp_outputs"][: args.eval_samples]

    # Optionally load base model for Jacobian
    mlp_fn = None
    base_model = None
    if args.jacobian:
        print(f"Loading {args.model_name} for Jacobian comparison...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map=args.device,
        )
        base_model.eval()
        mlp_layer = base_model.model.layers[args.layer_idx].mlp

        def mlp_fn(xi):
            return mlp_layer(xi.unsqueeze(0)).squeeze(0)

    # Load MOLT sweep results
    with open(args.molt_results) as f:
        molt_results = json.load(f)

    # Evaluate transcoders
    print("\n=== Evaluating Transcoders ===")
    transcoder_results = []
    for tc_config in TRANSCODER_CONFIGS:
        print(f"\nEvaluating {tc_config['label']}...")
        try:
            tc_result = evaluate_transcoder(
                release=tc_config["release"],
                sae_id=tc_config["sae_id"],
                x=x,
                target=target,
                mlp_fn=mlp_fn,
                jacobian_samples=args.jacobian_samples,
                device=args.device,
                label=tc_config["label"],
            )
            transcoder_results.append(tc_result)
            print(f"  L0={tc_result['l0']:.1f}, NMSE={tc_result['nmse']:.6f}")
            if "jacobian_cosine_sim" in tc_result:
                print(f"  Jacobian cos sim={tc_result['jacobian_cosine_sim']:.4f}")
        except Exception as e:
            print(f"  Failed to load {tc_config['label']}: {e}")

    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "molt": molt_results,
        "transcoders": transcoder_results,
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Plot combined Pareto
    plot_pareto(
        molt_results,
        transcoder_results=transcoder_results,
        save_path=str(output_dir / "pareto_comparison.png"),
    )

    # Print summary table
    print("\n=== Comparison Summary ===")
    print(f"{'Method':<35} {'L0':>8} {'NMSE':>12} {'Jac. Sim':>10}")
    print("-" * 65)
    for r in molt_results:
        lam = r.get("lambda", "?")
        jac = f"{r.get('jacobian_cosine_sim', 'N/A'):.4f}" if "jacobian_cosine_sim" in r else "N/A"
        print(f"MOLT (λ={lam}){'':<20} {r['l0']:>8.1f} {r['nmse']:>12.6f} {jac:>10}")
    for r in transcoder_results:
        jac = f"{r.get('jacobian_cosine_sim', 'N/A'):.4f}" if "jacobian_cosine_sim" in r else "N/A"
        print(f"{r['label']:<35} {r['l0']:>8.1f} {r['nmse']:>12.6f} {jac:>10}")

    if base_model is not None:
        del base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
