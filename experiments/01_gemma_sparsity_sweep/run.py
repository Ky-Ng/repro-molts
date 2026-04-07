#!/usr/bin/env python3
"""Parallel sparsity sweep on Gemma-3-1B — trains multiple lambda values concurrently.

Collects activations once into shared memory, then spawns concurrent training
workers via ThreadPoolExecutor.

Usage:
    uv run python experiments/01_gemma_sparsity_sweep/run.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.eval import compute_l0, compute_nmse
from molt.train import train_molt
from molt.utils.plotting import plot_l0_vs_nmse

EXPERIMENT_DIR = Path(__file__).parent
LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def train_single_lambda(lam, train_inputs, train_outputs, eval_inputs, eval_outputs, base_config_dict):
    config = MOLTConfig(**{**base_config_dict, "sparsity_coeff": lam})

    print(f"[lambda={lam:.0e}] Starting training (transforms={config.total_transforms})")
    t0 = time.time()

    save_dir = EXPERIMENT_DIR / "results"
    model, history = train_molt(config, train_inputs, train_outputs, save_dir=str(save_dir))

    l0 = compute_l0(model, eval_inputs)
    nmse = compute_nmse(model, eval_inputs, eval_outputs)
    elapsed = time.time() - t0
    print(f"[lambda={lam:.0e}] Done in {elapsed:.0f}s -- L0={l0:.1f}, NMSE={nmse:.4f}")

    with torch.no_grad():
        sample = eval_inputs[:256].to(config.device)
        _, aux = model(sample)
        all_gates = torch.cat(aux["gate_acts"], dim=1)
        per_transform_freq = (all_gates > 0).float().mean(dim=0).cpu().tolist()

    result = {
        "lambda": lam,
        "l0": l0,
        "nmse": nmse,
        "rank_multiplier": config.rank_multiplier,
        "training_time_s": elapsed,
        "final_mse": history[-1]["mse"] if history else None,
        "transform_frequencies": per_transform_freq,
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    max_workers = int(os.environ.get("MOLT_WORKERS", "3"))
    num_tokens = int(os.environ.get("MOLT_TOKENS", "10000000"))

    print(f"Parallel sweep: {len(LAMBDAS)} lambda values, {max_workers} workers, {num_tokens/1e6:.0f}M tokens")

    config = MOLTConfig(
        num_tokens=num_tokens,
        batch_size=64,
        device="cuda",
        log_every=2000,
        sparsity_warmup_frac=0.1,
    )
    base_config_dict = vars(config)

    # Collect activations
    print(f"\n=== Collecting activations ({num_tokens/1e6:.0f}M tokens, layer {config.layer_idx}) ===")
    token_chunks = stream_fineweb_tokens(config)
    mlp_inputs, mlp_outputs = collect_activations(config, token_chunks, cache_path=None)
    del token_chunks

    eval_size = min(50_000, len(mlp_inputs) // 10)
    eval_inputs = mlp_inputs[-eval_size:]
    eval_outputs = mlp_outputs[-eval_size:]
    train_inputs = mlp_inputs[:-eval_size]
    train_outputs = mlp_outputs[:-eval_size]

    # Train in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(train_single_lambda, lam, train_inputs, train_outputs,
                        eval_inputs, eval_outputs, base_config_dict): lam
            for lam in LAMBDAS
        }
        for future in as_completed(futures):
            lam = futures[future]
            try:
                all_results.append(future.result())
            except Exception as e:
                print(f"[lambda={lam:.0e}] FAILED: {e}")

    all_results.sort(key=lambda r: r["lambda"])

    # Save
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"sweep_N{config.rank_multiplier}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'lambda':>10} {'L0':>8} {'NMSE':>10} {'Time':>8}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r['lambda']:>10.0e} {r['l0']:>8.1f} {r['nmse']:>10.4f} {r['training_time_s']:>7.0f}s")

    # Plot
    plot_l0_vs_nmse(all_results, EXPERIMENT_DIR / "figures" / f"pareto_N{config.rank_multiplier}.png")


if __name__ == "__main__":
    main()
