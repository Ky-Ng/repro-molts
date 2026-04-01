#!/usr/bin/env python3
"""Parallel sparsity sweep — trains multiple λ values concurrently on a single GPU.

Collects activations once into shared memory, then spawns concurrent training
workers. Designed for GPUs with enough VRAM for multiple MOLT models (~200MB each).
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch
import torch.multiprocessing as mp

from molt.config import MOLTConfig
from molt.data import collect_activations, make_dataloader, stream_fineweb_tokens
from molt.eval import compute_l0, compute_nmse
from molt.model import MOLT
from molt.train import train_molt


def train_single_lambda(
    lam: float,
    train_inputs: torch.Tensor,
    train_outputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    eval_outputs: torch.Tensor,
    config_dict: dict,
) -> dict:
    """Train and evaluate a single λ value. Runs in a subprocess."""
    config = MOLTConfig(**{**config_dict, "sparsity_coeff": lam})
    config.save_dir = f"checkpoints/sweep_N{config.rank_multiplier}"

    print(f"[λ={lam:.0e}] Starting training (transforms={config.total_transforms})")
    t0 = time.time()

    model, history = train_molt(config, train_inputs, train_outputs)

    # Evaluate
    l0 = compute_l0(model, eval_inputs)
    nmse = compute_nmse(model, eval_inputs, eval_outputs)

    elapsed = time.time() - t0
    print(f"[λ={lam:.0e}] Done in {elapsed:.0f}s — L0={l0:.1f}, NMSE={nmse:.4f}")

    # Check which transforms are active
    with torch.no_grad():
        sample = eval_inputs[:256].to(config.device)
        _, aux = model(sample)
        all_gates = torch.cat(aux["gate_acts"], dim=1)
        active_mask = (all_gates > 0).float()
        per_transform_freq = active_mask.mean(dim=0).cpu().tolist()

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
    mp.set_start_method("spawn", force=True)

    # Configuration
    lambdas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    max_workers = int(os.environ.get("MOLT_WORKERS", "3"))
    num_tokens = int(os.environ.get("MOLT_TOKENS", "10000000"))

    print(f"Parallel sweep: {len(lambdas)} λ values, {max_workers} workers, {num_tokens/1e6:.0f}M tokens")

    config = MOLTConfig(
        num_tokens=num_tokens,
        batch_size=64,
        device="cuda",
        wandb_enabled=False,
        log_every=2000,
        sparsity_warmup_frac=0.1,
    )
    config_dict = {
        "model_name": config.model_name,
        "layer_idx": config.layer_idx,
        "d_model": config.d_model,
        "rank_multiplier": config.rank_multiplier,
        "activation": config.activation,
        "jumprelu_threshold": config.jumprelu_threshold,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "num_tokens": config.num_tokens,
        "seq_len": config.seq_len,
        "num_epochs": config.num_epochs,
        "seed": config.seed,
        "device": config.device,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "streaming": config.streaming,
        "wandb_project": config.wandb_project,
        "wandb_enabled": False,
        "log_every": config.log_every,
        "eval_every": config.eval_every,
        "save_dir": config.save_dir,
        "sparsity_warmup_frac": config.sparsity_warmup_frac,
    }

    # Step 1: Collect activations once
    print(f"\n=== Collecting activations ({num_tokens/1e6:.0f}M tokens, layer {config.layer_idx}) ===")
    token_chunks = stream_fineweb_tokens(config)
    print(f"Got {len(token_chunks)} chunks ({len(token_chunks) * config.seq_len} tokens)")

    mlp_inputs, mlp_outputs = collect_activations(config, token_chunks, cache_path=None)
    del token_chunks
    print(f"Activations: {mlp_inputs.shape}")

    # Train/eval split
    eval_size = min(50_000, len(mlp_inputs) // 10)
    eval_inputs = mlp_inputs[-eval_size:]
    eval_outputs = mlp_outputs[-eval_size:]
    train_inputs = mlp_inputs[:-eval_size]
    train_outputs = mlp_outputs[:-eval_size]
    print(f"Train: {train_inputs.shape[0]} tokens, Eval: {eval_inputs.shape[0]} tokens")

    # Move to shared memory for multiprocessing
    train_inputs.share_memory_()
    train_outputs.share_memory_()
    eval_inputs.share_memory_()
    eval_outputs.share_memory_()

    # Step 2: Train all λ values in parallel
    print(f"\n=== Training {len(lambdas)} λ values ({max_workers} concurrent) ===")
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                train_single_lambda,
                lam,
                train_inputs,
                train_outputs,
                eval_inputs,
                eval_outputs,
                config_dict,
            ): lam
            for lam in lambdas
        }

        for future in as_completed(futures):
            lam = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"[λ={lam:.0e}] FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Sort by lambda
    all_results.sort(key=lambda r: r["lambda"])

    # Save results
    output_dir = Path("results/sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"sweep_N{config.rank_multiplier}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved sweep results to {results_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'λ':>10} {'L0':>8} {'NMSE':>10} {'Time':>8}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r['lambda']:>10.0e} {r['l0']:>8.1f} {r['nmse']:>10.4f} {r['training_time_s']:>7.0f}s")

    # Plot Pareto frontier
    from molt.eval import plot_pareto
    plot_path = output_dir / f"pareto_N{config.rank_multiplier}.png"
    plot_pareto(all_results, save_path=str(plot_path))


if __name__ == "__main__":
    main()
