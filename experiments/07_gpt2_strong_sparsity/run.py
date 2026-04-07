#!/usr/bin/env python3
"""GPT-2 strong sparsity sweep: {ReLU, JumpReLU} x {Tanh, L0} x {lambda=1e-3..1e-1}.

Tests whether strong sparsity penalties push theta increasingly negative on GPT-2,
extending experiment 05 which only tested lambda up to 1e-4.

Runs in parallel (3 concurrent workers) using ThreadPoolExecutor to share GPU memory.

Usage:
    uv run python experiments/07_gpt2_strong_sparsity/run.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from molt.config import MOLTConfig
from molt.eval import compute_l0, compute_nmse
from molt.train import train_molt
from molt.utils.activations import load_cached_activations
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
MODEL = "openai-community/gpt2"
CACHE_PATH = "data/activations_openai_community_gpt2_2M.pt"

LAMBDAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

SETUPS = []
for lam in LAMBDAS:
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

BASE_CONFIG = dict(
    num_tokens=2_000_000,
    batch_size=64,
    device="cuda",
    log_every=100,
    sparsity_warmup_frac=0.1,
    jumprelu_threshold=0.0,
)


def run_single_setup(setup, mlp_inputs, mlp_outputs):
    """Train and evaluate a single setup. Runs in a thread."""
    name = setup["name"]
    overrides = {k: v for k, v in setup.items() if k != "name"}
    config = MOLTConfig.from_preset(MODEL, **{**BASE_CONFIG, **overrides})

    print(f"\n[{name}] Starting — {config.activation}, {config.sparsity_type}, "
          f"lambda={config.sparsity_coeff:.0e}")

    t0 = time.time()
    model, history = train_molt(config, mlp_inputs, mlp_outputs,
                                save_dir=str(RESULTS_DIR))
    elapsed = time.time() - t0

    eval_in = mlp_inputs[-10_000:]
    eval_out = mlp_outputs[-10_000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)
    final_theta = model.threshold.item() if model.threshold is not None else None

    print(f"[{name}] Done in {elapsed:.0f}s — L0={l0:.2f}, NMSE={nmse:.4f}"
          + (f", theta={final_theta:.4f}" if final_theta is not None else ""))

    # Per-transform activity
    active_transforms = []
    with torch.no_grad():
        x = eval_in[:512].to(config.device)
        _, aux = model(x)
        all_gates = torch.cat(aux["gate_acts"], dim=1)
        freq = (all_gates > 0).float().mean(dim=0)
        cumulative = 0
        for count, rank in config.rank_distribution:
            for j in range(count):
                f = freq[cumulative].item()
                if f > 0.001:
                    active_transforms.append({
                        "transform": cumulative, "rank": rank,
                        "frequency": round(f * 100, 1),
                    })
                cumulative += 1

    result = {
        "name": name,
        "activation": config.activation,
        "sparsity_type": config.sparsity_type,
        "learned_threshold": config.learned_threshold,
        "sparsity_coeff": config.sparsity_coeff,
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "final_threshold": round(final_theta, 4) if final_theta is not None else None,
        "num_active": len(active_transforms),
        "active_transforms": active_transforms,
        "training_time_s": round(elapsed, 1),
    }

    # Save per-run files
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"result_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return {**result, "history": history}


def main():
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)
    max_workers = int(os.environ.get("MOLT_WORKERS", "3"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(SETUPS)} setups with {max_workers} concurrent workers")

    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_single_setup, setup, mlp_inputs, mlp_outputs): setup
            for setup in SETUPS
        }
        for future in as_completed(futures):
            setup = futures[future]
            try:
                all_results.append(future.result())
            except Exception as e:
                print(f"[{setup['name']}] FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Sort by lambda then name
    all_results.sort(key=lambda r: (r["sparsity_coeff"], r["name"]))

    # Save combined summary
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(RESULTS_DIR / "sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Full summary table
    print(f"\n{'='*80}")
    print("GPT-2 Strong Sparsity Penalty Sweep")
    print(f"{'='*80}")
    print(f"{'Setup':<35} {'lambda':>8} {'L0':>6} {'NMSE':>8} {'theta':>8} {'#Act':>5}")
    print(f"{'-'*80}")
    for r in summary:
        theta_str = f"{r['final_threshold']:.4f}" if r.get("final_threshold") is not None else " fixed"
        print(f"{r['name']:<35} {r['sparsity_coeff']:>8.0e} {r['l0']:>6.2f} "
              f"{r['nmse']:>8.4f} {theta_str:>8} {r.get('num_active', 0):>5}")

    # Plot per-lambda comparisons
    for lam in LAMBDAS:
        lam_runs = {r["name"]: r["history"] for r in all_results
                    if r["sparsity_coeff"] == lam}
        if lam_runs:
            plot_multi_run_curves(
                lam_runs,
                f"GPT-2 Strong Sparsity (lambda={lam:.0e})",
                FIGURES_DIR / f"sparsity_sweep_lam{lam:.0e}.png",
            )

    # Print theta trajectory summary for JumpReLU runs
    jumprelu_results = [r for r in all_results if r["activation"] == "jumprelu"]
    if jumprelu_results:
        print(f"\n{'='*70}")
        print("THETA TRAJECTORY — JumpReLU runs only")
        print(f"{'='*70}")
        print(f"{'Setup':<35} {'lambda':>8} {'L0':>6} {'NMSE':>8} {'theta':>8}")
        print(f"{'-'*70}")
        for r in sorted(jumprelu_results, key=lambda x: (x["sparsity_type"], x["sparsity_coeff"])):
            theta = r.get("final_threshold")
            theta_str = f"{theta:.4f}" if theta is not None else "N/A"
            print(f"{r['name']:<35} {r['sparsity_coeff']:>8.0e} {r['l0']:>6.2f} "
                  f"{r['nmse']:>8.4f} {theta_str:>8}")


if __name__ == "__main__":
    main()
