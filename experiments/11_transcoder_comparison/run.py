#!/usr/bin/env python3
"""Experiment 11: MOLT vs Transcoder comparison on GPT-2.

Trains MOLTs and transcoders at matched parameter counts and training steps,
then compares L0 vs NMSE Pareto frontiers.

Uses direct GPU batching (no DataLoader) for ~7x speedup over CPU-based DataLoader.

Usage:
    uv run python experiments/11_transcoder_comparison/run.py
    uv run python experiments/11_transcoder_comparison/run.py --scale 1x
    uv run python experiments/11_transcoder_comparison/run.py --method molt
    uv run python experiments/11_transcoder_comparison/run.py --method transcoder
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from molt.config import MOLTConfig
from molt.eval import compute_l0, compute_nmse
from molt.model import MOLT
from molt.transcoder import TrainableTranscoder, evaluate_trainable_transcoder
from molt.utils.activations import load_cached_activations

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
CACHE_PATH = "data/activations_openai_community_gpt2_2M.pt"
MODEL = "openai-community/gpt2"
D_MODEL = 768
EVAL_SIZE = 10_000
TRAIN_TOKENS = 500_000  # use first 500K tokens (rest for eval)
BATCH_SIZE = 1024
LR = 1e-3
LOG_EVERY = 50

# Parameter-matched scales:
# Each 4x FLOPs = 2x params * 2x steps (epochs)
SCALES = {
    "1x": {"rank_multiplier": 1, "num_epochs": 1, "tc_features": 2573},
    "2x": {"rank_multiplier": 2, "num_epochs": 2, "tc_features": 5147},
    "4x": {"rank_multiplier": 4, "num_epochs": 4, "tc_features": 10295},
}

LAMBDAS = [0.0, 1e-4, 1e-3]


# ---------------------------------------------------------------------------
# GPU-based training loops (bypass DataLoader for ~7x speedup)
# ---------------------------------------------------------------------------

def train_molt_gpu(
    config: MOLTConfig,
    train_in: torch.Tensor,
    train_out: torch.Tensor,
) -> tuple[MOLT, list[dict]]:
    """Train MOLT with data pre-loaded on GPU."""
    torch.manual_seed(config.seed)
    model = MOLT(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    N = len(train_in)
    steps_per_epoch = N // BATCH_SIZE
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.sparsity_warmup_frac)

    history = []
    step = 0
    for epoch in range(config.num_epochs):
        perm = torch.randperm(N, device=train_in.device)
        pbar = tqdm(range(steps_per_epoch), desc=f"MOLT epoch {epoch+1}/{config.num_epochs}")
        for i in pbar:
            idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            bx = train_in[idx]
            bt = train_out[idx]

            sparsity_scale = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
            optimizer.zero_grad()
            loss, metrics = model.loss(bx, bt, sparsity_scale)
            loss.backward()
            optimizer.step()
            step += 1

            if step % LOG_EVERY == 0:
                log = {k: v.item() for k, v in metrics.items()}
                log["step"] = step
                log["epoch"] = epoch
                history.append(log)
                pbar.set_postfix(nmse=f"{log['nmse']:.4f}", l0=f"{log['l0']:.1f}")

    return model, history


def train_transcoder_gpu(
    n_features: int,
    num_epochs: int,
    sparsity_coeff: float,
    train_in: torch.Tensor,
    train_out: torch.Tensor,
) -> tuple[TrainableTranscoder, list[dict]]:
    """Train transcoder with data pre-loaded on GPU."""
    torch.manual_seed(42)
    model = TrainableTranscoder(D_MODEL, n_features).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    N = len(train_in)
    steps_per_epoch = N // BATCH_SIZE
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)

    history = []
    step = 0
    for epoch in range(num_epochs):
        perm = torch.randperm(N, device=train_in.device)
        pbar = tqdm(range(steps_per_epoch), desc=f"TC epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            bx = train_in[idx]
            bt = train_out[idx]

            sparsity_scale = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
            optimizer.zero_grad()
            loss, metrics = model.loss(bx, bt, sparsity_coeff, sparsity_scale)
            loss.backward()
            optimizer.step()
            step += 1

            if step % LOG_EVERY == 0:
                log = {k: v.item() for k, v in metrics.items()}
                log["step"] = step
                log["epoch"] = epoch
                history.append(log)
                pbar.set_postfix(nmse=f"{log['nmse']:.4f}", l0=f"{log['l0']:.1f}")

    return model, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_run(name: str, history: list[dict]) -> None:
    """Plot training curves for a single run."""
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "nmse", "NMSE", "#2563eb"),
        (axes[0, 1], "l0", "L0", "#dc2626"),
        (axes[1, 0], "sparsity_loss", "Sparsity Loss", "#16a34a"),
        (axes[1, 1], "mse", "MSE", "#9333ea"),
    ]
    for ax, key, ylabel, color in panels:
        if key in history[0]:
            ax.plot(steps, [h[key] for h in history], color=color, lw=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"train_{name}.png", dpi=120)
    plt.close(fig)


def plot_all_results(results: list[dict]) -> None:
    """Plot combined L0 vs NMSE for all completed runs."""
    if not results:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    molt_results = [r for r in results if r["method"] == "molt"]
    tc_results = [r for r in results if r["method"] == "transcoder"]

    fig, ax = plt.subplots(figsize=(10, 7))
    scale_colors = {"1x": "tab:blue", "2x": "tab:orange", "4x": "tab:green", "8x": "tab:red"}

    for scale, color in scale_colors.items():
        molt_s = [r for r in molt_results if r["scale"] == scale]
        tc_s = [r for r in tc_results if r["scale"] == scale]
        if molt_s:
            ax.scatter(
                [r["l0"] for r in molt_s], [r["nmse"] for r in molt_s],
                marker="o", s=80, color=color, label=f"MOLT {scale}",
                edgecolors="black", linewidths=0.5,
            )
        if tc_s:
            ax.scatter(
                [r["l0"] for r in tc_s], [r["nmse"] for r in tc_s],
                marker="x", s=100, color=color, label=f"Transcoder {scale}",
                linewidths=2,
            )

    ax.set_xlabel("L0 (Active Transforms / Features)", fontsize=12)
    ax.set_ylabel("Normalized MSE", fontsize=12)
    ax.set_title("L0 vs NMSE — MOLT vs Transcoder (GPT-2 Layer 6)", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "l0_vs_nmse_all.png", dpi=150)
    plt.close(fig)
    print(f"Saved combined plot to {FIGURES_DIR / 'l0_vs_nmse_all.png'}")


def load_all_results() -> list[dict]:
    """Load all existing result JSON files."""
    results = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("result_*.json")):
            with open(f) as fh:
                results.append(json.load(fh))
    return results


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_molt(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out):
    name = f"molt_{scale_name}_lam{lam:.0e}"
    result_path = RESULTS_DIR / f"result_{name}.json"
    if result_path.exists():
        print(f"Skipping {name} (already done)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}\nMOLT: {name}\n{'='*60}")

    config = MOLTConfig.from_preset(
        MODEL,
        rank_multiplier=scale_cfg["rank_multiplier"],
        num_epochs=scale_cfg["num_epochs"],
        sparsity_coeff=lam,
        sparsity_warmup_frac=0.1,
        activation="jumprelu",
        sparsity_type="tanh",
        batch_size=BATCH_SIZE,
        lr=LR,
        log_every=LOG_EVERY,
        device="cuda",
    )

    start = time.time()
    model, history = train_molt_gpu(config, train_in, train_out)
    train_time = time.time() - start

    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)

    result = {
        "name": name,
        "method": "molt",
        "scale": scale_name,
        "rank_multiplier": scale_cfg["rank_multiplier"],
        "num_epochs": scale_cfg["num_epochs"],
        "sparsity_coeff": lam,
        "l0": round(l0, 2),
        "nmse": round(nmse, 6),
        "params": sum(p.numel() for p in model.parameters()),
        "training_time_s": round(train_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_single_run(name, history)
    del model
    torch.cuda.empty_cache()

    print(f"  L0={l0:.2f}, NMSE={nmse:.6f}, time={train_time:.0f}s")
    return result


def run_transcoder(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out):
    name = f"tc_{scale_name}_lam{lam:.0e}"
    result_path = RESULTS_DIR / f"result_{name}.json"
    if result_path.exists():
        print(f"Skipping {name} (already done)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}\nTranscoder: {name}\n{'='*60}")
    n_features = scale_cfg["tc_features"]

    start = time.time()
    model, history = train_transcoder_gpu(
        n_features, scale_cfg["num_epochs"], lam, train_in, train_out,
    )
    train_time = time.time() - start

    metrics = evaluate_trainable_transcoder(model, eval_in, eval_out)

    result = {
        "name": name,
        "method": "transcoder",
        "scale": scale_name,
        "n_features": n_features,
        "num_epochs": scale_cfg["num_epochs"],
        "sparsity_coeff": lam,
        "l0": round(metrics["l0"], 2),
        "nmse": round(metrics["nmse"], 6),
        "params": model.param_count(),
        "training_time_s": round(train_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_single_run(name, history)
    del model
    torch.cuda.empty_cache()

    print(f"  L0={metrics['l0']:.2f}, NMSE={metrics['nmse']:.6f}, time={train_time:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=list(SCALES.keys()), help="Run single scale")
    parser.add_argument("--method", choices=["molt", "transcoder"], help="Run single method")
    args = parser.parse_args()

    # Load activations and move to GPU
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)
    train_in = mlp_inputs[:TRAIN_TOKENS].cuda()
    train_out = mlp_outputs[:TRAIN_TOKENS].cuda()
    eval_in = mlp_inputs[-EVAL_SIZE:]   # keep on CPU for eval (smaller)
    eval_out = mlp_outputs[-EVAL_SIZE:]
    del mlp_inputs, mlp_outputs
    print(f"Train: {train_in.shape} on GPU, Eval: {eval_in.shape} on CPU")

    scales = {args.scale: SCALES[args.scale]} if args.scale else SCALES

    for scale_name, scale_cfg in scales.items():
        for lam in LAMBDAS:
            if args.method != "transcoder":
                run_molt(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out)
                plot_all_results(load_all_results())

            if args.method != "molt":
                run_transcoder(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out)
                plot_all_results(load_all_results())

    # Final summary
    all_results = load_all_results()
    plot_all_results(all_results)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<30} {'Scale':>5} {'Lambda':>8} {'L0':>8} {'NMSE':>10} {'Params':>10}")
    print(f"{'-'*80}")
    for r in sorted(all_results, key=lambda x: (x["method"], x["scale"], x["sparsity_coeff"])):
        print(f"{r['name']:<30} {r['scale']:>5} {r['sparsity_coeff']:>8.0e} "
              f"{r['l0']:>8.2f} {r['nmse']:>10.6f} {r['params']:>10,}")


if __name__ == "__main__":
    main()
