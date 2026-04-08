#!/usr/bin/env python3
"""Experiment 12: L0 vs Jacobian faithfulness for MOLT vs Transcoder.

Extends experiment 11 by computing Jacobian cosine similarity against the true
GPT-2 MLP for each (method, scale, lambda) configuration. Retrains models from
scratch (exp 11 did not save checkpoints), evaluates Jacobian faithfulness, and
plots L0 vs Jacobian cosine similarity.

Usage:
    uv run python experiments/12_jacobian_comparison/run.py
    uv run python experiments/12_jacobian_comparison/run.py --scale 1x
    uv run python experiments/12_jacobian_comparison/run.py --method molt
    uv run python experiments/12_jacobian_comparison/run.py --plot-only
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
from transformers import AutoModelForCausalLM

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
LAYER_IDX = 6
EVAL_SIZE = 10_000
TRAIN_TOKENS = 500_000
BATCH_SIZE = 1024
LR = 1e-3
LOG_EVERY = 50
JACOBIAN_SAMPLES = 64  # number of samples for Jacobian evaluation

# Same scales as experiment 11
SCALES = {
    "1x": {"rank_multiplier": 1, "num_epochs": 1, "tc_features": 2573},
    "2x": {"rank_multiplier": 2, "num_epochs": 2, "tc_features": 5147},
    "4x": {"rank_multiplier": 4, "num_epochs": 4, "tc_features": 10295},
}

# Same lambdas as experiment 11 (core set)
LAMBDAS_MOLT = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
LAMBDAS_TC = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1]


# ---------------------------------------------------------------------------
# GPU-based training loops (same as experiment 11)
# ---------------------------------------------------------------------------

def train_molt_gpu(
    config: MOLTConfig,
    train_in: torch.Tensor,
    train_out: torch.Tensor,
) -> MOLT:
    """Train MOLT with data pre-loaded on GPU. Returns trained model."""
    torch.manual_seed(config.seed)
    model = MOLT(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    N = len(train_in)
    steps_per_epoch = N // BATCH_SIZE
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.sparsity_warmup_frac)

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
                pbar.set_postfix(
                    nmse=f"{metrics['nmse'].item():.4f}",
                    l0=f"{metrics['l0'].item():.1f}",
                )

    return model


def train_transcoder_gpu(
    n_features: int,
    num_epochs: int,
    sparsity_coeff: float,
    train_in: torch.Tensor,
    train_out: torch.Tensor,
) -> TrainableTranscoder:
    """Train transcoder with data pre-loaded on GPU. Returns trained model."""
    torch.manual_seed(42)
    model = TrainableTranscoder(D_MODEL, n_features).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    N = len(train_in)
    steps_per_epoch = N // BATCH_SIZE
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)

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
                pbar.set_postfix(
                    nmse=f"{metrics['nmse'].item():.4f}",
                    l0=f"{metrics['l0'].item():.1f}",
                )

    return model


# ---------------------------------------------------------------------------
# MLP extraction
# ---------------------------------------------------------------------------

def get_gpt2_mlp_fn(device: str = "cuda"):
    """Load GPT-2 and return the layer-6 MLP forward function."""
    print("Loading GPT-2 for Jacobian reference...")
    gpt2 = AutoModelForCausalLM.from_pretrained(MODEL)
    mlp = gpt2.transformer.h[LAYER_IDX].mlp
    mlp = mlp.to(device).eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    # Free the rest of GPT-2
    del gpt2
    torch.cuda.empty_cache()

    def mlp_fn(x):
        return mlp(x)

    return mlp_fn


# ---------------------------------------------------------------------------
# Jacobian evaluation (manual autograd — JumpReLU doesn't support functorch)
# ---------------------------------------------------------------------------

def compute_jacobian_manual(fn, x):
    """Compute Jacobian row-by-row using standard autograd.

    Args:
        fn: function mapping (d,) -> (d,)
        x: single input (d,)

    Returns:
        jacobian: (d, d) matrix
    """
    d = x.shape[0]
    x = x.detach().requires_grad_(True)
    y = fn(x)
    jac_rows = []
    for i in range(d):
        grad = torch.autograd.grad(y[i], x, retain_graph=(i < d - 1))[0]
        jac_rows.append(grad.detach())
    return torch.stack(jac_rows)  # (d, d)


def jacobian_cosine_sim(fn_a, fn_b, x_batch, device="cuda"):
    """Compute per-sample Jacobian cosine similarity between two functions.

    Processes one sample at a time to avoid OOM.
    """
    sims = []
    for i in range(len(x_batch)):
        xi = x_batch[i].to(device)
        j_a = compute_jacobian_manual(fn_a, xi)  # (d, d)
        j_b = compute_jacobian_manual(fn_b, xi)  # (d, d)
        sim = F.cosine_similarity(j_a.flatten().unsqueeze(0), j_b.flatten().unsqueeze(0)).item()
        sims.append(sim)
    return torch.tensor(sims)


def compute_jacobian_for_molt(model, mlp_fn, eval_in, device="cuda"):
    """Compute Jacobian faithfulness for a MOLT model."""
    jac_x = eval_in[:JACOBIAN_SAMPLES]

    def molt_fn(xi):
        out, _ = model(xi.unsqueeze(0))
        return out.squeeze(0)

    sims = jacobian_cosine_sim(molt_fn, mlp_fn, jac_x, device=device)
    return sims.mean().item(), sims.std().item()


def compute_jacobian_for_transcoder(model, mlp_fn, eval_in, device="cuda"):
    """Compute Jacobian faithfulness for a transcoder model."""
    jac_x = eval_in[:JACOBIAN_SAMPLES]

    def tc_fn(xi):
        out, _ = model(xi.unsqueeze(0))
        return out.squeeze(0)

    sims = jacobian_cosine_sim(tc_fn, mlp_fn, jac_x, device=device)
    return sims.mean().item(), sims.std().item()


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_molt(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out, mlp_fn):
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
    model = train_molt_gpu(config, train_in, train_out)
    train_time = time.time() - start

    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)

    print(f"  Computing Jacobian faithfulness ({JACOBIAN_SAMPLES} samples)...")
    jac_mean, jac_std = compute_jacobian_for_molt(model, mlp_fn, eval_in)

    result = {
        "name": name,
        "method": "molt",
        "scale": scale_name,
        "rank_multiplier": scale_cfg["rank_multiplier"],
        "num_epochs": scale_cfg["num_epochs"],
        "sparsity_coeff": lam,
        "l0": round(l0, 2),
        "nmse": round(nmse, 6),
        "jacobian_cosine_sim": round(jac_mean, 6),
        "jacobian_cosine_sim_std": round(jac_std, 6),
        "params": sum(p.numel() for p in model.parameters()),
        "training_time_s": round(train_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()

    print(f"  L0={l0:.2f}, NMSE={nmse:.6f}, Jacobian={jac_mean:.4f}±{jac_std:.4f}, time={train_time:.0f}s")
    return result


def run_transcoder(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out, mlp_fn):
    name = f"tc_{scale_name}_lam{lam:.0e}"
    result_path = RESULTS_DIR / f"result_{name}.json"
    if result_path.exists():
        print(f"Skipping {name} (already done)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}\nTranscoder: {name}\n{'='*60}")
    n_features = scale_cfg["tc_features"]

    start = time.time()
    model = train_transcoder_gpu(
        n_features, scale_cfg["num_epochs"], lam, train_in, train_out,
    )
    train_time = time.time() - start

    metrics = evaluate_trainable_transcoder(model, eval_in, eval_out)

    print(f"  Computing Jacobian faithfulness ({JACOBIAN_SAMPLES} samples)...")
    jac_mean, jac_std = compute_jacobian_for_transcoder(model, mlp_fn, eval_in)

    result = {
        "name": name,
        "method": "transcoder",
        "scale": scale_name,
        "n_features": n_features,
        "num_epochs": scale_cfg["num_epochs"],
        "sparsity_coeff": lam,
        "l0": round(metrics["l0"], 2),
        "nmse": round(metrics["nmse"], 6),
        "jacobian_cosine_sim": round(jac_mean, 6),
        "jacobian_cosine_sim_std": round(jac_std, 6),
        "params": model.param_count(),
        "training_time_s": round(train_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()

    print(f"  L0={metrics['l0']:.2f}, NMSE={metrics['nmse']:.6f}, Jacobian={jac_mean:.4f}±{jac_std:.4f}, time={train_time:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_all_results() -> list[dict]:
    """Load all existing result JSON files."""
    results = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("result_*.json")):
            with open(f) as fh:
                results.append(json.load(fh))
    return results


def plot_l0_vs_jacobian(results: list[dict]) -> None:
    """Plot L0 vs Jacobian cosine similarity for MOLT and transcoders."""
    if not results:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    molt_results = [r for r in results if r["method"] == "molt"]
    tc_results = [r for r in results if r["method"] == "transcoder"]

    scale_colors = {"1x": "tab:blue", "2x": "tab:orange", "4x": "tab:green"}

    # --- Plot 1: L0 vs Jacobian (main plot) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    for scale, color in scale_colors.items():
        molt_s = sorted([r for r in molt_results if r["scale"] == scale], key=lambda r: r["l0"])
        tc_s = sorted([r for r in tc_results if r["scale"] == scale], key=lambda r: r["l0"])

        if molt_s:
            l0s = [r["l0"] for r in molt_s]
            jacs = [r["jacobian_cosine_sim"] for r in molt_s]
            ax.scatter(l0s, jacs, marker="o", s=80, color=color, label=f"MOLT {scale}",
                       edgecolors="black", linewidths=0.5)
            ax.plot(l0s, jacs, "--", color=color, alpha=0.4, linewidth=1)

        if tc_s:
            l0s = [r["l0"] for r in tc_s]
            jacs = [r["jacobian_cosine_sim"] for r in tc_s]
            ax.scatter(l0s, jacs, marker="x", s=100, color=color, label=f"Transcoder {scale}",
                       linewidths=2)
            ax.plot(l0s, jacs, ":", color=color, alpha=0.4, linewidth=1)

    ax.set_xlabel("L0 (Active Transforms / Features)", fontsize=12)
    ax.set_ylabel("Jacobian Cosine Similarity", fontsize=12)
    ax.set_title("L0 vs Jacobian Faithfulness — MOLT vs Transcoder (GPT-2 Layer 6)", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "l0_vs_jacobian_all.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'l0_vs_jacobian_all.png'}")

    # --- Plot 2: NMSE vs Jacobian (supplementary) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    for scale, color in scale_colors.items():
        molt_s = sorted([r for r in molt_results if r["scale"] == scale], key=lambda r: r["nmse"])
        tc_s = sorted([r for r in tc_results if r["scale"] == scale], key=lambda r: r["nmse"])

        if molt_s:
            nmses = [r["nmse"] for r in molt_s]
            jacs = [r["jacobian_cosine_sim"] for r in molt_s]
            ax.scatter(nmses, jacs, marker="o", s=80, color=color, label=f"MOLT {scale}",
                       edgecolors="black", linewidths=0.5)

        if tc_s:
            nmses = [r["nmse"] for r in tc_s]
            jacs = [r["jacobian_cosine_sim"] for r in tc_s]
            ax.scatter(nmses, jacs, marker="x", s=100, color=color, label=f"Transcoder {scale}",
                       linewidths=2)

    ax.set_xlabel("Normalized MSE", fontsize=12)
    ax.set_ylabel("Jacobian Cosine Similarity", fontsize=12)
    ax.set_title("NMSE vs Jacobian Faithfulness — MOLT vs Transcoder (GPT-2 Layer 6)", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "nmse_vs_jacobian_all.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'nmse_vs_jacobian_all.png'}")

    # --- Plot 3: L0 vs NMSE (reproduce exp 11 for reference) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    for scale, color in scale_colors.items():
        molt_s = [r for r in molt_results if r["scale"] == scale]
        tc_s = [r for r in tc_results if r["scale"] == scale]
        if molt_s:
            ax.scatter([r["nmse"] for r in molt_s], [r["l0"] for r in molt_s],
                       marker="o", s=80, color=color, label=f"MOLT {scale}",
                       edgecolors="black", linewidths=0.5)
        if tc_s:
            ax.scatter([r["nmse"] for r in tc_s], [r["l0"] for r in tc_s],
                       marker="x", s=100, color=color, label=f"Transcoder {scale}",
                       linewidths=2)

    ax.set_xlabel("Normalized MSE", fontsize=12)
    ax.set_ylabel("L0 (Active Transforms / Features)", fontsize=12)
    ax.set_title("NMSE vs L0 — MOLT vs Transcoder (GPT-2 Layer 6)", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "l0_vs_nmse_all.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'l0_vs_nmse_all.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=list(SCALES.keys()), help="Run single scale")
    parser.add_argument("--method", choices=["molt", "transcoder"], help="Run single method")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots")
    args = parser.parse_args()

    if args.plot_only:
        all_results = load_all_results()
        plot_l0_vs_jacobian(all_results)
        return

    # Load activations
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)
    train_in = mlp_inputs[:TRAIN_TOKENS].cuda()
    train_out = mlp_outputs[:TRAIN_TOKENS].cuda()
    eval_in = mlp_inputs[-EVAL_SIZE:]
    eval_out = mlp_outputs[-EVAL_SIZE:]
    del mlp_inputs, mlp_outputs
    print(f"Train: {train_in.shape} on GPU, Eval: {eval_in.shape} on CPU")

    # Load GPT-2 MLP for Jacobian reference
    mlp_fn = get_gpt2_mlp_fn(device="cuda")

    scales = {args.scale: SCALES[args.scale]} if args.scale else SCALES

    for scale_name, scale_cfg in scales.items():
        if args.method != "transcoder":
            for lam in LAMBDAS_MOLT:
                run_molt(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out, mlp_fn)

        if args.method != "molt":
            for lam in LAMBDAS_TC:
                run_transcoder(scale_name, scale_cfg, lam, train_in, train_out, eval_in, eval_out, mlp_fn)

    # Final plots and summary
    all_results = load_all_results()
    plot_l0_vs_jacobian(all_results)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<35} {'Scale':>5} {'Lambda':>8} {'L0':>8} {'NMSE':>10} {'Jacobian':>10}")
    print(f"{'-'*80}")
    for r in sorted(all_results, key=lambda x: (x["method"], x["scale"], x["sparsity_coeff"])):
        print(f"{r['name']:<35} {r['scale']:>5} {r['sparsity_coeff']:>8.0e} "
              f"{r['l0']:>8.2f} {r['nmse']:>10.6f} {r['jacobian_cosine_sim']:>10.6f}")


if __name__ == "__main__":
    main()
