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
import gc
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

_MLP_MODULE = None  # global reference for dtype conversion


def get_gpt2_mlp_fn(device: str = "cuda"):
    """Load GPT-2 and return the layer-6 MLP forward function."""
    global _MLP_MODULE
    print("Loading GPT-2 for Jacobian reference...")
    gpt2 = AutoModelForCausalLM.from_pretrained(MODEL)
    mlp = gpt2.transformer.h[LAYER_IDX].mlp
    mlp = mlp.to(device).eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    _MLP_MODULE = mlp

    # Free the rest of GPT-2
    del gpt2
    torch.cuda.empty_cache()

    def mlp_fn(x):
        return mlp(x)

    return mlp_fn


# ---------------------------------------------------------------------------
# Jacobian evaluation
# ---------------------------------------------------------------------------
# JumpReLU uses autograd.Function which doesn't support functorch (vmap/jacrev).
# We build a functorch-compatible MOLT forward using the smooth surrogate directly.

def _make_fp16_mlp_fn(mlp_fn):
    """Wrap MLP to work with fp16 inputs by casting in/out."""
    def fn(x):
        return mlp_fn(x.float()).half()
    return fn


def _make_molt_smooth_fn(model):
    """Create a functorch-compatible single-input MOLT forward function.

    Uses the smooth surrogate x * sigmoid(x / tau) instead of JumpReLU's
    autograd.Function, which doesn't support functorch transforms.
    """
    tau = 0.1  # same as JumpReLU.SURROGATE_TAU
    threshold = model.config.jumprelu_threshold

    def fn(x):
        """Single input forward: x is (d,), returns (d,)."""
        output = torch.zeros(x.shape[0], device=x.device)
        for group in model.groups:
            pre_acts = x @ group.encoder.T - group.bias  # (n_transforms,)
            gate = pre_acts * torch.sigmoid((pre_acts - threshold) / tau)
            Vx = torch.einsum("nrd,d->nr", group.V, x)  # (n_transforms, rank)
            UVx = torch.einsum("ndr,nr->nd", group.U, Vx)  # (n_transforms, d)
            gated = UVx * gate.unsqueeze(-1)  # (n_transforms, d)
            output = output + gated.sum(dim=0)
        return output

    return fn


def _make_tc_fn(model):
    """Create a functorch-compatible single-input transcoder forward function."""
    def fn(x):
        h = F.relu(x @ model.W_enc.T + model.b_enc)
        return h @ model.W_dec.T + model.b_dec
    return fn


def compute_jacobian_faithfulness(model_fn, mlp_fn, eval_in, n_samples, device="cuda", use_fp16=False):
    """Compute Jacobian cosine similarity using jacrev with chunking.

    Args:
        model_fn: single-input function (d,) -> (d,), functorch-compatible
        mlp_fn: reference MLP function, single-input (d,) -> (d,)
        eval_in: evaluation inputs on CPU
        n_samples: number of samples to use
        device: compute device
        use_fp16: if True, compute in float16 (saves ~50% memory for large models)
    """
    from torch.func import jacrev

    dtype = torch.float16 if use_fp16 else torch.float32
    jac_x = eval_in[:n_samples].to(device=device, dtype=dtype)

    chunk = 32
    jac_model_fn = jacrev(model_fn, chunk_size=chunk)
    jac_mlp_fn = jacrev(mlp_fn, chunk_size=chunk)

    all_sims = []

    for i in range(n_samples):
        xi = jac_x[i]
        j_model = jac_model_fn(xi)  # (d, d)
        j_mlp = jac_mlp_fn(xi)      # (d, d)

        # Cosine similarity in float32 for numerical stability
        sim = F.cosine_similarity(
            j_model.float().flatten().unsqueeze(0),
            j_mlp.float().flatten().unsqueeze(0),
        )
        all_sims.append(sim.detach().cpu())
        del j_model, j_mlp
        torch.cuda.empty_cache()

    sims = torch.cat(all_sims)
    return sims.mean().item(), sims.std().item()


def compute_jacobian_for_molt(model, mlp_fn, eval_in, device="cuda", use_fp16=False):
    """Compute Jacobian faithfulness for a MOLT model."""
    if use_fp16:
        model = model.half()
    molt_fn = _make_molt_smooth_fn(model)
    return compute_jacobian_faithfulness(molt_fn, mlp_fn, eval_in, JACOBIAN_SAMPLES, device, use_fp16)


def compute_jacobian_for_transcoder(model, mlp_fn, eval_in, device="cuda", use_fp16=False):
    """Compute Jacobian faithfulness for a transcoder model."""
    if use_fp16:
        model = model.half()
    tc_fn = _make_tc_fn(model)
    return compute_jacobian_faithfulness(tc_fn, mlp_fn, eval_in, JACOBIAN_SAMPLES, device, use_fp16)


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_molt(scale_name, scale_cfg, lam, train_in_cpu, train_out_cpu, eval_in, eval_out, mlp_fn):
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

    # Train on GPU
    train_in_gpu = train_in_cpu.cuda()
    train_out_gpu = train_out_cpu.cuda()
    start = time.time()
    model = train_molt_gpu(config, train_in_gpu, train_out_gpu)
    train_time = time.time() - start
    del train_in_gpu, train_out_gpu
    gc.collect()
    torch.cuda.empty_cache()

    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)
    n_params = sum(p.numel() for p in model.parameters())

    # Save model, reload fresh to clear training memory artifacts
    print(f"  Computing Jacobian faithfulness ({JACOBIAN_SAMPLES} samples)...")
    ckpt_path = RESULTS_DIR / f"_tmp_{name}.pt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = MOLT(config).cuda().eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    ckpt_path.unlink()

    # Use fp16 for large models to avoid OOM during Jacobian computation
    use_fp16 = scale_name in ("4x", "8x")
    if use_fp16:
        mlp_fn_jac = _make_fp16_mlp_fn(mlp_fn)
    else:
        mlp_fn_jac = mlp_fn
    jac_mean, jac_std = compute_jacobian_for_molt(model, mlp_fn_jac, eval_in, use_fp16=use_fp16)

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
        "params": n_params,
        "training_time_s": round(train_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()

    print(f"  L0={l0:.2f}, NMSE={nmse:.6f}, Jacobian={jac_mean:.4f}±{jac_std:.4f}, time={train_time:.0f}s")
    return result


def run_transcoder(scale_name, scale_cfg, lam, train_in_cpu, train_out_cpu, eval_in, eval_out, mlp_fn):
    name = f"tc_{scale_name}_lam{lam:.0e}"
    result_path = RESULTS_DIR / f"result_{name}.json"
    if result_path.exists():
        print(f"Skipping {name} (already done)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}\nTranscoder: {name}\n{'='*60}")
    n_features = scale_cfg["tc_features"]

    # Train on GPU
    train_in_gpu = train_in_cpu.cuda()
    train_out_gpu = train_out_cpu.cuda()
    start = time.time()
    model = train_transcoder_gpu(
        n_features, scale_cfg["num_epochs"], lam, train_in_gpu, train_out_gpu,
    )
    train_time = time.time() - start
    del train_in_gpu, train_out_gpu
    gc.collect()
    torch.cuda.empty_cache()

    metrics = evaluate_trainable_transcoder(model, eval_in, eval_out)
    n_params = model.param_count()

    # Save model, reload fresh to clear training memory artifacts
    print(f"  Computing Jacobian faithfulness ({JACOBIAN_SAMPLES} samples)...")
    ckpt_path = RESULTS_DIR / f"_tmp_{name}.pt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = TrainableTranscoder(D_MODEL, n_features).cuda().eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    ckpt_path.unlink()

    use_fp16 = scale_name in ("4x", "8x")
    if use_fp16:
        mlp_fn_jac = _make_fp16_mlp_fn(mlp_fn)
    else:
        mlp_fn_jac = mlp_fn
    jac_mean, jac_std = compute_jacobian_for_transcoder(model, mlp_fn_jac, eval_in, use_fp16=use_fp16)

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
        "params": n_params,
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

    from molt.utils.plotting import plot_comparison

    # Plot 1: Jacobian vs L0 (main plot)
    plot_comparison(
        molt_results, tc_results,
        x_key="jacobian_cosine_sim", y_key="l0",
        save_path=FIGURES_DIR / "l0_vs_jacobian_all.png",
        title="Jacobian Faithfulness vs L0 — MOLT vs Transcoder (GPT-2 Layer 6)",
        x_label="Jacobian Cosine Similarity",
        y_label="L0 (Active Transforms / Features)",
    )

    # Plot 2: Jacobian vs NMSE (supplementary)
    plot_comparison(
        molt_results, tc_results,
        x_key="jacobian_cosine_sim", y_key="nmse",
        save_path=FIGURES_DIR / "nmse_vs_jacobian_all.png",
        title="Jacobian Faithfulness vs NMSE — MOLT vs Transcoder (GPT-2 Layer 6)",
        x_label="Jacobian Cosine Similarity",
        y_label="Normalized MSE",
    )

    # Plot 3: NMSE vs L0 (reproduce exp 11 for reference)
    plot_comparison(
        molt_results, tc_results,
        x_key="nmse", y_key="l0",
        save_path=FIGURES_DIR / "l0_vs_nmse_all.png",
        title="NMSE vs L0 — MOLT vs Transcoder (GPT-2 Layer 6)",
        x_label="Normalized MSE",
        y_label="L0 (Active Transforms / Features)",
    )


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

    # Load activations (keep on CPU, move to GPU only during training)
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)
    train_in = mlp_inputs[:TRAIN_TOKENS]
    train_out = mlp_outputs[:TRAIN_TOKENS]
    eval_in = mlp_inputs[-EVAL_SIZE:]
    eval_out = mlp_outputs[-EVAL_SIZE:]
    del mlp_inputs, mlp_outputs
    print(f"Train: {train_in.shape}, Eval: {eval_in.shape} (both on CPU)")

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
