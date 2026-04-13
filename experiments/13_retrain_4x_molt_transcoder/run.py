#!/usr/bin/env python3
"""Experiment 13 orchestrator: retrain MOLT + Transcoder, then delphi pipeline.

Stages (each independently re-runnable):
    --stage train    # (stage 0)  train + save checkpoints
    --stage upload   # (stage 0b) upload checkpoints to HF
    --stage cache    # (stage 1)  build delphi latent caches
    --stage explain  # (stage 2)  generate feature labels via OpenRouter
    --stage score    # (stage 3)  detection + fuzzing scorers

Slices:
    --slice smoke    # 1x MOLT + 1x Transcoder (cheap pipeline validation)
    --slice main     # 4x MOLT + 4x Transcoder (gated on smoke success)
    --slice all      # both

Cost safety:
    --confirm-budget # required when pre-flight estimate > $15 (75% of cap)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import torch
import torch.nn.functional as F
from tqdm import tqdm

# repo root imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Import budget first so its vllm stub is installed before any delphi import
import budget as _budget  # noqa: F401

from molt.config import MOLTConfig
from molt.eval import compute_l0, compute_nmse
from molt.model import MOLT
from molt.transcoder import TrainableTranscoder
from molt.utils.activations import load_cached_activations, split_train_eval
from molt.delphi_shim import MOLTShim, TranscoderShim

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

# Bulky/regenerable artifacts live under out/ (gitignored). `results/` and
# `figures/` stay at the experiment root, matching the experiment template.
OUT_DIR = EXPERIMENT_DIR / "out"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
LATENTS_DIR = OUT_DIR / "latents"
EXPLANATIONS_DIR = OUT_DIR / "explanations"
SCORES_DIR = OUT_DIR / "scores"
TOKEN_CACHE_DIR = OUT_DIR / "token_cache"

CACHE_PATH = REPO_ROOT / "data" / "activations_2M_gpt2-small-layer6.pt"

MODEL = "openai-community/gpt2"
D_MODEL = 768
EVAL_SIZE = 10_000
TRAIN_TOKENS = 500_000
BATCH_SIZE = 1024
LR = 1e-3
LOG_EVERY = 50
SEED = 42

# Slice definitions (configs to train) — matches the README table
SLICES = {
    "smoke": {
        "molt": {"rank_multiplier": 1, "num_epochs": 1, "lam": 1e-3},
        "tc":   {"n_features": 2573,   "num_epochs": 1, "lam": 1.0},
    },
    "main": {
        "molt": {"rank_multiplier": 4, "num_epochs": 4, "lam": 1e-3},
        "tc":   {"n_features": 10295,  "num_epochs": 4, "lam": 3.0},
    },
}

HF_REPO = "kylelovesllms/auto-repro-molts"

# Stage 1 (delphi cache) settings
# GPT-2 AutoModel (no LM head) uses hookpoint "h.6.mlp" (not "transformer.h.6.mlp")
DELPHI_HOOKPOINT = "h.6.mlp"
DELPHI_SEQ_LEN = 256
DELPHI_BATCH_SIZE = 8
DELPHI_N_SPLITS = 5
DELPHI_TOKENS = {"smoke_2M": 2_000_000, "main_2M": 2_000_000, "main_10M": 10_000_000}

# Cost model (OpenRouter Llama-3.1-70B-Instruct)
PRICE_IN_PER_M = 0.40
PRICE_OUT_PER_M = 0.40
COST_CAP_USD = 20.0
OPENROUTER_ENV = "OPEN_ROUTER_DEV_KEY"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def molt_name(scale_name: str, lam: float) -> str:
    return f"molt_{scale_name}_lam{lam:.0e}"


def tc_name(scale_name: str, lam: float) -> str:
    return f"tc_{scale_name}_lam{lam:.0e}"


def scale_name_for(method: str, cfg: dict) -> str:
    if method == "molt":
        return {1: "1x", 2: "2x", 4: "4x"}[cfg["rank_multiplier"]]
    if method == "tc":
        return {2573: "1x", 5147: "2x", 10295: "4x"}[cfg["n_features"]]
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Training loops (ported from experiments/11 with state_dict saving added)
# ---------------------------------------------------------------------------

def train_molt_gpu(
    config: MOLTConfig,
    lam: float,
    train_in: torch.Tensor,
    train_out: torch.Tensor,
) -> tuple[MOLT, list[dict]]:
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
        pbar = tqdm(range(steps_per_epoch), desc=f"MOLT λ={lam:.0e} ep {epoch+1}/{config.num_epochs}")
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
    torch.manual_seed(SEED)
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
        pbar = tqdm(range(steps_per_epoch), desc=f"TC λ={sparsity_coeff:.0e} ep {epoch+1}/{num_epochs}")
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
# Stage 0: train + checkpoint
# ---------------------------------------------------------------------------

def prepare_activations():
    """Load + split cached activations once per run."""
    assert CACHE_PATH.exists(), f"Missing activation cache at {CACHE_PATH}"
    mlp_in, mlp_out = load_cached_activations(CACHE_PATH)
    mlp_in = mlp_in[: TRAIN_TOKENS + EVAL_SIZE]
    mlp_out = mlp_out[: TRAIN_TOKENS + EVAL_SIZE]
    train_in, train_out, eval_in, eval_out = split_train_eval(mlp_in, mlp_out, EVAL_SIZE)
    # Move to GPU (fits: 500K * 768 * 4 * 2 ≈ 3 GB)
    train_in = train_in.cuda()
    train_out = train_out.cuda()
    eval_in = eval_in.cuda()
    eval_out = eval_out.cuda()
    return train_in, train_out, eval_in, eval_out


def stage_train(slices: list[str]):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_in, train_out, eval_in, eval_out = prepare_activations()

    for slice_name in slices:
        slice_cfg = SLICES[slice_name]
        scale_name = {"smoke": "1x", "main": "4x"}[slice_name]

        # ---- MOLT ----
        molt_cfg_dict = slice_cfg["molt"]
        lam = molt_cfg_dict["lam"]
        ckpt_path = CHECKPOINTS_DIR / f"{molt_name(scale_name, lam)}.pt"
        if ckpt_path.exists():
            print(f"[skip] {ckpt_path.name} already exists")
        else:
            cfg = MOLTConfig.from_preset(
                MODEL,
                rank_multiplier=molt_cfg_dict["rank_multiplier"],
                num_epochs=molt_cfg_dict["num_epochs"],
                sparsity_coeff=lam,
                sparsity_type="tanh",
                activation="jumprelu",
                lr=LR,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            print(f"\n=== Training {molt_name(scale_name, lam)} ===")
            t0 = time.time()
            model, history = train_molt_gpu(cfg, lam, train_in, train_out)
            train_time = time.time() - t0

            # eval
            with torch.no_grad():
                out_eval, aux = model(eval_in)
                mse = F.mse_loss(out_eval, eval_out).item()
                nmse = mse / (eval_out.var().item() + 1e-8)
                gate_concat = torch.cat([g.clamp_min(0.0) for g in aux["gate_acts"]], dim=-1)
                l0 = (gate_concat > 0).float().sum(dim=-1).mean().item()

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "history": history,
                    "eval": {"nmse": nmse, "l0": l0, "train_time_s": train_time},
                    "method": "molt",
                    "scale": scale_name,
                    "lam": lam,
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path.name}  NMSE={nmse:.4f}  L0={l0:.1f}  t={train_time:.1f}s")

        # ---- Transcoder ----
        tc_cfg_dict = slice_cfg["tc"]
        tc_lam = tc_cfg_dict["lam"]
        tc_ckpt_path = CHECKPOINTS_DIR / f"{tc_name(scale_name, tc_lam)}.pt"
        if tc_ckpt_path.exists():
            print(f"[skip] {tc_ckpt_path.name} already exists")
        else:
            print(f"\n=== Training {tc_name(scale_name, tc_lam)} ===")
            t0 = time.time()
            model, history = train_transcoder_gpu(
                tc_cfg_dict["n_features"], tc_cfg_dict["num_epochs"], tc_lam,
                train_in, train_out,
            )
            train_time = time.time() - t0

            with torch.no_grad():
                out_eval, aux = model(eval_in)
                mse = F.mse_loss(out_eval, eval_out)
                nmse = (mse / (eval_out.var() + 1e-8)).item()
                l0 = (aux["feature_acts"] > 0).float().sum(dim=-1).mean().item()

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "d_model": D_MODEL,
                        "n_features": tc_cfg_dict["n_features"],
                        "num_epochs": tc_cfg_dict["num_epochs"],
                        "lam": tc_lam,
                    },
                    "history": history,
                    "eval": {"nmse": nmse, "l0": l0, "train_time_s": train_time},
                    "method": "tc",
                    "scale": scale_name,
                    "lam": tc_lam,
                },
                tc_ckpt_path,
            )
            print(f"[save] {tc_ckpt_path.name}  NMSE={nmse:.4f}  L0={l0:.1f}  t={train_time:.1f}s")

    # Write a summary JSON
    summary = {}
    for ckpt in sorted(CHECKPOINTS_DIR.glob("*.pt")):
        d = torch.load(ckpt, weights_only=False, map_location="cpu")
        summary[ckpt.stem] = {
            "method": d.get("method"),
            "scale": d.get("scale"),
            "lam": d.get("lam"),
            "eval": d.get("eval"),
        }
    (RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] Stage 0 summary written to {RESULTS_DIR / 'train_summary.json'}")


# ---------------------------------------------------------------------------
# Stage 0b: upload to HF (deferred — placeholder)
# ---------------------------------------------------------------------------

def stage_upload(slices):
    print("Stage 'upload' not yet implemented — checkpoints exist locally at",
          CHECKPOINTS_DIR)


# ---------------------------------------------------------------------------
# Stages 1-3: placeholders (implemented in follow-up commits)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1: build delphi latent cache
# ---------------------------------------------------------------------------

def _load_coder(ckpt_path: Path):
    """Load a checkpoint and return (shim, method, scale, lam, name)."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    method = ckpt["method"]
    scale = ckpt["scale"]
    lam = ckpt["lam"]
    name = ckpt_path.stem
    if method == "molt":
        cfg = MOLTConfig.from_dict(ckpt["config"])
        coder = MOLT(cfg).cuda()
        coder.load_state_dict(ckpt["state_dict"])
        coder.eval()
        shim = MOLTShim(coder)
    elif method == "tc":
        coder = TrainableTranscoder(D_MODEL, ckpt["config"]["n_features"]).cuda()
        coder.load_state_dict(ckpt["state_dict"])
        coder.eval()
        shim = TranscoderShim(coder)
    else:
        raise ValueError(f"Unknown method {method} in {ckpt_path}")
    return shim, method, scale, lam, name


def _prepare_tokens(n_tokens: int, cache_dir: Path) -> torch.Tensor:
    """Stream FineWeb and tokenize into (n_seq, seq_len). Cache to disk so
    multiple coders at the same n_tokens reuse one tokenization pass."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fineweb_gpt2_n{n_tokens}_seq{DELPHI_SEQ_LEN}.pt"
    if cache_file.exists():
        print(f"[tokens] Loading cached {cache_file.name}")
        return torch.load(cache_file, weights_only=True)

    print(f"[tokens] Streaming {n_tokens:,} FineWeb tokens (seq_len={DELPHI_SEQ_LEN}) ...")
    cfg = MOLTConfig.from_preset(
        MODEL, seq_len=DELPHI_SEQ_LEN, num_tokens=n_tokens, seed=SEED,
    )
    from molt.data import stream_fineweb_tokens
    chunks = stream_fineweb_tokens(cfg, num_tokens=n_tokens)
    tokens = torch.stack(chunks, dim=0)  # (n_seq, seq_len)
    torch.save(tokens, cache_file)
    print(f"[tokens] Saved {tokens.shape} → {cache_file}")
    return tokens


def stage_cache(slices, args):
    """Build delphi's LatentCache for each checkpoint in the selected slices.

    Uses transcode=True so the hook captures the INPUT of h.6.mlp (what both
    MOLT and Transcoder consume). Output: latents/<ckpt_name>/<hookpoint>/*.safetensors
    """
    from transformers import AutoModel
    from delphi.latents import LatentCache

    n_tokens = getattr(args, "n_tokens", None) or 2_000_000
    tokens = _prepare_tokens(n_tokens, TOKEN_CACHE_DIR)

    print(f"[model] Loading {MODEL} on cuda...")
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
    model.eval()

    # Resolve which checkpoints to run
    want_names = set()
    for slice_name in slices:
        scfg = SLICES[slice_name]
        scale_name = {"smoke": "1x", "main": "4x"}[slice_name]
        want_names.add(f"molt_{scale_name}_lam{scfg['molt']['lam']:.0e}")
        want_names.add(f"tc_{scale_name}_lam{scfg['tc']['lam']:.0e}")

    for ckpt_path in sorted(CHECKPOINTS_DIR.glob("*.pt")):
        if ckpt_path.stem not in want_names:
            continue

        out_dir = LATENTS_DIR / f"n{n_tokens}" / ckpt_path.stem
        done_marker = out_dir / "DONE"
        if done_marker.exists():
            print(f"[skip] cache already exists at {out_dir}")
            continue

        shim, method, scale, lam, name = _load_coder(ckpt_path)
        print(f"\n=== Caching {name}  "
              f"(n_features={shim.n_features}, n_tokens={n_tokens:,}) ===")

        out_dir.mkdir(parents=True, exist_ok=True)
        cache = LatentCache(
            model=model,
            hookpoint_to_sparse_encode={DELPHI_HOOKPOINT: shim},
            batch_size=DELPHI_BATCH_SIZE,
            transcode=True,  # capture MLP INPUT, not output
        )

        t0 = time.time()
        cache.run(n_tokens=n_tokens, tokens=tokens)
        elapsed = time.time() - t0

        print(f"[save] Splitting into {DELPHI_N_SPLITS} shards → {out_dir}")
        cache.save_splits(n_splits=DELPHI_N_SPLITS, save_dir=out_dir)

        # Write the config.json delphi's LatentDataset expects in each module dir
        module_dir = out_dir / DELPHI_HOOKPOINT
        cache_config = {
            "model_name": MODEL,
            "ctx_len": DELPHI_SEQ_LEN,
            "dataset_repo": "HuggingFaceFW/fineweb",
            "dataset_split": "train",
            "dataset_name": "",
            "dataset_column": "text",
            "n_tokens": n_tokens,
            "batch_size": DELPHI_BATCH_SIZE,
            "n_splits": DELPHI_N_SPLITS,
        }
        (module_dir / "config.json").write_text(json.dumps(cache_config, indent=2))
        done_marker.write_text(json.dumps({
            "name": name,
            "method": method,
            "scale": scale,
            "lam": lam,
            "n_features": shim.n_features,
            "n_tokens": n_tokens,
            "seq_len": DELPHI_SEQ_LEN,
            "hookpoint": DELPHI_HOOKPOINT,
            "wall_time_s": elapsed,
        }, indent=2))
        print(f"[done] {name} cached in {elapsed:.1f}s")

        # Free GPU memory for the next coder
        del shim
        torch.cuda.empty_cache()

    print(f"\n[done] Stage 1 complete for slices {slices}")


# ---------------------------------------------------------------------------
# Stages 2 & 3: explain + score via delphi
# ---------------------------------------------------------------------------

EXPLAINER_MODEL = "meta-llama/llama-3.1-70b-instruct"
EXPLAINER_MAX_TOKENS = 300          # short labels
SCORER_MAX_TOKENS = 250
ACT_FREQ_LOW = 1e-4                 # ~200+ activations at 2M tokens
ACT_FREQ_HIGH = 0.2                 # drop hyper-dense features
BUDGET_CAP_USD = 20.0
BUDGET_CONFIRM_THRESHOLD_USD = 15.0
SAMPLER_N_EXAMPLES_TRAIN = 40
SAMPLER_N_EXAMPLES_TEST = 50


def _compute_activation_freq(cache_dir: Path, n_features: int) -> torch.Tensor:
    """Per-feature activation frequency from saved shards: nonzeros / total_tokens."""
    from safetensors.torch import load_file
    counts = torch.zeros(n_features, dtype=torch.int64)
    n_seq = n_seq_len = None
    for shard in sorted(cache_dir.glob("*.safetensors")):
        d = load_file(shard)
        locs = d["locations"].to(torch.int64)
        # uint16 storage stores (batch, seq, feature_within_shard) offset by start
        start = int(shard.stem.split("_")[0])
        feat_ids = locs[:, 2] + start
        counts += torch.bincount(feat_ids, minlength=n_features)
        if n_seq is None:
            n_seq, n_seq_len = d["tokens"].shape
    total = (n_seq or 0) * (n_seq_len or 0)
    return counts.float() / max(total, 1)


def _select_features(
    cache_dir: Path, n_features: int, method: str
) -> tuple[torch.Tensor, dict]:
    """Filter features by activation frequency. Bounds differ by method:

    - MOLT: transforms are structurally dense (few transforms, each fires
      on a large fraction of tokens). We keep all transforms that fire on
      at least ACT_FREQ_LOW of tokens — no upper bound.
    - Transcoder: dictionary-style features; apply the full
      (ACT_FREQ_LOW, ACT_FREQ_HIGH) filter.
    """
    freq = _compute_activation_freq(cache_dir, n_features)
    low_mask = freq > ACT_FREQ_LOW
    if method == "molt":
        keep_mask = low_mask
    else:  # tc
        keep_mask = low_mask & (freq < ACT_FREQ_HIGH)
    kept = torch.where(keep_mask)[0]
    stats = {
        "method": method,
        "n_total": n_features,
        "n_kept": int(keep_mask.sum().item()),
        "n_dead": int((~low_mask).sum().item()),
        "n_hyperdense": int((freq >= ACT_FREQ_HIGH).sum().item()),
        "freq_min": float(freq.min().item()),
        "freq_max": float(freq.max().item()),
        "freq_median": float(freq.median().item()),
        "freq_bounds": [ACT_FREQ_LOW, ACT_FREQ_HIGH if method == "tc" else None],
    }
    return kept, stats


def _ckpt_names_for_slices(slices: list[str]) -> list[str]:
    names = []
    for slice_name in slices:
        scfg = SLICES[slice_name]
        scale_name = {"smoke": "1x", "main": "4x"}[slice_name]
        names.append(molt_name(scale_name, scfg["molt"]["lam"]))
        names.append(tc_name(scale_name, scfg["tc"]["lam"]))
    return names


def _n_features_for(ckpt_name: str) -> int:
    ckpt = torch.load(CHECKPOINTS_DIR / f"{ckpt_name}.pt", weights_only=False, map_location="cpu")
    if ckpt["method"] == "molt":
        cfg = MOLTConfig.from_dict(ckpt["config"])
        return cfg.total_transforms
    return ckpt["config"]["n_features"]


def _require_openrouter_key() -> str:
    key = os.environ.get(OPENROUTER_ENV)
    if not key:
        print(f"[blocker] {OPENROUTER_ENV} not set.")
        sys.exit(2)
    return key


def stage_explain(slices, args):
    """Generate feature labels via delphi's DefaultExplainer + Llama-3.1-70B on OpenRouter."""
    import asyncio
    import orjson
    from functools import partial
    from transformers import AutoTokenizer

    from delphi.config import ConstructorConfig, SamplerConfig
    from delphi.explainers import DefaultExplainer
    from delphi.latents import LatentDataset
    from delphi.pipeline import Pipe, Pipeline, process_wrapper

    from budget import BudgetedOpenRouter, CostTracker, BudgetExceeded, check_openrouter_credits

    sys.path.insert(0, str(EXPERIMENT_DIR))  # ensure `budget` importable

    api_key = _require_openrouter_key()
    n_tokens = getattr(args, "n_tokens", None) or 2_000_000

    # Pre-flight credits + budget check
    try:
        credits = check_openrouter_credits(api_key)
        print(f"[credits] OpenRouter: {credits.get('data', credits)}")
    except Exception as e:
        print(f"[warn] credits check failed ({e}); proceeding")

    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    tracker = CostTracker(
        cap_usd=BUDGET_CAP_USD,
        log_path=RESULTS_DIR / f"budget_log_{args.stage}.jsonl",
    )

    for ckpt_name in _ckpt_names_for_slices(slices):
        cache_dir = LATENTS_DIR / f"n{n_tokens}" / ckpt_name / DELPHI_HOOKPOINT
        if not cache_dir.exists():
            print(f"[skip] no cache at {cache_dir}")
            continue

        n_feat = _n_features_for(ckpt_name)
        method = "molt" if ckpt_name.startswith("molt_") else "tc"
        kept, stats = _select_features(cache_dir, n_feat, method)
        print(f"\n=== Explain {ckpt_name}  ({stats['n_kept']}/{stats['n_total']} "
              f"features; dead={stats['n_dead']}, hyperdense={stats['n_hyperdense']}) ===")

        # Pre-flight estimate: ~$0.0017 per feature
        est_cost = stats["n_kept"] * 0.00084
        print(f"[estimate] ~${est_cost:.2f} for {stats['n_kept']} features "
              f"(remaining budget ${tracker.remaining():.2f})")
        if est_cost > BUDGET_CONFIRM_THRESHOLD_USD and not args.confirm_budget:
            print(f"[blocker] estimated ${est_cost:.2f} > ${BUDGET_CONFIRM_THRESHOLD_USD} "
                  f"— re-run with --confirm-budget")
            sys.exit(3)

        out_dir = EXPLANATIONS_DIR / ckpt_name
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = RESULTS_DIR / f"feature_stats_{ckpt_name}.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(
            {**stats, "kept_feature_ids": kept.tolist()}, indent=2))

        # Skip features already explained (delphi saves as "{module}_latent{idx}.txt")
        already = set()
        for p in out_dir.glob("*.txt"):
            if "_latent" in p.stem:
                try:
                    already.add(int(p.stem.rsplit("_latent", 1)[1]))
                except ValueError:
                    pass
        remaining = [int(x.item()) for x in kept if int(x.item()) not in already]
        if not remaining:
            print(f"[skip] all {len(kept)} explanations already exist for {ckpt_name}")
            continue
        kept_tensor = torch.tensor(remaining, dtype=torch.int64)

        dataset = LatentDataset(
            raw_dir=LATENTS_DIR / f"n{n_tokens}" / ckpt_name,
            modules=[DELPHI_HOOKPOINT],
            sampler_cfg=SamplerConfig(
                n_examples_train=SAMPLER_N_EXAMPLES_TRAIN,
                n_examples_test=SAMPLER_N_EXAMPLES_TEST,
            ),
            constructor_cfg=ConstructorConfig(
                non_activating_source="random",  # cheap baseline; upgrade to FAISS later
            ),
            tokenizer=tok,
            latents={DELPHI_HOOKPOINT: kept_tensor},
        )

        client = BudgetedOpenRouter(
            model=EXPLAINER_MODEL, tracker=tracker, api_key=api_key,
            max_tokens=EXPLAINER_MAX_TOKENS, temperature=0.0,
        )
        explainer = DefaultExplainer(client=client, threshold=0.3)

        # Use delphi's Latent.__str__ as the filename so delphi.explanation_loader
        # can find them in stage 3 without custom wiring.
        def save_explanation(result, out_dir=out_dir):
            fname = str(result.record.latent)  # e.g. "h.6.mlp_latent42"
            with (out_dir / f"{fname}.txt").open("wb") as f:
                f.write(orjson.dumps(result.explanation))
            return result

        pipeline = Pipeline(
            dataset,
            Pipe(process_wrapper(explainer, postprocess=save_explanation)),
        )
        print(f"[run] labeling {len(remaining)} features ...")
        try:
            asyncio.run(pipeline.run(1))   # OpenRouter → single-thread
        except BudgetExceeded as e:
            print(f"[stop] budget exceeded: {e}")
            break

        snap = tracker.snapshot()
        print(f"[done] {ckpt_name}: ${snap['spent']:.4f} spent "
              f"(in={snap['prompt_tokens']} out={snap['completion_tokens']})")

    (RESULTS_DIR / "budget_summary.json").write_text(json.dumps(tracker.snapshot(), indent=2))
    print(f"\n[final] total spend ${tracker.spent:.4f} / ${tracker.cap:.2f}")


def stage_score(slices, args):
    """Score saved explanations with DetectionScorer + FuzzingScorer."""
    import asyncio
    import orjson
    from functools import partial
    from transformers import AutoTokenizer

    from delphi.config import ConstructorConfig, SamplerConfig
    from delphi.explainers import explanation_loader
    from delphi.latents import LatentDataset
    from delphi.pipeline import Pipe, Pipeline, process_wrapper
    from delphi.scorers import DetectionScorer, FuzzingScorer

    from budget import BudgetedOpenRouter, CostTracker, BudgetExceeded, check_openrouter_credits

    sys.path.insert(0, str(EXPERIMENT_DIR))

    api_key = _require_openrouter_key()
    n_tokens = getattr(args, "n_tokens", None) or 2_000_000
    tok = AutoTokenizer.from_pretrained(MODEL)

    tracker = CostTracker(
        cap_usd=BUDGET_CAP_USD,
        log_path=RESULTS_DIR / f"budget_log_{args.stage}.jsonl",
    )

    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    for ckpt_name in _ckpt_names_for_slices(slices):
        cache_dir = LATENTS_DIR / f"n{n_tokens}" / ckpt_name / DELPHI_HOOKPOINT
        exp_dir = EXPLANATIONS_DIR / ckpt_name
        if not cache_dir.exists() or not exp_dir.exists():
            print(f"[skip] missing cache or explanations for {ckpt_name}")
            continue

        explained_ids = []
        for p in exp_dir.glob("*.txt"):
            if "_latent" in p.stem:
                try:
                    explained_ids.append(int(p.stem.rsplit("_latent", 1)[1]))
                except ValueError:
                    pass
        explained_ids = sorted(explained_ids)
        if not explained_ids:
            print(f"[skip] no explanations for {ckpt_name}")
            continue

        kept_tensor = torch.tensor(explained_ids, dtype=torch.int64)
        print(f"\n=== Score {ckpt_name}  ({len(explained_ids)} explained features) ===")

        dataset = LatentDataset(
            raw_dir=LATENTS_DIR / f"n{n_tokens}" / ckpt_name,
            modules=[DELPHI_HOOKPOINT],
            sampler_cfg=SamplerConfig(
                n_examples_train=SAMPLER_N_EXAMPLES_TRAIN,
                n_examples_test=SAMPLER_N_EXAMPLES_TEST,
            ),
            constructor_cfg=ConstructorConfig(
                non_activating_source="random",
            ),
            tokenizer=tok,
            latents={DELPHI_HOOKPOINT: kept_tensor},
        )

        client = BudgetedOpenRouter(
            model=EXPLAINER_MODEL, tracker=tracker, api_key=api_key,
            max_tokens=SCORER_MAX_TOKENS, temperature=0.0,
        )
        detection_dir = SCORES_DIR / "detection" / ckpt_name
        fuzzing_dir   = SCORES_DIR / "fuzzing" / ckpt_name
        detection_dir.mkdir(parents=True, exist_ok=True)
        fuzzing_dir.mkdir(parents=True, exist_ok=True)

        # explanation loader attaches the saved text to the record
        loader = partial(explanation_loader, explanation_dir=str(exp_dir))

        def scorer_preprocess(result):
            if isinstance(result, list):
                result = result[0]
            record = result.record
            record.explanation = result.explanation
            record.extra_examples = record.not_active
            return record

        def scorer_postprocess(result, score_dir):
            fname = str(result.record.latent)  # match explainer naming
            with (score_dir / f"{fname}.txt").open("wb") as f:
                f.write(orjson.dumps(result.score))
            return result

        detection = DetectionScorer(client, n_examples_shown=5, verbose=False)
        fuzzing   = FuzzingScorer(client,   n_examples_shown=5, verbose=False)

        pipeline = Pipeline(
            dataset,
            Pipe(process_wrapper(loader)),
            Pipe(
                process_wrapper(detection, preprocess=scorer_preprocess,
                                postprocess=partial(scorer_postprocess, score_dir=detection_dir)),
                process_wrapper(fuzzing, preprocess=scorer_preprocess,
                                postprocess=partial(scorer_postprocess, score_dir=fuzzing_dir)),
            ),
        )

        print(f"[run] scoring {len(explained_ids)} features ...")
        try:
            asyncio.run(pipeline.run(1))
        except BudgetExceeded as e:
            print(f"[stop] budget exceeded: {e}")
            break

        snap = tracker.snapshot()
        print(f"[done] {ckpt_name}: ${snap['spent']:.4f} spent "
              f"(in={snap['prompt_tokens']} out={snap['completion_tokens']})")

    (RESULTS_DIR / "budget_summary_score.json").write_text(json.dumps(tracker.snapshot(), indent=2))
    print(f"\n[final] total spend ${tracker.spent:.4f} / ${tracker.cap:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train", "upload", "cache", "explain", "score"])
    ap.add_argument("--slice", default="all",
                    choices=["smoke", "main", "all"])
    ap.add_argument("--confirm-budget", action="store_true",
                    help="Required for explain/score stages with estimated cost > $15")
    ap.add_argument("--n-tokens", type=int, default=2_000_000,
                    help="Tokens for delphi cache (default 2M; use 10M for main rerun)")
    args = ap.parse_args()

    slices = ["smoke", "main"] if args.slice == "all" else [args.slice]

    dispatch = {
        "train":   lambda: stage_train(slices),
        "upload":  lambda: stage_upload(slices),
        "cache":   lambda: stage_cache(slices, args),
        "explain": lambda: stage_explain(slices, args),
        "score":   lambda: stage_score(slices, args),
    }
    dispatch[args.stage]()


if __name__ == "__main__":
    main()
