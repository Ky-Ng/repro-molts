#!/usr/bin/env python3
"""Experiment 08: Anti-collapse interventions for Gemma-3-1B.

Tests four approaches to prevent L0 collapse on Gemma:
1. Fixed θ=0 with smooth surrogate (missing baseline from exp 06)
2. Threshold warmup: freeze θ for first N% of training, then unfreeze
3. Separate LR for θ: slow down threshold learning
4. Increased max rank (768) to match GPT-2's coverage ratio

All runs use λ=0 (no sparsity penalty) to isolate gating dynamics.

Usage:
    uv run python experiments/08_gemma_anti_collapse/run.py
"""

import os
import sys
from pathlib import Path

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.utils.activations import load_cached_activations
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent

# --- Configuration ---

# Setup 1: Missing baseline — fixed θ=0, smooth surrogate JumpReLU
# Setup 2: Threshold warmup — freeze θ for 25/50/75% of training
# Setup 3: Separate θ LR — slow θ down to 0.1x/0.01x/0.001x base LR
# Setup 4: Higher max rank (768) — match GPT-2's 67% coverage ratio
SETUPS = [
    # Setup 1: Fixed θ baseline
    {
        "name": "fixed_theta_surrogate",
        "activation": "jumprelu",
        "learned_threshold": False,
        "jumprelu_threshold": 0.0,
    },
    # Setup 2: Threshold warmup sweep
    {
        "name": "theta_warmup_25pct",
        "activation": "jumprelu",
        "learned_threshold": True,
        "threshold_freeze_frac": 0.25,
    },
    {
        "name": "theta_warmup_50pct",
        "activation": "jumprelu",
        "learned_threshold": True,
        "threshold_freeze_frac": 0.50,
    },
    {
        "name": "theta_warmup_75pct",
        "activation": "jumprelu",
        "learned_threshold": True,
        "threshold_freeze_frac": 0.75,
    },
    # Setup 3: Separate θ LR sweep
    {
        "name": "theta_lr_1e-4",
        "activation": "jumprelu",
        "learned_threshold": True,
        "threshold_lr": 1e-4,
    },
    {
        "name": "theta_lr_1e-5",
        "activation": "jumprelu",
        "learned_threshold": True,
        "threshold_lr": 1e-5,
    },
    {
        "name": "theta_lr_1e-6",
        "activation": "jumprelu",
        "learned_threshold": True,
        "threshold_lr": 1e-6,
    },
    # Setup 4: Increased max rank
    {
        "name": "max_rank_768",
        "activation": "jumprelu",
        "learned_threshold": False,
        "jumprelu_threshold": 0.0,
        "max_rank": 768,
    },
]

# Shared config: Gemma-3-1B-IT, 2M tokens, no sparsity penalty
BASE_CONFIG = dict(
    model_name="google/gemma-3-1b-it",
    d_model=1152,
    layer_idx=13,
    num_tokens=2_000_000,
    batch_size=64,
    lr=1e-3,
    device="cuda",
    sparsity_coeff=0.0,
    sparsity_warmup_frac=0.0,
    log_every=100,
)

CACHE_PATH = "data/activations_2M.pt"


def main():
    if not os.path.exists(CACHE_PATH):
        print("Cached activations not found — collecting from Gemma-3-1B-IT...")
        config = MOLTConfig(**BASE_CONFIG)
        token_chunks = stream_fineweb_tokens(config, num_tokens=2_000_000)
        collect_activations(config, token_chunks, cache_path=CACHE_PATH)

    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)

    # Support running a single setup by name: python run.py <setup_name>
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        setups = [s for s in SETUPS if s["name"] == target]
        if not setups:
            print(f"Unknown: {target}. Options: {[s['name'] for s in SETUPS]}")
            sys.exit(1)
    else:
        setups = SETUPS

    runner = ExperimentRunner(EXPERIMENT_DIR)

    for setup in setups:
        overrides = {k: v for k, v in setup.items() if k != "name"}
        config = MOLTConfig(**{**BASE_CONFIG, **overrides})
        runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)

    if not target:
        runner.save_summary(title="Exp 08: Gemma Anti-Collapse")

        # Plot comparison
        runs = {r["name"]: r["history"] for r in runner.all_results}
        plot_multi_run_curves(
            runs,
            "Exp 08: Gemma Anti-Collapse",
            runner.figures_dir / "comparison.png",
        )


if __name__ == "__main__":
    main()
