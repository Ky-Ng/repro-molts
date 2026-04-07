#!/usr/bin/env python3
"""Learned vs fixed JumpReLU threshold on Gemma-3-1B (lambda=0).

Tests whether making theta a learnable parameter resolves the L0=1 collapse.

Usage:
    uv run python experiments/03_learned_threshold/run.py
"""

import sys
from pathlib import Path

from molt.config import MOLTConfig
from molt.utils.activations import load_cached_activations
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent
CACHE_PATH = "data/activations_2M.pt"

SETUPS = [
    {"name": "tanh_jumprelu_fixed",   "sparsity_type": "tanh", "learned_threshold": False},
    {"name": "tanh_jumprelu_learned", "sparsity_type": "tanh", "learned_threshold": True},
    {"name": "l1_jumprelu_fixed",     "sparsity_type": "l1",   "learned_threshold": False},
    {"name": "l1_jumprelu_learned",   "sparsity_type": "l1",   "learned_threshold": True},
]

BASE_CONFIG = dict(
    num_tokens=2_000_000,
    batch_size=64,
    device="cuda",
    sparsity_coeff=0.0,
    sparsity_warmup_frac=0.0,
    activation="jumprelu",
    jumprelu_threshold=0.0,
    log_every=100,
)


def main():
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)

    # Optional: run a single setup by name
    target = sys.argv[1] if len(sys.argv) > 1 else None
    setups = [s for s in SETUPS if s["name"] == target] if target else SETUPS
    if target and not setups:
        print(f"Unknown: {target}. Options: {[s['name'] for s in SETUPS]}")
        sys.exit(1)

    runner = ExperimentRunner(EXPERIMENT_DIR)

    for setup in setups:
        overrides = {k: v for k, v in setup.items() if k != "name"}
        config = MOLTConfig(**{**BASE_CONFIG, **overrides})
        runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)

    runner.save_summary(title="Learned vs Fixed JumpReLU Threshold")

    # Plot
    runs = {r["name"]: r["history"] for r in runner.all_results}
    plot_multi_run_curves(runs, "Learned vs Fixed JumpReLU Threshold (lambda=0)",
                          runner.figures_dir / "learned_threshold_comparison.png")


if __name__ == "__main__":
    main()
