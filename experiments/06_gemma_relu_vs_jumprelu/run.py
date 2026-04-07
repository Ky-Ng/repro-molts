#!/usr/bin/env python3
"""Gemma-3-1B: ReLU vs JumpReLU (smooth surrogate, learned theta), lambda=0.

Tests whether the smooth surrogate that resolved collapse on GPT-2 also works on Gemma.

Usage:
    uv run python experiments/06_gemma_relu_vs_jumprelu/run.py [setup_name]
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
    {"name": "relu",             "activation": "relu",     "learned_threshold": False},
    {"name": "jumprelu_learned", "activation": "jumprelu", "learned_threshold": True},
]

BASE_CONFIG = dict(
    num_tokens=2_000_000,
    batch_size=64,
    device="cuda",
    sparsity_coeff=0.0,
    sparsity_warmup_frac=0.0,
    sparsity_type="tanh",
    jumprelu_threshold=0.0,
    log_every=100,
)


def main():
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)

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

    runner.save_summary(title="Gemma-3-1B: ReLU vs JumpReLU (Smooth Surrogate)")

    runs = {r["name"]: r["history"] for r in runner.all_results}
    plot_multi_run_curves(runs, "Gemma-3-1B: ReLU vs JumpReLU (lambda=0)",
                          runner.figures_dir / "relu_vs_jumprelu_gemma.png")


if __name__ == "__main__":
    main()
