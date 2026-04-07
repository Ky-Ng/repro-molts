#!/usr/bin/env python3
"""Template experiment — copy this folder to start a new experiment.

Usage:
    uv run python experiments/NN_my_experiment/run.py
"""

from pathlib import Path

from molt.config import MOLTConfig
from molt.utils.activations import load_cached_activations
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent

# --- Configuration ---
# Define your experiment setups here. Each setup is a dict of MOLTConfig overrides.
SETUPS = [
    {"name": "baseline", "activation": "jumprelu", "sparsity_type": "tanh"},
    # Add more setups...
]

# Shared config for all setups
BASE_CONFIG = dict(
    num_tokens=2_000_000,
    batch_size=64,
    device="cuda",
    sparsity_coeff=0.0,
    sparsity_warmup_frac=0.0,
    log_every=100,
)

# Path to cached activations
CACHE_PATH = "data/activations_2M.pt"


def main():
    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)
    runner = ExperimentRunner(EXPERIMENT_DIR)

    for setup in SETUPS:
        overrides = {k: v for k, v in setup.items() if k != "name"}
        config = MOLTConfig(**{**BASE_CONFIG, **overrides})
        runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)

    runner.save_summary(title="My Experiment")

    # Plot comparison
    runs = {r["name"]: r["history"] for r in runner.all_results}
    plot_multi_run_curves(runs, "My Experiment", runner.figures_dir / "comparison.png")


if __name__ == "__main__":
    main()
