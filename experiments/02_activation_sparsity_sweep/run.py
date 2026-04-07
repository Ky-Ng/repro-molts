#!/usr/bin/env python3
"""Sweep activation/sparsity setups on Gemma-3-1B with lambda=0.

Tests how gating activation (ReLU vs JumpReLU) and sparsity penalty type
(Tanh vs L1) affect training dynamics and collapse behavior.

Usage:
    uv run python experiments/02_activation_sparsity_sweep/run.py
"""

from pathlib import Path

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent

SETUPS = [
    {"name": "tanh_relu",   "sparsity_type": "tanh", "activation": "relu"},
    {"name": "l1_relu",     "sparsity_type": "l1",   "activation": "relu"},
    {"name": "l1_jumprelu", "sparsity_type": "l1",   "activation": "jumprelu"},
]

BASE_CONFIG = dict(
    num_tokens=2_000_000,
    batch_size=64,
    device="cuda",
    sparsity_coeff=0.0,
    sparsity_warmup_frac=0.0,
    log_every=100,
)


def main():
    # Collect activations
    base_config = MOLTConfig(num_tokens=2_000_000, batch_size=64, device="cuda")
    cache_path = "data/activations_2M.pt"

    print("Streaming tokens...")
    token_chunks = stream_fineweb_tokens(base_config)
    mlp_inputs, mlp_outputs = collect_activations(base_config, token_chunks, cache_path=cache_path)
    del token_chunks

    runner = ExperimentRunner(EXPERIMENT_DIR)

    for setup in SETUPS:
        overrides = {k: v for k, v in setup.items() if k != "name"}
        config = MOLTConfig(**{**BASE_CONFIG, **overrides})
        runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)

    runner.save_summary(title="Activation/Sparsity Setup Sweep (lambda=0)")

    # Plot
    runs = {r["name"]: r["history"] for r in runner.all_results}
    plot_multi_run_curves(runs, "Activation/Sparsity Setup Sweep (lambda=0)",
                          runner.figures_dir / "activation_sweep_comparison.png")


if __name__ == "__main__":
    main()
