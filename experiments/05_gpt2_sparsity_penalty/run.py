#!/usr/bin/env python3
"""GPT-2 sparsity penalty sweep: {ReLU, JumpReLU} x {Tanh, L0} x {lambda=1e-5, 1e-4}.

Tests whether non-zero sparsity penalties produce a meaningful L0 vs NMSE tradeoff.

Usage:
    uv run python experiments/05_gpt2_sparsity_penalty/run.py [setup_name]
"""

import sys
from pathlib import Path

from molt.config import MOLTConfig
from molt.utils.activations import load_cached_activations
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent
MODEL = "openai-community/gpt2"
CACHE_PATH = "data/activations_openai_community_gpt2_2M.pt"

SETUPS = []
for lam in [1e-5, 1e-4]:
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
        config = MOLTConfig.from_preset(MODEL, **{**BASE_CONFIG, **overrides})
        runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)

    runner.save_summary(title="GPT-2 Sparsity Penalty Sweep")

    # Plot per-lambda comparisons
    for lam in [1e-5, 1e-4]:
        lam_results = {r["name"]: r["history"] for r in runner.all_results
                       if r["sparsity_coeff"] == lam}
        if lam_results:
            plot_multi_run_curves(lam_results, f"GPT-2 Sparsity Sweep (lambda={lam:.0e})",
                                  runner.figures_dir / f"sparsity_sweep_lam{lam:.0e}.png")


if __name__ == "__main__":
    main()
