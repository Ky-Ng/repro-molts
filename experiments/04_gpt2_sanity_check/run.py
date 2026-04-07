#!/usr/bin/env python3
"""GPT-2 sanity check: {Tanh, L1} x {ReLU, JumpReLU} with lambda=0.

Tests whether L0 collapse is model-specific or universal, and validates
the smooth surrogate JumpReLU backward.

Usage:
    uv run python experiments/04_gpt2_sanity_check/run.py [setup_name]
"""

import sys
from pathlib import Path

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent
MODEL = "openai-community/gpt2"

SETUPS = [
    {"name": "tanh_relu",     "sparsity_type": "tanh", "activation": "relu"},
    {"name": "tanh_jumprelu", "sparsity_type": "tanh", "activation": "jumprelu"},
    {"name": "l1_relu",       "sparsity_type": "l1",   "activation": "relu"},
    {"name": "l1_jumprelu",   "sparsity_type": "l1",   "activation": "jumprelu"},
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
    # Load or collect activations
    cache_path = "data/activations_openai_community_gpt2_2M.pt"
    base_config = MOLTConfig.from_preset(MODEL, num_tokens=2_000_000, device="cuda")

    if Path(cache_path).exists():
        from molt.utils.activations import load_cached_activations
        mlp_inputs, mlp_outputs = load_cached_activations(cache_path)
    else:
        Path("data").mkdir(parents=True, exist_ok=True)
        token_chunks = stream_fineweb_tokens(base_config)
        mlp_inputs, mlp_outputs = collect_activations(base_config, token_chunks, cache_path=cache_path)
        del token_chunks

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

    runner.save_summary(title=f"GPT-2 Sanity Check ({MODEL})")

    # Plot
    runs = {r["name"]: r["history"] for r in runner.all_results}
    plot_multi_run_curves(runs, "GPT-2 Activation/Sparsity Sweep (lambda=0)",
                          runner.figures_dir / "sweep_comparison.png")


if __name__ == "__main__":
    main()
