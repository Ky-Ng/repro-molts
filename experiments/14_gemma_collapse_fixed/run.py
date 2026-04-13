#!/usr/bin/env python3
"""Experiment 14: Gemma-3-1B with the post-port MOLT (fixes from DIFFERENCES.md).

Re-runs a small Gemma-3-1B sparsity sweep after porting the MOLT model,
JumpReLU, and standardizers from `crosslayer-transcoder`. The point is to
check whether the L0=1 collapse that motivated experiments 01/02/06 still
occurs now that bugs A-F (per-feature learnable theta, true ||UV||_F,
per-token tanh sparsity, c_sparsity, input/output standardizers, ReLU mask
on the JumpReLU forward) have been fixed.

Usage:
    uv run python experiments/14_gemma_post_port/run.py
"""

from pathlib import Path

from molt.config import MOLTConfig
from molt.data import collect_activations, stream_fineweb_tokens
from molt.utils.experiment import ExperimentRunner
from molt.utils.plotting import plot_multi_run_curves

EXPERIMENT_DIR = Path(__file__).parent

# Small but non-trivial: 3 lambda values around the ported default (5e-4).
SETUPS = [
    {"name": "lambda_0",    "sparsity_coeff": 0.0},
    {"name": "lambda_5e-4", "sparsity_coeff": 5e-4},
    {"name": "lambda_5e-3", "sparsity_coeff": 5e-3},
]

BASE_CONFIG = dict(
    model_name="google/gemma-3-1b-it",
    d_model=1152,
    layer_idx=13,
    mlp_path="model.layers.{layer_idx}.mlp",
    model_dtype="bfloat16",
    num_tokens=1_000_000,
    batch_size=512,
    lr=1e-3,
    device="cuda",
    sparsity_warmup_frac=0.1,
    log_every=50,
    # defaults already match the ported reference (learned per-feature theta,
    # c_sparsity=100, use_tanh=True), but pin them explicitly for clarity.
    learned_threshold=True,
    jumprelu_threshold=0.0,
    jumprelu_bandwidth=1.0,
    c_sparsity=100.0,
    use_tanh=True,
)


def main():
    cfg_for_data = MOLTConfig(**BASE_CONFIG)
    print(f"=== Collecting {cfg_for_data.num_tokens/1e6:.1f}M tokens of "
          f"{cfg_for_data.model_name} layer {cfg_for_data.layer_idx} activations ===")
    token_chunks = stream_fineweb_tokens(cfg_for_data)
    mlp_inputs, mlp_outputs = collect_activations(cfg_for_data, token_chunks, cache_path=None)
    del token_chunks

    runner = ExperimentRunner(EXPERIMENT_DIR)
    for setup in SETUPS:
        overrides = {k: v for k, v in setup.items() if k != "name"}
        config = MOLTConfig(**{**BASE_CONFIG, **overrides})
        runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)

    runner.save_summary(title="Exp 14: Gemma-3-1B post-port sparsity sweep")

    runs = {r["name"]: r["history"] for r in runner.all_results}
    plot_multi_run_curves(
        runs,
        "Exp 14: Gemma-3-1B post-port sparsity sweep",
        runner.figures_dir / "comparison.png",
    )


if __name__ == "__main__":
    main()
