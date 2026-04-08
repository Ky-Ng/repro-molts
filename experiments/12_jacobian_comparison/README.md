# Experiment 12: L0 vs Jacobian Faithfulness — MOLT vs Transcoder

## Goal

Compute Jacobian cosine similarity for the MOLT vs transcoder comparison from experiment 11. Experiment 11 showed MOLTs Pareto-dominate transcoders in L0 vs NMSE, but did not compute Jacobian faithfulness (a more meaningful metric for mechanistic interpretability). This experiment fills that gap.

## Setup

- **Model:** GPT-2 (768 dims), layer 6 MLP
- **Data:** 500K FineWeb activations (train), 10K (eval), 128 samples for Jacobian
- **Architecture:** Same MOLT and transcoder configs as experiment 11
- **Training:** Same hyperparameters as experiment 11 (Adam, lr=1e-3, batch=1024)
- **Jacobian:** Cosine similarity of flattened Jacobian matrices (MOLT vs true MLP)

| Scale | MOLT Config | MOLT Params | Transcoder Features | Transcoder Params |
|-------|-------------|-------------|---------------------|-------------------|
| 1x    | N=1 (31 transforms) | ~3.96M | 2,573 | ~3.96M |
| 2x    | N=2 (62 transforms) | ~7.92M | 5,147 | ~7.92M |
| 4x    | N=4 (124 transforms) | ~15.8M | 10,295 | ~15.8M |

## Reproduction

```bash
uv run python experiments/12_jacobian_comparison/run.py
uv run python experiments/12_jacobian_comparison/run.py --scale 1x
uv run python experiments/12_jacobian_comparison/run.py --plot-only
```

## Results

*To be filled after experiment completes.*

## Artifacts

- `results/result_*.json` — Per-run metrics (L0, NMSE, Jacobian cosine sim)
- `figures/l0_vs_jacobian_all.png` — Main plot: L0 vs Jacobian faithfulness
- `figures/nmse_vs_jacobian_all.png` — Supplementary: NMSE vs Jacobian
- `figures/l0_vs_nmse_all.png` — Reference: L0 vs NMSE (reproduces exp 11)
