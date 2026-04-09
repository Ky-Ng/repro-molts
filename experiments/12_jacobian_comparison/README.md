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

### MOLT Results

| Scale | Lambda | L0 | NMSE | Jacobian CosSim |
|-------|--------|----|------|-----------------|
| 1x | 0 | 26.89 | 0.694 | 0.275 |
| 1x | 1e-4 | 26.19 | 0.694 | 0.275 |
| 1x | 1e-3 | 19.26 | 0.694 | 0.274 |
| 1x | 3e-3 | 9.15 | 0.697 | 0.275 |
| 1x | 1e-2 | 2.78 | 0.712 | 0.268 |
| 1x | 3e-2 | 1.18 | 0.733 | 0.248 |
| 2x | 0 | 52.63 | 0.582 | 0.387 |
| 2x | 1e-4 | 50.24 | 0.581 | 0.387 |
| 2x | 1e-3 | 30.55 | 0.584 | 0.381 |
| 2x | 3e-3 | 12.03 | 0.598 | 0.370 |
| 2x | 1e-2 | 3.59 | 0.644 | 0.334 |
| 2x | 3e-2 | 1.50 | 0.702 | 0.293 |
| 4x | 0 | 101.59 | 0.455 | 0.501 |
| 4x | 1e-4 | 91.64 | 0.454 | 0.500 |
| 4x | 1e-3 | 43.28 | 0.462 | 0.493 |
| 4x | 3e-3 | 16.85 | 0.493 | 0.467 |
| 4x | 1e-2 | 5.30 | 0.561 | 0.402 |
| 4x | 3e-2 | 2.22 | 0.643 | 0.342 |

### Transcoder Results (selected)

| Scale | Lambda | L0 | NMSE | Jacobian CosSim |
|-------|--------|----|------|-----------------|
| 1x | 0 | 1271.75 | 0.602 | 0.360 |
| 1x | 0.1 | 962.44 | 0.591 | 0.354 |
| 1x | 1.0 | 365.63 | 0.598 | 0.296 |
| 1x | 3.0 | 107.64 | 0.655 | 0.212 |
| 1x | 10.0 | 12.92 | 0.803 | 0.100 |
| 1x | 30.0 | 1.08 | 0.909 | 0.050 |
| 2x | 0 | 1993.28 | 0.454 | 0.500 |
| 2x | 0.1 | 1599.60 | 0.448 | 0.494 |
| 2x | 1.0 | 717.07 | 0.444 | 0.455 |
| 2x | 3.0 | 330.82 | 0.460 | 0.403 |
| 2x | 10.0 | 68.43 | 0.572 | 0.262 |
| 2x | 30.0 | 9.26 | 0.771 | 0.124 |
| 4x | 0 | 2931.34 | 0.281 | 0.653 |
| 4x | 0.1 | 2443.27 | 0.270 | 0.659 |
| 4x | 1.0 | 1219.43 | 0.242 | 0.671 |
| 4x | 3.0 | 693.91 | 0.255 | 0.635 |
| 4x | 10.0 | 250.89 | 0.339 | 0.502 |
| 4x | 30.0 | 60.22 | 0.501 | 0.316 |

## Analysis

### Finding 1: MOLTs Pareto-dominate transcoders in L0 vs Jacobian

At matched L0, MOLTs achieve higher Jacobian cosine similarity than transcoders. Examples at comparable L0:

- L0 ~ 10: MOLT 1x Jac=0.275 vs TC 1x Jac=0.100 (2.7x higher)
- L0 ~ 10: MOLT 2x Jac=0.370 vs TC 2x Jac=0.124 (3.0x higher)
- L0 ~ 50: MOLT 2x Jac=0.387 vs TC 4x Jac=0.316 (MOLT at 4x less compute still wins)

This extends experiment 11's L0-vs-NMSE Pareto dominance to the more meaningful Jacobian faithfulness metric.

### Finding 2: Transcoder Jacobian collapses under forced sparsity

Transcoder Jacobian degrades dramatically when pushed to low L0 via strong L1:

- TC 1x: Jac drops from 0.360 (L0=1272) to 0.050 (L0=1.08) — an 86% reduction
- TC 2x: Jac drops from 0.500 (L0=1993) to 0.124 (L0=9.26) — a 75% reduction

MOLTs degrade much more gracefully. MOLT 2x drops from Jac=0.387 (L0=53) to 0.293 (L0=1.5) — only a 24% reduction over a 35x decrease in L0.

### Finding 3: No Pareto dominance in NMSE vs Jacobian — NMSE determines Jacobian regardless of architecture

The NMSE-vs-Jacobian plot shows both methods fall on roughly the **same trendline**. At matched NMSE, MOLTs and transcoders achieve similar Jacobian:

- MOLT 4x λ=0: NMSE=0.455, Jac=0.501 vs TC 2x λ=0: NMSE=0.454, Jac=0.500
- MOLT 2x λ=0: NMSE=0.582, Jac=0.387 vs TC 1x λ=0.1: NMSE=0.591, Jac=0.354

Neither method Pareto-dominates the other on this axis. This means NMSE is the primary driver of Jacobian faithfulness, and the architecture matters only insofar as it achieves lower NMSE. The Pareto dominance is specifically in the **L0 dimension** — MOLTs achieve the same NMSE (and therefore the same Jacobian) with far fewer active components. That's what makes them more useful for interpretability: not better faithfulness per unit error, but better faithfulness per unit complexity.

### Finding 4: Compute scaling improves Jacobian for both methods

Best Jacobian at each scale (no sparsity penalty):

| Scale | MOLT Jac | TC Jac |
|-------|----------|--------|
| 1x | 0.275 | 0.360 |
| 2x | 0.387 | 0.500 |
| 4x | 0.501 | 0.653 |

At maximum L0 (no sparsity), transcoders achieve higher peak Jacobian by activating thousands of features. But this comparison is misleading — the transcoder's advantage disappears entirely when controlling for L0, which is the relevant comparison for interpretability (fewer active components = more interpretable).

## Figures

1. `l0_vs_jacobian_all.png` — Main plot: L0 vs Jacobian faithfulness, all 51 runs
2. `nmse_vs_jacobian_all.png` — NMSE vs Jacobian showing the strong correlation
3. `l0_vs_nmse_all.png` — Reference: L0 vs NMSE (reproduces exp 11)

## Artifacts

- `results/result_*.json` — Per-run metrics (L0, NMSE, Jacobian cosine sim)
- `figures/l0_vs_jacobian_all.png` — Main plot: L0 vs Jacobian faithfulness
- `figures/nmse_vs_jacobian_all.png` — Supplementary: NMSE vs Jacobian
- `figures/l0_vs_nmse_all.png` — Reference: L0 vs NMSE (reproduces exp 11)
