# Experiment 08: Gemma-3-1B Anti-Collapse Interventions

## Goal

Can we prevent L0 collapse on Gemma-3-1B by addressing the θ runaway and dimensionality bootstrapping problems identified in experiments 02-06?

## Setup

- **Model:** Gemma-3-1B-IT, layer 13 (d=1152)
- **Data:** 2M cached FineWeb activations
- **Architecture:** MOLT N=1 (31 transforms), smooth surrogate JumpReLU
- **Training:** batch=64, lr=1e-3, λ=0 (no sparsity), 1 epoch
- **Baseline:** Exp 06 — learned θ → L0=0 (complete collapse), fixed θ STE → L0=1 (winner-take-all)

### Setups

| # | Name | Intervention | Hypothesis |
|---|------|-------------|------------|
| 1 | `fixed_theta_surrogate` | Fixed θ=0, smooth surrogate | θ runaway is sole cause of L0=0 |
| 2a-c | `theta_warmup_{25,50,75}pct` | Freeze θ for first N% of steps | Transforms need time to learn before gating kicks in |
| 3a-c | `theta_lr_{1e-4,1e-5,1e-6}` | Separate (slower) LR for θ | θ scalar moves faster than U,V matrices |
| 4 | `max_rank_768` | Max rank 768 (fixed θ=0) | Higher coverage ratio (67% vs 44%) prevents collapse |

## Reproduction

```bash
uv run python experiments/08_gemma_anti_collapse/run.py
```

## Results

| Setup | L0 | NMSE | θ | Notes |
|-------|----|------|---|-------|
| fixed_theta_surrogate | | | 0.0 | |
| theta_warmup_25pct | | | | |
| theta_warmup_50pct | | | | |
| theta_warmup_75pct | | | | |
| theta_lr_1e-4 | | | | |
| theta_lr_1e-5 | | | | |
| theta_lr_1e-6 | | | | |
| max_rank_768 | | | 0.0 | |

## Analysis

*Fill after running.*

## Artifacts

- Results: `experiments/08_gemma_anti_collapse/results/`
- Figures: `experiments/08_gemma_anti_collapse/figures/`
