# Experiment 10: Gemma Gate Collapse Investigation

## Goal

Why do Gemma-3-1B MOLT gates collapse from ~99% open (exp 09) to L0≈0 within 100 training steps? Test two interventions from the hypotheses in `notes/hypothesis/GemmaCollapseHypothesis.md`:

1. **Diverse encoder initialization** — do random encoders in d=1152 start too similar, leaving transforms undifferentiated and vulnerable to winner-take-all?
2. **Gate freezing** — if we force all gates to 1.0 for the first N% of training, can UV matrices learn useful directions before gating competition kills them?

## Context

- Exp 08 `fixed_theta_surrogate`: fixed θ=0 + smooth surrogate → L0=0, NMSE≈1.0 (complete collapse, learns nothing)
- Exp 09: 30.73/31 gates open at init → collapse is a training dynamics problem
- Smooth surrogate makes collapse *worse* on Gemma (L0=0 vs L0=1 with hard JumpReLU)
- λ=0 throughout — no sparsity penalty, gates die from MSE gradient dynamics alone

## Setup

- **Model:** Gemma-3-1B-IT, layer 13 (d=1152)
- **Data:** 2M cached FineWeb activations
- **Architecture:** MOLT N=1 (31 transforms), smooth surrogate JumpReLU, fixed θ=0
- **Training:** batch=64, lr=1e-3, λ=0, 1 epoch (~31K steps)

### Setups

| # | Name | Init | Gate freeze | Tests |
|---|------|------|-------------|-------|
| 1 | `baseline_random_init` | random unit-norm | none | Reproduces exp 08 baseline |
| 2 | `orthogonal_encoders` | QR orthogonal | none | H: random encoders too similar |
| 3 | `pca_encoders` | Top PCA dirs | none | H: data-informed dirs help |
| 4 | `gate_freeze_10pct` | random | 10% (~3.1K steps) | H: UV needs time to learn |
| 5 | `gate_freeze_25pct` | random | 25% (~7.8K steps) | H: more freeze time helps |
| 6 | `gate_freeze_50pct` | random | 50% (~15.6K steps) | H: even more freeze time |
| 7 | `orthogonal_freeze_25pct` | QR orthogonal | 25% | Combined: best of both? |
| 8 | `pca_freeze_25pct` | Top PCA dirs | 25% | Combined: best of both? |

## Reproduction

```bash
# Run all setups
uv run python experiments/10_gemma_closing/run.py

# Run a single setup
uv run python experiments/10_gemma_closing/run.py orthogonal_encoders
```

## Results

| Setup | Init | Freeze | L0 | NMSE | Notes |
|-------|------|--------|----|------|-------|
| baseline_random_init | random | 0% | | | |
| orthogonal_encoders | orthogonal | 0% | | | |
| pca_encoders | PCA | 0% | | | |
| gate_freeze_10pct | random | 10% | | | |
| gate_freeze_25pct | random | 25% | | | |
| gate_freeze_50pct | random | 50% | | | |
| orthogonal_freeze_25pct | orthogonal | 25% | | | |
| pca_freeze_25pct | PCA | 25% | | | |

## Analysis

*Fill after running.*

## Artifacts

- Results: `experiments/10_gemma_closing/results/`
- Figures: `experiments/10_gemma_closing/figures/`
