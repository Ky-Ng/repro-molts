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
- **Training:** batch=4096, lr=1e-3, λ=0, 1 epoch (~489 steps), log_every=10

### Setups

| # | Name | Init | Gate freeze | Tests |
|---|------|------|-------------|-------|
| 1 | `baseline_random_init` | random unit-norm | none | Reproduces exp 08 baseline |
| 2 | `orthogonal_encoders` | QR orthogonal | none | H: random encoders too similar |
| 3 | `pca_encoders` | Top PCA dirs | none | H: data-informed dirs help |
| 4 | `gate_freeze_10pct` | random | 10% (~49 steps) | H: UV needs time to learn |
| 5 | `gate_freeze_25pct` | random | 25% (~122 steps) | H: more freeze time helps |
| 6 | `gate_freeze_50pct` | random | 50% (~245 steps) | H: even more freeze time |
| 7 | `orthogonal_freeze_25pct` | QR orthogonal | 25% | Combined: best of both? |
| 8 | `pca_freeze_25pct` | Top PCA dirs | 25% | Combined: best of both? |

## Reproduction

```bash
# Run all setups
uv run python experiments/10_gemma_closing/run.py

# Run a single setup
uv run python experiments/10_gemma_closing/run.py orthogonal_encoders

# Regenerate figures
uv run python experiments/10_gemma_closing/plot.py
```

## Results

| Setup | Init | Freeze | L0 | NMSE | Notes |
|-------|------|--------|----|------|-------|
| baseline_random_init | random | 0% | 0.25 | 1.054 | Complete collapse |
| orthogonal_encoders | orthogonal | 0% | 0.26 | 1.065 | No improvement |
| pca_encoders | PCA | 0% | 0.44 | 1.435 | Worse — PCA dirs increase gate variance |
| gate_freeze_10pct | random | 10% | 0.21 | 1.010 | MSE drops during freeze, then collapses |
| gate_freeze_25pct | random | 25% | 0.22 | 1.008 | Same pattern |
| gate_freeze_50pct | random | 50% | 0.29 | 1.011 | Same pattern |
| orthogonal_freeze_25pct | orthogonal | 25% | 0.22 | 1.009 | No benefit from orthogonal + freeze |
| pca_freeze_25pct | PCA | 25% | 0.75 | 1.152 | Slightly more L0 but still collapsed |

**Every setup collapses to L0 < 1 and NMSE ≈ 1.0 (learns nothing).**

## Analysis

### Neither diverse init nor gate freezing prevents collapse

Both hypotheses are ruled out as sufficient explanations:

**Diverse encoder init (H1 partial, H5 partial):** Orthogonal encoders produce the same collapse as random unit-norm encoders (L0=0.26 vs 0.25). This is unsurprising in hindsight — random unit vectors in d=1152 are already near-orthogonal (max cosine sim ≈ 0.095 for 31 vectors). Making them exactly orthogonal doesn't change the dynamics. PCA encoders are actually worse (NMSE=1.44) because they align with high-variance input directions, making gate pre-activations more variable and accelerating winner selection.

**Gate freezing (H1 direct test):** This is the most informative result. During the frozen period:
- L0 stays at 31 (all transforms contribute)
- MSE drops significantly (UV matrices learn useful reconstructions)
- But the moment gates are unfrozen, L0 **immediately** collapses to ~0 within ~10 steps

This means UV matrices *do* learn meaningful directions when forced to all contribute. But the learned UV directions are not sufficient to sustain gating — the JumpReLU dynamics overwhelm everything. Even 50% freeze (245 steps of UV learning) produces instant collapse upon unfreezing.

### What this rules out

- **Not an init problem:** Encoders start fine (exp 09), and alternative inits don't help
- **Not a bootstrap problem:** UV matrices learn during freeze, but this doesn't prevent gate collapse
- **Not a time-to-learn problem:** Even 50% of training with forced-open gates doesn't help

### What this points to

The collapse is a fundamental property of the **gating mechanism itself** interacting with Gemma's activation geometry. The smooth surrogate JumpReLU gradient drives all gates to zero on Gemma inputs, regardless of how well-trained the UV transforms are. This is consistent with **H2 (RMSNorm-compressed inputs)** — the post-RMSNorm inputs have such uniform norms that the gating signal provides no token-level discrimination. The gates can't learn to be selectively active because there's insufficient input diversity for token-conditional gating.

Next steps:
- Test H2 directly: try pre-RMSNorm (raw residual stream) activations
- Test H3: compare MLP output scale vs MOLT output scale across models
- Consider replacing JumpReLU entirely (e.g., top-k routing, or softmax gating)

## Artifacts

- Results: `experiments/10_gemma_closing/results/`
- Figures: `experiments/10_gemma_closing/figures/`
  - `comparison_all.png` — all 8 setups overlaid
  - `comparison_init.png` — baseline vs orthogonal vs PCA
  - `comparison_freeze.png` — baseline vs 10/25/50% gate freeze
  - `comparison_combined.png` — baseline vs combined interventions
  - `curves_*.png` — individual 4-panel training curves per setup
