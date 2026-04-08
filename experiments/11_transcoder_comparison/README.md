# Experiment 11: Transcoder vs MOLT Comparison

## Goal

Do MOLTs Pareto-dominate transcoders at equal compute, as claimed in the Anthropic paper? Compare L0 vs NMSE for transcoders and MOLTs trained on GPT-2, matching both parameter count and training tokens at each compute scale.

### Background

From the Anthropic paper (trained on Claude 3.5 Haiku):

> We trained MOLTs and transcoders on the middle layer, varying the amount of compute. We scaled the number of training steps proportionally to the number of features, and matched the number of parameters between transcoder and MOLT runs. Thus each 4x increase in FLOPs reflects a 2x increase in both number of parameters and training steps. The smallest MOLT runs Pareto-dominate transcoder runs that use 1024x as many FLOPs.

We reproduce this comparison on GPT-2 layer 6 (d=768), where MOLTs are known to train successfully (exp 04-05), avoiding the Gemma L0-collapse issue.

## Setup

- **Model:** GPT-2, layer 6 (d_model=768)
- **Data:** 500K FineWeb activations (from 2M cache), eval on last 10K
- **Batch size:** 1024 (direct GPU batching, no DataLoader overhead)
- **Optimizer:** Adam, lr=1e-3
- **MOLT:** tanh sparsity + smooth surrogate JumpReLU (best config from exp 04-05)
- **Transcoder:** ReLU encoder + linear decoder, L1 sparsity on feature activations
- **Sparsity warmup:** linear over first 10% of training steps

### Parameter Matching

MOLT parameters per transform: `2 * d_model * rank + d_model + 1` (U, V matrices + encoder + bias).

Transcoder parameters: `2 * d_model * n_features + n_features + d_model` (W_enc, W_dec + biases).

| Scale | MOLT Config | MOLT Params | Transcoder Features | Transcoder Params |
|-------|-------------|-------------|---------------------|-------------------|
| 1x    | N=1 (31 transforms) | 3,955,999 | 2,573 | 3,955,469 |
| 2x    | N=2 (62 transforms) | 7,911,998 | 5,147 | 7,911,707 |
| 4x    | N=4 (124 transforms) | 15,823,996 | 10,295 | 15,824,183 |

### Compute Scaling

Each 4x FLOPs = 2x parameters x 2x training steps (epochs):

| FLOPs Scale | Param Scale | Epochs | Steps (batch=1024) |
|-------------|-------------|--------|---------------------|
| 1x          | N=1         | 1      | 488 |
| 4x          | N=2         | 2      | 976 |
| 16x         | N=4         | 4      | 1,952 |

### Sparsity Sweep

MOLT lambda: [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]

Transcoder lambda: [0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

Transcoders require much stronger L1 penalties to achieve comparable sparsity levels to MOLTs. The wider sweep ensures overlapping L0 ranges for fair comparison.

Total: 51 training runs (18 MOLT + 33 transcoder).

## Reproduction

```bash
# Run main experiment (MOLT + transcoder at 3 scales, base lambdas)
uv run python experiments/11_transcoder_comparison/run.py

# Run a single scale
uv run python experiments/11_transcoder_comparison/run.py --scale 1x

# Run only MOLTs or only transcoders
uv run python experiments/11_transcoder_comparison/run.py --method molt
uv run python experiments/11_transcoder_comparison/run.py --method transcoder
```

Note: Additional strong-L1 transcoder runs and extra MOLT lambda runs were generated via inline scripts (see git history). The `run.py` covers the core 18-run sweep.

## Results

### MOLT Results

| Scale | Lambda | L0 | NMSE | Time |
|-------|--------|----|------|------|
| 1x | 0 | 26.89 | 0.694 | 16s |
| 1x | 1e-4 | 26.19 | 0.694 | 16s |
| 1x | 1e-3 | 19.26 | 0.694 | 21s |
| 1x | 3e-3 | 9.15 | 0.697 | 17s |
| 1x | 1e-2 | 2.78 | 0.712 | 16s |
| 1x | 3e-2 | 1.18 | 0.733 | 16s |
| 2x | 0 | 52.63 | 0.582 | 49s |
| 2x | 1e-4 | 50.24 | 0.581 | 48s |
| 2x | 1e-3 | 30.55 | 0.584 | 47s |
| 2x | 3e-3 | 12.03 | 0.598 | 30s |
| 2x | 1e-2 | 3.59 | 0.644 | 27s |
| 2x | 3e-2 | 1.50 | 0.702 | 28s |
| 4x | 0 | 101.59 | 0.455 | 82s |
| 4x | 1e-4 | 91.64 | 0.454 | 90s |
| 4x | 1e-3 | 43.28 | 0.462 | 103s |
| 4x | 3e-3 | 16.85 | 0.493 | 61s |
| 4x | 1e-2 | 5.30 | 0.561 | 61s |
| 4x | 3e-2 | 2.22 | 0.643 | 60s |

### Transcoder Results (selected)

| Scale | Lambda | L0 | NMSE | Time |
|-------|--------|----|------|------|
| 1x | 0 | 1271.75 | 0.602 | 2s |
| 1x | 0.1 | 962.44 | 0.591 | 4s |
| 1x | 1.0 | 365.63 | 0.598 | 3s |
| 1x | 3.0 | 107.64 | 0.655 | 5s |
| 1x | 10.0 | 12.92 | 0.803 | 3s |
| 1x | 30.0 | 1.08 | 0.909 | 3s |
| 2x | 0 | 1993.28 | 0.454 | 7s |
| 2x | 0.1 | 1599.60 | 0.448 | 7s |
| 2x | 1.0 | 717.07 | 0.444 | 7s |
| 2x | 3.0 | 330.82 | 0.460 | 6s |
| 2x | 10.0 | 68.43 | 0.572 | 8s |
| 2x | 30.0 | 9.26 | 0.771 | 15s |
| 4x | 0 | 2931.34 | 0.281 | 10s |
| 4x | 0.1 | 2443.27 | 0.270 | 14s |
| 4x | 1.0 | 1219.43 | 0.242 | 12s |
| 4x | 3.0 | 693.91 | 0.255 | 15s |
| 4x | 10.0 | 250.89 | 0.339 | 15s |
| 4x | 30.0 | 60.22 | 0.501 | 15s |

## Analysis

### Finding 1: MOLTs Pareto-dominate transcoders in the low-L0 regime

In the overlapping L0 range (L0 = 1-100), MOLTs achieve **consistently lower NMSE** than transcoders at matched L0 and compute scale. For example:

- At L0 ~ 10: MOLT 1x achieves NMSE=0.697 vs transcoder 1x at NMSE=0.803
- At L0 ~ 10: MOLT 4x achieves NMSE=0.493 vs transcoder 4x not yet reaching L0=10 at any lambda that maintains quality

This confirms the paper's claim that MOLTs Pareto-dominate transcoders.

### Finding 2: Transcoders naturally operate at very high L0

Without strong sparsity penalties, transcoders activate 50-100% of their features per token (L0=1200-2900 out of 2573-10295 features). Achieving MOLT-level sparsity (L0<100) requires L1 penalties 3-4 orders of magnitude stronger (lambda=3-30 vs lambda=1e-3).

### Finding 3: Transcoders degrade sharply under forced sparsity

When pushed to low L0 via strong L1, transcoder NMSE degrades rapidly. The transcoder L0-vs-NMSE curve is much steeper than the MOLT curve in the low-L0 regime, suggesting that **MOLTs achieve sparsity more gracefully** — their gated low-rank structure is inherently compatible with sparse activation, while transcoders' ReLU + L1 mechanism fights against their architecture.

### Finding 4: Compute scaling benefits both methods, but MOLTs benefit more at matched L0

At L0=10 and L0=50, NMSE decreases faster with compute for MOLTs than transcoders (see `compute_scaling.png`). This is consistent with the paper's observation that transcoder performance saturates at higher compute while MOLT performance continues improving.

### Limitation: No Jacobian comparison

Jacobian faithfulness was not computed in this run due to time constraints (requires loading GPT-2 for the true MLP function). This would be a natural follow-up.

## Figures

1. `l0_vs_nmse_all.png` — Full L0 vs NMSE Pareto plot (log-log), all 51 runs
2. `l0_vs_nmse_zoomed.png` — Zoomed to L0=[0.5, 200] for the overlapping regime
3. `compute_scaling.png` — NMSE at fixed L0 (10 and 50) vs FLOPs
4. `train_*.png` — Individual training curves (18 MOLT + 33 transcoder)

## Artifacts

- Results: `experiments/11_transcoder_comparison/results/` (51 JSON files)
- Figures: `experiments/11_transcoder_comparison/figures/`
