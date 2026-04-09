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

| Setup | L0 | NMSE | Final θ | Notes |
|-------|----|------|---------|-------|
| fixed_theta_surrogate | 0.0 | 1.001 | 0.0 (fixed) | Collapsed — θ runaway not the cause |
| theta_warmup_25pct | 0.0 | 1.001 | 0.780 | Collapsed — θ ran away after unfreeze |
| theta_warmup_50pct | 0.0 | 1.001 | 0.445 | Collapsed — same pattern |
| theta_warmup_75pct | 0.0 | 1.001 | — | OOM killed at 28%, already L0=0 |
| theta_lr_1e-4 | 0.0 | 1.001 | 0.114 | Collapsed — θ still drifted up |
| theta_lr_1e-5 | 0.0 | 1.001 | 0.015 | Collapsed — θ barely moved, still died |
| theta_lr_1e-6 | 0.0 | 1.001 | 0.002 | Collapsed — θ effectively frozen, still died |
| max_rank_768 | 0.0 | 1.001 | 0.0 (fixed) | Collapsed — higher rank did not help |

**All 8 setups collapsed to L0=0, NMSE≈1.0 (outputting nothing).**

## Analysis

### Key finding: the collapse is NOT caused by θ dynamics

The most important result is `fixed_theta_surrogate` and `theta_lr_1e-6`: even with θ permanently at 0.0 (or effectively frozen at 0.002), all transforms still die. This rules out θ runaway as the root cause of collapse on Gemma.

### The smooth surrogate backward pass itself is the problem

The collapse happens purely through the gating gradient dynamics:
- The smooth surrogate backward computes `σ(x/τ) + x·σ'(x/τ)/τ` with τ=0.1
- On Gemma's narrow pre-activation distribution (std≈0.45), this gradient landscape creates a winner-take-all dynamic that kills transforms even at fixed θ=0
- This contrasts with the hard STE (exp 06), which at least preserves L0=1

### Higher rank (768) did not help

Increasing max rank from 512→768 to match GPT-2's 67% coverage ratio made no difference. The dimensionality bootstrapping hypothesis alone does not explain the collapse — the gating mechanism itself is fundamentally incompatible with Gemma's activation statistics.

### θ warmup confirms the timeline

The warmup runs show that collapse happens *while θ is frozen*:
- At 14% progress (well within the 25% freeze window), `theta_warmup_25pct` was already at L0=0
- The transforms die from the smooth surrogate gradient alone, before θ ever moves
- After unfreezing, θ then runs away to wherever the dead-gate gradient pushes it (0.78 for 25pct, 0.44 for 50pct — less time to drift = lower final θ)

### Implications for next experiments

The smooth surrogate JumpReLU is incompatible with Gemma-3-1B at this scale. Possible next directions:
1. **Use hard STE (identity backward) with fixed θ=0** — this at least achieves L0=1 (exp 06), then focus on breaking the winner-take-all to L0>1
2. **Replace JumpReLU entirely** — try sigmoid gating, softmax gating, or top-k routing
3. **Auxiliary diversity loss** — penalize gate correlation or reward multiple active transforms
4. **Much longer training** — the paper uses far more tokens; perhaps 2M is insufficient for Gemma's scale

## Artifacts

- Results: `experiments/08_gemma_anti_collapse/results/`
- Figures: `experiments/08_gemma_anti_collapse/figures/`
