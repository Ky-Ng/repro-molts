# L0=1 Collapse Analysis

## Observation

Across all experiments on Gemma-3-1B — every lambda value in the sparsity sweep and a lambda=0 control — the model collapses to exactly L0=1.0 (one active transform per token).

## Sparsity Sweep Results (N=1, 10M FineWeb tokens)

| lambda | L0 | NMSE |
|--------|-----|------|
| 1e-5 | 1.0 | 0.135 |
| 3e-5 | 1.0 | 0.191 |
| 1e-4 | 1.0 | 0.140 |
| 3e-4 | 1.0 | 0.375 |
| 1e-3 | 1.0 | 0.155 |
| 3e-3 | 1.0 | 0.143 |
| 1e-2 | 1.0 | 0.168 |
| **0.0** | **1.0** | **0.120** |

## Root Cause

The collapse is **not caused by the sparsity penalty**. It is an optimization dynamic arising from JumpReLU hard gating + MSE objective:

1. **Winner-take-all competition:** JumpReLU zeros inactive transforms in the forward pass. Even with full STE, the forward contribution of inactive transforms is exactly zero.
2. **Positive feedback loop:** The transform that captures the most variance early gets the strongest MSE gradients, improves, captures more variance, and crowds out the rest.
3. **The full STE is necessary but not sufficient:** It prevents permanent gate death (Bug 2) but doesn't prevent competitive exclusion.
4. **Rank-512 has a structural advantage:** Highest capacity single transform, but the specific winner varies by seed.

## Cross-Model Comparison

| Model | ReLU | JumpReLU (full STE, fixed theta) | JumpReLU (smooth surrogate, learned theta) |
|-------|------|------|------|
| **GPT-2** (d=768) | L0=18.8, NMSE=0.49 | L0=1.6, NMSE=0.74 | **L0=14.5, NMSE=0.47** |
| **Gemma-3-1B** (d=1152) | L0=0.0, NMSE=1.00 | L0=1.0, NMSE=0.12 | L0=0.0, NMSE=1.00 |

## Why Smooth Surrogate Works on GPT-2 but Not Gemma

1. **GPT-2's lower d=768:** Individual transforms have higher relative capacity (rank-512 covers 67% vs 44% on Gemma)
2. **Transforms fail to learn fast enough on Gemma:** Initial reconstruction is worse, so MSE gradient for keeping gates open is weaker
3. **Threshold runaway is faster on Gemma:** Transforms contribute more noise than signal early → raising theta genuinely reduces MSE initially → irreversible death spiral

## Activation/Sparsity Setup Sweep Results (lambda=0)

| Sparsity | Activation | L0 | NMSE |
|----------|------------|-----|------|
| Tanh | JumpReLU | 1.0 | 0.120 |
| Tanh | ReLU | 0.0 | 1.001 |
| L1 | ReLU | 0.0 | 1.001 |
| L1 | JumpReLU | 1.0 | 0.166 |

**Key:** ReLU collapses to L0=0 (dead gates), JumpReLU collapses to L0=1 (winner-take-all). The gating activation determines the collapse mode, not the sparsity penalty type.

## Learned Threshold Results

Making theta learnable makes collapse **worse** (L0=0 instead of L0=1). The threshold runs away to +0.87, killing all gates via an oscillatory instability. Root cause: theta is a single scalar that moves faster than high-dimensional transform weights can learn useful features.
