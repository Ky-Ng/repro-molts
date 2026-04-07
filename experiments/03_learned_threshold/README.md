# Learned JumpReLU Cutoff Experiment

## Experiment Design

**Question:** Does making the JumpReLU threshold θ a learnable parameter (shared across all transforms) help resolve the L0=1 collapse, or does the optimizer exploit the threshold to create a new collapse mode?

**Setup:** 4 configurations, all JumpReLU activation with λ=0, varying threshold (fixed vs learned) and sparsity type (Tanh vs L1):

| | Fixed θ=0 | Learned θ (init=0) |
|---|---|---|
| **Tanh** | Baseline | New |
| **L1** | Baseline | New |

**Data:** Cached activations from `data/activations_2M.pt` (2M FineWeb tokens, Gemma-3-1B layer 13).

**Hyperparameters:**
- Architecture: N=1 (31 transforms), d_model=1152
- Batch size: 64, LR: 1e-3 (Adam), 1 epoch (~31K steps)
- λ=0 (no sparsity penalty), no warmup
- Learned threshold: single scalar `nn.Parameter` shared across all 31 transforms, initialized to 0.0

**Implementation:**
- New `LearnedJumpReLU` autograd function with gradient for θ via smoothed Heaviside derivative (logistic kernel density, bandwidth=0.1)
- Forward: `x * 1[x > θ]` (same hard gating as fixed JumpReLU)
- Backward for x: full STE (unconditional)
- Backward for θ: `d/dθ = -Σ grad_output * x * σ'((x-θ)/bw) / bw` where σ' is the logistic density

## Results

| Setup | Learned θ | L0 | NMSE | θ_final | Winner |
|-------|-----------|-----|------|---------|--------|
| Tanh + JumpReLU | No (fixed 0.0) | 1.0 | 0.166 | 0.0 | T0 (rank-512, 99.4%) |
| Tanh + JumpReLU | **Yes** | **0.0** | **1.001** | **0.873** | **None (all dead)** |
| L1 + JumpReLU | No (fixed 0.0) | 1.0 | 0.166 | 0.0 | T0 (rank-512, 99.4%) |
| L1 + JumpReLU | **Yes** | **0.0** | **1.001** | **0.873** | **None (all dead)** |

### Observations

1. **Fixed threshold runs are identical** (both L0=1.0, NMSE=0.166, T0 wins). At λ=0 the sparsity type is irrelevant.

2. **Learned threshold runs are also identical** (both L0=0.0, NMSE=1.001, θ→0.873). Again sparsity type is irrelevant at λ=0.

3. **The learned threshold introduces a new, worse collapse mode** (L0=0 instead of L0=1).

## Threshold Trajectory Analysis

The learned θ trajectory reveals an oscillatory instability before permanent collapse:

```
Step       L0     NMSE    θ
   100    0.64   1.499   0.042    ← early training, many gates active
  3100    0.98   0.171   0.991    ← θ rises fast, L0→1 (winner-take-all)
  6100    0.98   0.147   1.054    ← θ still rising, winner T0 survives
  9100    0.94   0.168   1.137    ← θ overshoots, some tokens lose T0
 12100    1.00   0.137   1.078    ← brief recovery
 15100    0.00   1.001   0.984    ← COLLAPSE: θ kills even the winner
 18100    0.89   0.216   1.152    ← STE gradient revives gates briefly
 21100    0.94   0.166   0.973    ← oscillation continues
 24100    0.00   1.001   0.873    ← PERMANENT COLLAPSE
 27100    0.00   1.001   0.873    ← stuck — no recovery
 31200    0.00   1.001   0.873    ← final state
```

### Three phases:

1. **Threshold rise (steps 0–12K):** θ climbs from 0 to ~1.1. This initially helps by pruning noisy low-confidence gates, leaving only the strongest transform (T0) active. NMSE improves to 0.137.

2. **Oscillation (steps 12K–24K):** θ is near the edge where even T0's gate activations can't consistently exceed it. The system oscillates: when θ overshoots and kills T0, NMSE jumps to 1.0. The STE gradient then pulls θ down (since zero output is terrible for MSE), gates reactivate briefly, but θ drifts up again because intermediate-confidence gates add noise.

3. **Permanent death (steps 24K+):** After repeated cycles, the transform parameters drift while gates are off. When θ eventually decreases enough to reactivate gates, the transform weights no longer produce useful output (they haven't been training). The MSE gradient from bad transform output pushes θ back up, and the system locks into L0=0.

## Root Cause: Threshold Runaway

The learned threshold creates a **shortcut for the optimizer**: instead of improving transform weights to reduce MSE, it's cheaper to raise θ and silence noisy transforms entirely. This is a form of reward hacking — the model finds that `output ≈ 0` (NMSE≈1.0) is a local minimum that's easier to reach via threshold than via learning 31 useful decompositions.

The fundamental issue is an **asymmetry in gradient timescales**:
- θ is a single scalar that receives gradient from every gate across every token → it moves fast
- U, V (transform weights) are high-dimensional and only receive useful gradient when their gate is on → they move slowly
- The threshold can shut off gates faster than the transforms can learn to produce useful output

This is analogous to the "lazy neuron" problem in deep learning: if there's a cheap parameter that can reduce loss by zeroing out a contribution, the optimizer will prefer it over the expensive path of actually learning useful features.

## Implications

1. **A shared learned threshold does not help with L0 collapse** — it makes it strictly worse (L0=0 vs L0=1).

2. **Per-transform thresholds** might behave differently but risk the same issue at each individual gate.

3. **The threshold needs counter-pressure** to prevent runaway:
   - An auxiliary loss that penalizes high θ (but this conflicts with sparsity goals)
   - Threshold warmup (start high, anneal down) — opposite of sparsity warmup
   - Separate learning rates: slower LR for θ than for transform weights
   - Gumbel-softmax or other soft gating that avoids hard thresholds entirely

4. **The fixed θ=0 setup is actually better** because it prevents this failure mode while still allowing the full STE to pass gradients. The L0=1 collapse with fixed θ is an optimization dynamics issue, not a threshold placement issue.

## Artifacts

- Checkpoints: `checkpoints/learned_threshold/{tanh,l1}_jumprelu_{fixed,learned}/`
- Results: `results/learned_threshold/result_*.json`
- Training histories: `results/learned_threshold/history_*.json`
- Comparison plot: `results/learned_threshold/learned_threshold_comparison.png`
- Script: `scripts/run_learned_threshold.py`

## Code Changes

- `molt/config.py`: Added `learned_threshold: bool = False`
- `molt/model.py`:
  - Added `LearnedJumpReLU` autograd function with gradient for θ
  - `MOLT.__init__`: creates `self.threshold` as `nn.Parameter` when `learned_threshold=True`
  - `MOLT.forward`: passes learned threshold to groups
  - `MOLT.loss`: logs threshold value in metrics
  - `TransformGroup.forward`: accepts `learned` flag to dispatch to `LearnedJumpReLU`
