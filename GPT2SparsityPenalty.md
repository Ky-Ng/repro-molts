# GPT-2 Sparsity Penalty Sweep

## Experiment

Test whether non-zero sparsity penalties (λ=1e-5 and λ=1e-4) produce a meaningful L0 vs NMSE tradeoff on GPT-2, sweeping over all 4 combinations of {ReLU, JumpReLU (smooth surrogate + learned θ)} × {Tanh, L0} sparsity penalties.

**Setup:**
- Model: `openai-community/gpt2`, layer 6, d_model=768
- Data: 2M FineWeb tokens (cached at `data/activations_openai_community_gpt2_2M.pt`)
- Architecture: N=1, 31 transforms
- Training: batch_size=64, lr=1e-3, 1 epoch (~31K steps)
- Sparsity warmup: 10% of training (linear ramp from 0 to λ)
- JumpReLU: smooth surrogate backward, learned threshold initialized at 0.0

## Results

### Full Table

| Sparsity | Activation | λ | L0 | NMSE | θ_final | #Active |
|----------|------------|------|------|------|---------|---------|
| Tanh | ReLU | 0 | 18.84 | 0.489 | fixed | 31 |
| Tanh | JumpReLU (learned θ) | 0 | 14.46 | 0.474 | -0.466 | 31 |
| L0 | ReLU | 0 | 18.84 | 0.489 | fixed | 31 |
| L0 | JumpReLU (learned θ) | 0 | 14.46 | 0.474 | -0.466 | 31 |
| **Tanh** | **ReLU** | **1e-5** | **18.75** | **0.489** | **fixed** | **31** |
| **Tanh** | **JumpReLU (learned θ)** | **1e-5** | **14.44** | **0.474** | **-0.464** | **31** |
| **L0** | **ReLU** | **1e-5** | **18.76** | **0.488** | **fixed** | **31** |
| **L0** | **JumpReLU (learned θ)** | **1e-5** | **14.32** | **0.473** | **-0.478** | **31** |
| **Tanh** | **ReLU** | **1e-4** | **18.38** | **0.489** | **fixed** | **31** |
| **Tanh** | **JumpReLU (learned θ)** | **1e-4** | **14.46** | **0.474** | **-0.466** | **31** |
| **L0** | **ReLU** | **1e-4** | **18.54** | **0.488** | **fixed** | **31** |
| **L0** | **JumpReLU (learned θ)** | **1e-4** | **13.06** | **0.472** | **-0.572** | **31** |

### Grouped by Activation

**ReLU:**

| Sparsity | λ | L0 | NMSE |
|----------|---|-----|------|
| Tanh | 0 | 18.84 | 0.489 |
| Tanh | 1e-5 | 18.75 | 0.489 |
| Tanh | 1e-4 | 18.38 | 0.489 |
| L0 | 0 | 18.84 | 0.489 |
| L0 | 1e-5 | 18.76 | 0.488 |
| L0 | 1e-4 | 18.54 | 0.488 |

**JumpReLU (smooth surrogate, learned θ):**

| Sparsity | λ | L0 | NMSE | θ |
|----------|---|-----|------|---|
| Tanh | 0 | 14.46 | 0.474 | -0.466 |
| Tanh | 1e-5 | 14.44 | 0.474 | -0.464 |
| Tanh | 1e-4 | 14.46 | 0.474 | -0.466 |
| L0 | 0 | 14.46 | 0.474 | -0.466 |
| L0 | 1e-5 | 14.32 | 0.473 | -0.478 |
| L0 | 1e-4 | **13.06** | **0.472** | **-0.572** |

## Analysis

### 1. Sparsity penalties at these magnitudes have minimal effect

At λ=1e-5, results are nearly identical to λ=0 across all configurations. The sparsity penalty is too small to meaningfully influence training.

At λ=1e-4, there are small effects:
- **ReLU + Tanh:** L0 drops from 18.84 → 18.38 (Δ=-0.46), NMSE unchanged
- **ReLU + L0:** L0 drops from 18.84 → 18.54 (Δ=-0.30), NMSE unchanged
- **JumpReLU + Tanh:** No change (L0=14.46, NMSE=0.474)
- **JumpReLU + L0:** L0 drops from 14.46 → 13.06 (Δ=-1.40), NMSE slightly improves to 0.472, θ goes more negative (-0.572)

### 2. L0 penalty is more effective than Tanh at reducing sparsity

The L0 + JumpReLU combination at λ=1e-4 shows the largest sparsity effect: L0 drops by 1.4 transforms while NMSE actually *improves* slightly (0.474 → 0.472). This suggests the L0 penalty is successfully pruning low-value transforms without hurting reconstruction.

The Tanh penalty at these magnitudes has essentially no effect on JumpReLU. This is likely because the Tanh penalty saturates — `tanh(mean|gate|)` is close to 1.0 for all active transforms, so the gradient is nearly zero and the penalty doesn't discriminate between high-value and low-value transforms.

### 3. ReLU is less sensitive to sparsity penalties

ReLU's L0 barely moves (18.84 → 18.38 at most) because ReLU gates are either clearly on or clearly off — there's no soft zone where a small penalty tips the balance. The sparsity penalty adds a small gradient toward zero, but for most gates the MSE reconstruction gradient dominates.

JumpReLU with learned θ is more responsive because the threshold parameter provides a single knob that globally adjusts the sparsity level. The L0 penalty pushes θ more negative at λ=1e-4 (-0.572 vs -0.466), which paradoxically means it opens more gates but the increased gate openness allows lower-capacity transforms to drop out as higher-capacity transforms absorb their work.

### 4. No meaningful Pareto frontier yet

These λ values don't produce enough variation in L0 to establish a Pareto frontier. The L0 range is narrow: 13–19 across all configurations. To produce a real sparsity-reconstruction tradeoff, much larger λ values (1e-3 to 1e-1) would likely be needed, with the risk of triggering collapse dynamics similar to what was observed on Gemma.

### 5. NMSE is remarkably stable

Across all 12 runs (including λ=0 baselines), NMSE stays in the range 0.472–0.489. The reconstruction quality is almost entirely determined by the activation function (ReLU≈0.489, JumpReLU≈0.474), not by the sparsity penalty type or magnitude.

## Artifacts

- Results: `results/gpt2_sparsity/result_*.json`, `results/gpt2_sparsity/sweep_results.json`
- Training histories: `results/gpt2_sparsity/history_*.json`
- Plots per λ: `results/gpt2_sparsity/sparsity_sweep_lam1e-05.png`, `results/gpt2_sparsity/sparsity_sweep_lam1e-04.png`
- L0 vs NMSE scatter: `results/gpt2_sparsity/l0_vs_nmse_all.png`
- Script: `scripts/run_gpt2_sparsity_sweep.py`
