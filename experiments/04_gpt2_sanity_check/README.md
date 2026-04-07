# GPT-2 Sanity Check: MOLT Activation/Sparsity Sweeps

## Purpose

Sanity-check whether the L0 collapse observed on Gemma-3-1B is model-specific or universal, and test whether the smooth surrogate JumpReLU backward pass resolves the collapse.

## Setup

- **Model:** `openai-community/gpt2` (124M params, d_model=768, 12 layers)
- **Layer:** 6 (mid-layer), MLP path: `transformer.h.6.mlp`
- **Data:** 2M FineWeb tokens, cached at `data/activations_openai_community_gpt2_2M.pt` (12GB)
- **Architecture:** N=1, 31 transforms (1×rank-512, 2×rank-256, 4×rank-128, 8×rank-64, 16×rank-32)
- **Training:** batch_size=64, lr=1e-3, 1 epoch (~31K steps), λ=0

## Infrastructure Changes

Generalized the codebase to support multiple model architectures:

- `molt/config.py`:
  - Added `MODEL_PRESETS` dict mapping model names → {d_model, layer_idx, mlp_path, model_dtype}
  - Added `mlp_path` and `model_dtype` fields to `MOLTConfig`
  - Added `MOLTConfig.from_preset(model_name, **overrides)` classmethod
  - `rank_distribution` now filters out ranks > d_model

- `molt/data.py`:
  - Added `_resolve_mlp_module(model, config)` — resolves dot-separated path (e.g. `transformer.h.6.mlp`) to the actual module
  - `collect_activations()` uses configurable `model_dtype` and `mlp_path` instead of hardcoded Gemma paths

- `scripts/run_model_sweep.py`: Generalized sweep script that takes model name as CLI argument

---

## Experiment 1: Fixed θ=0 Sweep (Full STE Backward)

Ran with the original full-STE JumpReLU backward (grad passes unconditionally).

### Results

| Sparsity | Activation | L0 | NMSE | Active Transforms |
|----------|------------|------|------|-------------------|
| Tanh | ReLU | **18.85** | **0.489** | **All 31** (25–88% each) |
| L1 | ReLU | **18.85** | **0.489** | **All 31** (25–88% each) |
| Tanh | JumpReLU (full STE) | 1.59 | 0.739 | T0 (99.6%) + sparse low-rank |
| L1 | JumpReLU (full STE) | 1.59 | 0.739 | T0 (99.6%) + sparse low-rank |

### Comparison to Gemma-3-1B

| | Gemma-3-1B (d=1152) | GPT-2 (d=768) |
|---|---|---|
| **ReLU, λ=0** | L0=0.0, NMSE=1.001 (all dead) | L0=18.85, NMSE=0.489 (all alive) |
| **JumpReLU (full STE), λ=0** | L0=1.0, NMSE=0.120 (winner-take-all) | L0=1.59, NMSE=0.739 (near-collapse) |

**Key finding:** ReLU works excellently on GPT-2 (all 31 transforms alive) but fails completely on Gemma. JumpReLU with full STE collapses toward L0≈1 on both models, confirming the full STE's biased gradient drives winner-take-all dynamics regardless of model.

---

## Experiment 2: Smooth Surrogate Backward + Learned θ

Replaced the JumpReLU backward pass with a **smooth surrogate gradient**:

```
Forward:  y = x * 1[x > θ]           (hard threshold, unchanged)
Backward: dy/dx = σ(x/τ) + x·σ(x/τ)·(1-σ(x/τ))/τ   (gradient of x·σ(x/τ))
```

Where σ is sigmoid and τ=0.1 is the temperature. This provides:
- Active gates (x >> θ): gradient ≈ 1 (correct)
- Near-threshold: gradient ≈ 0.5 (can reactivate)
- Deeply-off (x << θ): gradient ≈ 0 (honestly reflects zero contribution)

Also added **L0 sparsity penalty**: `Σ_t σ(gate_t/τ) · ||U_t V_t||_F` — a smoothed count of active gates, differentiable via the same sigmoid surrogate.

### Results

| Sparsity | Activation | Learned θ | L0 | NMSE | θ_final | #Active |
|----------|------------|-----------|------|------|---------|---------|
| Tanh | ReLU | No | 18.84 | 0.489 | fixed | 31/31 |
| **Tanh** | **JumpReLU (surrogate)** | **Yes** | **14.46** | **0.474** | **-0.466** | **31/31** |
| L0 | ReLU | No | 18.84 | 0.489 | fixed | 31/31 |
| **L0** | **JumpReLU (surrogate)** | **Yes** | **14.46** | **0.474** | **-0.466** | **31/31** |

### Threshold Trajectory (smooth surrogate)

```
Step       L0    NMSE    θ
   100   15.36  0.903   +0.100    ← starts positive, begins rising
  5100   15.50  0.547   -0.504    ← quickly goes negative (opens gates)
 10100   15.56  0.518   -0.546    ← stabilizes around -0.5
 20100   14.34  0.500   -0.496
 31200   15.34  0.501   -0.469    ← stable, θ ≈ -0.47
```

### Analysis

1. **The smooth surrogate completely prevents the JumpReLU collapse.** With the full STE, JumpReLU collapsed to L0=1.6 on GPT-2. With the smooth surrogate, it maintains L0≈14.5 with **all 31 transforms active** and achieves the **best NMSE (0.474)** of any configuration tested.

2. **The learned threshold goes negative (θ=-0.47).** Unlike the full-STE experiment where θ ran away to +0.87 and killed all gates, the smooth surrogate θ moves to -0.47. A negative threshold means `gate = x * 1[x > -0.47]`, which is more permissive than the default θ=0 — it allows slightly negative pre-activations to pass. The optimizer found that being more inclusive improves reconstruction.

3. **Why does the surrogate prevent threshold runaway?** With the full STE, deeply-off gates (x << 0) still received full gradient, creating phantom signals that destabilized training. The smooth surrogate gives near-zero gradient to deeply-off gates, so the optimizer can't "cheat" by raising θ — there's no gradient pressure from already-dead gates to kill more. Only near-threshold gates influence θ, creating a stable equilibrium.

4. **JumpReLU (surrogate) slightly outperforms ReLU.** NMSE 0.474 vs 0.489. The hard gating in the forward pass appears to provide a small benefit — transforms that are confidently off contribute exactly zero noise, while ReLU lets small positive gate values pass noisy transform outputs.

5. **Sparsity type is irrelevant at λ=0** — all pairs of (Tanh, L0) produce identical results within each activation type.

### Comparison: All JumpReLU Backward Variants on GPT-2

| JumpReLU Backward | Learned θ | L0 | NMSE | θ_final | Outcome |
|---|---|---|---|---|---|
| Full STE | No (fixed 0) | 1.59 | 0.739 | 0.0 | Near-collapse, T0 dominates |
| Full STE | Yes | 0.0 | 1.001 | +0.873 | Total collapse, θ runaway |
| **Smooth surrogate** | **Yes** | **14.46** | **0.474** | **-0.466** | **Healthy, all 31 active** |

The smooth surrogate is the only variant that avoids collapse while also improving reconstruction over ReLU.

## Artifacts

- Cached activations: `data/activations_openai_community_gpt2_2M.pt` (12GB)
- Experiment 1 results: `results/openai_community_gpt2/`
- Experiment 2 results: `results/gpt2_surrogate/`
- Comparison plots: `results/openai_community_gpt2/sweep_comparison.png`, `results/gpt2_surrogate/surrogate_sweep.png`
- Scripts: `scripts/run_model_sweep.py`, `scripts/run_gpt2_surrogate_sweep.py`

## Code Changes

- `molt/model.py`:
  - `JumpReLU.backward`: replaced full STE with smooth surrogate gradient `σ(x/τ) + x·σ'(x/τ)/τ`
  - `LearnedJumpReLU.backward`: same smooth surrogate for x, kernel density for θ
  - Added L0 sparsity: `σ(gate/τ)` smoothed count weighted by Frobenius norms
- `molt/config.py`: `sparsity_type` now accepts "l0" in addition to "tanh" and "l1"
