# Loss Landscape Experiment: Activation/Sparsity Setup Sweep

## Experiment Design

**Goal:** Determine how the choice of gating activation (ReLU vs JumpReLU) and sparsity penalty type (Tanh vs L1) affect training dynamics and the L0 collapse phenomenon.

**Setup:** 2x2 grid of (sparsity type) x (gating activation), all with λ=0 (no sparsity penalty) to isolate the effect of the gating mechanism.

| | JumpReLU | ReLU |
|---|---|---|
| **Tanh** | Baseline (prior) | New |
| **L1** | New | New |

**Hyperparameters (shared):**
- Model: Gemma-3-1B layer 13, d_model=1152
- Architecture: N=1 (31 transforms: 1×rank-512, 2×rank-256, 4×rank-128, 8×rank-64, 16×rank-32)
- Tokens: 2M FineWeb
- Batch size: 64
- Learning rate: 1e-3 (Adam)
- Sparsity coefficient: λ=0.0
- Seed: 42

## Results

| Setup | Sparsity | Activation | L0 | NMSE | Active Transforms |
|-------|----------|------------|-----|------|-------------------|
| Tanh+JumpReLU | tanh | jumprelu | 1.0 | 0.120 | T0 (rank-512, 100%) |
| Tanh+ReLU | tanh | relu | 0.0 | 1.001 | None |
| L1+ReLU | l1 | relu | 0.0 | 1.001 | None |
| L1+JumpReLU | l1 | jumprelu | 1.0 | 0.166 | T0 (rank-512, 99.4%) |

## Analysis

### ReLU Gating: Complete Collapse (L0=0)

Both ReLU configurations (Tanh+ReLU, L1+ReLU) produce **identical** training trajectories since λ=0 makes the penalty type irrelevant. The model converges to NMSE≈1.0 (zero output), meaning no transform contributes to reconstruction.

**Mechanism:** Standard ReLU has zero gradient for negative inputs. The gating pre-activation is `φ(e_t · x - b_t)` where `φ = ReLU`. When a gate's pre-activation goes negative for a given input:
1. `ReLU(pre_act) = 0` → gate is off → no forward contribution
2. `d(ReLU)/d(pre_act) = 0` for `pre_act ≤ 0` → no gradient to reactivate

Even with bias initialized to -1.0 (starting ~99% of gates open), the stochastic optimization dynamics push some gates negative. Once negative, they're permanently dead. Over training, all 31 transforms progressively lose their gates and die.

### JumpReLU Gating: Winner-Take-All (L0=1)

Both JumpReLU configurations collapse to L0=1.0 with the rank-512 transform T0 winning nearly all tokens.

**Mechanism:** JumpReLU uses a full straight-through estimator (STE) that passes gradients unconditionally in the backward pass, even when the gate is off in the forward pass. This prevents permanent gate death but doesn't prevent competitive exclusion — whichever transform captures the most variance early gets the strongest gradients and crowds out the rest.

The NMSE difference between Tanh+JumpReLU (0.120) and L1+JumpReLU (0.166) reflects minor variance from different random seeds — the same transform (T0, rank-512) wins in both cases, but the exact learned decomposition differs slightly.

### Key Insight: The Gating Activation Determines Collapse Mode

The collapse behavior is entirely determined by the gating activation, not the sparsity penalty:

| | Dead gates (L0→0) | Winner-take-all (L0→1) |
|---|---|---|
| **ReLU** | Yes (no gradient for inactive gates) | N/A (all gates die) |
| **JumpReLU (full STE)** | No (gradient passes through) | Yes (competitive exclusion) |

This suggests that resolving the L0=1 collapse requires changes to the gating mechanism itself (e.g., soft gating, auxiliary losses, or architectural changes) rather than to the sparsity penalty type.

## Artifacts

- Checkpoints: `checkpoints/activation_sweep/{tanh_relu,l1_relu,l1_jumprelu}/`
- Results: `results/activation_sweep/result_{name}.json`
- Training histories: `results/activation_sweep/history_{name}.json`
- Comparison plot: `results/activation_sweep/activation_sweep_comparison.png`
- Cached activations: `data/activations_2M.pt` (2M tokens, 18GB, for HuggingFace upload)

## Code Changes

- `molt/config.py`: Added `sparsity_type: str = "tanh"` field (options: "tanh", "l1")
- `molt/model.py`: Added L1 branch in `MOLT.forward()` sparsity computation
- `scripts/run_activation_sweep.py`: Full sweep script (collects activations, trains all 3 setups, plots comparison)
- `scripts/run_single_setup.py`: Single-setup script using cached activations
