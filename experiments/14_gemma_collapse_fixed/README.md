# Experiment 14: Gemma-3-1B post-port sparsity sweep

## Goal

Check whether the L0=1 winner-take-all collapse previously observed on
Gemma-3-1B (experiments 01, 02, 06) still happens with the post-port MOLT
that carries the fixes described in [DIFFERENCES.md](../../DIFFERENCES.md)
(bugs A-F): true `||UV||_F`, `mean(tanh(...))` sparsity, `c_sparsity=100`
inside the `tanh`, per-feature learnable JumpReLU thresholds, `input > 0`
guard on the gate, and dimensionwise input/output standardizers.

## Setup

- **Model:** `google/gemma-3-1b-it`, MLP layer 13 (mid-stack)
- **Data:** 1M FineWeb tokens, streamed → in-memory activation tensors
- **Architecture:** `MOLTConfig` defaults (rank multiplier 1, 31 transforms)
- **Training:** 1 epoch, batch 512, lr 1e-3, 10% linear sparsity warmup

Sweep over `sparsity_coeff ∈ {0, 5e-4, 5e-3}` — 0 tests whether the collapse
is gate-dynamic rather than penalty-driven, 5e-4 matches the ported default,
5e-3 probes 10× the default.

## Reproduction

```bash
uv run python experiments/14_gemma_post_port/run.py
```

## Results

| Setup | λ | L0 | NMSE | θ̄ | #Active |
|-------|---|----|------|----|---------|
| lambda_0    | 0     | 18.56 | 0.0847 | 0.0000 | 31 |
| lambda_5e-4 | 5e-4  | 13.25 | 0.0854 | 0.0000 | 31 |
| lambda_5e-3 | 5e-3  | 5.67  | 0.0911 | 0.0000 | 31 |

## Analysis

The port fixes the Gemma L0=1 collapse. Compared to exp 01 (same model /
layer, L0 ≈ 1.0 across every λ), every λ here sustains L0 > 5 and all 31
transforms remain active at every λ.

1. **Collapse fixed.** Even at λ=5e-3 (10× the ported default), L0 stays at
   5.67 with all 31 transforms active — there is no winner-take-all.
2. **NMSE is ~10× lower than exp 01.** exp 01's best NMSE on Gemma was
   around 0.8–0.9; here NMSE ≈ 0.085 at the same token budget. Most of this
   is because MSE is now computed in the standardized output space (bug F),
   so NMSE reflects reconstruction quality on the per-dim rescaled target
   rather than raw activations.
3. **θ̄ stays at 0.** Per-feature thresholds are learnable but the mean did
   not drift after 1M tokens — consistent with the `input > 0` guard in the
   ported JumpReLU keeping θ near its init without requiring a fixed clamp
   (contrast exp 03, where a shared scalar θ ran away to +0.87).
4. **Sparsity penalty has a real effect.** L0 drops monotonically from 18.6
   → 13.3 → 5.7 as λ ramps 0 → 5e-4 → 5e-3, with only a ~8% NMSE cost — the
   penalty now discriminates between transforms (per-token `tanh(·c)` + true
   `||UV||_F`, bugs B/D/E) instead of collapsing everything to the same
   compressed mean.

## Artifacts

- Results: `experiments/14_gemma_post_port/results/`
- Figures: `experiments/14_gemma_post_port/figures/`
