# Experiment 09: Gemma-3-1B Initial Gate Openness

## Goal

How many MOLT transform gates are open at initialization (before the first training update) on Gemma-3-1B? This establishes that L0 collapse is a training dynamics problem, not an initialization problem.

## Setup

- **Model:** Gemma-3-1B-IT, layer 13 (d_model=1152)
- **Data:** 256 tokens from cached FineWeb activations + synthetic inputs at varying norms
- **Architecture:** Default MOLT (N=1, 31 transforms: 1×512 + 2×256 + 4×128 + 8×64 + 16×32)
- **Gating:** JumpReLU with threshold=0.0, bias init=-1.0, unit-norm encoders
- **No training** — this measures the freshly initialized model

## Reproduction

```bash
uv run python experiments/09_gemma_default_open/run.py
```

## Results

### Real Gemma activations (||x|| ≈ 15)

| Group | Transforms | Rank | Active/token | Fraction open |
|-------|-----------|------|-------------|---------------|
| 0 | 1 | 512 | 1.0/1 | 99.6% |
| 1 | 2 | 256 | 2.0/2 | 100.0% |
| 2 | 4 | 128 | 3.9/4 | 97.7% |
| 3 | 8 | 64 | 8.0/8 | 99.8% |
| 4 | 16 | 32 | 15.8/16 | 99.0% |
| **Total** | **31** | | **30.73/31** | **~99%** |

### Sensitivity to input norm

| Input | Norm | L0 | Frac open |
|-------|------|----|-----------|
| Real Gemma activations | 15 | 30.73/31 | 99.1% |
| Synthetic (norm=5) | 5 | 31.00/31 | 100.0% |
| Synthetic (norm=10) | 10 | 30.98/31 | 99.9% |
| Synthetic (norm=15, Gemma-like) | 15 | 30.96/31 | 99.9% |
| Synthetic (norm=34, std normal) | 34 | 26.07/31 | 84.1% |
| Synthetic (norm=50) | 50 | 21.33/31 | 68.8% |

At higher input norms, the projection `e·x` has larger variance, so more pre-activations can land below zero despite the +1 bias offset. Real Gemma activations have modest norms (~15), keeping nearly all gates open.

## Analysis

### Why ~99% of gates are open at init

The gate pre-activation is:

```
pre_act = e · x - bias = e · x + 1.0
```

where `e` is a unit-norm encoder and `bias = -1.0` (Bug 3 fix). Since `e` is unit-norm, the projection `e · x` follows approximately `N(0, σ)` where `σ ≈ ||x|| / √d_model`.

For real Gemma activations:
- `||x|| ≈ 15`, `d_model = 1152` → `σ ≈ 15/√1152 ≈ 0.44`
- Pre-activations ≈ `N(1.0, 0.44)` — the +1 offset dominates
- P(pre_act < 0) ≈ P(z < -2.3) ≈ 1% → ~99% gates open

### Implication for L0 collapse

The collapse to L0 ≈ 0.375 seen by step 100 in experiments 01/08 happens **entirely during training**, not from initialization. The Bug 3 fix (bias=-1.0) successfully ensures near-universal initial activation. The winner-take-all JumpReLU dynamic kills transforms within the first ~100 gradient steps.

## Artifacts

- Results: `experiments/09_gemma_default_open/results/initial_gate_stats.json`
