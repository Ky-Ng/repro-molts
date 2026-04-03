# GPT-2 Sanity Check: MOLT Activation/Sparsity Sweep

## Purpose

Sanity-check whether the L0 collapse observed on Gemma-3-1B is model-specific or universal. Running the same 2×2 sweep ({Tanh, L1} × {ReLU, JumpReLU}, all λ=0) on GPT-2 (768 hidden dim, 12 layers) to isolate model-dependent effects from MOLT architecture issues.

## Setup

- **Model:** `openai-community/gpt2` (124M params, d_model=768, 12 layers)
- **Layer:** 6 (mid-layer), MLP path: `transformer.h.6.mlp`
- **Data:** 2M FineWeb tokens, cached at `data/activations_openai_community_gpt2_2M.pt` (12GB)
- **Architecture:** N=1, 31 transforms (1×rank-512, 2×rank-256, 4×rank-128, 8×rank-64, 16×rank-32)
- **Training:** batch_size=64, lr=1e-3, 1 epoch (~31K steps), λ=0, no warmup

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

- `scripts/run_model_sweep.py`: Generalized sweep script that takes model name as CLI argument. Works for any model with a preset.

## Results

| Sparsity | Activation | L0 | NMSE | Active Transforms |
|----------|------------|------|------|-------------------|
| Tanh | ReLU | **18.85** | **0.489** | **All 31** (25–88% each) |
| L1 | ReLU | **18.85** | **0.489** | **All 31** (25–88% each) |
| Tanh | JumpReLU | 1.59 | 0.739 | T0 (99.6%) + sparse low-rank |
| L1 | JumpReLU | 1.59 | 0.739 | T0 (99.6%) + sparse low-rank |

As expected, sparsity type is irrelevant at λ=0 (Tanh≡L1 within each activation type).

### Comparison to Gemma-3-1B

| | Gemma-3-1B (d=1152) | GPT-2 (d=768) |
|---|---|---|
| **ReLU, λ=0** | L0=0.0, NMSE=1.001 (all dead) | L0=18.85, NMSE=0.489 (all alive) |
| **JumpReLU, λ=0** | L0=1.0, NMSE=0.120 (winner-take-all) | L0=1.59, NMSE=0.739 (near-collapse) |

## Analysis

### ReLU on GPT-2: No Collapse

All 31 transforms remain active with healthy utilization (25–88% per transform). L0 stabilizes around 19 by step 10K and NMSE reaches 0.489. This is the expected behavior of a mixture model — many transforms contribute, each specializing in different input subspaces.

**Why does ReLU work on GPT-2 but not Gemma?** The key difference is the activation distribution at the MLP input:
- GPT-2 layer 6 MLP inputs are float32 with moderate variance
- Gemma-3-1B layer 13 MLP inputs are converted from bfloat16 with mean≈-0.008 and std≈0.45

The gating pre-activation `e_t · x - b_t` depends on the distribution of `e_t · x`. With GPT-2's activation distribution, the dot products stay in a range where ReLU gradients remain non-zero for a large fraction of tokens. On Gemma, the dot products cluster closer to zero, making ReLU gates fragile — once negative they die permanently.

### JumpReLU on GPT-2: Partial Collapse

JumpReLU collapses toward L0≈1.6 (down from 25.5 at initialization), with T0 (rank-512) dominating at 99.6%. However, unlike Gemma's clean L0=1.0, several low-rank transforms retain sparse activity (1–10%). The collapse is still happening but is less severe — likely because GPT-2's MLP is lower-dimensional (768 vs 1152) so even a partially trained rank-512 transform covers a larger fraction of the Jacobian.

**Training trajectory (JumpReLU):**
```
Step    L0     NMSE
  100  25.55  0.900   ← all transforms active at init
 5100   5.62  0.666   ← rapid competitive elimination
10100   3.97  0.695   ← still slowly collapsing
20100   2.25  0.750   ← approaching winner-take-all
31200   1.70  0.810   ← near-converged, T0 dominates
```

Note: NMSE *worsens* as L0 decreases (0.67 → 0.81), confirming that the collapse hurts reconstruction quality. The model would benefit from keeping more transforms active.

### JumpReLU NMSE Is Worse Than ReLU

On GPT-2, **ReLU (NMSE=0.489) substantially outperforms JumpReLU (NMSE=0.739)**. This is the opposite of what might be expected from the Gemma results (where JumpReLU at least produced NMSE=0.12 while ReLU gave NMSE=1.0). The reason: on GPT-2, ReLU keeps all 31 transforms active, giving much more representational capacity than JumpReLU's single dominant transform.

### Key Takeaway: The L0=1 Collapse Is Not Universal

The collapse to L0=1 with JumpReLU appears to be a general optimization issue (present on both models), but its severity depends on the model. The total collapse to L0=0 with ReLU is Gemma-specific, likely related to the activation distribution at that layer.

**This suggests the MOLT architecture itself is capable of using multiple transforms** — the problem is specifically in the JumpReLU gating dynamics, not in the transform learning capacity.

## Artifacts

- Cached activations: `data/activations_openai_community_gpt2_2M.pt` (12GB)
- Checkpoints: `checkpoints/openai_community_gpt2/{tanh_relu,tanh_jumprelu,l1_relu,l1_jumprelu}/`
- Results: `results/openai_community_gpt2/result_*.json`
- Training histories: `results/openai_community_gpt2/history_*.json`
- Comparison plot: `results/openai_community_gpt2/sweep_comparison.png`
