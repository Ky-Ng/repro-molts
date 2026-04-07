# Experiment 07: GPT-2 Strong Sparsity Penalty Sweep

## Goal

Determine whether strong sparsity penalties push theta (learned JumpReLU threshold) increasingly negative on GPT-2. Experiment 05 showed theta went to -0.572 at lambda=1e-4 with L0 penalty. This experiment tests 10-1000x stronger penalties.

## Setup

- **Model:** `openai-community/gpt2`, layer 6, d_model=768
- **Data:** 2M FineWeb tokens (cached at `data/activations_openai_community_gpt2_2M.pt`)
- **Architecture:** N=1, 31 transforms
- **Training:** batch_size=64, lr=1e-3, 1 epoch (~31K steps), sparsity warmup 10%
- **JumpReLU:** smooth surrogate backward, learned threshold initialized at 0.0

**Lambda values:** 1e-3, 3e-3, 1e-2, 3e-2, 1e-1

**Setup types (4 per lambda, 20 total):**
- {Tanh, L0} sparsity penalty x {ReLU, JumpReLU (learned theta)} activation

## Reproduction

```bash
uv run python experiments/07_gpt2_strong_sparsity/run.py

# Run a single setup:
uv run python experiments/07_gpt2_strong_sparsity/run.py l0_jumprelu_lam1e-02
```

## Prior Results (from Experiment 05)

| Sparsity | lambda | theta_final |
|----------|--------|-------------|
| Tanh | 0 | -0.466 |
| Tanh | 1e-5 | -0.464 |
| Tanh | 1e-4 | -0.466 |
| L0 | 0 | -0.466 |
| L0 | 1e-5 | -0.478 |
| L0 | 1e-4 | **-0.572** |

## Results

_To be filled after running._

## Artifacts

- Results: `experiments/07_gpt2_strong_sparsity/results/`
- Figures: `experiments/07_gpt2_strong_sparsity/figures/`
