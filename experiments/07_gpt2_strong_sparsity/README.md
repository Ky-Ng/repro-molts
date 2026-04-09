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

## Results (10/20 runs complete — lambda=1e-3, 3e-3, plus tanh at 1e-2)

### Full Table

| Sparsity | Activation | lambda | L0 | NMSE | theta_final | #Active |
|----------|------------|--------|------|------|-------------|---------|
| Tanh | ReLU | 1e-3 | 14.95 | 0.493 | fixed | 31 |
| Tanh | JumpReLU | 1e-3 | 14.07 | 0.475 | -0.406 | 31 |
| L0 | ReLU | 1e-3 | 14.52 | 0.503 | fixed | 31 |
| L0 | JumpReLU | 1e-3 | 7.50 | 0.477 | **-1.110** | 31 |
| Tanh | ReLU | 3e-3 | 10.88 | 0.502 | fixed | 31 |
| Tanh | JumpReLU | 3e-3 | 15.00 | 0.479 | -0.449 | 31 |
| L0 | ReLU | 3e-3 | 11.52 | 0.527 | fixed | 31 |
| L0 | JumpReLU | 3e-3 | 2.82 | 0.493 | **-2.319** | 30 |
| Tanh | ReLU | 1e-2 | 4.36 | 0.561 | fixed | 15 |
| Tanh | JumpReLU | 1e-2 | 13.39 | 0.522 | -0.116 | 31 |

### Theta Trajectory (JumpReLU + L0 penalty)

Combining with Experiment 05 baselines:

| lambda | theta | L0 | NMSE |
|--------|-------|-----|------|
| 0 | -0.466 | 14.46 | 0.474 |
| 1e-5 | -0.478 | 14.32 | 0.473 |
| 1e-4 | -0.572 | 13.06 | 0.472 |
| 1e-3 | **-1.110** | 7.50 | 0.477 |
| 3e-3 | **-2.319** | 2.82 | 0.493 |

### Preliminary Analysis

1. **L0 + JumpReLU: theta goes strongly negative** under increasing sparsity pressure. The trend is clear and accelerating: theta roughly doubles in magnitude for each 3x increase in lambda.

2. **Tanh + JumpReLU: theta is non-monotonic.** It went from -0.449 (lambda=3e-3) to -0.116 (lambda=1e-2). The Tanh penalty does not consistently drive theta negative — it saturates because `tanh(mean|gate|)` is near 1.0 for active transforms, providing near-zero gradient to discriminate between them.

3. **L0 penalty is far more effective at sparsity.** At lambda=3e-3, L0+JumpReLU achieves L0=2.82 while Tanh+JumpReLU stays at L0=15.0. The L0 penalty pushes theta more negative (opening more gates) while simultaneously reducing the *number* of active transforms — the remaining transforms absorb the work of pruned ones.

4. **NMSE degrades slowly.** Even at L0=2.82, NMSE is only 0.493 (vs 0.474 at L0=14.5 baseline). The sparsity-reconstruction tradeoff is favorable.

5. **Tanh+ReLU at lambda=1e-2 shows transform death.** Only 15/31 transforms remain active, with L0 dropping to 4.36. This is the ReLU dead-gate dynamic being amplified by the sparsity penalty.

### Remaining runs (in progress)

- L0 penalty: lambda=1e-2 (both ReLU and JumpReLU)
- All 4 setups at lambda=3e-2 and lambda=1e-1

## Artifacts

- Results: `experiments/07_gpt2_strong_sparsity/results/`
- Figures: `experiments/07_gpt2_strong_sparsity/figures/`

## Bug
- According to Lindsey et al. 2025, the sparsity penalty should be L1 not L0. We will need to remove L0 as a sparsity penalty in the loss function in future iterations and re-run these sweeps.