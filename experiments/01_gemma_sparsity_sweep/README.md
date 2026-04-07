# Experiment 01: Gemma-3-1B Sparsity Sweep

## Goal

Sweep lambda values on Gemma-3-1B to find the L0 vs NMSE Pareto frontier.

## Setup

- **Model:** Gemma-3-1B, layer 13, d_model=1152
- **Data:** 10M FineWeb tokens (streamed, not cached)
- **Architecture:** N=1, 31 transforms, JumpReLU gating
- **Lambda values:** 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
- **Training:** batch_size=64, lr=1e-3, 1 epoch, 10% sparsity warmup
- **Parallelism:** ThreadPoolExecutor, 3 concurrent workers on single GPU

## Reproduction

```bash
uv run python experiments/01_gemma_sparsity_sweep/run.py
```

Environment variables: `MOLT_WORKERS=3`, `MOLT_TOKENS=10000000`

## Results

All lambda values collapsed to L0=1.0 (one active transform per token). This is the JumpReLU winner-take-all dynamic, not a sparsity penalty effect.

## Artifacts

- Results: `experiments/01_gemma_sparsity_sweep/results/`
- Pareto plot: `experiments/01_gemma_sparsity_sweep/figures/`
