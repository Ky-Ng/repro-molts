# repro-molts

Reproducing Anthropic's [Sparse Mixtures of Linear Transforms (MOLTs)](https://transformer-circuits.pub/2025/bulk-update/index.html).

## Quick Start

```bash
./setup.sh                    # install uv + dependencies
uv run pytest tests/          # verify everything works
```

## What This Does

Trains MOLT decompositions of MLP layers and evaluates faithfulness via:
1. Jacobian cosine similarity
2. L0 (active transforms) vs Normalized MSE tradeoffs

Currently tested on Gemma-3-1B and GPT-2 with FineWeb data.

## Running Experiments

Each experiment lives in `experiments/NN_name/` with a self-contained `run.py`:

```bash
uv run python experiments/04_gpt2_sanity_check/run.py
```

See `experiments/README.md` for details on creating new experiments.

## Repository Structure

```
src/molt/          Core library (model, training, evaluation)
experiments/       Numbered experiment folders (run.py + README.md each)
scripts/           Standalone CLI tools
tests/             pytest suite (38 tests)
notes/             Research docs and bug documentation
data/              Activation caches + checkpoints (gitignored)
```

## Environment Variables

- `HF_TOKEN` — Required for Gemma-3-1B (gated model)
- `WANDB_API_KEY` — Optional, for experiment tracking

## Goals

1. Qualitatively evaluate MOLT transforms by visualizing interactions with SAE
2. Compare MOLTs to Transcoders (Skip/Non-Skip)
3. Replicate quantitative analysis: Jacobian faithfulness + L0 vs NMSE Pareto frontier
