# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction of Anthropic's [Sparse Mixtures of Linear Transforms (MOLTs)](https://transformer-circuits.pub/2025/bulk-update/index.html). Trains MOLT decompositions of MLP layers in Gemma-3-1B using FineWeb data, then evaluates faithfulness via Jacobian cosine similarity and L0 vs NMSE tradeoffs.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a single test
pytest tests/test_model.py::test_molt_forward

# Lint
ruff check molt/ scripts/ tests/

# Train (full pipeline: stream data → collect activations → train)
python scripts/train_molt.py --config configs/default.yaml

# Override config from CLI
python scripts/train_molt.py --sparsity_coeff 1e-4 --num_tokens 1000000

# Evaluate a checkpoint
python scripts/eval_molt.py

# Sparsity sweep (parallel, 3 concurrent)
python scripts/sweep_parallel.py
```

## Architecture

The pipeline has three stages: (1) stream FineWeb tokens and collect MLP input/output activations from Gemma via hooks (`molt/data.py`), (2) train MOLT to reconstruct MLP outputs from inputs (`molt/train.py`), (3) evaluate with NMSE, L0, and Jacobian faithfulness (`molt/eval.py`).

**MOLT model** (`molt/model.py`): A `MOLT` contains multiple `TransformGroup`s, each a batch of low-rank transforms at the same rank. With rank multiplier N, this creates N×512 + 2N×256 + 4N×128 + 8N×64 + 16N×32 = 31N transforms. Each transform computes `gate(e·x - b) * (U @ V @ x)` where gate is JumpReLU with a full straight-through estimator. The sparsity penalty is `tanh(mean|gate|) * ||UV||_F` (Frobenius norm normalized by sqrt(d_model * rank)).

**Config** (`molt/config.py`): Single `MOLTConfig` dataclass. YAML config in `configs/default.yaml` serves as base; any field can be overridden from CLI via `--field_name value`.

**Activation collection** (`molt/data.py`): Pre-allocates output tensors and writes in-place to avoid OOM (10M tokens × 1152 dims × float32 ≈ 43GB per tensor). Disk cache is typically not used due to 50GB disk limit.

## Key context (CONTEXT.md)

- The model currently collapses to L0=1 (one active transform per token) regardless of sparsity coefficient, including λ=0. This is a JumpReLU winner-take-all dynamic, not a bug — see CONTEXT.md "L0=1 Collapse" for full analysis.
- Gemma-3-1B is a gated model; requires `HF_TOKEN` env var.
- Activations are kept in RAM only (not cached to disk) due to workspace disk limits.
- CONTEXT.md documents six infrastructure bugs that were found and fixed; read it before modifying initialization, gating, sparsity penalty, or activation collection code.
