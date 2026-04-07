# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction of Anthropic's [Sparse Mixtures of Linear Transforms (MOLTs)](https://transformer-circuits.pub/2025/bulk-update/index.html). Trains MOLT decompositions of MLP layers in Gemma-3-1B using FineWeb data, then evaluates faithfulness via Jacobian cosine similarity and L0 vs NMSE tradeoffs.

## Quick Start

```bash
./setup.sh                              # install uv, sync deps, check env vars
uv run pytest tests/                    # run all tests (38 tests)
```

## Commands

```bash
# Install (via uv)
uv sync --extra dev

# Run tests
uv run pytest tests/
uv run pytest tests/test_model.py::test_molt_forward

# Lint
uv run ruff check src/ experiments/ scripts/ tests/

# Run an experiment
uv run python experiments/01_gemma_sparsity_sweep/run.py

# Evaluate a checkpoint
uv run python scripts/eval_molt.py

# Upload to HuggingFace
uv run python -c "from molt.utils.hf_upload import upload_experiment; upload_experiment('experiments/01_gemma_sparsity_sweep', 'my-repo')"
```

## Repository Layout

```
src/molt/               Core library (import as `molt`)
  config.py             MOLTConfig dataclass with model presets
  model.py              MOLT, TransformGroup, JumpReLU
  train.py              Training loop with wandb integration
  data.py               FineWeb streaming + activation collection
  eval.py               NMSE, L0, Jacobian faithfulness
  interpret.py          Interpretability utilities (delphi)
  transcoder.py         Transcoder baseline comparison
  utils/                Shared utilities
    plotting.py         Training curves, sweep plots, L0-vs-NMSE
    experiment.py       ExperimentRunner: train->eval->save->cleanup
    activations.py      Load/split cached activation tensors
    hf_upload.py        Upload artifacts to HuggingFace Hub

experiments/            Numbered, immutable experiment folders
  _template/            Copy to start a new experiment
  01-06_*/              Completed experiments (each has run.py, README.md)

scripts/                Standalone CLI tools
  eval_molt.py          Generic checkpoint evaluation
  compare_baselines.py  MOLT vs transcoder comparison

notes/                  Research documentation
  read/                 Paper reading notes
  meeting-notes/        Team meeting notes
  bugs/                 Infrastructure bug documentation

tests/                  pytest suite (38 tests covering all external functions)
data/                   Activation caches + checkpoints (gitignored)
```

## Architecture

The pipeline has three stages: (1) stream FineWeb tokens and collect MLP input/output activations via hooks (`molt/data.py`), (2) train MOLT to reconstruct MLP outputs from inputs (`molt/train.py`), (3) evaluate with NMSE, L0, and Jacobian faithfulness (`molt/eval.py`).

**MOLT model** (`src/molt/model.py`): A `MOLT` contains multiple `TransformGroup`s, each a batch of low-rank transforms at the same rank. With rank multiplier N, this creates N×512 + 2N×256 + 4N×128 + 8N×64 + 16N×32 = 31N transforms. Each transform computes `gate(e·x - b) * (U @ V @ x)` where gate is JumpReLU with a smooth surrogate backward. The sparsity penalty is `tanh(mean|gate|) * ||UV||_F` (Frobenius norm normalized by sqrt(d_model * rank)).

**Config** (`src/molt/config.py`): Single `MOLTConfig` dataclass with model presets for Gemma-3-1B and GPT-2. Configs are hardcoded in each experiment's `run.py` (no external YAML files).

**Activation collection** (`src/molt/data.py`): Pre-allocates output tensors and writes in-place to avoid OOM (10M tokens × 1152 dims × float32 ≈ 43GB per tensor).

**Shared utilities** (`src/molt/utils/`): `ExperimentRunner` eliminates the ~80-line boilerplate repeated across experiments (train, eval, save JSON, compute transform activity, cleanup GPU). `plotting.py` provides standardized 4-panel training curve plots and Pareto frontier visualizations.

## Experiment Workflow

1. Copy `experiments/_template/` to `experiments/NN_name/`
2. Edit `run.py` with hardcoded config (all hyperparameters inline, no YAML)
3. Run: `uv run python experiments/NN_name/run.py`
4. Outputs go to `results/`, `figures/`, `logs/` (all gitignored, regenerable from run.py)
5. Update `README.md` with results and analysis

**Rules:** Never modify completed experiments. Create new numbered folders instead. Use `molt.utils` instead of duplicating boilerplate.

## Key Context: Infrastructure Bugs (6 Found & Fixed)

These bugs are documented in detail at `notes/bugs/infrastructure_bugs.md`. Read before modifying initialization, gating, sparsity penalty, or activation collection code.

1. **Bug 1 (Kaiming Init):** Initial Frobenius norms too large → sparsity penalty dominated → all transforms died. **Fix:** Scaled normal init + Frobenius norm normalization by sqrt(d_model * rank).
2. **Bug 2 (Dead STE):** Hard STE masked gradients for inactive gates → permanent death. **Fix:** Full STE passing gradients unconditionally; later replaced by smooth surrogate.
3. **Bug 3 (Zero Bias):** Only 30-40% gates fired at init. **Fix:** Bias initialized to -1.0, unit-norm encoder → 99% initial activation.
4. **Bug 4 (No Warmup):** λ at full strength from step 0 → late training collapse. **Fix:** Linear warmup over first 10% of steps.
5. **Bug 5 (NMSE Scaling):** Off by d_model factor (1152x). **Fix:** Removed extra multiplication.
6. **Bug 6 (OOM on torch.cat):** List + concatenation peaked at 180GB. **Fix:** Pre-allocate + in-place writes.

## Key Context: L0=1 Collapse

The model collapses to L0=1 (one active transform per token) regardless of sparsity coefficient, including λ=0. This is a JumpReLU winner-take-all dynamic:

- **On Gemma-3-1B:** JumpReLU collapses to L0=1, ReLU collapses to L0=0 (all gates die)
- **On GPT-2:** The smooth surrogate JumpReLU resolves the collapse (L0≈14.5, all 31 transforms active)
- **Root cause on Gemma:** Higher dimensionality (d=1152 vs 768) means transforms need more training steps to become useful, but gating dynamics kill them before they learn

See `notes/bugs/l0_collapse_analysis.md` for the full analysis and cross-model comparison.

## Completed Experiments

| # | Name | Key Finding |
|---|------|-------------|
| 01 | Gemma sparsity sweep | All λ values collapse to L0=1.0 |
| 02 | Activation/sparsity sweep | ReLU→L0=0 (all dead), JumpReLU→L0=1 (winner-take-all) |
| 03 | Learned threshold | Learned θ makes collapse worse (L0=0, θ runs away to +0.87) |
| 04 | GPT-2 sanity check | Smooth surrogate resolves collapse on GPT-2 (L0=14.5, NMSE=0.474) |
| 05 | GPT-2 sparsity penalty | L0 + JumpReLU at λ=1e-4 shows small sparsity effect |
| 06 | Gemma ReLU vs JumpReLU | Smooth surrogate fails on Gemma (θ runs away, all gates die) |

## Environment Requirements

- `HF_TOKEN` — Required for Gemma-3-1B (gated model)
- `WANDB_API_KEY` — Optional, for experiment tracking via wandb
- Activations kept in RAM only (not cached to disk) due to 50GB workspace disk limit
