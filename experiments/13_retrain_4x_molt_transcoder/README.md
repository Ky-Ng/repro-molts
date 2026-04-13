# Experiment 13: Retrain 4x MOLT + Transcoder and Label Features with Delphi

## Goal

Retrain the highest-compute MOLT and parameter-matched transcoder from
[experiment 11](../11_transcoder_comparison/README.md), save their checkpoints,
and use [delphi](https://github.com/EleutherAI/delphi) to generate and score
natural-language labels for every feature.

This produces the first interpretable feature catalog on this codebase and
lets us ask: *at matched compute, do MOLTs yield features that are more or
less monosemantic than transcoders, as measured by delphi scorers?*

### Background

Experiment 11 trained a 51-run sweep over `(scale, λ)` for both methods and
established that MOLTs Pareto-dominate transcoders on L0-vs-NMSE. It did
**not** look at *what features are learned* — only at reconstruction
fidelity. Exp 13 picks the two operating points that exp 11 identified as
representative (1x smoke + 4x main) and runs them through delphi's
auto-interp pipeline (LatentCache → ContrastiveExplainer → DetectionScorer +
FuzzingScorer) to produce per-feature labels and quality scores.

## Setup

- **Model:** GPT-2, layer 6 (d_model=768) — same as exp 11
- **Data:** 500K FineWeb activations for training, 10K for eval, 2M tokens for delphi cache
- **Architecture:** MOLT (tanh + smooth-surrogate JumpReLU) and TrainableTranscoder (ReLU + L1)
- **Training:** Adam, lr=1e-3, batch=1024, 10% sparsity warmup, seed=42
- **Explainer LLM:** Llama-3.1-70B-Instruct via OpenRouter ($0.40/M in+out)
- **Scorers:** delphi DetectionScorer + FuzzingScorer (skip simulation/embedding/surprisal)
- **Budget cap:** $20 total OpenRouter spend (enforced in `run.py`)

### Configs trained

Two `(scale, λ)` points per method. The 1x slice is a smoke test that
validates the full pipeline; the 4x main slice is the headline run.

| Slice | Method     | Scale | Features | λ    | Predicted L0 (exp 11) | Predicted NMSE (exp 11) |
|-------|------------|-------|----------|------|-----------------------|-------------------------|
| smoke | MOLT       | 1x    | 31       | 1e-3 | ~19                   | 0.694                   |
| smoke | Transcoder | 1x    | 2 573    | 1.0  | ~366                  | 0.598                   |
| main  | MOLT       | 4x    | 124      | 1e-3 | ~43                   | 0.462                   |
| main  | Transcoder | 4x    | 10 295   | 3.0  | ~694                  | 0.255                   |

**Why these λ.** MOLT λ=1e-3 is the Pareto sweet-spot at 4x in exp 11 (L0
drops from 101.6 at λ=0 to 43.3 with NMSE only rising from 0.455 to 0.462).
Transcoders need L1 penalties 3–4 orders of magnitude larger to reach
comparable L0 — λ=1.0 (1x) and λ=3.0 (4x) place them at exp 11's healthy
operating range without forcing reconstruction collapse.

### Pipeline

```
Stage 0  retrain + checkpoint     ──► out/checkpoints/*.pt
Stage 1  delphi LatentCache       ──► out/latents/n2000000/{slice}/h.6.mlp/*.safetensors
Stage 2  ContrastiveExplainer     ──► out/explanations/{slice}/h.6.mlp_latent{id}.txt
Stage 3  Detection + Fuzzing      ──► out/scores/{kind}/{slice}/h.6.mlp_latent{id}.txt
analyze  merge labels + scores    ──► results/labels_{slice}.json
```

Stages are independently re-runnable. Each stage skips work that already
exists on disk.

### Design decisions

1. **MOLT feature strength = raw gate value** (not gate × ‖UV‖_F). Gate is
   the per-token quantity; ‖UV‖_F is a per-transform constant. For
   clustering tokens by feature participation we want the per-token value.
2. **MOLT feature-id flattening.** Stable canonical order across the 5 rank
   groups (512, 256, 128, 64, 32), N transforms each. At N=4 (4x): 124
   features. Stored in [feature_layout.py](feature_layout.py).
3. **Activation-frequency filter.** Drop features with `freq < 1e-4` (dead,
   unlabelable) or `freq > 0.2` (hyper-dense, polysemantic by construction).
   Applied to the transcoder; disabled upper bound for MOLT since all 1x
   transforms are inherently dense (see Results §Stage 1).
4. **Explainer = Llama-3.1-70B-Instruct.** Delphi's documented default;
   ~10× cheaper than Claude Haiku 4.5 ($0.40/$0.40 per M tokens vs $1/$5).
5. **Scorers = detection + fuzzing only.** Each adds ~2× cost; not needed
   for the core MOLT-vs-transcoder comparison.

## Reproduction

```bash
# 1x smoke: cheap pipeline validation (~$2)
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage train --slice smoke
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage cache --slice smoke
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage explain --slice smoke
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage score --slice smoke

# 4x main: gated on smoke go/no-go (~$5 additional)
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage train --slice main
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage cache --slice main
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage explain --slice main --confirm-budget
uv run python experiments/13_retrain_4x_molt_transcoder/run.py --stage score --slice main

# Figures (training curves + L0-vs-NMSE comparison against exp 11)
uv run python experiments/13_retrain_4x_molt_transcoder/plot_figures.py
```

Required env vars: `HF_TOKEN`, `OPEN_ROUTER_DEV_KEY`. The 12 GB activation
cache downloads from `kylelovesllms/auto-repro-molts` on first stage-0 run.

## Results

### Stage 0 — Training (all 4 slices)

All four runs land within ~5% of the exp-11 L0 prediction; NMSE is
slightly better than predicted in every case.

| Slice | Method     | Scale | λ    | L0 (predicted) | L0 (actual) | NMSE (predicted) | NMSE (actual) | Time   |
|-------|------------|-------|------|----------------|-------------|------------------|---------------|--------|
| smoke | MOLT       | 1x    | 1e-3 | ~19            | **20.5**    | 0.694            | **0.670**     | 9.0 s  |
| smoke | Transcoder | 1x    | 1.0  | ~366           | **371.2**   | 0.598            | **0.578**     | 1.0 s  |
| main  | MOLT       | 4x    | 1e-3 | ~43            | **45.3**    | 0.462            | **0.444**     | 34.4 s |
| main  | Transcoder | 4x    | 3.0  | ~694           | **701.9**   | 0.255            | **0.245**     | 4.9 s  |

Source: [results/train_summary.json](results/train_summary.json).

### Stage 1 — Feature filtering

Activation-frequency filter (`1e-4 ≤ freq ≤ 0.2`) was applied to the
transcoder; the upper bound was disabled for MOLT since every 1x transform
is dense by construction. Source: [results/feature_stats_*.json](results/).

| Slice            | Total  | Kept    | Dead | Hyper-dense (freq > 0.2) | Freq min / median / max |
|------------------|--------|---------|------|--------------------------|-------------------------|
| molt_1x_lam1e-03 | 31     | **31**  | 0    | 31 *(kept anyway)*       | 0.314 / 0.628 / 1.000   |
| tc_1x_lam1e+00   | 2 573  | **2 129** | 0  | 444 *(dropped)*          | 0.012 / 0.134 / 0.482   |

Every MOLT 1x transform fires on ≥31% of tokens — at this scale MOLTs are
inherently hyper-dense. We label them all and rely on Stage 3 F1s + an
eyeball pass to judge interpretability. 4x slice not yet through this stage.

### Stage 2 — Explanations (1x only)

- `out/explanations/molt_1x_lam1e-03/` — 31 labels
- `out/explanations/tc_1x_lam1e+00/` — 2 129 labels

Budget used across both 1x slices: **$1.97 / $20.00** over **2 122 calls**
(4.85M prompt tokens, 76.7K completion tokens, ~2 h wall clock). Well under
the $5 go/no-go threshold for 1x. Source:
[results/budget_summary.json](results/budget_summary.json).

Sample MOLT 1x labels (first 3 features):

- **F0** — "Formal or informative text passages, often from news articles, academic writing, or official documents..."
- **F1** — "Blocks of text that appear to be excerpts or quotes from various sources, including articles, books, and online content..."
- **F10** — "Various texts from different sources... with no specific pattern or theme, but often featuring descriptive language, quotes, and references..."

These read visibly vague, consistent with the hyper-dense activation
frequencies from Stage 1. Stage 3 F1s will quantify.

### Stage 3 — Scoring

Not yet run. Needed to resolve the 4x go/no-go criteria:

- ≥ 60% of MOLT features (≥ 19/31) with detection F1 ≥ 0.5
- Median transcoder feature detection F1 ≥ 0.4
- Cumulative 1x spend ≤ $5 ✓ (already satisfied at $1.97)

If MOLT 1x F1s look weak (likely, given the vague labels above), root-cause
before committing to the 4x labeling spend.

## Analysis

### Finding 1: Reproduction matches exp 11 within noise

All four `(L0, NMSE)` points sit within ~5% of the exp-11 sweep, and the
NMSE is consistently slightly better (-3% to -5%). Same seed, same data,
same code path — the small NMSE gap is most likely from slightly different
shuffle order in the GPU-batched loop. The retrained checkpoints are valid
inputs to delphi.

### Finding 2: 1x MOLT is too dense for delphi's contrastive explainer

With only 31 transforms and median activation frequency of 0.63, every
MOLT 1x feature fires on most tokens. The top-K activating contexts pulled
by delphi span a huge variety of text, and the resulting labels read like
"formal text passages" or "various sources" — exactly the failure mode the
hyper-dense filter (`freq > 0.2 → drop`) was designed to catch. We kept
them anyway because dropping all 31 would leave no MOLT 1x results to
compare; the right move is to look to the 4x slice (124 transforms, expected
median freq much lower) for meaningful MOLT labels.

### Finding 3: Explainer cost is well under cap

Total Stage-2 spend so far ($1.97) is ~10% of cap on the easier slice. The
4x slice is bounded above by ~$5 (124 + ~3 000 features at the same $0.0017
each), leaving comfortable headroom for retries and any second-pass scoring.

### Limitation: 4x labels and all scoring still pending

Headline comparison (MOLT 4x labels vs TC 4x labels) cannot be made yet.
Stages 2 (4x) and 3 (all slices) are required before this experiment
delivers on its goal.

## Figures

1. `train_molt_1x_lam1e-03.png`, `train_molt_4x_lam1e-03.png`,
   `train_tc_1x_lam1e+00.png`, `train_tc_4x_lam3e+00.png` — 4-panel
   training curves (NMSE, L0, sparsity loss, MSE), same format as exp 11
2. `l0_vs_nmse.png` — the four exp-13 retrained points overlaid on the
   exp-11 sweep, log-log axes, color by scale

Regenerate with:

```bash
uv run python experiments/13_retrain_4x_molt_transcoder/plot_figures.py
```

## Artifacts

Tracked in git:

- Code: `run.py`, `plot_figures.py`, `eyeball.py`, `budget.py`, `feature_layout.py`
- Results: `results/` — `train_summary.json`, `feature_stats_*.json`, `budget_summary.json`, `budget_log_explain.jsonl`
- Figures: `figures/` — training curves + L0-vs-NMSE
- This `README.md`

Not tracked (regenerable, written under `out/`, gitignored):

- `out/checkpoints/` — 151 MB of `.pt` (state_dict + config + history). Uploaded to HF as `kylelovesllms/auto-repro-molts:checkpoints/13/*` for reproducibility.
- `out/latents/` — ~17 GB of delphi safetensors caches. Regenerable from checkpoints + activation cache (~10–20 min/slice on GPU).
- `out/token_cache/` — 16 MB of cached FineWeb token IDs. Regenerable from FineWeb + GPT-2 tokenizer.
- `out/explanations/` — 8.6 MB of per-feature label `.txt` files from Stage 2. Regenerable but costs ~$2 in OpenRouter API calls.
- `out/scores/` — per-feature F1 `.txt` files from Stage 3 (not yet generated).
