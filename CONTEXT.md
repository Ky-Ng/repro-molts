# MOLT Reproduction — Context & Infrastructure Notes

## Current State (2026-04-02)

### Completed
- **Phase 1 PoC training** on Gemma-3-1B layer 13, N=1, 10M FineWeb tokens
  - Final metrics: NMSE=0.132, L0=1.0, Jacobian cosine_sim=0.015
  - Checkpoint: `checkpoints/full_N1/molt_N1_lam0.001.pt`
  - Training curve: `results/training_curve_N1_lam1e-3.png`
- **Sparsity sweep** (λ = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
  - All 7 values collapsed to L0=1.0 — no Pareto frontier produced
  - Results: `results/sweep/sweep_N1.json`, plot: `results/sweep/pareto_N1.png`
  - Parallel training via `scripts/sweep_parallel.py` using ThreadPoolExecutor (3 concurrent)
- **L0=1 collapse diagnosis** — confirmed collapse is architectural, not from sparsity penalty
  - λ=0 control experiment: still collapses to L0=1.0, NMSE=0.120
  - Plot: `results/training_no_sparsity.png`
  - Checkpoint: `checkpoints/no_sparsity/molt_N1_lam0.0.pt`
- **Activation/sparsity setup sweep** (λ=0, 2M tokens) — 3 additional setups tested
  - Added `sparsity_type` config option ("tanh" or "l1") to `molt/config.py` and `molt/model.py`
  - Results in `results/activation_sweep/`, plot: `results/activation_sweep/activation_sweep_comparison.png`
  - Cached 2M token activations: `data/activations_2M.pt` (18GB)
  - See "Activation/Sparsity Setup Sweep" section below for full results

### Blocked
- **Sparsity sweep is non-informative** — all λ values produce L0=1.0 (see "L0=1 Collapse" below)
- Jacobian faithfulness comparison, transcoder baselines, and interpretability all depend on resolving the collapse

### Not Yet Started
- Transcoder baseline comparison (Gemma Scope 2 skip/non-skip)
- Interpretability analysis with delphi
- N=2, N=4 rank multiplier experiments

---

## Infrastructure Bugs Found & Fixed

### Bug 1: Dead Transforms — Kaiming Init Produces Frobenius Norm Explosion

**File:** `molt/model.py` — `TransformGroup._init_weights()`

**Symptom:** All 31 transforms die (L0→0) within the first few thousand training steps. NMSE converges to ~1.0 (model outputs near-zero).

**Root Cause:** The original code used `nn.init.kaiming_uniform_` for the U, V, and encoder matrices:

```python
def _init_weights(self):
    nn.init.kaiming_uniform_(self.V.view(-1, self.d_model))
    nn.init.kaiming_uniform_(self.U.view(-1, self.rank))
    nn.init.kaiming_uniform_(self.encoder)
```

Kaiming init produces per-element values scaled by `sqrt(2/fan_in)`, but the Frobenius norm of U (shape `[d_model, rank]`) grows as `||U||_F ~ sqrt(d_model * rank)`. The sparsity penalty multiplies `tanh(mean|gate|)` by `||U||_F * ||V||_F`, which evaluates to ~500-1500 at init for the various rank groups. With λ=1e-3, the initial `λ * sparsity_loss ≈ 5.4` is comparable to `MSE ≈ 6.7`, causing the sparsity gradient to dominate immediately and kill all gates before transforms learn anything useful.

**Fix:** Two changes:
1. Scaled normal init for U, V with `std = (1 / (d_model * rank))^0.25`, producing smaller initial transforms
2. Normalized Frobenius norms by `sqrt(d_model * rank)` so the penalty is dimension-independent:

```python
frobenius_norms = u_norms * v_norms / (self.d_model * self.rank) ** 0.5
```

This makes the initial sparsity loss ~0.02 (vs MSE ~7.5), giving the model room to learn before sparsity kicks in.

---

### Bug 2: JumpReLU Straight-Through Estimator Kills Gradient for Inactive Gates

**File:** `molt/model.py` — `JumpReLU.backward()`

**Symptom:** Once a gate's pre-activation drops below the threshold (0.0), it can never reactivate. Even after fixing Bug 1, transforms still collapse to L0=0 after ~4000 steps.

**Root Cause:** The original STE only passes gradients where `x > threshold`:

```python
@staticmethod
def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return grad_output * (x > ctx.threshold).float(), None
```

This creates a "dead ReLU" problem for gates. The output is `gate * UVx`. When `gate = 0` (pre-act < threshold), the gradient `d(output)/d(pre_act) = 0` because the STE masks it. The MSE loss cannot push a dead gate back on — there's no gradient signal. Meanwhile, there's also no sparsity gradient for dead gates (since `|gate|=0 → tanh(0)=0`). Dead gates remain permanently dead.

**Fix:** Use a full straight-through estimator that passes gradients unconditionally:

```python
@staticmethod
def backward(ctx, grad_output):
    return grad_output, None
```

This allows the MSE loss to "pull" gates back on when a transform would improve reconstruction. The forward still uses hard thresholding (`x * (x > 0)`), so the sparsity pattern is discrete at inference time — only the training gradient is relaxed.

---

### Bug 3: Gate Bias Initialized to Zero — Insufficient Initial Activation

**File:** `molt/model.py` — `TransformGroup.__init__()`

**Symptom:** At initialization with `bias=0`, only ~30-40% of gates fire (pre-acts have negative mean), compounding Bug 2's dead gate problem.

**Root Cause:** The gating pre-activation is `e_t · x - bias`. With encoder vectors from Kaiming init and typical MLP input activations (mean≈-0.008, std≈0.45), the dot product `e_t · x` has a slightly negative mean and moderate variance. With `bias=0`, roughly 30-40% of pre-activations are positive. The negative-mean inputs combined with random encoder directions means many gates start near-dead.

**Fix:** Initialize bias to -1.0 so that `e_t · x - (-1) = e_t · x + 1`, shifting the pre-activation positive. Also normalize encoder vectors to unit norm for consistent gating scale:

```python
nn.init.constant_(self.bias, -1.0)
nn.init.normal_(self.encoder)
with torch.no_grad():
    self.encoder.div_(self.encoder.norm(dim=1, keepdim=True))
```

Result: ~99% of gates fire at initialization, giving all transforms a chance to learn before sparsity prunes them.

---

### Bug 4: Missing Sparsity Warmup — λ Applied at Full Strength from Step 0

**File:** `molt/train.py`, `molt/config.py`, `molt/model.py`

**Symptom:** Even after Bugs 1-3 were fixed, the model showed promising early training (NMSE→0.11 by step 10K, L0→2) but then collapsed (NMSE→1.05, L0→0.3) by the end of training. The sparsity penalty overwhelmed the few remaining active transforms.

**Root Cause:** The sparsity coefficient λ was applied at full strength from the very first step. During early training when the model is still learning basic reconstruction, the sparsity penalty competes with MSE gradient and pushes gates toward zero before the transforms have learned useful decompositions.

**Fix:** Added linear warmup for the sparsity coefficient:

```python
# config.py
sparsity_warmup_frac: float = 0.1  # ramp λ from 0 over first 10% of steps

# train.py
sparsity_scale = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
loss, metrics = model.loss(batch_inputs, batch_targets, sparsity_scale)

# model.py  
effective_lambda = self.config.sparsity_coeff * sparsity_scale
total = mse + effective_lambda * sparsity
```

Result: Stable training throughout. NMSE converges to ~0.13 with L0=1.0 and no late-training collapse.

---

### Bug 5: `compute_nmse` Off by Factor of `d_model` (1152x)

**File:** `molt/eval.py` — `compute_nmse()`

**Symptom:** Evaluation NMSE returned 155.7 instead of ~0.135 on the same distribution that showed NMSE=0.12 in manual checks.

**Root Cause:** The function used `reduction="sum"` for MSE, counted total elements, then multiplied by `target.shape[1]`:

```python
mse = F.mse_loss(output, bt, reduction="sum").item()
total_mse += mse
count += bt.numel()
mean_mse = total_mse / count * target.shape[1]  # BUG: extra * d_model
```

`total_mse / count` gives per-element MSE. Multiplying by `target.shape[1]` (=d_model=1152) gives per-sample MSE, but `target.var()` computes per-element variance. Dividing per-sample MSE by per-element variance inflates NMSE by d_model.

**Fix:** Remove the `* target.shape[1]`:

```python
mean_mse = total_mse / count  # per-element MSE
```

Now NMSE = per-element MSE / per-element variance, matching the training-time NMSE computed via `F.mse_loss(output, target) / target.var()`.

---

### Bug 6: OOM During `torch.cat` of Activation Lists

**File:** `molt/data.py` — `collect_activations()`

**Symptom:** Process killed (OOM) after 100% of activations collected for 10M tokens. No cache file written.

**Root Cause:** The original code appended each chunk's activations to a Python list, then concatenated:

```python
mlp_inputs_all.append(captured["mlp_input"].squeeze(0).cpu())
...
mlp_inputs = torch.cat(mlp_inputs_all, dim=0)
```

For 10M tokens: 39063 chunks × (256, 1152) float32 = ~45 GB per list. `torch.cat` allocates a new ~45 GB tensor while the list is still alive → peak ~90 GB per variable, ~180 GB total. With 236 GB available RAM, this was marginal and failed.

**Fix:** Pre-allocate output tensors and write in-place:

```python
mlp_inputs = torch.empty(total_tokens, config.d_model, dtype=torch.float32)
...
mlp_inputs[start:end] = captured["mlp_input"].squeeze(0).cpu()
```

Peak memory is now exactly 2 × (10M × 1152 × 4 bytes) = ~86 GB with no temporary allocation spike.

---

### Infrastructure Note: Disk Space Limitation

The workspace has a 50 GB disk overlay. 10M token activations in float32 are ~86 GB and cannot be cached to disk. The current approach keeps activations in RAM only. For the sparsity sweep, activations are collected once and shared across training threads via `ThreadPoolExecutor` (not `ProcessPoolExecutor`, which would require `/dev/shm` shared memory — limited to 31 GB).

---

### Infrastructure Note: Gemma-3-1B is Gated

`google/gemma-3-1b-it` requires HuggingFace authentication. The token is passed via `HF_TOKEN` environment variable (never stored in code). The `transformers` library reads it automatically.

---

## L0=1 Collapse

### Observation

Across all experiments — every λ value in the sparsity sweep and a λ=0 control — the model collapses to exactly L0=1.0 (one active transform per token). This was confirmed with a control experiment using zero sparsity penalty.

### Sparsity Sweep Results (N=1, 10M FineWeb tokens)

| λ | L0 | NMSE |
|---|-----|------|
| 1e-5 | 1.0 | 0.135 |
| 3e-5 | 1.0 | 0.191 |
| 1e-4 | 1.0 | 0.140 |
| 3e-4 | 1.0 | 0.375 |
| 1e-3 | 1.0 | 0.155 |
| 3e-3 | 1.0 | 0.143 |
| 1e-2 | 1.0 | 0.168 |
| **0.0** | **1.0** | **0.120** |

### Which transform wins

In the λ=0 experiment, Transform T0 (rank-512, the highest-capacity transform) wins 100% of tokens. In the λ=1e-3 full run, Transform T5 (rank-128) won 100%. The specific winner varies, but the pattern is the same: one transform dominates all tokens.

### Root cause

The collapse is **not caused by the sparsity penalty**. It is an optimization dynamic arising from the interaction between JumpReLU hard gating and the MSE objective:

1. **JumpReLU creates a winner-take-all competition.** The forward pass zeros inactive transforms (`gate * UVx = 0` when `gate ≤ 0`). Even though the backward pass uses a full straight-through estimator that passes gradients unconditionally, the forward contribution of inactive transforms is exactly zero — so the MSE loss has no signal about what those transforms would have contributed if they were active.

2. **Positive feedback loop.** Whichever transform captures the most variance early in training receives the strongest MSE gradients (since its output is the only non-zero contribution). It improves, captures even more variance, and pushes other transforms' gating pre-activations further negative via implicit competition for the reconstruction target. Once other transforms are inactive for most tokens, they have no forward-pass contribution to generate gradient signal, and they remain dead.

3. **The full STE is necessary but not sufficient.** Without the full STE, transforms die immediately (Bug 2). With it, transforms can technically receive gradient even when gated off — but the gradient only flows through the gating pre-activation, not through the transform output (which is zeroed in the forward pass). The MSE loss can push a gate to open, but it has no information about whether the transform behind that gate would actually help reconstruction, since the transform's parameters haven't been trained on the current data distribution while inactive.

4. **The rank-512 transform has a structural advantage at N=1.** With 31 transforms total and only one at rank 512, that transform has the highest capacity. It can approximate a larger subspace of the MLP Jacobian than any other individual transform, so it tends to win the early-training competition. However, the collapse to L0=1 is not specific to rank-512 — the λ=1e-3 run collapsed to T5 (rank-128) instead, suggesting that random initialization and early gradient dynamics determine the winner.

### Consequence

The sparsity sweep does not produce a meaningful L0 vs NMSE Pareto frontier. All points cluster at L0=1.0 with NMSE varying from 0.120 to 0.375 (the NMSE variation reflects which specific transform won and its rank, not a sparsity-reconstruction tradeoff). The Jacobian faithfulness is correspondingly low (cosine_sim ≈ 0.015) because a single low-rank transform can only match a small subspace of the full MLP Jacobian.

---

## Activation/Sparsity Setup Sweep (2026-04-02)

### Experiment

Sweep 4 configurations of (sparsity penalty type × gating activation) with λ=0 to isolate the effect of the gating activation on the L0 collapse. Since λ=0, the sparsity penalty type has no effect on training — the only variable that matters is the gating activation (ReLU vs JumpReLU).

### Results

| Sparsity | Activation | λ | L0 | NMSE | Winner Transform |
|----------|------------|---|-----|------|-----------------|
| Tanh | JumpReLU | 0.0 | 1.0 | 0.120 | T0 (rank-512, 100%) |
| **Tanh** | **ReLU** | **0.0** | **0.0** | **1.001** | **None** |
| **L1** | **ReLU** | **0.0** | **0.0** | **1.001** | **None** |
| **L1** | **JumpReLU** | **0.0** | **1.0** | **0.166** | **T0 (rank-512, 99.4%)** |

(Bold = new experiments from this sweep. Row 1 is the prior baseline from `checkpoints/no_sparsity/`.)

### Key Findings

1. **ReLU gating collapses to L0=0** (all transforms die). With ReLU activation and λ=0, the model fails to learn entirely — NMSE≈1.0 means the output is approximately zero, no better than predicting the mean. This occurs regardless of sparsity penalty type (Tanh or L1), confirming that the sparsity type is irrelevant at λ=0.

2. **JumpReLU gating collapses to L0=1** (one winner). Both JumpReLU setups (Tanh+JumpReLU and L1+JumpReLU) converge to exactly L0=1.0 with the rank-512 transform T0 winning ~100% of tokens. NMSE is 0.120–0.166, indicating meaningful reconstruction through a single transform.

3. **ReLU vs JumpReLU: dead gates vs winner-take-all.** The critical difference is in gradient flow for inactive gates:
   - **ReLU** has zero gradient for x≤0. Once a gate goes negative, the gradient `d(gate)/d(pre_act) = 0`, so the gate stays dead permanently. All 31 transforms progressively die during training.
   - **JumpReLU** (with full STE) passes gradients unconditionally in the backward pass, allowing dead gates to reactivate. This prevents total collapse but still produces winner-take-all dynamics (see "L0=1 Collapse" section above).

4. **Confirmation: sparsity penalty type is orthogonal to the collapse mechanism.** The two ReLU runs (Tanh+ReLU and L1+ReLU) are identical at λ=0, and the two JumpReLU runs (Tanh+JumpReLU and L1+JumpReLU) differ only in NMSE magnitude (0.120 vs 0.166), likely due to random seed effects on which transform wins.

### Infrastructure Changes

- Added `sparsity_type: str = "tanh"` to `MOLTConfig` (options: "tanh", "l1")
- L1 penalty in `MOLT.forward()`: `mean(|gate_t|) * ||U_t V_t||_F` (replaces tanh wrapping)
- Scripts: `scripts/run_activation_sweep.py`, `scripts/run_single_setup.py`
- Cached activations: `data/activations_2M.pt` (2M tokens, 18GB)
