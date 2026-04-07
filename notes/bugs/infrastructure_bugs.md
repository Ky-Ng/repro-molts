# Infrastructure Bugs Found & Fixed

Six critical infrastructure bugs were identified and fixed during initial development. Read this before modifying initialization, gating, sparsity penalty, or activation collection code.

## Bug 1: Dead Transforms — Kaiming Init Produces Frobenius Norm Explosion

**File:** `src/molt/model.py` — `TransformGroup._init_weights()`

**Symptom:** All 31 transforms die (L0→0) within the first few thousand training steps. NMSE converges to ~1.0 (model outputs near-zero).

**Root Cause:** Kaiming init produces per-element values scaled by `sqrt(2/fan_in)`, but the Frobenius norm of U (shape `[d_model, rank]`) grows as `||U||_F ~ sqrt(d_model * rank)`. The sparsity penalty multiplies `tanh(mean|gate|)` by `||U||_F * ||V||_F`, which evaluates to ~500-1500 at init. With λ=1e-3, the initial `λ * sparsity_loss ≈ 5.4` is comparable to `MSE ≈ 6.7`, causing the sparsity gradient to dominate immediately.

**Fix:** Scaled normal init for U, V with `std = (1 / (d_model * rank))^0.25` + normalized Frobenius norms by `sqrt(d_model * rank)`.

---

## Bug 2: JumpReLU STE Kills Gradient for Inactive Gates

**File:** `src/molt/model.py` — `JumpReLU.backward()`

**Symptom:** Once a gate's pre-activation drops below threshold, it can never reactivate.

**Root Cause:** Original STE only passed gradients where `x > threshold`. Dead gates received zero gradient from both MSE loss and sparsity penalty.

**Fix:** Full STE passing gradients unconditionally; later replaced by smooth surrogate gradient `σ(x/τ) + x·σ'(x/τ)/τ`.

---

## Bug 3: Gate Bias Initialized to Zero

**File:** `src/molt/model.py` — `TransformGroup.__init__()`

**Symptom:** Only ~30-40% of gates fire at init.

**Root Cause:** `e_t · x - bias` with `bias=0` and negative-mean input distribution.

**Fix:** Bias initialized to -1.0, encoder vectors normalized to unit norm → 99% initial activation.

---

## Bug 4: Missing Sparsity Warmup

**Files:** `src/molt/train.py`, `src/molt/config.py`, `src/molt/model.py`

**Symptom:** Promising early training (NMSE→0.11) followed by late collapse (NMSE→1.05).

**Root Cause:** λ applied at full strength from step 0.

**Fix:** Linear warmup over first 10% of steps via `sparsity_warmup_frac` config field.

---

## Bug 5: `compute_nmse` Off by Factor of d_model (1152x)

**File:** `src/molt/eval.py` — `compute_nmse()`

**Symptom:** Evaluation NMSE returned 155.7 instead of ~0.135.

**Root Cause:** Extra `* target.shape[1]` multiplication converting per-element MSE to per-sample MSE, but dividing by per-element variance.

**Fix:** Removed the multiplication.

---

## Bug 6: OOM During `torch.cat` of Activation Lists

**File:** `src/molt/data.py` — `collect_activations()`

**Symptom:** Process killed after 100% of activations collected for 10M tokens.

**Root Cause:** List + concatenation peaked at ~180GB (list alive during cat allocation).

**Fix:** Pre-allocate output tensors and write in-place. Peak memory now ~86GB.

---

## Infrastructure Notes

- **Disk Space:** 50GB workspace limit means activations stay in RAM only (10M tokens at float32 = ~86GB).
- **Gemma Authentication:** `google/gemma-3-1b-it` requires `HF_TOKEN` env var.
- **Thread vs Process:** Parallel sweeps use `ThreadPoolExecutor` (shared memory) not `ProcessPoolExecutor` (/dev/shm limited to 31GB).
