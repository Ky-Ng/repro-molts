# Differences between `repro-molts` and `crosslayer-transcoder`

Comparison of the experimental reproduction (`repro-molts`) with the working reference
implementation (`crosslayer-transcoder`), focused on the MOLT model and training path.

> **Status (post-port):** Bugs A–F and architectural items D1–D6, D8 have been
> fixed by porting `crosslayer-transcoder`'s Molt model, JumpReLU, and
> standardizers directly into `src/molt/`. See the
> ["Port Summary"](#port-summary) section at the bottom for the full list of
> files that changed and what was carried over. All 38 tests pass. Items D7,
> D9–D12 remain unchanged (framework / data-pipeline / LR-schedule choices).

- **repro-molts MOLT model:** [src/molt/model.py](src/molt/model.py)
- **crosslayer-transcoder MOLT model:** `/workspace/crosslayer-transcoder/model/molt.py`
- **crosslayer-transcoder training module:** `/workspace/crosslayer-transcoder/model/clt_lightning.py` (`MoltModule`)
- **crosslayer-transcoder standardizers:** `/workspace/crosslayer-transcoder/model/standardize.py`
- **crosslayer-transcoder JumpReLU:** `/workspace/crosslayer-transcoder/model/jumprelu.py`

---

## Likely Bugs in `repro-molts`

### Bug A — JumpReLU backward surrogate ignores the threshold
[src/molt/model.py:38-45](src/molt/model.py#L38-L45)

```python
@staticmethod
def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    tau = JumpReLU.SURROGATE_TAU
    sig = torch.sigmoid(x / tau)                       # <-- uses x, not (x - threshold)
    surrogate_grad = sig + x * sig * (1 - sig) / tau
    return grad_output * surrogate_grad, None
```

Forward is `x * 1[x > threshold]`. A faithful smooth surrogate of this is
`x * σ((x - θ)/τ)`. The backward uses `σ(x/τ)` instead, which is only correct when
`θ == 0`. Any non-zero `jumprelu_threshold` (or later migration to a learnable
threshold that happens to be non-zero) yields a gradient that does not match the
forward — gates centered around `x ≈ θ` will receive gradient as if `θ = 0`.

Note: the `LearnedJumpReLU` variant at [src/molt/model.py:70-83](src/molt/model.py#L70-L83)
does this correctly (`sigmoid((x - threshold) / tau)`). The fixed-threshold path
is the buggy one.

**Fix:** also save `threshold` in `ctx` (or just accept it as a float) and use
`sigmoid((x - threshold) / tau)` in the backward.

---

### Bug B — Frobenius norm of `UV` approximated by the product of Frobenius norms
[src/molt/model.py:172-176](src/molt/model.py#L172-L176)

```python
u_norms = self.U.flatten(1).norm(dim=1)       # ||U_t||_F
v_norms = self.V.flatten(1).norm(dim=1)       # ||V_t||_F
frobenius_norms = u_norms * v_norms / (self.d_model * self.rank) ** 0.5
```

`||UV||_F ≠ ||U||_F · ||V||_F` in general — the product is only an upper bound
(via Cauchy-Schwarz). The `crosslayer-transcoder` reference computes the correct
quantity in `Molt.transform_norm`:

```python
uv = einops.einsum(U, V, "n r do, n di r -> n di do")
norms.append(torch.norm(uv, dim=(1, 2)))        # actual ||UV||_F
```

Because this quantity multiplies the gate in the sparsity penalty (the idea being
"penalize transforms in proportion to how much they can move the reconstruction"),
the approximation mis-attributes penalty between transforms that align their U/V
directions versus those that do not. Normalizing by `sqrt(d_model * rank)` makes
the scale roughly O(1) at initialization but does not fix the per-transform
direction mismatch.

**Fix:** compute `||UV||_F` exactly via the einsum shown above (same as
`crosslayer-transcoder`).

---

### Bug C — L0 count is inconsistent with gate output when threshold is negative
[src/molt/model.py:248-250](src/molt/model.py#L248-L250) and [:35](src/molt/model.py#L35)

The forward gate is `x * 1[x > threshold]`; L0 is counted as `(gate > 0)`.
If a learned threshold drifts negative (no clamping — see
[src/molt/model.py:200-202](src/molt/model.py#L200-L202)), negative `x` values pass
the `x > threshold` check and the gate becomes a **negative** real number. Those
transforms contribute to the reconstruction (with negated sign) and to the
sparsity penalty via `gate.abs()`, but are not counted in `total_active`:

```python
active = (gate > 0).float().sum(dim=1).mean()   # misses gate < 0 contributions
```

The `crosslayer-transcoder` JumpReLU forward forbids this case:
```python
feature_mask = torch.logical_and(input > theta, input > 0.0)   # extra input > 0 check
```
so gate is always non-negative and L0 is always well-defined.

**Fix:** either add the `input > 0` guard to the repro's JumpReLU forward, or
change the L0 count to `(gate != 0)`. The former matches the reference.

---

### Bug D — Sparsity penalty applies `tanh` after batch-averaging
[src/molt/model.py:236-245](src/molt/model.py#L236-L245)

```python
mean_abs_gate = gate.abs().mean(dim=0)            # mean over batch first
sparsity = (torch.tanh(mean_abs_gate) * frob_norms).sum()
```

The working version applies `tanh` per-token, then averages
([model/clt_lightning.py:557-563 in crosslayer-transcoder]):

```python
weighted_norms = norms * gate                                # (batch, n_transforms)
weighted_norms = torch.tanh(weighted_norms * self.c)         # per-token non-linearity
sparsity = lambda_cur * weighted_norms.sum(dim=-1).mean()    # then mean over batch
```

These are materially different non-linearities — `tanh(mean(·))` compresses far
less aggressively than `mean(tanh(·))`. With a "winner-takes-all" gate
distribution (which the README explicitly calls out as the failure mode on
Gemma), the batch mean is small and `tanh` stays in its linear regime, so the
penalty barely distinguishes "one token firing hard" from "every token firing
gently." The reference version penalizes each token's contribution individually.

The repro also drops the `c_sparsity` multiplier inside `tanh` (see Bug E).

**Fix:** compute `tanh((gate * frob_norms) * c).sum(dim=-1).mean()` to match.

---

### Bug E — Missing `c_sparsity` hyperparameter
[src/molt/model.py:245](src/molt/model.py#L245)

The reference has a dedicated `c_sparsity` (100 in the shipped config) that
multiplies the argument to `tanh`:

```python
weighted_norms = torch.tanh(weighted_norms * self.c)   # c=100 in config
```

This controls how far into the saturated regime of `tanh` the penalty sits,
independently of `lambda_sparsity` (which scales after the `tanh`). `repro-molts`
has only `sparsity_coeff` which multiplies after the `tanh` — it cannot recover
the same loss landscape because it has one degree of freedom where the reference
has two.

**Fix:** introduce a `c_sparsity`-like scalar and multiply it into the `tanh`
argument.

---

### Bug F — No input/output standardization
[src/molt/model.py:181-298](src/molt/model.py#L181-L298) (full `MOLT` class has no standardizer).

`crosslayer-transcoder` wraps the forward pass with standardizers:

```python
def forward(self, acts, layer):
    acts = self.input_standardizer(acts, layer)           # (batch) - zero-center, unit-std per dim
    pre_actvs = self.e(acts)
    ...
    recons = self.output_standardizer(recons_norm, layer) # invert std of MLP-out
```

The default `DimensionwiseInputStandardizer` / `DimensionwiseOutputStandardizer`
are initialized from the first batch and store per-dim mean/std as buffers.
This matters because (a) the JumpReLU threshold, the `-1.0` bias init, and the
unit-norm encoder all assume inputs with O(1) norm per dim; raw residual-stream
activations can have very anisotropic per-dim scales, especially on Gemma-3-1B.
(b) MSE is computed in the standardized output space in the reference, making it
dimension-invariant; in `repro-molts` MSE is on raw activations, so `nmse = MSE /
Var(target)` is scale-corrected but loss weighting between MSE and sparsity is
not.

**Fix:** add input/output standardizers (or at least normalize activations once
up-front before training).

---

## Architectural Differences (not bugs per se, but may matter)

### D1 — Encoder structure
- **crosslayer**: single `nn.Linear(d_acts, n_features)` shared across all rank groups
  ([model/molt.py:35]). One matmul for gating.
- **repro-molts**: each `TransformGroup` owns its own `encoder` parameter
  `(num_transforms, d_model)` and `bias` ([src/molt/model.py:108-113](src/molt/model.py#L108-L113)).
  Separate matmul per group.

Mathematically equivalent (concatenation of per-group encoders == one big encoder),
but it affects init and means the two implementations will not load each other's
checkpoints.

### D2 — Bias sign convention
- **crosslayer**: `pre = W·x + b` (standard `nn.Linear`).
- **repro-molts**: `pre = W·x - b`, bias init at `-1.0` so `-b = +1.0`
  ([src/molt/model.py:128,151](src/molt/model.py#L128)).

Both can learn the same function; just different sign conventions. This matters
only for cross-loading weights.

### D3 — Weight initialization
- **crosslayer**: `xavier_uniform_` on both `U` and `V`; default `nn.Linear` init
  for encoder.
- **repro-molts**:
  - `U`, `V` ~ `N(0, (d·r)^{-1/4})` so that `||UVx||` is O(1)
  - encoder rows normalized to unit L2
  - bias constant at `-1.0`

The repro's init was introduced specifically to fix Bugs 1 and 3 listed in
`CLAUDE.md`. It is load-bearing for JumpReLU + fixed-threshold training.

### D4 — Shape conventions for `U` and `V`
- **crosslayer**: `U: (n, rank, d_acts)`, `V: (n, d_acts, rank)`; einsum contracts
  `d_acts` first, then `rank`.
- **repro-molts**: `U: (n, d_model, rank)`, `V: (n, rank, d_model)`.

Semantically the same low-rank transform; but again prevents checkpoint interop.

### D5 — JumpReLU threshold: scalar vs per-feature
- **crosslayer** `JumpReLU.theta` is a learnable parameter of shape
  `(1, n_layers, d_features)` — one threshold **per feature per layer**.
- **repro-molts**: at most a single **shared scalar** threshold
  (`self.threshold = nn.Parameter(torch.tensor(...))`,
  [src/molt/model.py:200-202](src/molt/model.py#L200-L202)).

A shared scalar cannot specialize to different features' activation magnitudes.
This likely contributes to the L0 collapse described in `notes/bugs/` — a single
threshold that works for one transform starves others.

### D6 — JumpReLU backward style
- **crosslayer**: rectangle-kernel STE for `θ`; straight-through for input with
  `grad_input[input < 0] = 0` (ReLU-like mask on input grad).
- **repro-molts**: smooth sigmoid surrogate `f(x) = x·σ(x/τ)` with τ=0.1.

Both are valid choices. The repro's smooth surrogate is documented as the fix
for Bug 2 in `CLAUDE.md`. Different bandwidth conventions (`bandwidth=1.0` vs
`tau=0.1`) mean gradients have different effective widths.

### D7 — Sparsity applied per-layer / per-batch-element
- **crosslayer** `MoltModule` runs on a single hard-coded layer (`layer = 8` in
  [model/clt_lightning.py:543]) and uses the layer-aware standardizer.
- **repro-molts** model itself has no notion of layers; layer selection happens
  in the data pipeline.

Not a bug, but the repro cannot do multi-layer MOLT as currently structured.

### D8 — Loss placement
- **crosslayer**: model returns `(gate, recons_norm, recons)`; loss and sparsity
  penalty are built in the Lightning module.
- **repro-molts**: `MOLT.forward` returns `(output, aux)` where `aux`
  already includes `sparsity_loss` and `l0`; `MOLT.loss()` combines everything.

Just a refactoring difference. The repro's tighter coupling makes it harder to
swap the sparsity formulation without editing the model.

### D9 — Dead-feature / dead-transform tracking
- **crosslayer** registers `last_active` buffer and logs dead features each step
  ([model/clt_lightning.py:236-242]); has a dedicated `DeadFeatures` metric.
- **repro-molts**: no dead-feature bookkeeping. Given that the "L0 collapse" note
  is a central finding of the repro, the absence of a direct dead-count metric
  during training makes the collapse harder to observe/diagnose.

### D10 — Framework / training loop
- **crosslayer**: PyTorch Lightning + wandb + LR warmup scheduler + AdamW option +
  `torch.compile` support + mixed precision.
- **repro-molts**: plain PyTorch + tqdm + Adam only + no compile + fp32.

### D11 — Data pipeline
- **crosslayer**: shared-memory multi-process streaming generator
  (`data/shared_memory.py`, `data/generation_loop.py`) producing activations on the fly.
- **repro-molts**: single-pass FineWeb streaming → full in-memory tensors
  ([src/molt/data.py](src/molt/data.py)). Limits total tokens to what fits in RAM,
  but simpler.

### D12 — `sparsity_warmup_frac` vs `max_steps`-based ramp
- **crosslayer** ramps `lambda` linearly from 0 over the entire `max_steps`:
  `cur_lambda = self._lambda * (current_step / n_steps)`.
- **repro-molts** ramps over `sparsity_warmup_frac * total_steps` (default 10%):
  `sparsity_scale = min(1.0, step / warmup_steps)`.

So the reference uses a *permanent* ramp while the repro holds at full strength
after 10%. The README's "Bug 4" says the 10% warmup was chosen as a fix, but note
the reference actually never reaches a steady `lambda` in normal training.

---

## Summary

Six items are likely bugs that affect training dynamics in `repro-molts`:

| # | Bug | Impact |
|---|-----|--------|
| A | JumpReLU backward ignores threshold | wrong gradients when `θ ≠ 0` |
| B | `||UV||_F` approximated by `||U||_F · ||V||_F` | mis-weighted sparsity per transform |
| C | L0 miscount when gate < 0 | under-reported L0, silently broken metric with learned θ < 0 |
| D | `tanh(mean(·))` vs `mean(tanh(·))` | sparsity penalty non-linearity collapses for WTA gates |
| E | No `c_sparsity` multiplier inside `tanh` | one too few degrees of freedom in the penalty |
| F | No input/output standardization | raw activations break init assumptions, especially on Gemma |

The remaining items in the "Architectural Differences" section are design
decisions that would need to be aligned if the goal is parity with the working
`crosslayer-transcoder` implementation — in particular, D5 (per-feature learnable
thresholds) and D9 (dead-feature tracking) look most relevant to the "L0
collapse" failure mode described in the repro's own notes.

---

## Port Summary

The model, nonlinearity, and standardizers were replaced with ports of
`crosslayer-transcoder`:

| New file | Source | Purpose |
|---|---|---|
| [src/molt/model.py](src/molt/model.py) | `crosslayer-transcoder/model/molt.py` + the sparsity loss from `clt_lightning.py::MoltModule.training_step` | Shared-encoder `MOLT` class, true `||UV||_F`, `tanh(weighted * c).sum(-1).mean()` sparsity |
| [src/molt/jumprelu.py](src/molt/jumprelu.py) | `crosslayer-transcoder/model/jumprelu.py` | Per-feature learnable threshold with rectangle-kernel STE and `input > 0 AND input > θ` forward |
| [src/molt/standardize.py](src/molt/standardize.py) | `crosslayer-transcoder/model/standardize.py` | `DimensionwiseInput/OutputStandardizer` (with `_initialized` as a buffer so they survive state_dict roundtrips) |

Supporting edits to integrate the port:

- [src/molt/config.py](src/molt/config.py) — added `c_sparsity` (default 100.0),
  `use_tanh`, `jumprelu_bandwidth`; changed `sparsity_coeff` default to `5e-4` and
  `learned_threshold` default to `True` to match `config/molt.yaml` from the reference.
- [src/molt/train.py](src/molt/train.py) — calls `model.initialize_standardizers(...)`
  on the first batch; threshold param lookup now uses `model.nonlinearity.theta`.
- [src/molt/eval.py](src/molt/eval.py) — adapted to the new `(gate, recons_norm,
  recons)` forward signature; `compute_nmse` uses `recons` (un-standardized) against
  the raw target; `compute_l0` counts `gate > 0`.
- [src/molt/__init__.py](src/molt/__init__.py) — re-exports `JumpReLU` and the
  standardizer classes.
- [tests/test_model.py](tests/test_model.py), [tests/test_eval.py](tests/test_eval.py),
  [tests/test_train.py](tests/test_train.py) — updated for the new forward signature
  and `initialize_standardizers` requirement.

### What was NOT ported

- The PyTorch Lightning harness (`clt_lightning.py`) and the wandb logging inside
  `MoltModule.training_step` — `repro-molts` keeps its plain-PyTorch training loop.
- The shared-memory activation generator (`data/` directory in the reference) —
  `repro-molts` keeps its in-RAM FineWeb streaming pipeline.
- Replacement-model / dead-feature metrics — still not implemented in `repro-molts`.
- The full-training sparsity ramp (`cur_lambda = lam * step / max_steps`) — the
  repro keeps the 10% warmup-then-hold schedule via `sparsity_warmup_frac`. Set
  `sparsity_warmup_frac = 1.0` in `MOLTConfig` to match the reference's ramp.

### How the fixes map to the bug list

| Bug | Fix |
|---|---|
| A. JumpReLU backward ignores threshold | Replaced with `jumprelu.py` (rectangle STE on `(x - θ)/bandwidth`, straight-through on positive `x`). |
| B. `||UV||_F` ≈ `||U||_F · ||V||_F` | `MOLT.transform_norm` computes `einsum(U, V)` then `torch.norm(·, dim=(1,2))`. |
| C. L0 miscount when `gate < 0` | JumpReLU forward enforces `input > 0 AND input > θ` so `gate ≥ 0` always. |
| D. `tanh(mean(·))` vs `mean(tanh(·))` | `MOLT.loss` now does `tanh(gate * norms * c).sum(-1).mean()` (per-token `tanh`, then batch mean). |
| E. Missing `c_sparsity` | Added `c_sparsity` config field, multiplied inside `tanh`. |
| F. No standardization | `MOLT` now owns `input_standardizer` and `output_standardizer` and applies them in `forward`; `initialize_standardizers(x, target)` is called on the first batch in `train_molt`. |
