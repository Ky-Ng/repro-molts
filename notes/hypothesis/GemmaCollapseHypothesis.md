# Gemma-3-1B Gate Collapse Hypotheses

**Date:** April 7, 2026
**Context:** Experiment 09 confirmed that ~99% of gates (30.73/31) are open at initialization on real Gemma activations. Yet by training step 100, L0 collapses to ~0.375. This document catalogs hypotheses for why collapse happens during training, not at init.

**Background — Gemma decoder layer flow:**
```
residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)   # RMSNorm
hidden_states = self.mlp(hidden_states)                     # hooks capture input/output here
hidden_states = post_feedforward_layernorm(hidden_states)   # RMSNorm (NOT in our target)
hidden_states = residual + hidden_states
```

Our MOLT is trained to reconstruct `mlp(RMSNorm(x))` — the post-pre-norm input and the raw MLP output (before post-feedforward RMSNorm and before residual addition).

---

## H1: Winner-take-all gradient competition

All 31 transforms start with random UV matrices producing near-zero output. A few transforms happen to align slightly better with the target early on. Backprop reinforces these — their gates get pushed open harder while losing transforms get weaker gradients. JumpReLU's hard threshold makes this a cliff: once a gate's pre-activation dips below 0, it gets zero forward signal and (even with the smooth surrogate) much-reduced backward signal. Within ~100 steps, one or two "winners" capture most of the reconstruction task and the rest die.

**Testable prediction:** If we freeze gates open (bypass JumpReLU, force all gates to 1.0) during early training, transforms should all learn useful directions and then survive when gating is re-enabled.

---

## H2: RMSNorm-compressed input produces homogeneous gate pre-activations

Because `pre_feedforward_layernorm` constrains all inputs to similar norms (~15 with std ~1.2), the gate pre-activations `e·x + 1` have very low variance (sigma ~0.44). This means all tokens push a given transform's gate in nearly the same direction — there's no per-token diversity to keep different transforms alive for different tokens. A transform either fires on everything or nothing, and the optimizer collapses to one winner.

Contrast with GPT-2 where LayerNorm (which subtracts the mean) may produce more directional variance even at normalized scale.

**Testable prediction:** Gate pre-activation variance on Gemma should be significantly lower than on GPT-2. Alternatively, feeding raw (pre-RMSNorm) residual stream activations to MOLT should show less collapse because the input has more norm variance.

---

## H3: Scale mismatch between MLP output and initial MOLT output

The Gemma MLP is a SwiGLU network (gate_proj, up_proj, down_proj) with a large intermediate size. Its output magnitudes may be significantly larger than what a freshly initialized MOLT produces. If the initial MOLT output is tiny relative to the target, the MSE gradient overwhelmingly favors whichever one or two transforms happen to align best, starving the rest.

On GPT-2 (d=768, simpler 2-layer MLP with GELU), the scale mismatch may be less severe, explaining why collapse doesn't happen there.

**Testable prediction:** Measure `||mlp_output|| / ||molt_output_at_init||` ratio for both Gemma and GPT-2. If Gemma's ratio is much larger, that supports this hypothesis.

---

## H4: Raw MLP output (our target) may be pathologically structured

Our reconstruction target is the raw MLP output before `post_feedforward_layernorm`. If this signal has high variance, heavy tails, or structure that's hard to decompose into a sparse mixture of low-rank transforms, the model may default to approximating it with a single dominant transform rather than distributing work across many.

**Testable prediction:** Compare the singular value spectrum of batched MLP outputs on Gemma vs GPT-2. A more concentrated spectrum (few dominant directions) on Gemma would support this.

---

## H5: Dimensionality makes transforms slow to specialize, but gating kills them fast

Gemma has d_model=1152 vs GPT-2's d_model=768 — a 2.25x increase in the dimensionality that each transform must operate in. This has compounding effects:

**The core tension:** Each transform's encoder `e` (a unit vector in R^d) needs to find a meaningful direction in the input space — one that selects for tokens where its low-rank UV matrix is useful. Simultaneously, UV needs to learn a useful rank-r approximation of the MLP's behavior for those tokens. Both problems are harder in higher dimensions:

1. **Encoder search space:** The encoder lives on the unit sphere S^{d-1}. In 1152 dimensions, two random unit vectors have expected cosine similarity ~0 with std ~1/sqrt(d) ~0.03. The encoder must navigate this vast space to find a discriminative direction, but early gradients are near-random because UV hasn't yet learned anything useful. In d=768, the search space is 1.5x smaller in each dimension (but exponentially smaller in volume).

2. **UV learning speed:** Each transform's V (rank x d) and U (d x rank) together have `2 * d * rank` parameters. At rank 128 on Gemma that's 2 * 1152 * 128 = 295K parameters vs 2 * 768 * 128 = 197K on GPT-2. More parameters to align before the transform produces a useful output.

3. **Random projection concentration:** In high dimensions, random projections concentrate — `e·x` for random unit `e` has variance `||x||^2/d`, which shrinks per-dimension. This means the initial "signal" each transform gets from the input is more uniform across tokens. The encoder needs larger updates to break out of this isotropy and find a token-selective direction, but the gating dynamics may kill it before that happens.

4. **Critical race condition:** There's a race between (a) transforms learning useful directions (slow, scales with d) and (b) gating dynamics selecting winners and killing losers (fast, driven by relative magnitudes of a few gate pre-activations). In higher dimensions, (a) takes longer while (b) operates at the same speed — the gate has the same JumpReLU threshold regardless of d. GPT-2 at d=768 may sit just below the critical threshold where transforms specialize fast enough to survive.

**Testable prediction:** Training the same MOLT architecture on Gemma with a lower effective dimensionality (e.g., PCA-projecting activations to d=768) should reduce or eliminate collapse. Alternatively, increasing GPT-2's effective dimensionality (padding with noise) should induce collapse.

---

## Priority for testing

| Hypothesis | Effort | Discriminating power |
|-----------|--------|---------------------|
| H3 (scale mismatch) | Low — just measure norms | High if ratio differs drastically between models |
| H2 (RMSNorm homogeneity) | Low — compare pre-act variance | Moderate — explains token-level uniformity |
| H5 (dimensionality race) | Medium — PCA experiment | High — directly tests the mechanism |
| H1 (winner-take-all) | Medium — gate freezing experiment | High — but mechanism, not root cause |
| H4 (target structure) | Medium — SVD analysis | Moderate — descriptive rather than causal |

H3 and H2 are cheapest to test and would narrow the field quickly before committing to the more involved H5 experiment.
