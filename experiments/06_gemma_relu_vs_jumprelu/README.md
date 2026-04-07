# Gemma-3-1B: ReLU vs JumpReLU (Smooth Surrogate)

## Experiment

Test whether the smooth surrogate JumpReLU backward (which resolved the collapse on GPT-2) also helps on Gemma-3-1B.

**Setup:**
- Model: Gemma-3-1B layer 13, d_model=1152
- Data: 2M FineWeb tokens (cached at `data/activations_2M.pt`)
- Architecture: N=1, 31 transforms
- Training: batch_size=64, lr=1e-3, 1 epoch (~31K steps), λ=0
- JumpReLU backward: smooth surrogate `σ(x/τ) + x·σ'(x/τ)/τ` with τ=0.1

## Results

| Activation | Learned θ | L0 | NMSE | θ_final |
|------------|-----------|-----|------|---------|
| ReLU | No | 0.0 | 1.001 | fixed |
| JumpReLU (smooth surrogate) | Yes | 0.0 | 1.001 | +0.953 |

**Both configurations collapse completely on Gemma-3-1B.** Neither produces any active transforms.

## JumpReLU Threshold Trajectory

```
Step       L0    NMSE    θ
   100    0.33  1.198   +0.038    ← some gates active initially
  5100    0.02  1.002   +0.335    ← θ rises, gates dying
 10100    0.00  1.001   +0.595    ← all gates dead
 15100    0.00  1.001   +0.670    ← θ keeps drifting up
 20100    0.00  1.001   +0.734
 25100    0.00  1.001   +0.879
 31200    0.00  1.001   +0.953    ← final: θ still rising slowly
```

The threshold runs away to +0.95, identical behavior to the full-STE learned θ experiment (which converged to +0.87). The smooth surrogate did not prevent the collapse on Gemma.

## Why the Smooth Surrogate Fails on Gemma but Works on GPT-2

The smooth surrogate's gradient for near-threshold gates is `≈ 0.5`, and for deeply-off gates it decays to `≈ 0`. This works when there are enough gates *near* the threshold to sustain gradient signal — which is the case on GPT-2, where the activation distribution keeps many gate pre-activations in the transition zone.

On Gemma-3-1B, the problem is different:

1. **The gate pre-activation distribution is narrower.** Gemma's MLP inputs at layer 13 have mean≈-0.008 and std≈0.45. The encoder dot products `e_t · x` cluster in a narrow band. With bias=-1.0, initial pre-activations (`e·x + 1`) are around 0.5–1.5 for most tokens, but they quickly converge during training.

2. **Transforms fail to learn useful decompositions early enough.** On GPT-2 (d=768), the rank-512 transform covers 67% of the hidden dimension — it can approximate the MLP reasonably well even with random-ish weights. On Gemma (d=1152), the same rank-512 transform covers only 44%. The initial reconstruction is worse, so the MSE gradient signal for keeping gates open is weaker.

3. **The threshold runaway is faster.** Because transforms contribute more noise than signal in early training (NMSE > 1.0 at step 100), raising θ *genuinely reduces MSE* by silencing bad outputs. By step 5100, θ has already reached 0.33 and only 2% of gates survive. Once gates are off, the transforms stop receiving gradient through their weights (even with the surrogate, the gradient through `gate * UVx` is `≈ 0` because `gate ≈ 0`), so they can't improve. This creates an irreversible death spiral.

4. **The smooth surrogate helps near-threshold gates, but not deeply-off ones.** The key difference from the full STE is that deeply-off gates get `≈ 0` gradient instead of full gradient. On GPT-2 this was beneficial (removed phantom signals). On Gemma, it means once a gate drifts past the threshold, there's almost no gradient to pull it back — the surrogate gradient decays exponentially with distance from threshold.

## Comparison Across Models

| Model | ReLU | JumpReLU (full STE, fixed θ) | JumpReLU (smooth surr., learned θ) |
|-------|------|------|------|
| **GPT-2** (d=768) | L0=18.8, NMSE=0.49 | L0=1.6, NMSE=0.74 | **L0=14.5, NMSE=0.47** |
| **Gemma-3-1B** (d=1152) | L0=0.0, NMSE=1.00 | L0=1.0, NMSE=0.12 | L0=0.0, NMSE=1.00 |

The smooth surrogate is a clear win on GPT-2 but has no effect on Gemma. The fixed-θ full-STE JumpReLU remains the only configuration that produces any learning on Gemma (L0=1, NMSE=0.12), despite its collapse to a single transform.

## Implications

The Gemma collapse appears to be a **model-scale problem**, not just a gating mechanism problem:
- GPT-2's lower dimensionality (768) means individual transforms have higher relative capacity
- Gemma's higher dimensionality (1152) means transforms need more training steps to become useful, but the gating dynamics don't give them enough time
- Possible remedies specific to larger models: higher initial learning rate for transform weights, frozen θ with periodic unfreezing, or auxiliary reconstruction losses that keep transforms training even when gated off

## Artifacts

- Plot: `results/gemma_relu_vs_jumprelu/relu_vs_jumprelu_gemma.png`
- Results: `results/gemma_relu_vs_jumprelu/result_*.json`
- Histories: `results/gemma_relu_vs_jumprelu/history_*.json`
- Script: `scripts/run_gemma_relu_vs_jumprelu.py`
