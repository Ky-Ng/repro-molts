# Experiment 11: Transcoder vs MOLT Comparison

## Goal

Do MOLTs Pareto-dominate transcoders at equal compute, as claimed in the Anthropic paper? Compare L0 vs NMSE and L0 vs Jacobian correlation for transcoders and MOLTs trained on GPT-2, matching both parameter count and training tokens at each compute scale.

### Background

From the Anthropic paper (trained on Claude 3.5 Haiku):

> We trained MOLTs and transcoders on the middle layer, varying the amount of compute. We scaled the number of training steps proportionally to the number of features, and matched the number of parameters between transcoder and MOLT runs. Thus each 4x increase in FLOPs reflects a 2x increase in both number of parameters and training steps. The smallest MOLT runs Pareto-dominate transcoder runs that use 1024x as many FLOPs.

We reproduce this comparison on GPT-2 layer 6 (d=768), where MOLTs are known to train successfully (exp 04-05), avoiding the Gemma L0-collapse issue.

## Setup

- **Model:** GPT-2, layer 6 (d_model=768)
- **Data:** FineWeb activations (streamed)
- **Metrics:** L0, NMSE, Jacobian cosine similarity
- **Sparsity:** tanh penalty with smooth surrogate JumpReLU (best config from exp 04-05)

### Parameter Matching

MOLT parameters per transform: `2 * d_model * rank + d_model + 1` (U, V matrices + encoder + bias).

Transcoder parameters: `2 * d_model * n_features + n_features + d_model` (W_enc, W_dec + biases).

| Scale | MOLT Config | MOLT Params | Transcoder Features | Transcoder Params |
|-------|-------------|-------------|---------------------|-------------------|
| 1x    | N=1 (31 transforms) | ~3.96M | 2,576 | ~3.96M |
| 2x    | N=2 (62 transforms) | ~7.92M | 5,152 | ~7.92M |
| 4x    | N=4 (124 transforms) | ~15.8M | 10,304 | ~15.8M |
| 8x    | N=8 (248 transforms) | ~31.7M | 20,608 | ~31.7M |

### Compute Scaling

Following the paper, each 4x FLOPs increase = 2x parameters x 2x training steps:

| FLOPs Scale | Param Scale | Tokens | Steps (batch=64, seq=256) |
|-------------|-------------|--------|---------------------------|
| 1x          | N=1         | 2M     | ~31K |
| 4x          | N=2         | 4M     | ~62K |
| 16x         | N=4         | 8M     | ~125K |
| 64x         | N=8         | 16M    | ~250K |

This gives 4 MOLT runs x 4 transcoder runs = 8 total training runs. Each run sweeps multiple sparsity coefficients to trace a Pareto curve.

### Sparsity Sweep

Each training run is repeated across multiple lambda values to produce L0-vs-NMSE curves:

| Lambda | Expected Effect |
|--------|-----------------|
| 0      | Baseline (no sparsity pressure) |
| 1e-5   | Mild sparsity |
| 1e-4   | Moderate sparsity |
| 1e-3   | Strong sparsity |

Total runs: 4 scales x 4 lambdas x 2 methods = 32 training runs.

## Implementation Notes

### Transcoder Training (New)

No transcoder training code exists yet. Need to implement a simple transcoder:

```
class Transcoder(nn.Module):
    # Encode: h = ReLU(W_enc @ x + b_enc)    shape: (d_model,) -> (n_features,)
    # Decode: y = W_dec @ h + b_dec           shape: (n_features,) -> (d_model,)
```

- Use TopK or JumpReLU activation to control L0 directly
- Train with same MSE loss as MOLT (reconstruct MLP output from MLP input)
- Same optimizer (Adam), same learning rate schedule
- Share activation collection code from `molt.data`

### Fair Comparison Considerations

1. **Same data:** Both methods see identical activation batches
2. **Same optimizer:** Adam with identical LR and schedule
3. **Parameter matching:** Transcoder feature count chosen to match MOLT param count at each scale
4. **Token matching:** Both methods train for the same number of tokens at each scale
5. **L0 control:** Sweep sparsity coefficient for both; compare at matched L0 values via Pareto frontier

## Reproduction

```bash
# Run all comparisons
uv run python experiments/11_transcoder_comparison/run.py

# Run a single scale
uv run python experiments/11_transcoder_comparison/run.py --scale 1x

# Run only MOLTs or only transcoders
uv run python experiments/11_transcoder_comparison/run.py --method molt
uv run python experiments/11_transcoder_comparison/run.py --method transcoder
```

## Expected Results

| Setup | L0 Range | NMSE Range | Jacobian Corr |
|-------|----------|------------|---------------|
| MOLT 1x | ? | ? | ? |
| MOLT 2x | ? | ? | ? |
| MOLT 4x | ? | ? | ? |
| MOLT 8x | ? | ? | ? |
| Transcoder 1x | ? | ? | ? |
| Transcoder 2x | ? | ? | ? |
| Transcoder 4x | ? | ? | ? |
| Transcoder 8x | ? | ? | ? |

## Expected Figures

1. **L0 vs NMSE Pareto plot** — All MOLT and transcoder runs overlaid, colored by method and sized by compute scale. Key question: does the smallest MOLT curve dominate the largest transcoder curve?
2. **L0 vs Jacobian cosine similarity** — Same layout. Tests whether MOLTs are more faithful to the true MLP Jacobian at matched L0.
3. **Compute scaling curves** — NMSE at fixed L0 (e.g., L0=5, L0=10) as a function of FLOPs, separate lines for MOLT vs transcoder. Tests whether transcoders saturate while MOLTs continue improving.

## Analysis

*Fill after running.*

Key questions:
- Do MOLTs Pareto-dominate transcoders at equal compute on GPT-2?
- At what compute multiplier (if any) do transcoders match MOLT performance?
- Do transcoders show the saturation effect observed in the Anthropic paper?
- Is the Jacobian faithfulness advantage consistent with the NMSE advantage?

## Artifacts

- Results: `experiments/11_transcoder_comparison/results/`
- Figures: `experiments/11_transcoder_comparison/figures/`
