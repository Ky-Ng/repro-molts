# repro-molts
Reproducing Anthropic's Sparse Mixtures of Linear Transforms (MOLTs)

## Motivation
1. Mixture of Linear Transforms (MOLTs) introduced in [Sparse Mixtures of Linear Transforms](https://transformer-circuits.pub/2025/bulk-update/index.html) are possibly
	1. More faithful
	2. More Interpretable
	3. More Compute efficient
2. Goal: Reproduce the MOLTs paper and run further evaluations on MOLTs as follow ups to Anthropic's work

## Goal
1. Qualitatively evaluate MOLT transforms by visualizing MOLT transforms interactions with SAE
2. Qualitatively compare MOLTs to an existing strategy such as Transcoders
3. Replicate Quantitative analysis of (1) Faithfulness via the Jacobian and (2) L0 vs. Normalized MSE

### Out of Scope
1. Pseudofeature Decomposition of SAE Error
2. MOLT transform "steering" (this is an unclear definition)
3. Reproduction of known transforms (requires more than a single layer MOLT)
	1. See if any of the Addition or translation MOLT transforms are detectable
4. Compute Attention OV feature interactions across layers

## Details
Proposed Pretrained SAE / Transcoders
1. Use Gemma Scope 2 on Gemma3-1B
	1. JumpReLU SAE
	2. Transcoders (Skip/Non-skip) 
2. Use [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) text
3. Train MOLT for only a single layer (e.g. in Gemma3-1B mid-layer layer [26](https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json)/2 = 13)
## Phases
1. Training Proof of Concept (Infrastructure)
	1. Train MOLTs on a [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) Subset with Gemma Scope SAE features
		1. Use Tanh sparsity penalty
		2. Use `N transforms of rank 512, 2N of rank 256, 4N of rank 128, 8N of rank 64, and 16N of rank 32`
		3. Start with N=1 and sweep lambda sparsities
			1. Note if results are promising move to N=2, N=4
	2. Evaluation
		1. Evaluate reconstruction loss
		2. Evaluate MOLT faithfulness via cosine similarity of flattened Jacobian matrices
	3. Single Layer Interpretability
		1. Use `EleutherAI/delphi` to label the contexts when a Transform is active
		2. Note: we do not interpret the function of transforms for single layers as detailed in the paper; we leave Transform function interpretability for multi-layer attribution graphs because of "[interference weights](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#global-weights)"
	4. Comparisons to Baselines (Transcoders)
		1. Qualitatively Compare MOLTs+SAEs to Transcoders (both Skip/Non-Skip)
		2. Compare Jacobian faithfulness
		3. Calculate L0 vs. MSE for Transcoders vs. MOLTs+SAE
2. Attribution Graphs
	1. Extend MOLT Training to multiple layers (starting first with mid-layer onwards, if this phase goes well, extend to all layers)
	2. Compute SAE feature interactions across layers
	3. Label transform functions using an open source model (use existing Delphi input/output features but unclear if Delphi can label transforms)

