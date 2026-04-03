from dataclasses import dataclass, field


# Presets for known model families
MODEL_PRESETS: dict[str, dict] = {
    "google/gemma-3-1b-it": {
        "d_model": 1152,
        "layer_idx": 13,  # mid-layer of 26
        "mlp_path": "model.layers.{layer_idx}.mlp",
        "model_dtype": "bfloat16",
    },
    "openai-community/gpt2": {
        "d_model": 768,
        "layer_idx": 6,  # mid-layer of 12
        "mlp_path": "transformer.h.{layer_idx}.mlp",
        "model_dtype": "float32",
    },
}


@dataclass
class MOLTConfig:
    """Configuration for MOLT training and architecture."""

    # Model
    model_name: str = "google/gemma-3-1b-it"
    layer_idx: int = 13
    d_model: int = 1152
    mlp_path: str = ""  # dot-separated path to MLP module, e.g. "model.layers.{layer_idx}.mlp"
    model_dtype: str = "bfloat16"  # dtype for loading the source model

    # MOLT architecture
    # Rank multiplier N: creates N×512, 2N×256, 4N×128, 8N×64, 16N×32
    rank_multiplier: int = 1
    activation: str = "jumprelu"  # "relu" or "jumprelu"
    jumprelu_threshold: float = 0.0
    learned_threshold: bool = False  # if True, JumpReLU threshold is a shared learnable parameter
    sparsity_type: str = "tanh"  # "tanh", "l1", or "l0"

    # Training
    sparsity_coeff: float = 1e-3
    sparsity_warmup_frac: float = 0.1  # fraction of training steps to linearly ramp λ from 0
    lr: float = 1e-3
    batch_size: int = 64
    num_tokens: int = 10_000_000
    seq_len: int = 256
    num_epochs: int = 1
    seed: int = 42
    device: str = "cuda"

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_split: str = "train"
    streaming: bool = True

    # Logging
    wandb_project: str = "repro-molts"
    wandb_enabled: bool = False
    log_every: int = 100
    eval_every: int = 1000

    # Checkpointing
    save_dir: str = "checkpoints"

    @property
    def rank_distribution(self) -> list[tuple[int, int]]:
        """Returns list of (num_transforms, rank) tuples.

        Ranks are capped at d_model and groups with rank > d_model are dropped.
        """
        n = self.rank_multiplier
        base_ranks = [
            (1 * n, 512),
            (2 * n, 256),
            (4 * n, 128),
            (8 * n, 64),
            (16 * n, 32),
        ]
        # Filter out ranks that exceed d_model
        return [(count, rank) for count, rank in base_ranks if rank <= self.d_model]

    @property
    def total_transforms(self) -> int:
        return sum(count for count, _ in self.rank_distribution)

    @classmethod
    def from_preset(cls, model_name: str, **overrides) -> "MOLTConfig":
        """Create config from a model preset with optional overrides."""
        preset = MODEL_PRESETS.get(model_name, {})
        kwargs = {"model_name": model_name, **preset, **overrides}
        return cls(**kwargs)
