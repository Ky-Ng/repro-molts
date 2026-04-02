from dataclasses import dataclass, field


@dataclass
class MOLTConfig:
    """Configuration for MOLT training and architecture."""

    # Model
    model_name: str = "google/gemma-3-1b-it"
    layer_idx: int = 13
    d_model: int = 1152

    # MOLT architecture
    # Rank multiplier N: creates N×512, 2N×256, 4N×128, 8N×64, 16N×32
    rank_multiplier: int = 1
    activation: str = "jumprelu"  # "relu" or "jumprelu"
    jumprelu_threshold: float = 0.0
    sparsity_type: str = "tanh"  # "tanh" or "l1"

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
        """Returns list of (num_transforms, rank) tuples."""
        n = self.rank_multiplier
        return [
            (1 * n, 512),
            (2 * n, 256),
            (4 * n, 128),
            (8 * n, 64),
            (16 * n, 32),
        ]

    @property
    def total_transforms(self) -> int:
        return sum(count for count, _ in self.rank_distribution)
