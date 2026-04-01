"""Training loop for MOLT."""

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from molt.config import MOLTConfig
from molt.data import make_dataloader
from molt.model import MOLT


def train_molt(
    config: MOLTConfig,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
) -> tuple[MOLT, list[dict]]:
    """Train a MOLT model on cached MLP activations.

    Args:
        config: MOLT config
        mlp_inputs: (N, d_model) MLP input activations
        mlp_outputs: (N, d_model) MLP output activations

    Returns:
        model: trained MOLT
        history: list of metric dicts per logging step
    """
    torch.manual_seed(config.seed)

    model = MOLT(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataloader = make_dataloader(mlp_inputs, mlp_outputs, config.batch_size)

    # Optional wandb
    if config.wandb_enabled:
        import wandb

        wandb.init(project=config.wandb_project, config=vars(config))

    history: list[dict] = []
    step = 0
    total_steps = len(dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.sparsity_warmup_frac)

    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch_inputs, batch_targets in pbar:
            batch_inputs = batch_inputs.to(config.device)
            batch_targets = batch_targets.to(config.device)

            # Linear sparsity warmup
            sparsity_scale = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0

            optimizer.zero_grad()
            loss, metrics = model.loss(batch_inputs, batch_targets, sparsity_scale)
            loss.backward()
            optimizer.step()

            step += 1

            if step % config.log_every == 0:
                log = {k: v.item() for k, v in metrics.items()}
                log["step"] = step
                log["epoch"] = epoch
                history.append(log)

                pbar.set_postfix(
                    mse=f"{log['mse']:.4f}",
                    nmse=f"{log['nmse']:.4f}",
                    l0=f"{log['l0']:.1f}",
                )

                if config.wandb_enabled:
                    import wandb

                    wandb.log(log, step=step)

    # Save checkpoint
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"molt_N{config.rank_multiplier}_lam{config.sparsity_coeff}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "history": history,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    # Save history
    history_path = save_dir / f"history_N{config.rank_multiplier}_lam{config.sparsity_coeff}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if config.wandb_enabled:
        import wandb

        wandb.finish()

    return model, history


def load_molt(checkpoint_path: str, device: str = "cuda") -> tuple[MOLT, MOLTConfig]:
    """Load a trained MOLT from checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = MOLTConfig(**ckpt["config"])
    config.device = device
    model = MOLT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config
