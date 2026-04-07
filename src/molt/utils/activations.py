"""Shared activation loading and splitting utilities."""

from __future__ import annotations

from pathlib import Path

import torch


def load_cached_activations(
    cache_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached MLP activations from a .pt file.

    Expected format: dict with 'mlp_inputs' and 'mlp_outputs' tensors.

    Args:
        cache_path: path to the .pt cache file

    Returns:
        (mlp_inputs, mlp_outputs) tensors
    """
    print(f"Loading cached activations from {cache_path}")
    data = torch.load(cache_path, weights_only=True)
    mlp_inputs = data["mlp_inputs"]
    mlp_outputs = data["mlp_outputs"]
    del data
    print(f"Activations: {mlp_inputs.shape}")
    return mlp_inputs, mlp_outputs


def split_train_eval(
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    eval_size: int = 10_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split activations into train and eval sets.

    Eval samples are taken from the end of the tensor.

    Args:
        mlp_inputs: (N, d_model) input activations
        mlp_outputs: (N, d_model) output activations
        eval_size: number of samples to reserve for evaluation

    Returns:
        (train_inputs, train_outputs, eval_inputs, eval_outputs)
    """
    train_in = mlp_inputs[:-eval_size]
    train_out = mlp_outputs[:-eval_size]
    eval_in = mlp_inputs[-eval_size:]
    eval_out = mlp_outputs[-eval_size:]
    return train_in, train_out, eval_in, eval_out
