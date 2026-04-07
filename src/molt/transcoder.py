"""Transcoder baseline: load Gemma Scope 2 transcoders and compute metrics."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap


def load_transcoder(
    release: str,
    sae_id: str,
    device: str = "cuda",
):
    """Load a Gemma Scope 2 transcoder via sae-lens.

    Example releases:
        - "gemma-scope-2-1b-pt-transcoder" (non-skip)
        - "gemma-scope-2-1b-pt-transcoder-skip" (skip)

    Args:
        release: sae-lens release name
        sae_id: specific SAE/transcoder ID within the release
        device: torch device

    Returns:
        transcoder: loaded SAE/transcoder object
    """
    from sae_lens import SAE

    transcoder, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    transcoder.eval()
    return transcoder, cfg_dict


def transcoder_forward(transcoder, x: torch.Tensor) -> torch.Tensor:
    """Run transcoder forward pass.

    For a standard transcoder: encode input -> decode to output space.
    For a skip transcoder: adds a skip connection from input.

    Args:
        transcoder: loaded sae-lens transcoder
        x: (batch, d_model) MLP input activations

    Returns:
        output: (batch, d_model) transcoder reconstruction
    """
    # sae-lens transcoders have encode() and decode() methods
    feature_acts = transcoder.encode(x)
    output = transcoder.decode(feature_acts)
    return output


def transcoder_l0(transcoder, x: torch.Tensor, batch_size: int = 256) -> float:
    """Compute average L0 for transcoder (number of active features per token)."""
    total_l0 = 0.0
    count = 0

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(next(transcoder.parameters()).device)
            feature_acts = transcoder.encode(batch)
            l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
            total_l0 += l0 * len(batch)
            count += len(batch)

    return total_l0 / count


def transcoder_nmse(
    transcoder, x: torch.Tensor, target: torch.Tensor, batch_size: int = 256
) -> float:
    """Compute normalized MSE for transcoder."""
    total_mse = 0.0
    count = 0
    device = next(transcoder.parameters()).device

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            bx = x[i : i + batch_size].to(device)
            bt = target[i : i + batch_size].to(device)
            output = transcoder_forward(transcoder, bx)
            mse = F.mse_loss(output, bt, reduction="sum").item()
            total_mse += mse
            count += bt.numel()

    mean_mse = total_mse / count * target.shape[1]
    target_var = target.var().item()
    return mean_mse / (target_var + 1e-8)


def transcoder_jacobian_faithfulness(
    transcoder,
    mlp_fn: Callable,
    x: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute Jacobian faithfulness for a transcoder.

    Args:
        transcoder: loaded transcoder
        mlp_fn: true MLP function (d_model,) -> (d_model,)
        x: (N, d_model) input activations (requires_grad=True)
        batch_size: batch for Jacobian computation

    Returns:
        cosine_similarities: (N,) per-input faithfulness scores
    """

    def tc_fn(xi):
        return transcoder_forward(transcoder, xi.unsqueeze(0)).squeeze(0)

    all_sims = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]

        j_tc = vmap(jacrev(tc_fn))(batch)
        j_mlp = vmap(jacrev(mlp_fn))(batch)

        j_tc_flat = j_tc.flatten(1)
        j_mlp_flat = j_mlp.flatten(1)

        sim = F.cosine_similarity(j_tc_flat, j_mlp_flat, dim=1)
        all_sims.append(sim.detach().cpu())

    return torch.cat(all_sims)


def evaluate_transcoder(
    release: str,
    sae_id: str,
    x: torch.Tensor,
    target: torch.Tensor,
    mlp_fn: Callable | None = None,
    jacobian_samples: int = 64,
    device: str = "cuda",
    label: str = "Transcoder",
) -> dict[str, float]:
    """Full evaluation of a transcoder baseline.

    Args:
        release: sae-lens release name
        sae_id: transcoder ID
        x: (N, d_model) MLP inputs
        target: (N, d_model) MLP outputs
        mlp_fn: optional for Jacobian eval
        jacobian_samples: number of samples for Jacobian
        device: torch device
        label: display label for plots

    Returns:
        dict with l0, nmse, optionally jacobian_cosine_sim, and label
    """
    transcoder, _ = load_transcoder(release, sae_id, device)

    results = {
        "l0": transcoder_l0(transcoder, x),
        "nmse": transcoder_nmse(transcoder, x, target),
        "label": label,
    }

    if mlp_fn is not None:
        jac_x = x[:jacobian_samples].to(device).requires_grad_(True)
        sims = transcoder_jacobian_faithfulness(transcoder, mlp_fn, jac_x)
        results["jacobian_cosine_sim"] = sims.mean().item()
        results["jacobian_cosine_sim_std"] = sims.std().item()

    # Free memory
    del transcoder
    torch.cuda.empty_cache()

    return results
