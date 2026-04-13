"""Evaluation metrics for MOLTs: Jacobian faithfulness, L0, Normalized MSE."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap


def compute_jacobian(
    fn: Callable,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute Jacobian matrix J[i,j] = d(fn(x))_i / dx_j.

    Args:
        fn: function mapping (d_model,) -> (d_model,)
        x: (batch, d_model) input

    Returns:
        jacobian: (batch, d_model, d_model)
    """
    # Use vmap + jacrev for efficient batched Jacobian
    jac_fn = vmap(jacrev(fn))
    return jac_fn(x)


def jacobian_faithfulness(
    molt_fn: Callable,
    mlp_fn: Callable,
    x: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute Jacobian faithfulness as cosine similarity of flattened Jacobians.

    For each input x_i, compute:
        cos_sim(flatten(J_molt(x_i)), flatten(J_mlp(x_i)))

    Args:
        molt_fn: MOLT forward function (d_model,) -> (d_model,)
        mlp_fn: true MLP forward function (d_model,) -> (d_model,)
        x: (N, d_model) input activations
        batch_size: batch size for Jacobian computation

    Returns:
        cosine_similarities: (N,) per-input faithfulness scores
    """
    all_sims = []

    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]

        j_molt = compute_jacobian(molt_fn, batch)  # (B, d, d)
        j_mlp = compute_jacobian(mlp_fn, batch)  # (B, d, d)

        # Flatten and compute cosine similarity
        j_molt_flat = j_molt.flatten(1)  # (B, d*d)
        j_mlp_flat = j_mlp.flatten(1)  # (B, d*d)

        sim = F.cosine_similarity(j_molt_flat, j_mlp_flat, dim=1)  # (B,)
        all_sims.append(sim.detach().cpu())

    return torch.cat(all_sims)


def compute_l0(molt_model, x: torch.Tensor, batch_size: int = 256) -> float:
    """Compute average L0 (number of active transforms per token).

    Args:
        molt_model: trained MOLT
        x: (N, d_model) input activations

    Returns:
        mean_l0: average active transforms per token
    """
    total_l0 = 0.0
    count = 0

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(next(molt_model.parameters()).device)
            gate, _, _ = molt_model(batch)
            active = (gate > 0).float().sum(dim=1).mean().item()
            total_l0 += active * len(batch)
            count += len(batch)

    return total_l0 / count


def compute_nmse(
    molt_model, x: torch.Tensor, target: torch.Tensor, batch_size: int = 256
) -> float:
    """Compute normalized MSE = MSE / Var(target).

    Args:
        molt_model: trained MOLT
        x: (N, d_model) input activations
        target: (N, d_model) true MLP outputs

    Returns:
        nmse: normalized mean squared error
    """
    total_mse = 0.0
    count = 0
    device = next(molt_model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            bx = x[i : i + batch_size].to(device)
            bt = target[i : i + batch_size].to(device)
            # `recons` is in raw activation space (output_standardizer un-standardizes).
            _, _, recons = molt_model(bx)
            mse = F.mse_loss(recons, bt, reduction="sum").item()
            total_mse += mse
            count += bt.numel()

    mean_mse = total_mse / count  # per-element MSE
    target_var = target.var().item()
    return mean_mse / (target_var + 1e-8)


@torch.no_grad()
def evaluate_molt(
    molt_model,
    x: torch.Tensor,
    target: torch.Tensor,
    mlp_fn: Callable | None = None,
    jacobian_samples: int = 64,
) -> dict[str, float]:
    """Run full evaluation suite.

    Args:
        molt_model: trained MOLT
        x: (N, d_model) input activations
        target: (N, d_model) true MLP outputs
        mlp_fn: optional callable for Jacobian comparison
        jacobian_samples: number of samples for Jacobian eval

    Returns:
        dict with l0, nmse, and optionally jacobian_cosine_sim
    """
    results = {
        "l0": compute_l0(molt_model, x),
        "nmse": compute_nmse(molt_model, x, target),
    }

    if mlp_fn is not None:
        device = next(molt_model.parameters()).device
        jac_x = x[:jacobian_samples].to(device).requires_grad_(True)

        def molt_fn(xi):
            _, _, out = molt_model(xi.unsqueeze(0))
            return out.squeeze(0)

        sims = jacobian_faithfulness(molt_fn, mlp_fn, jac_x, batch_size=8)
        results["jacobian_cosine_sim"] = sims.mean().item()
        results["jacobian_cosine_sim_std"] = sims.std().item()

    return results


def plot_pareto(
    results: list[dict[str, float]],
    transcoder_results: list[dict[str, float]] | None = None,
    save_path: str = "pareto.png",
):
    """Plot L0 vs Normalized MSE Pareto frontier.

    Args:
        results: list of dicts with 'l0' and 'nmse' keys (one per λ)
        transcoder_results: optional transcoder baseline points
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    # MOLT points
    l0s = [r["l0"] for r in results]
    nmses = [r["nmse"] for r in results]
    ax.scatter(nmses, l0s, label="MOLT", marker="o", s=60)

    # Connect Pareto frontier
    sorted_pts = sorted(zip(nmses, l0s), key=lambda p: p[0])
    pareto_nmse, pareto_l0 = [], []
    min_l0 = float("inf")
    for nmse, l0 in sorted_pts:
        if l0 < min_l0:
            pareto_nmse.append(nmse)
            pareto_l0.append(l0)
            min_l0 = l0
    ax.plot(pareto_nmse, pareto_l0, "--", alpha=0.5, color="tab:blue")

    if transcoder_results:
        tc_l0s = [r["l0"] for r in transcoder_results]
        tc_nmses = [r["nmse"] for r in transcoder_results]
        labels = [r.get("label", "Transcoder") for r in transcoder_results]
        for l0, nmse, label in zip(tc_l0s, tc_nmses, labels):
            ax.scatter([nmse], [l0], marker="x", s=100, label=label)

    ax.set_xlabel("Normalized MSE")
    ax.set_ylabel("L0 (Active Transforms / Features)")
    ax.set_title("Normalized MSE vs. L0 — MOLT vs. Baselines")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved Pareto plot to {save_path}")
