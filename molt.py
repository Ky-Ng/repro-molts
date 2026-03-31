"""Sparse Mixtures of Linear Transforms (MOLT).

Reimplementation of https://transformer-circuits.pub/2025/bulk-update/index.html

Core idea: decompose an MLP layer into a sparse mixture of conditionally-active
low-rank linear transforms:

    f(x) = Σ_t [φ(e_t · x - b_t) · (U_t V_t x)]

Each transform t has:
  - encoder vector e_t and bias b_t for gating
  - low-rank factorization U_t V_t of rank k_t
  - nonlinearity φ (ReLU or JumpReLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MOLTConfig:
    d_model: int = 768
    # (rank, num_transforms) pairs
    rank_distribution: list[tuple[int, int]] | None = None
    nonlinearity: str = "relu"  # "relu" or "jumprelu"
    jumprelu_threshold: float = 0.0
    # N for default rank distribution: N×64, 2N×32, 4N×16, 8N×8
    base_n: int = 4

    def __post_init__(self):
        if self.rank_distribution is None:
            n = self.base_n
            self.rank_distribution = [
                (64, n),
                (32, 2 * n),
                (16, 4 * n),
                (8, 8 * n),
            ]

    @property
    def total_transforms(self) -> int:
        return sum(count for _, count in self.rank_distribution)


class JumpReLU(torch.autograd.Function):
    """JumpReLU: φ(x) = x * (x > θ) with straight-through estimator."""

    @staticmethod
    def forward(ctx, x, threshold):
        mask = x > threshold
        ctx.save_for_backward(mask)
        return x * mask.float()

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        # Straight-through: pass gradient through where active
        return grad_output * mask.float(), None


def jumprelu(x, threshold=0.0):
    return JumpReLU.apply(x, threshold)


class TransformGroup(nn.Module):
    """A group of transforms sharing the same rank k.

    Batches all transforms of the same rank for efficient computation.
    """

    def __init__(self, d_model: int, rank: int, n_transforms: int):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_transforms = n_transforms

        # Encoder: computes gating scores for all transforms in this group
        # e_t · x - b_t for each transform t
        self.encoder = nn.Linear(d_model, n_transforms, bias=True)

        # Low-rank transform matrices
        # V: down-project from d_model -> rank (per transform)
        # U: up-project from rank -> d_model (per transform)
        self.V = nn.Parameter(torch.empty(n_transforms, rank, d_model))
        self.U = nn.Parameter(torch.empty(n_transforms, d_model, rank))

        self._init_weights()

    def _init_weights(self):
        # Kaiming init scaled by 1/sqrt(n_transforms) to keep output scale stable
        nn.init.kaiming_uniform_(self.V)
        nn.init.kaiming_uniform_(self.U)
        scale = 1.0 / (self.n_transforms ** 0.5)
        self.V.data *= scale
        self.U.data *= scale
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.encoder.weight)

    def forward(self, x, nonlinearity="relu", jumprelu_threshold=0.0):
        """
        Args:
            x: (batch, d_model)
            nonlinearity: "relu" or "jumprelu"
            jumprelu_threshold: threshold for JumpReLU

        Returns:
            output: (batch, d_model) — sum of gated low-rank transforms
            activations: (batch, n_transforms) — gating activations (post-nonlinearity)
        """
        # Compute gating: (batch, n_transforms)
        gate = self.encoder(x)

        # Apply nonlinearity
        if nonlinearity == "jumprelu":
            activations = jumprelu(gate, jumprelu_threshold)
        else:
            activations = F.relu(gate)

        # Compute low-rank transforms for all transforms in batch
        # x: (batch, d_model)
        # V: (n_transforms, rank, d_model)
        # Vx: (batch, n_transforms, rank) = einsum("bd, nrd -> bnr")
        Vx = torch.einsum("bd, nrd -> bnr", x, self.V)

        # U @ Vx: (batch, n_transforms, d_model) = einsum("ndr, bnr -> bnd")
        UVx = torch.einsum("ndr, bnr -> bnd", self.U, Vx)

        # Gate and sum over transforms
        # activations: (batch, n_transforms) -> (batch, n_transforms, 1)
        # output: (batch, d_model)
        output = (activations.unsqueeze(-1) * UVx).sum(dim=1)

        return output, activations

    def frobenius_norms(self):
        """Compute ||U_t V_t||_F for each transform, approximated as ||U_t||_F * ||V_t||_F.

        This is an upper bound (by submultiplicativity) but cheap to compute.
        Returns: (n_transforms,)
        """
        u_norms = self.U.flatten(1).norm(dim=1)  # (n_transforms,)
        v_norms = self.V.flatten(1).norm(dim=1)  # (n_transforms,)
        return u_norms * v_norms


class MOLT(nn.Module):
    """Sparse Mixture of Linear Transforms.

    Decomposes an MLP layer into conditionally-active low-rank linear transforms.
    """

    def __init__(self, config: MOLTConfig):
        super().__init__()
        self.config = config

        self.groups = nn.ModuleList([
            TransformGroup(config.d_model, rank, n_transforms)
            for rank, n_transforms in config.rank_distribution
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, d_model) — MLP input (residual stream)

        Returns:
            output: (batch, d_model) — reconstructed MLP output
            aux: dict with activations and stats for loss computation
        """
        output = torch.zeros_like(x)
        all_activations = []
        all_frob_norms = []

        for group in self.groups:
            group_out, activations = group.forward(
                x,
                nonlinearity=self.config.nonlinearity,
                jumprelu_threshold=self.config.jumprelu_threshold,
            )
            output = output + group_out
            all_activations.append(activations)
            all_frob_norms.append(group.frobenius_norms())

        # Concatenate across all groups
        all_activations = torch.cat(all_activations, dim=1)  # (batch, total_transforms)
        all_frob_norms = torch.cat(all_frob_norms, dim=0)  # (total_transforms,)

        # Stats
        l0 = (all_activations > 0).float().sum(dim=1).mean()

        return output, {
            "activations": all_activations,
            "frob_norms": all_frob_norms,
            "l0": l0,
        }

    def sparsity_loss(self, activations, frob_norms):
        """Compute sparsity penalty: mean over batch of Σ_t ||U_t V_t||_F · φ(e_t·x - b_t).

        Args:
            activations: (batch, total_transforms) — post-nonlinearity gating values
            frob_norms: (total_transforms,) — Frobenius norms of U_t V_t

        Returns:
            scalar sparsity loss
        """
        # (batch, total_transforms) * (total_transforms,) -> (batch, total_transforms)
        weighted = activations * frob_norms.unsqueeze(0)
        return weighted.sum(dim=1).mean()
