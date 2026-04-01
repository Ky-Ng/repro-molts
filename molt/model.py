"""MOLT: Mixture of Linear Transforms.

Implements the architecture from "Sparse Mixtures of Linear Transforms"
(https://transformer-circuits.pub/2025/bulk-update/index.html).

Each transform is a low-rank matrix U_t @ V_t gated by φ(e_t · x - b_t):
    f(x) = Σ_t [φ(e_t · x - b_t) * (U_t @ V_t @ x)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from molt.config import MOLTConfig


class JumpReLU(torch.autograd.Function):
    """JumpReLU with full straight-through estimator.

    Forward: hard threshold (x * 1[x > θ])
    Backward: full STE — gradients pass through unconditionally so that
    dead gates can reactivate via MSE gradient signal.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Full straight-through: always pass gradient so dead gates can reactivate
        return grad_output, None


def jumprelu(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return JumpReLU.apply(x, threshold)


class TransformGroup(nn.Module):
    """A group of transforms sharing the same rank.

    Batches all transforms of a given rank for efficient computation.

    Parameters:
        num_transforms: Number of transforms in this group
        d_model: Model hidden dimension
        rank: Rank of the low-rank decomposition
    """

    def __init__(self, num_transforms: int, d_model: int, rank: int):
        super().__init__()
        self.num_transforms = num_transforms
        self.d_model = d_model
        self.rank = rank

        # Low-rank transform: U @ V, where U: (n, d, r), V: (n, r, d)
        self.V = nn.Parameter(torch.empty(num_transforms, rank, d_model))
        self.U = nn.Parameter(torch.empty(num_transforms, d_model, rank))

        # Gating: φ(encoder · x - bias)
        self.encoder = nn.Parameter(torch.empty(num_transforms, d_model))
        self.bias = nn.Parameter(torch.zeros(num_transforms))

        self._init_weights()

    def _init_weights(self):
        # Scale U, V so initial transform output is O(1/sqrt(num_transforms))
        # Each transform contributes U @ V @ x; we want ||U_t V_t x|| small at init
        scale = (1.0 / (self.d_model * self.rank)) ** 0.25
        nn.init.normal_(self.V, std=scale)
        nn.init.normal_(self.U, std=scale)
        # Unit-norm encoder directions so gating pre-acts scale with input norm
        nn.init.normal_(self.encoder)
        with torch.no_grad():
            self.encoder.div_(self.encoder.norm(dim=1, keepdim=True))
        # Negative bias so gates start open (pre_acts = e·x - b, b < 0 → easier to activate)
        nn.init.constant_(self.bias, -1.0)

    def forward(
        self, x: torch.Tensor, activation_fn: str = "jumprelu", threshold: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, d_model)
            activation_fn: "relu" or "jumprelu"
            threshold: JumpReLU threshold

        Returns:
            output: (batch, d_model) — sum of gated transforms
            gate_acts: (batch, num_transforms) — gating activations (pre-activation for sparsity)
            frobenius_norms: (num_transforms,) — ||U_t V_t||_F approximated as ||U_t||_F * ||V_t||_F
        """
        # Gating: (batch, d_model) @ (d_model, num_transforms) -> (batch, num_transforms)
        pre_acts = x @ self.encoder.T - self.bias  # (batch, num_transforms)

        if activation_fn == "jumprelu":
            gate = jumprelu(pre_acts, threshold)
        else:
            gate = F.relu(pre_acts)

        # Transform: V @ x -> (num_transforms, rank, batch) then U @ that
        # x: (batch, d_model) -> (1, batch, d_model)
        # V: (num_transforms, rank, d_model)
        Vx = torch.einsum("nrd,bd->nbr", self.V, x)  # (num_transforms, batch, rank)
        UVx = torch.einsum("ndr,nbr->nbd", self.U, Vx)  # (num_transforms, batch, d_model)

        # Apply gating and sum over transforms
        # gate: (batch, num_transforms) -> (num_transforms, batch, 1)
        gated = UVx * gate.T.unsqueeze(-1)  # (num_transforms, batch, d_model)
        output = gated.sum(dim=0)  # (batch, d_model)

        # Frobenius norms for sparsity penalty, normalized by expected init scale
        # so that λ operates in a dimension-independent range
        u_norms = self.U.flatten(1).norm(dim=1)  # (num_transforms,)
        v_norms = self.V.flatten(1).norm(dim=1)  # (num_transforms,)
        frobenius_norms = u_norms * v_norms / (self.d_model * self.rank) ** 0.5

        return output, gate, frobenius_norms


class MOLT(nn.Module):
    """Mixture of Linear Transforms.

    Decomposes an MLP layer into a sparse mixture of low-rank transforms
    with varying ranks: N×512, 2N×256, 4N×128, 8N×64, 16N×32.
    """

    def __init__(self, config: MOLTConfig):
        super().__init__()
        self.config = config

        self.groups = nn.ModuleList()
        for num_transforms, rank in config.rank_distribution:
            self.groups.append(
                TransformGroup(num_transforms, config.d_model, rank)
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through all transform groups.

        Args:
            x: (batch, d_model) — input activations (pre-MLP residual stream)

        Returns:
            output: (batch, d_model) — MOLT reconstruction of MLP output
            aux: dict with keys:
                - sparsity_loss: scalar, tanh sparsity penalty
                - l0: scalar, average number of active transforms
                - gate_acts: list of (batch, n_transforms) per group
        """
        output = torch.zeros_like(x)
        total_sparsity = torch.tensor(0.0, device=x.device)
        total_active = torch.tensor(0.0, device=x.device)
        all_gate_acts = []

        for group in self.groups:
            group_out, gate, frob_norms = group.forward(
                x, self.config.activation, self.config.jumprelu_threshold
            )
            output = output + group_out

            # Tanh sparsity penalty: Σ_t tanh(mean |gate_t|) * ||U_t V_t||_F
            mean_abs_gate = gate.abs().mean(dim=0)  # (num_transforms,)
            sparsity = (torch.tanh(mean_abs_gate) * frob_norms).sum()
            total_sparsity = total_sparsity + sparsity

            # L0: count active transforms (gate > 0)
            active = (gate > 0).float().sum(dim=1).mean()  # avg active per token
            total_active = total_active + active

            all_gate_acts.append(gate)

        aux = {
            "sparsity_loss": total_sparsity,
            "l0": total_active,
            "gate_acts": all_gate_acts,
        }
        return output, aux

    def loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        sparsity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute training loss = MSE + λ * sparsity_scale * sparsity.

        Args:
            x: (batch, d_model) — MLP input activations
            target: (batch, d_model) — true MLP output activations
            sparsity_scale: multiplier for sparsity coeff (0→1 during warmup)

        Returns:
            total_loss: scalar
            metrics: dict with mse, sparsity_loss, l0, total_loss
        """
        output, aux = self.forward(x)

        mse = F.mse_loss(output, target)
        sparsity = aux["sparsity_loss"]
        effective_lambda = self.config.sparsity_coeff * sparsity_scale
        total = mse + effective_lambda * sparsity

        # Normalized MSE (MSE / variance of target)
        target_var = target.var()
        nmse = mse / (target_var + 1e-8)

        metrics = {
            "mse": mse.detach(),
            "nmse": nmse.detach(),
            "sparsity_loss": sparsity.detach(),
            "l0": aux["l0"].detach(),
            "total_loss": total.detach(),
        }
        return total, metrics
