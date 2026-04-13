"""MOLT: Mixture of Linear Transforms.

Ported from crosslayer-transcoder/model/molt.py to fix the bugs documented in
[DIFFERENCES.md](../../DIFFERENCES.md). The architecture now matches the
reference implementation byte-for-byte:

  - Shared encoder `nn.Linear(d_acts, n_features)` producing pre-activations
    for all transforms in one matmul.
  - `JumpReLU` nonlinearity with learnable per-feature threshold and the
    rectangle-kernel STE from the reference (see `jumprelu.py`).
  - True `||UV||_F` in `transform_norm()` (no product-of-norms approximation).
  - Per-(layer, dim) standardizers on input and output (see `standardize.py`).
  - Sparsity penalty `mean_over_batch(sum(tanh(gate * ||UV||_F * c)))`, matching
    `crosslayer-transcoder/model/clt_lightning.py::MoltModule.training_step`.

Single-layer is represented internally as `n_layers=1`, and all calls use
`layer=0`. This keeps the module interoperable with the multi-layer reference
while matching the repro's single-layer data pipeline.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from molt.config import MOLTConfig
from molt.jumprelu import JumpReLU
from molt.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
    Standardizer,
)


class MOLT(nn.Module):
    """Mixture of Linear Transforms.

    Low-rank transforms with varying ranks
    (N×512, 2N×256, 4N×128, 8N×64, 16N×32) gated by a single shared JumpReLU
    encoder. Forward returns `(gate, recons_norm, recons)`:

        - `gate`: (batch, n_features) — JumpReLU features after the encoder.
        - `recons_norm`: (batch, d_model) — reconstruction in standardized space.
        - `recons`: (batch, d_model) — reconstruction in raw activation space.

    Use `MOLT.loss(...)` to combine MSE + sparsity into a scalar loss.
    """

    def __init__(
        self,
        config: MOLTConfig,
        nonlinearity: nn.Module | None = None,
        input_standardizer: Standardizer | None = None,
        output_standardizer: Standardizer | None = None,
    ):
        super().__init__()
        self.config = config

        ranks: list[tuple[int, int]] = config.rank_distribution
        Us: list[nn.Parameter] = []
        Vs: list[nn.Parameter] = []
        n_features = 0
        d_latents = 0
        for num_transforms, rank in ranks:
            Us.append(nn.Parameter(torch.empty(num_transforms, rank, config.d_model)))
            Vs.append(nn.Parameter(torch.empty(num_transforms, config.d_model, rank)))
            n_features += num_transforms
            d_latents += num_transforms * rank

        self.d_acts = config.d_model
        self.n_features = n_features
        self.d_latents = d_latents
        self.Us = nn.ParameterList(Us)
        self.Vs = nn.ParameterList(Vs)
        self.e = nn.Linear(config.d_model, n_features)

        if nonlinearity is None:
            nonlinearity = JumpReLU(
                theta=config.jumprelu_threshold,
                bandwidth=config.jumprelu_bandwidth,
                n_layers=1,
                d_features=n_features,
            )
        self.nonlinearity = nonlinearity

        if input_standardizer is None:
            input_standardizer = DimensionwiseInputStandardizer(
                n_layers=1, activation_dim=config.d_model
            )
        if output_standardizer is None:
            output_standardizer = DimensionwiseOutputStandardizer(
                n_layers=1, activation_dim=config.d_model
            )
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer

        self.reset_parameters()

    def reset_parameters(self):
        for U in self.Us:
            nn.init.xavier_uniform_(U)
        for V in self.Vs:
            nn.init.xavier_uniform_(V)

    # ------------------------------------------------------------------
    # Core forward / norms
    # ------------------------------------------------------------------

    def transform_norm(self) -> torch.Tensor:
        """Compute `||U_t V_t||_F` for every transform (true Frobenius norm)."""
        norms = []
        for U, V in zip(self.Us, self.Vs):
            uv = einops.einsum(
                U,
                V,
                "n_transforms d_transform d_acts_out, n_transforms d_acts_in d_transform "
                "-> n_transforms d_acts_in d_acts_out",
            )
            norms.append(torch.norm(uv, dim=(1, 2)))
        return torch.cat(norms, dim=0)

    def forward(
        self,
        acts: Float[torch.Tensor, "batch d_acts"],
        layer: int = 0,
    ) -> tuple[
        Float[torch.Tensor, "batch n_features"],
        Float[torch.Tensor, "batch d_acts"],
        Float[torch.Tensor, "batch d_acts"],
    ]:
        acts = self.input_standardizer(acts, layer)
        pre_actvs = self.e(acts)
        gate = self.nonlinearity(pre_actvs)

        raw_recons: list[torch.Tensor] = []
        for U, V in zip(self.Us, self.Vs):
            latents = einops.einsum(
                acts,
                V,
                "batch d_acts, n_transforms d_acts d_transform -> batch n_transforms d_transform",
            )
            raw_recons.append(
                einops.einsum(
                    latents,
                    U,
                    "batch n_transforms d_transform, n_transforms d_transform d_acts "
                    "-> batch n_transforms d_acts",
                )
            )

        raw_recons_t = torch.cat(raw_recons, dim=1)
        weighted_recons = gate.unsqueeze(-1) * raw_recons_t
        recons_norm = weighted_recons.sum(dim=1)
        recons = self.output_standardizer(recons_norm, layer)
        return gate, recons_norm, recons

    # ------------------------------------------------------------------
    # Loss (MSE + sparsity), matching MoltModule.training_step
    # ------------------------------------------------------------------

    def loss(
        self,
        x: Float[torch.Tensor, "batch d_acts"],
        target: Float[torch.Tensor, "batch d_acts"],
        sparsity_scale: float = 1.0,
        layer: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute training loss = MSE(standardized) + λ·scale · tanh-sparsity.

        Mirrors `crosslayer-transcoder/model/clt_lightning.py::MoltModule.training_step`:

            mse         = (recons_norm - standardize(target)) ** 2
            norms       = ||UV||_F                                     (n_features,)
            weighted    = norms * gate                                 (batch, n_features)
            if use_tanh: weighted = tanh(weighted * c_sparsity)
            sparsity    = lambda_cur * weighted.sum(-1).mean()
            loss        = mse.mean() + sparsity
        """
        gate, recons_norm, recons = self.forward(x, layer)

        target_norm = self.output_standardizer.standardize(target, layer)
        mse = ((recons_norm - target_norm) ** 2).mean()

        norms = self.transform_norm()
        weighted = norms * gate
        if self.config.use_tanh:
            weighted = torch.tanh(weighted * self.config.c_sparsity)

        effective_lambda = self.config.sparsity_coeff * sparsity_scale
        sparsity = effective_lambda * weighted.sum(dim=-1).mean()

        total = mse + sparsity

        # Metrics
        l0 = (gate > 0).float().sum(dim=1).mean()
        nmse = mse / (target_norm.var() + 1e-8)

        metrics = {
            "mse": mse.detach(),
            "nmse": nmse.detach(),
            "sparsity_loss": sparsity.detach(),
            "l0": l0.detach(),
            "total_loss": total.detach(),
        }
        if hasattr(self.nonlinearity, "theta"):
            metrics["threshold_mean"] = self.nonlinearity.theta.detach().mean()
        return total, metrics

    # ------------------------------------------------------------------
    # Standardizer initialization (call with the first training batch)
    # ------------------------------------------------------------------

    def initialize_standardizers(
        self,
        x: Float[torch.Tensor, "batch d_acts"],
        target: Float[torch.Tensor, "batch d_acts"],
    ) -> None:
        """Initialize input/output standardizers from a representative batch.

        Constructs the `(batch, io=2, n_layers=1, d_acts)` tensor that the
        reference standardizers expect.
        """
        batch = torch.stack([x, target], dim=1).unsqueeze(2)  # (B, 2, 1, d)
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)
