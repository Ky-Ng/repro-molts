"""Activation standardizers.

Ported from crosslayer-transcoder/model/standardize.py. These modules buffer
per-dimension mean/std from the first training batch and apply them to inputs
and outputs so that MOLT trains on unit-scaled activations (matching the
reference implementation). The `layer` argument supports both multi-layer setups
(when `layer="all"`) and single-layer indexing (when `layer` is an int).
"""

import torch
import torch.nn as nn
from jaxtyping import Float


class Standardizer(nn.Module):
    """No-op standardizer base / interface."""

    def __init__(self, **kwargs):
        super().__init__()

    def initialize_from_batch(self, batch: Float[torch.Tensor, "batch io n_layers d"]):
        pass

    def forward(self, batch: Float[torch.Tensor, "batch n_layers d"], layer="all"):
        return batch

    def standardize(self, batch: Float[torch.Tensor, "batch n_layers d"], layer="all"):
        return batch


class DimensionwiseInputStandardizer(Standardizer):
    """Per-(layer, dimension) z-score standardizer for encoder inputs."""

    def __init__(self, n_layers: int, activation_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers, activation_dim))
        self.register_buffer("std", torch.empty(n_layers, activation_dim))
        # Buffer so initialized state survives state_dict save/load.
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialize_from_batch(self, batch: Float[torch.Tensor, "batch io n_layers d"]):
        inputs = batch[:, 0]
        self.mean.data = inputs.mean(dim=0)
        self.std.data = inputs.std(dim=0)
        self.std.data.clamp_(min=1e-8)
        self._initialized.fill_(True)

    def forward(self, batch, layer="all"):
        if not bool(self._initialized):
            raise ValueError("Standardizer not initialized")
        if layer == "all":
            return (batch - self.mean) / self.std
        return (batch - self.mean[layer]) / self.std[layer]


class DimensionwiseOutputStandardizer(Standardizer):
    """Per-(layer, dimension) z-score standardizer for decoder outputs.

    `forward` un-standardizes (model-space → activation-space) so the reconstruction
    can be compared to raw MLP outputs. `standardize` standardizes raw outputs for
    loss computation in model space.
    """

    def __init__(self, n_layers: int, activation_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers, activation_dim))
        self.register_buffer("std", torch.empty(n_layers, activation_dim))
        # Buffer so initialized state survives state_dict save/load.
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialize_from_batch(self, batch: Float[torch.Tensor, "batch io n_layers d"]):
        outputs = batch[:, 1]
        self.mean.data = outputs.mean(dim=0)
        self.std.data = outputs.std(dim=0)
        self.std.data.clamp_(min=1e-8)
        self._initialized.fill_(True)

    def forward(self, batch, layer="all"):
        if not bool(self._initialized):
            raise ValueError("Standardizer not initialized")
        if layer == "all":
            return (batch * self.std) + self.mean
        return (batch * self.std[layer]) + self.mean[layer]

    def standardize(self, mlp_out, layer="all"):
        if not bool(self._initialized):
            raise ValueError("Standardizer not initialized")
        if layer == "all":
            return (mlp_out - self.mean) / self.std
        return (mlp_out - self.mean[layer]) / self.std[layer]
