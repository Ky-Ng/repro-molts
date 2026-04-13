"""SAE-shaped wrappers around MOLT and Transcoder for delphi compatibility.

Delphi's LatentCache hooks a submodule in the base model, captures its input,
and runs that input through a callable (the "SAE") to get feature activations
of shape (B, S, n_features). The nonzeros are then harvested and sharded as
safetensors.

This module provides two adapters:

- MOLTShim: flattens aux["gate_acts"] into a single (B, S, n_features) tensor
  using the canonical flat ordering defined in feature_layout.FeatureLayout.
  Strength = raw gate value (pre-activation clipped at JumpReLU threshold),
  NOT gate * ||U V||_F — see experiment 13 README §Design decisions.

- TranscoderShim: returns encoder(x) post-activation, the standard transcoder
  latent.

Both conform to the minimal SAE interface delphi expects:
    .encode(x: Tensor[..., d_model]) -> Tensor[..., n_features]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molt.model import MOLT
from molt.transcoder import TrainableTranscoder


class SparseCoderShim(nn.Module):
    """Base class — subclasses implement .encode(x)."""

    n_features: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class MOLTShim(SparseCoderShim):
    """Flatten MOLT gate activations into (..., n_features) in canonical order.

    The canonical order is defined by the MOLT's rank_distribution:
        feature_id in group g, transform t
            = sum(group_sizes[:g]) + t
    which matches experiments/13_retrain_4x_molt_transcoder/feature_layout.py.
    """

    def __init__(self, molt: MOLT):
        super().__init__()
        self.molt = molt
        self.n_features = molt.config.total_transforms
        # Cache the flat ordering once — it's stable as long as the config doesn't change
        self._group_sizes = [g.num_transforms for g in molt.groups]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map MLP input x to flat gate activations.

        Args:
            x: (..., d_model) — can be (B, d) or (B, S, d)

        Returns:
            (..., n_features) — non-negative, gate value when active else 0
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        else:
            x_flat = x

        # Run MOLT forward to get aux["gate_acts"] list-of-(B, n_transforms)
        # Clamp to >= 0: with JumpReLU fixed θ=0, gate is either 0 or the
        # pre-activation value. Sometimes the returned gate can be slightly
        # negative during surrogate training; clamp for delphi's nonzero-mask
        # semantics (which treats any nonzero as "active").
        _, aux = self.molt(x_flat)
        parts = [g.clamp_min(0.0) for g in aux["gate_acts"]]
        feats_flat = torch.cat(parts, dim=-1)  # (B*S, n_features)
        assert feats_flat.shape[-1] == self.n_features

        if x.dim() == 3:
            return feats_flat.reshape(B, S, self.n_features)
        return feats_flat


class TranscoderShim(SparseCoderShim):
    """Return transcoder encoder output (post-ReLU) as the latent activations."""

    def __init__(self, transcoder: TrainableTranscoder):
        super().__init__()
        self.transcoder = transcoder
        self.n_features = transcoder.n_features

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map MLP input x to post-ReLU feature activations.

        Mirrors TrainableTranscoder.forward encoder pass:
            feats = relu(W_enc @ x + b_enc)
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        else:
            x_flat = x

        pre = x_flat @ self.transcoder.W_enc.T + self.transcoder.b_enc
        feats_flat = torch.relu(pre)

        if x.dim() == 3:
            return feats_flat.reshape(B, S, self.n_features)
        return feats_flat
