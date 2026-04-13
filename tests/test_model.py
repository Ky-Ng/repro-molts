"""Smoke tests for MOLT model (ported from crosslayer-transcoder)."""

import torch
from molt.config import MOLTConfig
from molt.jumprelu import JumpReLU
from molt.model import MOLT


def test_jumprelu_forward():
    # The ported JumpReLU requires input > theta AND input > 0.
    layer = JumpReLU(theta=0.5, bandwidth=1.0, n_layers=1, d_features=5)
    x = torch.tensor([[-1.0, 0.0, 0.5, 1.0, 2.0]])
    out = layer(x)
    assert out[0, 0] == 0.0  # negative -> 0
    assert out[0, 1] == 0.0  # exactly 0
    assert out[0, 2] == 0.0  # exactly at threshold (not strictly greater)
    assert out[0, 3] == 1.0
    assert out[0, 4] == 2.0


def test_molt_forward():
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)
    x = torch.randn(8, 64)
    t = torch.randn(8, 64)
    model.initialize_standardizers(x, t)

    gate, recons_norm, recons = model(x)
    assert gate.shape == (8, model.n_features)
    assert recons_norm.shape == (8, 64)
    assert recons.shape == (8, 64)
    assert (gate >= 0).all()  # JumpReLU output is non-negative


def test_molt_transform_norm():
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)
    norms = model.transform_norm()
    assert norms.shape == (model.n_features,)
    assert (norms >= 0).all()


def test_molt_loss():
    config = MOLTConfig(d_model=64, rank_multiplier=1, sparsity_coeff=1e-3, device="cpu")
    model = MOLT(config)
    x = torch.randn(8, 64)
    target = torch.randn(8, 64)
    model.initialize_standardizers(x, target)

    loss, metrics = model.loss(x, target)
    assert loss.shape == ()
    assert loss.requires_grad
    assert metrics["mse"].item() > 0
    assert metrics["l0"].item() >= 0

    # Check backward works for all parameters with requires_grad.
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_rank_distribution():
    config = MOLTConfig(rank_multiplier=1)
    assert config.total_transforms == 31
    assert config.rank_distribution == [
        (1, 512), (2, 256), (4, 128), (8, 64), (16, 32),
    ]

    config2 = MOLTConfig(rank_multiplier=2)
    assert config2.total_transforms == 62


def test_molt_training_step():
    """End-to-end single training step with standardizer init."""
    config = MOLTConfig(d_model=64, rank_multiplier=1, lr=1e-3, device="cpu")
    model = MOLT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x = torch.randn(16, 64)
    target = torch.randn(16, 64)
    model.initialize_standardizers(x, target)

    loss1, _ = model.loss(x, target)
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss2, _ = model.loss(x, target)
    assert loss2.item() < loss1.item() * 2  # Not exploding
