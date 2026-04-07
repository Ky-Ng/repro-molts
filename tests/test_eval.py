"""Tests for evaluation metrics."""

import torch
from molt.config import MOLTConfig
from molt.eval import compute_jacobian, compute_l0, compute_nmse, evaluate_molt
from molt.model import MOLT


def test_compute_jacobian():
    """Test Jacobian computation on a known linear function."""
    W = torch.randn(32, 32)

    def linear_fn(x):
        return x @ W.T

    x = torch.randn(4, 32)
    jac = compute_jacobian(linear_fn, x)
    assert jac.shape == (4, 32, 32)

    # For a linear function, Jacobian should be W at every point
    for i in range(4):
        assert torch.allclose(jac[i], W, atol=1e-5)


def test_compute_l0():
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)
    x = torch.randn(32, 64)
    l0 = compute_l0(model, x, batch_size=16)
    assert 0 <= l0 <= config.total_transforms


def test_compute_nmse():
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)
    x = torch.randn(32, 64)
    target = torch.randn(32, 64)
    nmse = compute_nmse(model, x, target, batch_size=16)
    assert nmse > 0


def test_evaluate_molt():
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)
    x = torch.randn(32, 64)
    target = torch.randn(32, 64)

    results = evaluate_molt(model, x, target)
    assert "l0" in results
    assert "nmse" in results
