"""Smoke tests for MOLT model."""

import torch
from molt.config import MOLTConfig
from molt.model import MOLT, TransformGroup, jumprelu


def test_jumprelu():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    out = jumprelu(x, threshold=0.5)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 0.0  # exactly at threshold, not >
    assert out[3] == 1.0
    assert out[4] == 2.0


def test_transform_group_shapes():
    group = TransformGroup(num_transforms=4, d_model=64, rank=16)
    x = torch.randn(8, 64)
    output, gate, frob = group.forward(x, activation_fn="relu")
    assert output.shape == (8, 64)
    assert gate.shape == (8, 4)
    assert frob.shape == (4,)


def test_molt_forward():
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)
    x = torch.randn(8, 64)
    output, aux = model(x)
    assert output.shape == (8, 64)
    assert "sparsity_loss" in aux
    assert "l0" in aux
    assert aux["l0"].item() >= 0


def test_molt_loss():
    config = MOLTConfig(d_model=64, rank_multiplier=1, sparsity_coeff=1e-3, device="cpu")
    model = MOLT(config)
    x = torch.randn(8, 64)
    target = torch.randn(8, 64)
    loss, metrics = model.loss(x, target)
    assert loss.shape == ()
    assert loss.requires_grad
    assert metrics["mse"].item() > 0
    assert metrics["l0"].item() >= 0

    # Check backward works
    loss.backward()
    for p in model.parameters():
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
    """Test a single training step end-to-end."""
    config = MOLTConfig(d_model=64, rank_multiplier=1, lr=1e-3, device="cpu")
    model = MOLT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x = torch.randn(16, 64)
    target = torch.randn(16, 64)

    # Step 1
    loss1, _ = model.loss(x, target)
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Step 2
    loss2, _ = model.loss(x, target)
    # Loss should generally decrease (not guaranteed but likely with small model)
    assert loss2.item() < loss1.item() * 2  # At least not exploding
