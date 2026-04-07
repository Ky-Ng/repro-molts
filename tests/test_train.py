"""Tests for training loop and checkpoint save/load."""

import torch

from molt.config import MOLTConfig
from molt.model import MOLT
from molt.train import train_molt, load_molt


def test_train_molt_smoke(tmp_path):
    """Smoke test: train for a few steps on tiny data."""
    config = MOLTConfig(
        d_model=64,
        rank_multiplier=1,
        device="cpu",
        batch_size=8,
        num_epochs=1,
        log_every=1,
        save_dir=str(tmp_path),
        sparsity_coeff=1e-3,
        sparsity_warmup_frac=0.1,
    )
    inputs = torch.randn(32, 64)
    outputs = torch.randn(32, 64)

    model, history = train_molt(config, inputs, outputs)

    assert isinstance(model, MOLT)
    assert len(history) > 0
    assert "mse" in history[0]
    assert "l0" in history[0]
    assert "step" in history[0]


def test_train_molt_saves_checkpoint(tmp_path):
    """Verify checkpoint and history files are saved."""
    config = MOLTConfig(
        d_model=64,
        rank_multiplier=1,
        device="cpu",
        batch_size=16,
        num_epochs=1,
        log_every=1,
        save_dir=str(tmp_path),
    )
    inputs = torch.randn(32, 64)
    outputs = torch.randn(32, 64)

    train_molt(config, inputs, outputs)

    # Check files were created
    pt_files = list(tmp_path.glob("*.pt"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(pt_files) == 1
    assert len(json_files) == 1


def test_train_molt_save_dir_override(tmp_path):
    """Test that save_dir parameter overrides config.save_dir."""
    override_dir = tmp_path / "override"
    config = MOLTConfig(
        d_model=64,
        rank_multiplier=1,
        device="cpu",
        batch_size=16,
        num_epochs=1,
        log_every=1,
        save_dir=str(tmp_path / "wrong"),
    )
    inputs = torch.randn(32, 64)
    outputs = torch.randn(32, 64)

    train_molt(config, inputs, outputs, save_dir=str(override_dir))

    assert len(list(override_dir.glob("*.pt"))) == 1


def test_load_molt_roundtrip(tmp_path):
    """Test save → load roundtrip preserves model outputs."""
    config = MOLTConfig(
        d_model=64,
        rank_multiplier=1,
        device="cpu",
        batch_size=16,
        num_epochs=1,
        log_every=1,
        save_dir=str(tmp_path),
    )
    inputs = torch.randn(32, 64)
    outputs = torch.randn(32, 64)

    model, _ = train_molt(config, inputs, outputs)

    # Get predictions before save
    with torch.no_grad():
        x = inputs[:4]
        expected, _ = model(x)

    # Load from checkpoint
    ckpt_path = list(tmp_path.glob("*.pt"))[0]
    loaded_model, loaded_config = load_molt(str(ckpt_path), device="cpu")

    # Verify config restored
    assert loaded_config.d_model == 64
    assert loaded_config.rank_multiplier == 1

    # Verify predictions match
    with torch.no_grad():
        actual, _ = loaded_model(x)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_load_molt_ignores_unknown_keys(tmp_path):
    """Test that load_molt handles checkpoints with extra keys (via from_dict)."""
    config = MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")
    model = MOLT(config)

    # Save with extra keys in config
    config_dict = vars(config)
    config_dict["fake_field"] = True
    ckpt_path = tmp_path / "test.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config_dict,
    }, ckpt_path)

    # Should load without error
    loaded_model, loaded_config = load_molt(str(ckpt_path), device="cpu")
    assert loaded_config.d_model == 64
