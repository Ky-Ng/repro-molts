"""Shared test fixtures for MOLT tests."""

import pytest
import torch

from molt.config import MOLTConfig
from molt.model import MOLT


@pytest.fixture
def small_config():
    """A small MOLTConfig for fast testing (d_model=64, CPU)."""
    return MOLTConfig(d_model=64, rank_multiplier=1, device="cpu")


@pytest.fixture
def small_model(small_config):
    """A small MOLT model for testing."""
    return MOLT(small_config)


@pytest.fixture
def sample_tensors():
    """Sample input/output tensors for testing (32 tokens, d_model=64)."""
    torch.manual_seed(42)
    return torch.randn(32, 64), torch.randn(32, 64)


@pytest.fixture
def sample_activations(tmp_path):
    """Create a temporary activation cache file and return its path."""
    torch.manual_seed(42)
    mlp_inputs = torch.randn(100, 64)
    mlp_outputs = torch.randn(100, 64)
    cache_path = tmp_path / "activations.pt"
    torch.save({"mlp_inputs": mlp_inputs, "mlp_outputs": mlp_outputs}, cache_path)
    return cache_path, mlp_inputs, mlp_outputs
