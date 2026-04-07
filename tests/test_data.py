"""Tests for data pipeline utilities."""

import torch

from molt.config import MOLTConfig
from molt.data import make_dataloader, _resolve_mlp_module


def test_make_dataloader():
    inputs = torch.randn(100, 64)
    outputs = torch.randn(100, 64)
    dl = make_dataloader(inputs, outputs, batch_size=16)
    batch_in, batch_out = next(iter(dl))
    assert batch_in.shape == (16, 64)
    assert batch_out.shape == (16, 64)


def test_make_dataloader_last_batch():
    inputs = torch.randn(50, 64)
    outputs = torch.randn(50, 64)
    dl = make_dataloader(inputs, outputs, batch_size=16, shuffle=False)
    batches = list(dl)
    # 50 / 16 = 3 full batches + 1 partial
    assert len(batches) == 4
    assert batches[-1][0].shape[0] == 2  # 50 - 3*16 = 2


def test_make_dataloader_no_shuffle():
    inputs = torch.arange(20).float().unsqueeze(1)
    outputs = torch.arange(20).float().unsqueeze(1)
    dl = make_dataloader(inputs, outputs, batch_size=10, shuffle=False)
    batch_in, _ = next(iter(dl))
    assert torch.equal(batch_in, inputs[:10])


class MockModel:
    """Mock model with nested attribute access for testing _resolve_mlp_module."""
    class MLP:
        pass

    class Layer:
        def __init__(self):
            self.mlp = MockModel.MLP()

    class Layers:
        def __init__(self):
            self._layers = [MockModel.Layer() for _ in range(26)]

        def __getitem__(self, idx):
            return self._layers[idx]

    class Model:
        def __init__(self):
            self.layers = MockModel.Layers()

    def __init__(self):
        self.model = self.Model()


def test_resolve_mlp_module():
    config = MOLTConfig(mlp_path="model.layers.{layer_idx}.mlp", layer_idx=13)
    mock = MockModel()
    mlp = _resolve_mlp_module(mock, config)
    assert isinstance(mlp, MockModel.MLP)


def test_resolve_mlp_module_fallback():
    config = MOLTConfig(mlp_path="", layer_idx=0)
    mock = MockModel()
    mlp = _resolve_mlp_module(mock, config)
    assert isinstance(mlp, MockModel.MLP)
