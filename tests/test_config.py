"""Tests for MOLTConfig."""

from molt.config import MOLTConfig


def test_from_preset_gemma():
    config = MOLTConfig.from_preset("google/gemma-3-1b-it")
    assert config.d_model == 1152
    assert config.layer_idx == 13
    assert config.model_dtype == "bfloat16"


def test_from_preset_gpt2():
    config = MOLTConfig.from_preset("openai-community/gpt2")
    assert config.d_model == 768
    assert config.layer_idx == 6
    assert config.model_dtype == "float32"


def test_from_preset_with_overrides():
    config = MOLTConfig.from_preset("openai-community/gpt2", lr=5e-4, batch_size=128)
    assert config.d_model == 768
    assert config.lr == 5e-4
    assert config.batch_size == 128


def test_from_preset_unknown_model():
    config = MOLTConfig.from_preset("unknown/model", d_model=512)
    assert config.model_name == "unknown/model"
    assert config.d_model == 512


def test_from_dict_basic():
    d = {"d_model": 256, "lr": 5e-4, "activation": "relu"}
    config = MOLTConfig.from_dict(d)
    assert config.d_model == 256
    assert config.lr == 5e-4
    assert config.activation == "relu"


def test_from_dict_ignores_unknown_keys():
    d = {"d_model": 256, "unknown_field": True, "another_fake": 42}
    config = MOLTConfig.from_dict(d)
    assert config.d_model == 256
    assert not hasattr(config, "unknown_field")


def test_from_dict_empty():
    config = MOLTConfig.from_dict({})
    assert config.d_model == 1152  # default


def test_rank_distribution_filters_large_ranks():
    config = MOLTConfig(d_model=64, rank_multiplier=1)
    # d_model=64 means only rank<=64 groups survive: 8×64 and 16×32
    assert config.rank_distribution == [(8, 64), (16, 32)]
    assert config.total_transforms == 24


def test_rank_distribution_n2():
    config = MOLTConfig(rank_multiplier=2)
    assert config.total_transforms == 62
    # Verify counts double
    for count, rank in config.rank_distribution:
        assert count % 2 == 0


def test_default_save_dir():
    config = MOLTConfig()
    assert config.save_dir == "data"
