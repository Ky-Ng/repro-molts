"""Tests for shared utilities (plotting, experiment runner, activations)."""

import json

import torch

from molt.utils.activations import load_cached_activations, split_train_eval
from molt.utils.plotting import plot_training_curves, plot_multi_run_curves, plot_l0_vs_nmse


# --- Activations ---

def test_load_cached_activations(sample_activations):
    cache_path, expected_inputs, expected_outputs = sample_activations
    inputs, outputs = load_cached_activations(cache_path)
    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(outputs, expected_outputs)


def test_split_train_eval():
    inputs = torch.randn(100, 64)
    outputs = torch.randn(100, 64)
    train_in, train_out, eval_in, eval_out = split_train_eval(inputs, outputs, eval_size=20)
    assert train_in.shape == (80, 64)
    assert train_out.shape == (80, 64)
    assert eval_in.shape == (20, 64)
    assert eval_out.shape == (20, 64)
    # Eval comes from end
    assert torch.equal(eval_in, inputs[-20:])


# --- Plotting ---

def _make_history(n_steps=10):
    return [
        {"step": i * 100, "mse": 1.0 / (i + 1), "nmse": 0.5 / (i + 1),
         "l0": 5.0 + i * 0.5, "sparsity_loss": 0.1 * i}
        for i in range(n_steps)
    ]


def test_plot_training_curves(tmp_path):
    history = _make_history()
    save_path = tmp_path / "curves.png"
    plot_training_curves(history, "Test Run", save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_multi_run_curves(tmp_path):
    runs = {"run_a": _make_history(), "run_b": _make_history()}
    save_path = tmp_path / "multi.png"
    plot_multi_run_curves(runs, "Multi Run", save_path)
    assert save_path.exists()


def test_plot_l0_vs_nmse(tmp_path):
    results = [{"l0": 1.0, "nmse": 0.5}, {"l0": 5.0, "nmse": 0.1}, {"l0": 10.0, "nmse": 0.05}]
    save_path = tmp_path / "pareto.png"
    plot_l0_vs_nmse(results, save_path)
    assert save_path.exists()


def test_plot_l0_vs_nmse_with_transcoder(tmp_path):
    results = [{"l0": 1.0, "nmse": 0.5}]
    tc_results = [{"l0": 3.0, "nmse": 0.2, "label": "TC-64"}]
    save_path = tmp_path / "pareto_tc.png"
    plot_l0_vs_nmse(results, save_path, transcoder_results=tc_results)
    assert save_path.exists()


# --- Experiment Runner ---

def test_experiment_runner_dirs(tmp_path):
    from molt.utils.experiment import ExperimentRunner
    ExperimentRunner(tmp_path)
    assert (tmp_path / "results").is_dir()
    assert (tmp_path / "figures").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_experiment_runner_save_summary(tmp_path):
    from molt.utils.experiment import ExperimentRunner
    runner = ExperimentRunner(tmp_path)
    runner.all_results = [
        {"name": "test", "l0": 5.0, "nmse": 0.1, "sparsity_coeff": 1e-3,
         "final_threshold": None, "num_active": 3, "history": []},
    ]
    runner.save_summary(title="Test")
    summary_path = tmp_path / "results" / "sweep_results.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["name"] == "test"
    assert "history" not in data[0]
