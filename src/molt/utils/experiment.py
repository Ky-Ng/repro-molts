"""Shared experiment runner that eliminates boilerplate across experiment scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from molt.config import MOLTConfig
from molt.eval import compute_l0, compute_nmse
from molt.train import train_molt


class ExperimentRunner:
    """Manages the train -> eval -> save -> cleanup cycle for MOLT experiments.

    Usage:
        runner = ExperimentRunner(Path(__file__).parent)
        for setup in SETUPS:
            config = MOLTConfig(...)
            result = runner.run_config(setup["name"], config, mlp_inputs, mlp_outputs)
        runner.save_summary()
    """

    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / "results"
        self.figures_dir = self.experiment_dir / "figures"
        self.logs_dir = self.experiment_dir / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: list[dict] = []

    def run_config(
        self,
        name: str,
        config: MOLTConfig,
        mlp_inputs: torch.Tensor,
        mlp_outputs: torch.Tensor,
        eval_size: int = 10_000,
    ) -> dict:
        """Train a MOLT, evaluate it, save results, and clean up GPU memory.

        Args:
            name: identifier for this run (used in filenames)
            config: MOLT configuration
            mlp_inputs: (N, d_model) MLP input activations
            mlp_outputs: (N, d_model) MLP output activations
            eval_size: number of samples for evaluation (taken from end)

        Returns:
            result dict with metrics and metadata
        """
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        print(f"  activation={config.activation}, sparsity={config.sparsity_type}, "
              f"learned_threshold={config.learned_threshold}, lambda={config.sparsity_coeff}")

        start_time = time.time()
        model, history = train_molt(config, mlp_inputs, mlp_outputs)
        training_time = time.time() - start_time

        # Evaluate
        eval_in = mlp_inputs[-eval_size:]
        eval_out = mlp_outputs[-eval_size:]
        l0 = compute_l0(model, eval_in)
        nmse = compute_nmse(model, eval_in, eval_out)

        # Threshold (if learned)
        final_threshold = model.threshold.item() if model.threshold is not None else None

        print(f"  Final -- L0: {l0:.2f}, NMSE: {nmse:.4f}, time: {training_time:.0f}s"
              + (f", theta: {final_threshold:.4f}" if final_threshold is not None else ""))

        # Per-transform activity frequencies
        active_transforms = self.compute_transform_activity(model, config, eval_in)

        result = {
            "name": name,
            "activation": config.activation,
            "sparsity_type": config.sparsity_type,
            "learned_threshold": config.learned_threshold,
            "sparsity_coeff": config.sparsity_coeff,
            "l0": round(l0, 2),
            "nmse": round(nmse, 4),
            "final_threshold": round(final_threshold, 4) if final_threshold is not None else None,
            "num_active": len(active_transforms),
            "active_transforms": active_transforms,
            "training_time_s": round(training_time, 1),
        }

        # Save per-run files
        with open(self.results_dir / f"result_{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(self.results_dir / f"history_{name}.json", "w") as f:
            json.dump(history, f, indent=2)

        self.all_results.append({**result, "history": history})

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return result

    def compute_transform_activity(
        self,
        model,
        config: MOLTConfig,
        eval_inputs: torch.Tensor,
        sample_size: int = 512,
    ) -> list[dict]:
        """Compute per-transform activation frequencies.

        Args:
            model: trained MOLT
            config: MOLT configuration
            eval_inputs: evaluation input activations
            sample_size: number of samples to use

        Returns:
            list of dicts with transform index, rank, and frequency (%)
        """
        active_transforms = []
        with torch.no_grad():
            x = eval_inputs[:sample_size].to(next(model.parameters()).device)
            _, aux = model(x)
            all_gates = torch.cat(aux["gate_acts"], dim=1)
            freq = (all_gates > 0).float().mean(dim=0)

            cumulative = 0
            for count, rank in config.rank_distribution:
                for j in range(count):
                    f = freq[cumulative].item()
                    if f > 0.001:
                        active_transforms.append({
                            "transform": cumulative,
                            "rank": rank,
                            "frequency": round(f * 100, 1),
                        })
                    cumulative += 1

        return active_transforms

    def save_summary(self, title: str = "Experiment Summary") -> None:
        """Save combined sweep_results.json and print a summary table."""
        summary = [{k: v for k, v in r.items() if k != "history"} for r in self.all_results]
        with open(self.results_dir / "sweep_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.print_summary_table(summary, title)

    def print_summary_table(self, results: list[dict], title: str) -> None:
        """Print a formatted summary table of experiment results."""
        print(f"\n{'='*80}")
        print(title)
        print(f"{'='*80}")
        print(f"{'Setup':<35} {'lambda':>8} {'L0':>6} {'NMSE':>8} {'theta':>8} {'#Act':>5}")
        print(f"{'-'*80}")
        for r in results:
            theta_str = f"{r['final_threshold']:.4f}" if r.get("final_threshold") is not None else " fixed"
            print(f"{r['name']:<35} {r['sparsity_coeff']:>8.0e} {r['l0']:>6.2f} "
                  f"{r['nmse']:>8.4f} {theta_str:>8} {r.get('num_active', 0):>5}")
