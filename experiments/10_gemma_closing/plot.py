#!/usr/bin/env python3
"""Generate figures for experiment 10."""

import json
from pathlib import Path

from molt.utils.plotting import plot_multi_run_curves, plot_training_curves

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SETUPS = [
    "baseline_random_init", "orthogonal_encoders", "pca_encoders",
    "gate_freeze_10pct", "gate_freeze_25pct", "gate_freeze_50pct",
    "orthogonal_freeze_25pct", "pca_freeze_25pct",
]


def main():
    all_histories = {}
    all_results = []
    for name in SETUPS:
        hist_path = RESULTS_DIR / f"history_{name}.json"
        res_path = RESULTS_DIR / f"result_{name}.json"
        if hist_path.exists():
            with open(hist_path) as f:
                all_histories[name] = json.load(f)
        if res_path.exists():
            with open(res_path) as f:
                all_results.append(json.load(f))

    # 1. All runs comparison
    plot_multi_run_curves(
        all_histories,
        "Exp 10: Gemma Gate Collapse — All Setups",
        FIGURES_DIR / "comparison_all.png",
    )

    # 2. Init-only comparison
    init_runs = {k: v for k, v in all_histories.items()
                 if k in ["baseline_random_init", "orthogonal_encoders", "pca_encoders"]}
    plot_multi_run_curves(
        init_runs,
        "Exp 10: Encoder Initialization Comparison",
        FIGURES_DIR / "comparison_init.png",
    )

    # 3. Gate freeze comparison
    freeze_runs = {k: v for k, v in all_histories.items()
                   if k in ["baseline_random_init", "gate_freeze_10pct",
                            "gate_freeze_25pct", "gate_freeze_50pct"]}
    plot_multi_run_curves(
        freeze_runs,
        "Exp 10: Gate Freezing Comparison",
        FIGURES_DIR / "comparison_freeze.png",
    )

    # 4. Combined interventions
    combined_runs = {k: v for k, v in all_histories.items()
                     if k in ["baseline_random_init", "orthogonal_freeze_25pct",
                              "pca_freeze_25pct"]}
    plot_multi_run_curves(
        combined_runs,
        "Exp 10: Combined Interventions",
        FIGURES_DIR / "comparison_combined.png",
    )

    # 5. Per-setup individual training curves
    for name, hist in all_histories.items():
        plot_training_curves(hist, f"Exp 10: {name}", FIGURES_DIR / f"curves_{name}.png")

    # Summary table
    print(f"\n{'Setup':<30} {'Init':<12} {'Freeze':>7} {'L0':>6} {'NMSE':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<30} {r['init']:<12} {r['gate_freeze_frac']:>6.0%} "
              f"{r['l0']:>6.2f} {r['nmse']:>8.4f}")


if __name__ == "__main__":
    main()
