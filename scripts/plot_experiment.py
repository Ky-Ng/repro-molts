#!/usr/bin/env python3
"""Plot training curves and summary charts for any experiment.

Reads history_*.json and result_*.json files from an experiment's results/ dir.

Usage:
    uv run python scripts/plot_experiment.py experiments/07_gpt2_strong_sparsity

    # Only plot specific configs (substring match on filename):
    uv run python scripts/plot_experiment.py experiments/07_gpt2_strong_sparsity --filter jumprelu
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_histories(results_dir: Path, name_filter: str = ""):
    """Load all history_*.json files, return {name: history_list}."""
    runs = {}
    for f in sorted(results_dir.glob("history_*.json")):
        name = f.stem.replace("history_", "")
        if name_filter and name_filter not in name:
            continue
        with open(f) as fh:
            runs[name] = json.load(fh)
    return runs


def load_results(results_dir: Path, name_filter: str = ""):
    """Load all result_*.json files, return {name: result_dict}."""
    results = {}
    for f in sorted(results_dir.glob("result_*.json")):
        name = f.stem.replace("result_", "")
        if name_filter and name_filter not in name:
            continue
        with open(f) as fh:
            results[name] = json.load(fh)
    return results


def plot_per_config(runs: dict, figures_dir: Path, results: dict | None = None):
    """One figure per config: 2-3 panel (L0, NMSE, theta if present)."""
    for name, history in runs.items():
        if not history:
            continue
        steps = [h["step"] for h in history]
        has_theta = "threshold" in history[0]
        n_panels = 3 if has_theta else 2

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))

        # Build descriptive title from result metadata if available
        r = (results or {}).get(name, {})
        title = name.replace("_", " ").title()
        subtitle_parts = []
        if "sparsity_coeff" in r:
            subtitle_parts.append(f"\u03bb={r['sparsity_coeff']}")
        if "sparsity_type" in r:
            subtitle_parts.append(r["sparsity_type"])
        if "activation" in r:
            subtitle_parts.append(r["activation"])
        if subtitle_parts:
            title += f"\n({', '.join(subtitle_parts)})"
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # L0
        axes[0].plot(steps, [h["l0"] for h in history], color="#2563eb", lw=1.5)
        axes[0].set_ylabel("L0 (active transforms)")
        axes[0].set_xlabel("Step")
        axes[0].set_title("L0")
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

        # NMSE
        axes[1].plot(steps, [h["nmse"] for h in history], color="#dc2626", lw=1.5)
        axes[1].set_ylabel("NMSE")
        axes[1].set_xlabel("Step")
        axes[1].set_title("Normalized MSE")
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

        # Theta (if learned)
        if has_theta:
            axes[2].plot(steps, [h["threshold"] for h in history], color="#9333ea", lw=1.5)
            axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            axes[2].set_ylabel("Threshold (theta)")
            axes[2].set_xlabel("Step")
            axes[2].set_title("Learned Threshold")
            axes[2].grid(True, alpha=0.3)
            axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

        plt.tight_layout()
        save_path = figures_dir / f"config_{name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_theta_vs_lambda(results: dict, figures_dir: Path):
    """Summary: theta_final vs lambda for each sparsity type (JumpReLU only)."""
    jumprelu = {k: v for k, v in results.items() if "jumprelu" in k}
    if not jumprelu:
        return

    # Group by sparsity type
    groups = {}
    for name, r in jumprelu.items():
        sp = r.get("sparsity_type", "unknown")
        if sp not in groups:
            groups[sp] = {"lambdas": [], "thetas": [], "l0s": []}
        groups[sp]["lambdas"].append(r["sparsity_coeff"])
        groups[sp]["thetas"].append(r.get("final_threshold", 0))
        groups[sp]["l0s"].append(r["l0"])

    colors = {"tanh": "#2563eb", "l0": "#dc2626"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("JumpReLU: Theta & L0 vs Lambda", fontsize=13, fontweight="bold")

    for sp, data in sorted(groups.items()):
        order = sorted(range(len(data["lambdas"])), key=lambda i: data["lambdas"][i])
        lams = [data["lambdas"][i] for i in order]
        thetas = [data["thetas"][i] for i in order]
        l0s = [data["l0s"][i] for i in order]
        c = colors.get(sp, "gray")

        axes[0].plot(lams, thetas, "o-", color=c, label=f"{sp} penalty", lw=2, ms=8)
        axes[1].plot(lams, l0s, "o-", color=c, label=f"{sp} penalty", lw=2, ms=8)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("Theta (final)")
    axes[0].set_title("Learned Threshold vs Lambda")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("L0 (final)")
    axes[1].set_title("Active Transforms vs Lambda")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_path = figures_dir / "theta_vs_lambda.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_l0_vs_nmse(results: dict, figures_dir: Path):
    """Scatter: final L0 vs NMSE, colored by config type."""
    markers = {"tanh_relu": ("o", "#2563eb"), "tanh_jumprelu": ("s", "#60a5fa"),
               "l0_relu": ("^", "#dc2626"), "l0_jumprelu": ("D", "#f87171")}

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, r in results.items():
        sp = r.get("sparsity_type", "")
        act = r.get("activation", "")
        key = f"{sp}_{act}"
        marker, color = markers.get(key, ("x", "gray"))
        lam = r.get("sparsity_coeff", 0)
        ax.scatter(r["l0"], r["nmse"], marker=marker, color=color, s=80, zorder=3)
        ax.annotate(f"{lam:.0e}", (r["l0"], r["nmse"]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), alpha=0.7)

    # Legend entries
    for key, (marker, color) in markers.items():
        label = key.replace("_", " + ").replace("tanh", "Tanh").replace("l0", "L0").replace("relu", "ReLU").replace("jump", "Jump")
        ax.scatter([], [], marker=marker, color=color, s=80, label=label)

    ax.set_xlabel("L0 (Active Transforms)")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("L0 vs NMSE — All Configurations")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = figures_dir / "l0_vs_nmse_scatter.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_overlay_by_lambda(runs: dict, results: dict, figures_dir: Path):
    """Overlay all 4 config types on shared axes, one figure per lambda."""
    # Group runs by lambda
    lambda_groups = {}
    for name, history in runs.items():
        r = results.get(name, {})
        lam = r.get("sparsity_coeff")
        if lam is None:
            # Try to parse from name
            m = re.search(r"lam([\d.e+-]+)", name)
            if m:
                lam = float(m.group(1))
        if lam is not None:
            lambda_groups.setdefault(lam, {})[name] = history

    colors = {"tanh_relu": "#2563eb", "tanh_jumprelu": "#60a5fa",
              "l0_relu": "#dc2626", "l0_jumprelu": "#f87171"}

    for lam, lam_runs in sorted(lambda_groups.items()):
        has_theta = any("threshold" in h[0] for h in lam_runs.values() if h)
        n_panels = 3 if has_theta else 2

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
        fig.suptitle(f"Lambda = {lam:.0e}", fontsize=13, fontweight="bold")

        for name, history in sorted(lam_runs.items()):
            if not history:
                continue
            steps = [h["step"] for h in history]
            # Determine color
            color = "gray"
            for key, c in colors.items():
                if key in name:
                    color = c
                    break
            label = name.replace(f"_lam{lam:.0e}", "").replace("_", " + ")

            axes[0].plot(steps, [h["l0"] for h in history], color=color, lw=1.5, label=label)
            axes[1].plot(steps, [h["nmse"] for h in history], color=color, lw=1.5, label=label)

            if has_theta and "threshold" in history[0]:
                axes[2].plot(steps, [h["threshold"] for h in history], color=color, lw=1.5, label=label)

        axes[0].set_ylabel("L0")
        axes[0].set_title("Active Transforms (L0)")
        axes[1].set_ylabel("NMSE")
        axes[1].set_title("Normalized MSE")
        if has_theta:
            axes[2].set_ylabel("Theta")
            axes[2].set_title("Learned Threshold")
            axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.4)

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlabel("Step")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

        plt.tight_layout()
        save_path = figures_dir / f"overlay_lam{lam:.0e}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment training curves and summaries")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment folder")
    parser.add_argument("--filter", type=str, default="", help="Substring filter on run names")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    results_dir = experiment_dir / "results"
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = load_histories(results_dir, args.filter)
    results = load_results(results_dir, args.filter)
    print(f"Loaded {len(runs)} histories, {len(results)} results from {results_dir}")

    if not runs:
        print("No history files found.")
        return

    # 1. Per-config plots (L0, NMSE, theta over steps)
    print("\n--- Per-config training curves ---")
    plot_per_config(runs, figures_dir, results)

    # 2. Overlay by lambda (all 4 configs on shared axes)
    print("\n--- Overlay by lambda ---")
    plot_overlay_by_lambda(runs, results, figures_dir)

    # 3. Theta vs lambda summary (JumpReLU only)
    print("\n--- Theta vs lambda summary ---")
    plot_theta_vs_lambda(results, figures_dir)

    # 4. L0 vs NMSE scatter
    print("\n--- L0 vs NMSE scatter ---")
    plot_l0_vs_nmse(results, figures_dir)

    print(f"\nAll plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
