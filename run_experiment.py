"""Run MOLT experiments across sparsity coefficients and plot Pareto frontier."""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from molt import MOLT, MOLTConfig
from train import collect_mlp_data


def train_molt(mlp_inputs, mlp_outputs, config, sparsity_coeff, device,
               lr=3e-4, batch_size=256, n_epochs=15):
    molt = MOLT(config).to(device)
    optimizer = torch.optim.AdamW(molt.parameters(), lr=lr)

    dataset = TensorDataset(mlp_inputs.to(device), mlp_outputs.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    baseline_mse = mlp_outputs.pow(2).mean().item()

    history = []
    for epoch in range(n_epochs):
        epoch_mse, epoch_l0, epoch_steps = 0.0, 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            y_hat, aux = molt(x)
            mse = F.mse_loss(y_hat, y)
            sparsity = molt.sparsity_loss(aux["activations"], aux["frob_norms"])
            loss = mse + sparsity_coeff * sparsity
            loss.backward()
            optimizer.step()
            epoch_mse += mse.item()
            epoch_l0 += aux["l0"].item()
            epoch_steps += 1

        avg_mse = epoch_mse / epoch_steps
        avg_l0 = epoch_l0 / epoch_steps
        ev = 1.0 - avg_mse / baseline_mse
        history.append({"epoch": epoch + 1, "mse": avg_mse, "l0": avg_l0, "explained_var": ev})
        print(f"  epoch {epoch+1:2d} | mse {avg_mse:.6f} | L0 {avg_l0:.1f} | EV {ev:.4f}")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Collecting MLP data from GPT-2 layer 6...")
    mlp_inputs, mlp_outputs = collect_mlp_data(
        model_name="gpt2", layer_idx=6,
        max_samples=2048, seq_len=128, device=device,
    )
    d_model = mlp_inputs.shape[-1]
    baseline_mse = mlp_outputs.pow(2).mean().item()

    sparsity_coeffs = [0.0, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    all_results = {}

    for sc in sparsity_coeffs:
        print(f"\n{'='*60}")
        print(f"Training with sparsity_coeff={sc}")
        print(f"{'='*60}")
        config = MOLTConfig(d_model=d_model, base_n=4)
        history = train_molt(mlp_inputs, mlp_outputs, config, sc, device, n_epochs=15)
        all_results[str(sc)] = history

    # Save results
    with open("results.json", "w") as f:
        json.dump({"baseline_mse": baseline_mse, "runs": all_results}, f, indent=2)
    print("\nSaved results to results.json")

    # Plot
    plot_results(all_results, baseline_mse)


def plot_results(all_results, baseline_mse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Plot 1: MSE over training ---
    ax = axes[0]
    for sc, history in all_results.items():
        epochs = [h["epoch"] for h in history]
        mses = [h["mse"] for h in history]
        ax.plot(epochs, mses, marker="o", markersize=3, label=f"λ={sc}")
    ax.axhline(baseline_mse, color="gray", linestyle="--", alpha=0.5, label="baseline (zero)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error over Training")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: L0 over training ---
    ax = axes[1]
    for sc, history in all_results.items():
        epochs = [h["epoch"] for h in history]
        l0s = [h["l0"] for h in history]
        ax.plot(epochs, l0s, marker="o", markersize=3, label=f"λ={sc}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L0 (active transforms)")
    ax.set_title("Sparsity (L0) over Training")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Pareto frontier (L0 vs MSE at final epoch) ---
    ax = axes[2]
    final_l0s, final_mses, labels = [], [], []
    for sc, history in all_results.items():
        final = history[-1]
        final_l0s.append(final["l0"])
        final_mses.append(final["mse"])
        labels.append(f"λ={sc}")
    ax.scatter(final_l0s, final_mses, s=60, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (final_l0s[i], final_mses[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    # Connect points for Pareto frontier visualization
    sorted_pts = sorted(zip(final_l0s, final_mses))
    ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
            "b--", alpha=0.4)
    ax.set_xlabel("L0 (active transforms)")
    ax.set_ylabel("MSE")
    ax.set_title("Pareto Frontier: Sparsity vs Reconstruction")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("molt_results.png", dpi=150)
    print("Saved plot to molt_results.png")


if __name__ == "__main__":
    main()
