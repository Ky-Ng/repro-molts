"""Training script for MOLT.

Trains a MOLT to reconstruct the output of a transformer MLP layer,
following the methodology from:
https://transformer-circuits.pub/2025/bulk-update/index.html
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from molt import MOLT, MOLTConfig


def collect_mlp_data(
    model_name: str,
    layer_idx: int,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    max_samples: int = 4096,
    seq_len: int = 128,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (input, output) pairs from a transformer MLP layer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize data
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    texts = [ex["text"] for ex in dataset if ex["text"].strip()][:max_samples]
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=seq_len,
    )

    # Hook into the target MLP layer to capture inputs and outputs
    mlp_inputs, mlp_outputs = [], []
    mlp = model.transformer.h[layer_idx].mlp

    def hook_fn(module, inp, out):
        # inp is a tuple; first element is the hidden states
        mlp_inputs.append(inp[0].detach().cpu())
        mlp_outputs.append(out.detach().cpu())

    handle = mlp.register_forward_hook(hook_fn)

    # Run forward passes in batches
    batch_size = 16
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            batch_mask = attention_mask[i : i + batch_size].to(device)
            model(input_ids=batch_ids, attention_mask=batch_mask)

    handle.remove()

    # Flatten sequence dimension: (n_batches, batch, seq, d_model) -> (N, d_model)
    all_inputs = torch.cat(mlp_inputs, dim=0).reshape(-1, mlp_inputs[0].shape[-1])
    all_outputs = torch.cat(mlp_outputs, dim=0).reshape(-1, mlp_outputs[0].shape[-1])

    # Filter out padding positions (zero vectors)
    mask = all_inputs.norm(dim=-1) > 1e-6
    all_inputs = all_inputs[mask]
    all_outputs = all_outputs[mask]

    print(f"Collected {all_inputs.shape[0]} MLP (input, output) pairs "
          f"from layer {layer_idx} of {model_name}")

    return all_inputs, all_outputs


def train(
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    config: MOLTConfig,
    lr: float = 3e-4,
    sparsity_coeff: float = 1e-3,
    batch_size: int = 256,
    n_epochs: int = 10,
    device: str = "cpu",
    log_every: int = 50,
):
    """Train a MOLT to reconstruct MLP outputs."""
    molt = MOLT(config).to(device)
    optimizer = torch.optim.AdamW(molt.parameters(), lr=lr)

    dataset = TensorDataset(mlp_inputs.to(device), mlp_outputs.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Compute baseline MSE (predicting zero)
    baseline_mse = mlp_outputs.pow(2).mean().item()
    print(f"Baseline MSE (zero predictor): {baseline_mse:.6f}")
    print(f"MOLT config: {config.total_transforms} transforms, "
          f"ranks: {config.rank_distribution}")
    print(f"Training with lr={lr}, sparsity_coeff={sparsity_coeff}, "
          f"batch_size={batch_size}, n_epochs={n_epochs}")
    print("-" * 70)

    step = 0
    for epoch in range(n_epochs):
        epoch_mse = 0.0
        epoch_l0 = 0.0
        epoch_steps = 0

        for x, y in loader:
            optimizer.zero_grad()

            y_hat, aux = molt(x)

            # Reconstruction loss
            mse = F.mse_loss(y_hat, y)

            # Sparsity penalty: Σ_t ||U_t V_t||_F · φ(e_t · x - b_t)
            sparsity = molt.sparsity_loss(aux["activations"], aux["frob_norms"])

            loss = mse + sparsity_coeff * sparsity
            loss.backward()
            optimizer.step()

            epoch_mse += mse.item()
            epoch_l0 += aux["l0"].item()
            epoch_steps += 1
            step += 1

            if step % log_every == 0:
                explained_var = 1.0 - mse.item() / baseline_mse
                print(
                    f"step {step:5d} | "
                    f"mse {mse.item():.6f} | "
                    f"sparsity {sparsity.item():.6f} | "
                    f"L0 {aux['l0'].item():.1f}/{config.total_transforms} | "
                    f"explained_var {explained_var:.4f}"
                )

        avg_mse = epoch_mse / epoch_steps
        avg_l0 = epoch_l0 / epoch_steps
        explained_var = 1.0 - avg_mse / baseline_mse
        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"avg_mse {avg_mse:.6f} | "
            f"avg_L0 {avg_l0:.1f}/{config.total_transforms} | "
            f"explained_var {explained_var:.4f}"
        )
        print("-" * 70)

    return molt


def main():
    parser = argparse.ArgumentParser(description="Train MOLT on a transformer MLP layer")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--layer", type=int, default=6, help="MLP layer index to decompose")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--max-samples", type=int, default=2048, help="Max text samples")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--base-n", type=int, default=4, help="Base N for rank distribution")
    parser.add_argument("--nonlinearity", default="relu", choices=["relu", "jumprelu"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sparsity-coeff", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default=None, help="Path to save trained MOLT")
    args = parser.parse_args()

    # Collect MLP data
    mlp_inputs, mlp_outputs = collect_mlp_data(
        model_name=args.model,
        layer_idx=args.layer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        device=args.device,
    )

    d_model = mlp_inputs.shape[-1]
    config = MOLTConfig(
        d_model=d_model,
        base_n=args.base_n,
        nonlinearity=args.nonlinearity,
    )

    molt = train(
        mlp_inputs=mlp_inputs,
        mlp_outputs=mlp_outputs,
        config=config,
        lr=args.lr,
        sparsity_coeff=args.sparsity_coeff,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=args.device,
    )

    if args.save:
        torch.save({"config": config, "state_dict": molt.state_dict()}, args.save)
        print(f"Saved MOLT to {args.save}")


if __name__ == "__main__":
    main()
