"""Transcoder baseline: load pretrained transcoders and train from scratch."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap


def load_transcoder(
    release: str,
    sae_id: str,
    device: str = "cuda",
):
    """Load a Gemma Scope 2 transcoder via sae-lens.

    Example releases:
        - "gemma-scope-2-1b-pt-transcoder" (non-skip)
        - "gemma-scope-2-1b-pt-transcoder-skip" (skip)

    Args:
        release: sae-lens release name
        sae_id: specific SAE/transcoder ID within the release
        device: torch device

    Returns:
        transcoder: loaded SAE/transcoder object
    """
    from sae_lens import SAE

    transcoder, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    transcoder.eval()
    return transcoder, cfg_dict


def transcoder_forward(transcoder, x: torch.Tensor) -> torch.Tensor:
    """Run transcoder forward pass.

    For a standard transcoder: encode input -> decode to output space.
    For a skip transcoder: adds a skip connection from input.

    Args:
        transcoder: loaded sae-lens transcoder
        x: (batch, d_model) MLP input activations

    Returns:
        output: (batch, d_model) transcoder reconstruction
    """
    # sae-lens transcoders have encode() and decode() methods
    feature_acts = transcoder.encode(x)
    output = transcoder.decode(feature_acts)
    return output


def transcoder_l0(transcoder, x: torch.Tensor, batch_size: int = 256) -> float:
    """Compute average L0 for transcoder (number of active features per token)."""
    total_l0 = 0.0
    count = 0

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(next(transcoder.parameters()).device)
            feature_acts = transcoder.encode(batch)
            l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
            total_l0 += l0 * len(batch)
            count += len(batch)

    return total_l0 / count


def transcoder_nmse(
    transcoder, x: torch.Tensor, target: torch.Tensor, batch_size: int = 256
) -> float:
    """Compute normalized MSE for transcoder."""
    total_mse = 0.0
    count = 0
    device = next(transcoder.parameters()).device

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            bx = x[i : i + batch_size].to(device)
            bt = target[i : i + batch_size].to(device)
            output = transcoder_forward(transcoder, bx)
            mse = F.mse_loss(output, bt, reduction="sum").item()
            total_mse += mse
            count += bt.numel()

    mean_mse = total_mse / count * target.shape[1]
    target_var = target.var().item()
    return mean_mse / (target_var + 1e-8)


def transcoder_jacobian_faithfulness(
    transcoder,
    mlp_fn: Callable,
    x: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute Jacobian faithfulness for a transcoder.

    Args:
        transcoder: loaded transcoder
        mlp_fn: true MLP function (d_model,) -> (d_model,)
        x: (N, d_model) input activations (requires_grad=True)
        batch_size: batch for Jacobian computation

    Returns:
        cosine_similarities: (N,) per-input faithfulness scores
    """

    def tc_fn(xi):
        return transcoder_forward(transcoder, xi.unsqueeze(0)).squeeze(0)

    all_sims = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]

        j_tc = vmap(jacrev(tc_fn))(batch)
        j_mlp = vmap(jacrev(mlp_fn))(batch)

        j_tc_flat = j_tc.flatten(1)
        j_mlp_flat = j_mlp.flatten(1)

        sim = F.cosine_similarity(j_tc_flat, j_mlp_flat, dim=1)
        all_sims.append(sim.detach().cpu())

    return torch.cat(all_sims)


def evaluate_transcoder(
    release: str,
    sae_id: str,
    x: torch.Tensor,
    target: torch.Tensor,
    mlp_fn: Callable | None = None,
    jacobian_samples: int = 64,
    device: str = "cuda",
    label: str = "Transcoder",
) -> dict[str, float]:
    """Full evaluation of a transcoder baseline.

    Args:
        release: sae-lens release name
        sae_id: transcoder ID
        x: (N, d_model) MLP inputs
        target: (N, d_model) MLP outputs
        mlp_fn: optional for Jacobian eval
        jacobian_samples: number of samples for Jacobian
        device: torch device
        label: display label for plots

    Returns:
        dict with l0, nmse, optionally jacobian_cosine_sim, and label
    """
    transcoder, _ = load_transcoder(release, sae_id, device)

    results = {
        "l0": transcoder_l0(transcoder, x),
        "nmse": transcoder_nmse(transcoder, x, target),
        "label": label,
    }

    if mlp_fn is not None:
        jac_x = x[:jacobian_samples].to(device).requires_grad_(True)
        sims = transcoder_jacobian_faithfulness(transcoder, mlp_fn, jac_x)
        results["jacobian_cosine_sim"] = sims.mean().item()
        results["jacobian_cosine_sim_std"] = sims.std().item()

    # Free memory
    del transcoder
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Trainable transcoder
# ---------------------------------------------------------------------------


class TrainableTranscoder(nn.Module):
    """Simple transcoder: ReLU encoder -> linear decoder.

    Maps MLP input (d_model) -> sparse features (n_features) -> MLP output (d_model).
    Sparsity is controlled via L1 penalty on feature activations.
    """

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.W_enc = nn.Parameter(torch.empty(n_features, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(d_model, n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for both encoder and decoder
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, d_model) -> (batch, n_features)"""
        return F.relu(x @ self.W_enc.T + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, n_features) -> (batch, d_model)"""
        return h @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass returning output and auxiliary info.

        Returns:
            output: (batch, d_model)
            aux: dict with l0, feature_acts, sparsity_loss
        """
        h = self.encode(x)
        output = self.decode(h)
        l0 = (h > 0).float().sum(dim=-1).mean()
        # L1 sparsity on feature activations, normalized by n_features
        sparsity_loss = h.abs().mean()
        return output, {"l0": l0, "feature_acts": h, "sparsity_loss": sparsity_loss}

    def loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        sparsity_coeff: float = 0.0,
        sparsity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute MSE + lambda * L1 sparsity loss.

        Returns:
            total_loss: scalar
            metrics: dict with mse, nmse, l0, sparsity_loss, total_loss
        """
        output, aux = self.forward(x)
        mse = F.mse_loss(output, target)
        sparsity = aux["sparsity_loss"]
        effective_lambda = sparsity_coeff * sparsity_scale
        total = mse + effective_lambda * sparsity

        target_var = target.var()
        nmse = mse / (target_var + 1e-8)

        metrics = {
            "mse": mse.detach(),
            "nmse": nmse.detach(),
            "sparsity_loss": sparsity.detach(),
            "l0": aux["l0"].detach(),
            "total_loss": total.detach(),
        }
        return total, metrics

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_transcoder(
    d_model: int,
    n_features: int,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    sparsity_coeff: float = 0.0,
    sparsity_warmup_frac: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 1,
    device: str = "cuda",
    log_every: int = 100,
    seed: int = 42,
) -> tuple[TrainableTranscoder, list[dict]]:
    """Train a transcoder on cached MLP activations.

    Returns:
        model: trained TrainableTranscoder
        history: list of metric dicts
    """
    from molt.data import make_dataloader

    torch.manual_seed(seed)
    model = TrainableTranscoder(d_model, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = make_dataloader(mlp_inputs, mlp_outputs, batch_size)

    history: list[dict] = []
    step = 0
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * sparsity_warmup_frac)

    for epoch in range(num_epochs):
        from tqdm import tqdm

        pbar = tqdm(dataloader, desc=f"Transcoder epoch {epoch + 1}/{num_epochs}")
        for batch_inputs, batch_targets in pbar:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            sparsity_scale = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0

            optimizer.zero_grad()
            loss, metrics = model.loss(
                batch_inputs, batch_targets, sparsity_coeff, sparsity_scale
            )
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_every == 0:
                log = {k: v.item() for k, v in metrics.items()}
                log["step"] = step
                log["epoch"] = epoch
                history.append(log)
                pbar.set_postfix(
                    mse=f"{log['mse']:.4f}",
                    nmse=f"{log['nmse']:.4f}",
                    l0=f"{log['l0']:.1f}",
                )

    return model, history


def evaluate_trainable_transcoder(
    model: TrainableTranscoder,
    x: torch.Tensor,
    target: torch.Tensor,
    batch_size: int = 256,
) -> dict:
    """Evaluate a trained transcoder: L0 and NMSE."""
    device = next(model.parameters()).device
    total_l0 = 0.0
    total_mse = 0.0
    count = 0
    numel = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            bx = x[i : i + batch_size].to(device)
            bt = target[i : i + batch_size].to(device)
            output, aux = model.forward(bx)
            total_l0 += aux["l0"].item() * len(bx)
            total_mse += F.mse_loss(output, bt, reduction="sum").item()
            count += len(bx)
            numel += bt.numel()

    mean_mse = total_mse / numel
    target_var = target.var().item()
    return {
        "l0": total_l0 / count,
        "nmse": mean_mse / (target_var + 1e-8),
    }
