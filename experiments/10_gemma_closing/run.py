#!/usr/bin/env python3
"""Experiment 10: Investigating Gemma-3-1B gate collapse mechanisms.

Tests two interventions to prevent L0 collapse:

1. Diverse encoder init: orthogonal encoders (via QR) and PCA-informed encoders
   so transforms start with different "views" of the input space.

2. Gate freezing: force all gates to 1.0 for the first N% of training steps,
   giving UV matrices time to learn useful directions before gating competition
   begins. This is distinct from exp 08's threshold freezing — here the gate
   output is literally 1.0, not just a frozen threshold.

Both interventions target the hypothesis that transforms die before they learn
because (a) random encoders in d=1152 are too similar and (b) gating dynamics
kill losers faster than UV can specialize.

Usage:
    uv run python experiments/10_gemma_closing/run.py [setup_name]
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from molt.config import MOLTConfig
from molt.data import make_dataloader
from molt.eval import compute_l0, compute_nmse
from molt.model import MOLT

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
CACHE_PATH = "data/activations_2M.pt"

BASE_CONFIG = dict(
    model_name="google/gemma-3-1b-it",
    d_model=1152,
    layer_idx=13,
    num_tokens=2_000_000,
    batch_size=4096,
    lr=1e-3,
    device="cuda",
    sparsity_coeff=0.0,
    sparsity_warmup_frac=0.0,
    log_every=10,
    activation="jumprelu",
    learned_threshold=False,
    jumprelu_threshold=0.0,
)

SETUPS = [
    # Baseline: reproduce exp 08 fixed_theta_surrogate for comparison
    {"name": "baseline_random_init"},
    # --- Diverse initialization ---
    {"name": "orthogonal_encoders", "init": "orthogonal"},
    {"name": "pca_encoders", "init": "pca"},
    # --- Gate freezing ---
    {"name": "gate_freeze_10pct", "gate_freeze_frac": 0.10},
    {"name": "gate_freeze_25pct", "gate_freeze_frac": 0.25},
    {"name": "gate_freeze_50pct", "gate_freeze_frac": 0.50},
    # --- Combined: diverse init + gate freezing ---
    {"name": "orthogonal_freeze_25pct", "init": "orthogonal", "gate_freeze_frac": 0.25},
    {"name": "pca_freeze_25pct", "init": "pca", "gate_freeze_frac": 0.25},
]


# ---------------------------------------------------------------------------
# Custom encoder initialization
# ---------------------------------------------------------------------------

def init_orthogonal_encoders(model: MOLT):
    """Replace random unit-norm encoders with orthogonal directions via QR.

    For groups with num_transforms <= d_model, we sample a random matrix and
    take its Q factor to get exactly orthogonal encoder directions. For groups
    with more transforms than dimensions, we tile orthogonal blocks.
    """
    d = model.config.d_model
    with torch.no_grad():
        for group in model.groups:
            n = group.num_transforms
            if n <= d:
                q, _ = torch.linalg.qr(torch.randn(d, n))
                group.encoder.copy_(q[:, :n].T)
            else:
                blocks = []
                remaining = n
                while remaining > 0:
                    take = min(remaining, d)
                    q, _ = torch.linalg.qr(torch.randn(d, take))
                    blocks.append(q[:, :take].T)
                    remaining -= take
                group.encoder.copy_(torch.cat(blocks, dim=0)[:n])


def init_pca_encoders(model: MOLT, mlp_inputs: torch.Tensor, sample_size: int = 50_000):
    """Initialize encoders to the top PCA directions of the input activations.

    This gives each encoder a direction that captures maximal variance in the
    actual input data, rather than a random direction.
    """
    d = model.config.d_model
    x = mlp_inputs[:sample_size].float()
    x = x - x.mean(dim=0)

    _, _, Vh = torch.linalg.svd(x, full_matrices=False)
    # Vh rows are the principal directions, sorted by variance

    with torch.no_grad():
        offset = 0
        for group in model.groups:
            n = group.num_transforms
            indices = (torch.arange(n) + offset) % d
            group.encoder.copy_(Vh[indices])
            offset += n


# ---------------------------------------------------------------------------
# Forward with frozen gates
# ---------------------------------------------------------------------------

def _forward_frozen_gates(model: MOLT, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Forward pass with all gates forced to 1.0 (bypassing JumpReLU)."""
    output = torch.zeros_like(x)
    total_sparsity = torch.tensor(0.0, device=x.device)
    total_active = torch.tensor(0.0, device=x.device)
    all_gate_acts = []

    for group in model.groups:
        Vx = torch.einsum("nrd,bd->nbr", group.V, x)
        UVx = torch.einsum("ndr,nbr->nbd", group.U, Vx)

        gate = torch.ones(x.shape[0], group.num_transforms, device=x.device)
        gated = UVx * gate.T.unsqueeze(-1)
        output = output + gated.sum(dim=0)

        total_active = total_active + group.num_transforms
        all_gate_acts.append(gate)

        u_norms = group.U.flatten(1).norm(dim=1)
        v_norms = group.V.flatten(1).norm(dim=1)
        frob = u_norms * v_norms / (group.d_model * group.rank) ** 0.5
        total_sparsity = total_sparsity + frob.sum()

    aux = {"sparsity_loss": total_sparsity, "l0": total_active, "gate_acts": all_gate_acts}
    return output, aux


# ---------------------------------------------------------------------------
# Training loop with custom init + gate freezing
# ---------------------------------------------------------------------------

def train_custom(
    config: MOLTConfig,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    init_mode: str = "default",
    gate_freeze_frac: float = 0.0,
) -> tuple[MOLT, list[dict]]:
    """Train MOLT with optional custom encoder init and gate freezing."""
    torch.manual_seed(config.seed)
    model = MOLT(config).to(config.device)

    if init_mode == "orthogonal":
        init_orthogonal_encoders(model)
    elif init_mode == "pca":
        init_pca_encoders(model, mlp_inputs)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataloader = make_dataloader(mlp_inputs, mlp_outputs, config.batch_size)

    history: list[dict] = []
    step = 0
    total_steps = len(dataloader) * config.num_epochs
    freeze_steps = int(total_steps * gate_freeze_frac)

    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch_inputs, batch_targets in pbar:
            batch_inputs = batch_inputs.to(config.device)
            batch_targets = batch_targets.to(config.device)

            optimizer.zero_grad()
            gates_frozen = step < freeze_steps

            if gates_frozen:
                output, aux = _forward_frozen_gates(model, batch_inputs)
            else:
                output, aux = model.forward(batch_inputs)

            mse = F.mse_loss(output, batch_targets)
            mse.backward()
            optimizer.step()
            step += 1

            if step % config.log_every == 0:
                target_var = batch_targets.var()
                nmse = mse / (target_var + 1e-8)

                # Always measure real (unfrozen) L0 for logging
                with torch.no_grad():
                    _, real_aux = model.forward(batch_inputs)
                    real_l0 = (
                        (torch.cat(real_aux["gate_acts"], dim=1) > 0)
                        .float().sum(dim=1).mean()
                    )

                log = {
                    "mse": mse.item(),
                    "nmse": nmse.item(),
                    "sparsity_loss": aux["sparsity_loss"].item(),
                    "l0": real_l0.item(),
                    "total_loss": mse.item(),
                    "step": step,
                    "epoch": epoch,
                    "gates_frozen": gates_frozen,
                }
                history.append(log)
                frozen_str = " [FROZEN]" if gates_frozen else ""
                pbar.set_postfix(
                    mse=f"{mse.item():.4f}",
                    nmse=f"{nmse.item():.4f}",
                    l0=f"{real_l0.item():.1f}{frozen_str}",
                )

    return model, history


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_setup(
    name: str,
    config: MOLTConfig,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    init_mode: str = "default",
    gate_freeze_frac: float = 0.0,
) -> dict:
    """Run a single setup: init, train, eval, save."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  init={init_mode}, gate_freeze_frac={gate_freeze_frac}")
    print(f"{'='*60}")

    start_time = time.time()
    model, history = train_custom(
        config, mlp_inputs, mlp_outputs,
        init_mode=init_mode,
        gate_freeze_frac=gate_freeze_frac,
    )
    training_time = time.time() - start_time

    # Evaluate
    eval_in = mlp_inputs[-10_000:]
    eval_out = mlp_outputs[-10_000:]
    l0 = compute_l0(model, eval_in)
    nmse = compute_nmse(model, eval_in, eval_out)

    print(f"  Final — L0: {l0:.2f}, NMSE: {nmse:.4f}, time: {training_time:.0f}s")

    result = {
        "name": name,
        "init": init_mode,
        "gate_freeze_frac": gate_freeze_frac,
        "l0": round(l0, 2),
        "nmse": round(nmse, 4),
        "training_time_s": round(training_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"result_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return result


def main():
    from molt.utils.activations import load_cached_activations
    from molt.utils.plotting import plot_multi_run_curves

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mlp_inputs, mlp_outputs = load_cached_activations(CACHE_PATH)

    # Support running a single setup by name
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        setups = [s for s in SETUPS if s["name"] == target]
        if not setups:
            print(f"Unknown: {target}. Options: {[s['name'] for s in SETUPS]}")
            sys.exit(1)
    else:
        setups = SETUPS

    all_results = []
    all_histories = {}

    for setup in setups:
        config = MOLTConfig(**BASE_CONFIG)
        result = run_setup(
            name=setup["name"],
            config=config,
            mlp_inputs=mlp_inputs,
            mlp_outputs=mlp_outputs,
            init_mode=setup.get("init", "default"),
            gate_freeze_frac=setup.get("gate_freeze_frac", 0.0),
        )
        all_results.append(result)

        hist_path = RESULTS_DIR / f"history_{setup['name']}.json"
        if hist_path.exists():
            with open(hist_path) as f:
                all_histories[setup["name"]] = json.load(f)

    # Summary
    if not target:
        with open(RESULTS_DIR / "sweep_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print("Exp 10: Gemma Gate Collapse Investigation")
        print(f"{'='*80}")
        print(f"{'Setup':<30} {'Init':<12} {'Freeze':>7} {'L0':>6} {'NMSE':>8}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['name']:<30} {r['init']:<12} {r['gate_freeze_frac']:>6.0%} "
                f"{r['l0']:>6.2f} {r['nmse']:>8.4f}"
            )

        if all_histories:
            plot_multi_run_curves(
                all_histories,
                "Exp 10: Gemma Gate Collapse Investigation",
                FIGURES_DIR / "comparison.png",
            )


if __name__ == "__main__":
    main()
