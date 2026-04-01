"""Interpretability: collect top-activating contexts per transform, label with delphi."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from molt.config import MOLTConfig
from molt.model import MOLT


@dataclass
class TransformContext:
    """A context where a specific transform was highly active."""

    transform_group_idx: int
    transform_idx: int
    rank: int
    activation: float
    token_ids: list[int]
    text: str
    token_position: int


def collect_top_contexts(
    model: MOLT,
    config: MOLTConfig,
    mlp_inputs: torch.Tensor,
    token_chunks: list[torch.Tensor],
    top_k: int = 20,
    batch_size: int = 256,
) -> dict[tuple[int, int], list[TransformContext]]:
    """Collect top-k activating contexts for each transform.

    Args:
        model: trained MOLT
        config: MOLT config
        mlp_inputs: (total_tokens, d_model) flattened MLP inputs
        token_chunks: list of (seq_len,) original token chunks
        top_k: number of top contexts per transform
        batch_size: batch size for forward passes

    Returns:
        dict mapping (group_idx, transform_idx) -> list of TransformContext
    """
    model.eval()
    device = next(model.parameters()).device

    # For each transform, maintain a min-heap of top-k activations
    # Structure: (group_idx, transform_idx) -> list of (activation, global_token_idx)
    top_acts: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for g_idx, group in enumerate(model.groups):
        for t_idx in range(group.num_transforms):
            top_acts[(g_idx, t_idx)] = []

    # Forward pass to collect all gate activations
    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(mlp_inputs), batch_size), desc="Collecting gate activations"
        ):
            batch = mlp_inputs[batch_start : batch_start + batch_size].to(device)
            _, aux = model(batch)

            for g_idx, gate_acts in enumerate(aux["gate_acts"]):
                # gate_acts: (batch, num_transforms)
                for t_idx in range(gate_acts.shape[1]):
                    acts = gate_acts[:, t_idx].cpu()
                    for local_idx in range(len(acts)):
                        act_val = acts[local_idx].item()
                        if act_val <= 0:
                            continue
                        global_idx = batch_start + local_idx
                        key = (g_idx, t_idx)
                        heap = top_acts[key]

                        if len(heap) < top_k:
                            heap.append((act_val, global_idx))
                            heap.sort(key=lambda x: x[0])
                        elif act_val > heap[0][0]:
                            heap[0] = (act_val, global_idx)
                            heap.sort(key=lambda x: x[0])

    # Convert global indices back to contexts
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    seq_len = config.seq_len

    result: dict[tuple[int, int], list[TransformContext]] = {}
    for (g_idx, t_idx), heap in top_acts.items():
        contexts = []
        rank = config.rank_distribution[g_idx][1]

        for act_val, global_idx in sorted(heap, key=lambda x: -x[0]):
            chunk_idx = global_idx // seq_len
            token_pos = global_idx % seq_len

            if chunk_idx < len(token_chunks):
                token_ids = token_chunks[chunk_idx].tolist()
                text = tokenizer.decode(token_ids)
                contexts.append(
                    TransformContext(
                        transform_group_idx=g_idx,
                        transform_idx=t_idx,
                        rank=rank,
                        activation=act_val,
                        token_ids=token_ids,
                        text=text,
                        token_position=token_pos,
                    )
                )

        result[(g_idx, t_idx)] = contexts

    return result


def save_contexts(
    contexts: dict[tuple[int, int], list[TransformContext]],
    save_path: str,
):
    """Save collected contexts to JSON."""
    serializable = {}
    for (g_idx, t_idx), ctx_list in contexts.items():
        key = f"group{g_idx}_transform{t_idx}"
        serializable[key] = [
            {
                "group_idx": c.transform_group_idx,
                "transform_idx": c.transform_idx,
                "rank": c.rank,
                "activation": c.activation,
                "text": c.text,
                "token_position": c.token_position,
            }
            for c in ctx_list
        ]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {len(serializable)} transform contexts to {save_path}")


def label_transforms_with_delphi(
    contexts: dict[tuple[int, int], list[TransformContext]],
    save_path: str | None = None,
) -> dict[str, str]:
    """Use delphi to generate labels for transform contexts.

    This requires the delphi-ai package to be installed.

    Args:
        contexts: dict of (group_idx, transform_idx) -> list of TransformContext
        save_path: optional path to save labels

    Returns:
        dict mapping transform key -> generated label
    """
    try:
        from delphi import DefaultExplainer
    except ImportError:
        print(
            "delphi-ai not installed. Install with: pip install delphi-ai\n"
            "Falling back to manual context display."
        )
        return _display_contexts_manually(contexts)

    explainer = DefaultExplainer()
    labels = {}

    for (g_idx, t_idx), ctx_list in tqdm(
        contexts.items(), desc="Labeling transforms"
    ):
        if not ctx_list:
            continue

        # Format activating examples for delphi
        examples = []
        for ctx in ctx_list[:10]:  # Top 10 examples
            examples.append(
                {
                    "text": ctx.text,
                    "position": ctx.token_position,
                    "activation": ctx.activation,
                }
            )

        key = f"group{g_idx}_transform{t_idx} (rank={ctx_list[0].rank})"
        try:
            label = explainer.explain(examples)
            labels[key] = label
        except Exception as e:
            labels[key] = f"[Error: {e}]"

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved {len(labels)} transform labels to {save_path}")

    return labels


def _display_contexts_manually(
    contexts: dict[tuple[int, int], list[TransformContext]],
) -> dict[str, str]:
    """Fallback: display top contexts per transform without delphi."""
    labels = {}
    for (g_idx, t_idx), ctx_list in contexts.items():
        if not ctx_list:
            continue
        key = f"group{g_idx}_transform{t_idx} (rank={ctx_list[0].rank})"
        top_texts = [
            f"  [{c.activation:.3f}] ...{c.text[max(0, c.token_position*4-40):c.token_position*4+40]}..."
            for c in ctx_list[:5]
        ]
        summary = f"Top activating contexts:\n" + "\n".join(top_texts)
        labels[key] = summary
        print(f"\n{key}:\n{summary}")
    return labels
