"""Data pipeline: stream FineWeb, collect MLP (input, output) activations from Gemma."""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from molt.config import MOLTConfig


def stream_fineweb_tokens(
    config: MOLTConfig,
    num_tokens: int | None = None,
) -> list[torch.Tensor]:
    """Stream FineWeb and tokenize into fixed-length chunks.

    Returns:
        List of (seq_len,) token tensors
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ds = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        streaming=config.streaming,
    )

    num_tokens = num_tokens or config.num_tokens
    all_ids: list[int] = []
    chunks: list[torch.Tensor] = []

    for example in ds:
        text = example["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

        while len(all_ids) >= config.seq_len:
            chunk = torch.tensor(all_ids[: config.seq_len], dtype=torch.long)
            chunks.append(chunk)
            all_ids = all_ids[config.seq_len :]

            if len(chunks) * config.seq_len >= num_tokens:
                return chunks

    return chunks


@torch.no_grad()
def collect_activations(
    config: MOLTConfig,
    token_chunks: list[torch.Tensor],
    cache_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Gemma and collect MLP input/output activations at target layer.

    Uses hooks to capture activations without modifying the model.

    Args:
        config: MOLT config
        token_chunks: list of (seq_len,) token tensors
        cache_path: optional path to cache activations on disk

    Returns:
        mlp_inputs: (total_tokens, d_model)
        mlp_outputs: (total_tokens, d_model)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached activations from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["mlp_inputs"], data["mlp_outputs"]

    print(f"Loading model {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
    )
    model.eval()

    # Pre-allocate output tensors to avoid OOM from list + cat
    total_tokens = len(token_chunks) * config.seq_len
    mlp_inputs = torch.empty(total_tokens, config.d_model, dtype=torch.float32)
    mlp_outputs = torch.empty(total_tokens, config.d_model, dtype=torch.float32)

    # Storage for hook captures
    captured: dict[str, torch.Tensor] = {}

    def hook_mlp_input(module, args, kwargs):
        captured["mlp_input"] = args[0].detach().float()
        return None

    def hook_mlp_output(module, args, output):
        captured["mlp_output"] = output.detach().float()
        return None

    # Register hooks on the target layer's MLP
    layer = model.model.layers[config.layer_idx]
    h_in = layer.mlp.register_forward_pre_hook(hook_mlp_input, with_kwargs=True)
    h_out = layer.mlp.register_forward_hook(hook_mlp_output)

    try:
        for i, chunk in enumerate(tqdm(token_chunks, desc="Collecting activations")):
            input_ids = chunk.unsqueeze(0).to(config.device)
            model(input_ids)

            start = i * config.seq_len
            end = start + config.seq_len
            mlp_inputs[start:end] = captured["mlp_input"].squeeze(0).cpu()
            mlp_outputs[start:end] = captured["mlp_output"].squeeze(0).cpu()
    finally:
        h_in.remove()
        h_out.remove()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching activations to {cache_path}")
        torch.save({"mlp_inputs": mlp_inputs, "mlp_outputs": mlp_outputs}, cache_path)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    return mlp_inputs, mlp_outputs


def make_dataloader(
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from activation tensors."""
    dataset = TensorDataset(mlp_inputs, mlp_outputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
