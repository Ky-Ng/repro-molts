#!/usr/bin/env python3
"""Render feature label + top activating contexts with the firing token highlighted.

Used for the 5+5 human spot-check in the smoke-slice go/no-go gate
(see README §Spot-checking labels).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "out"
HOOKPOINT = "transformer.h.6.mlp"


def load_feature_topk(
    slice_name: str,
    feature_id: int,
    top_k: int = 10,
) -> list[tuple[float, int, int, torch.Tensor]]:
    """Return top-k (activation, batch_idx, seq_pos, tokens_tensor) for this feature.

    Looks across all shards under latents/<slice_name>/<hookpoint>/.
    """
    latents_dir = OUT_DIR / "latents" / slice_name / HOOKPOINT
    if not latents_dir.exists():
        raise FileNotFoundError(f"No cache at {latents_dir}. Run stage 'cache' first.")

    all_entries: list[tuple[float, int, int, torch.Tensor]] = []
    for shard in sorted(latents_dir.glob("*.safetensors")):
        data = load_file(shard)
        locs = data["locations"]       # [nnz, 3] = (batch_idx, seq_idx, feature_id)
        acts = data["activations"]     # [nnz]
        tokens = data["tokens"]        # [n_seq, seq_len]
        mask = locs[:, 2] == feature_id
        if not mask.any():
            continue
        f_locs = locs[mask]
        f_acts = acts[mask]
        for i in range(len(f_acts)):
            all_entries.append(
                (f_acts[i].item(), int(f_locs[i, 0].item()), int(f_locs[i, 1].item()), tokens)
            )

    # Global top-k across shards
    all_entries.sort(key=lambda e: -e[0])
    return all_entries[:top_k]


def read_text_artifact(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    # delphi writes orjson-dumped strings; try json.loads first, fall back to raw text
    try:
        return json.loads(raw)
    except Exception:
        return raw.decode("utf-8", errors="replace").strip()


def render_feature(
    slice_name: str,
    feature_id: int,
    tokenizer,
    window: int = 10,
    top_k: int = 10,
) -> None:
    # delphi saves as "{hookpoint}_latent{feature_id}.txt"
    fname = f"{HOOKPOINT}_latent{feature_id}.txt"
    label = read_text_artifact(
        OUT_DIR / "explanations" / slice_name / fname
    ) or "(no label)"
    det = read_text_artifact(
        OUT_DIR / "scores" / "detection" / slice_name / fname
    ) or "(no score)"
    fuzz = read_text_artifact(
        OUT_DIR / "scores" / "fuzzing" / slice_name / fname
    ) or "(no score)"

    print(f"\n{'='*80}")
    print(f"FEATURE {feature_id}  ({slice_name})")
    print(f"  LABEL:        {label}")
    print(f"  DETECTION:    {det}")
    print(f"  FUZZING:      {fuzz}")
    print(f"{'='*80}")

    try:
        entries = load_feature_topk(slice_name, feature_id, top_k=top_k)
    except FileNotFoundError as e:
        print(f"  (no cache available: {e})")
        return

    if not entries:
        print("  (no activations for this feature — likely dead)")
        return

    for act, batch_idx, seq_pos, tokens in entries:
        seq_tokens = tokens[batch_idx].tolist()
        lo, hi = max(0, seq_pos - window), min(len(seq_tokens), seq_pos + window + 1)
        before = tokenizer.decode(seq_tokens[lo:seq_pos]).replace("\n", " ")
        firing = tokenizer.decode([seq_tokens[seq_pos]]).replace("\n", " ")
        after = tokenizer.decode(seq_tokens[seq_pos + 1 : hi]).replace("\n", " ")
        print(f"  [{act:5.2f}]  ...{before}[[{firing}]]{after}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True,
                    help="e.g. molt_1x_lam1e-03 or tc_1x_lam1e+00")
    ap.add_argument("--features", nargs="+", type=int,
                    help="specific feature ids to render")
    ap.add_argument("--sample", type=int, default=0,
                    help="random N features (ignored if --features given)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--window", type=int, default=10,
                    help="tokens of context on each side of the firing token")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    if args.features:
        feats = args.features
    else:
        # Infer feature count from slice name (crude): molt 1x=31, 4x=124;
        # tc 1x=2573, 4x=10295. User should pass --features for the precise set.
        if "molt_1x" in args.slice:
            n = 31
        elif "molt_4x" in args.slice:
            n = 124
        elif "tc_1x" in args.slice:
            n = 2573
        elif "tc_4x" in args.slice:
            n = 10295
        else:
            raise SystemExit(f"Cannot infer feature count from slice name '{args.slice}'")
        random.seed(42)
        feats = random.sample(range(n), min(args.sample or 5, n))

    for fid in feats:
        render_feature(args.slice, fid, tokenizer, window=args.window, top_k=args.top_k)


if __name__ == "__main__":
    main()
