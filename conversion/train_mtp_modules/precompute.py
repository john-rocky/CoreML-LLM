#!/usr/bin/env python3
"""Path C training Phase 1: precompute last-layer hidden states for a corpus.

For each sequence in the dataset:
  1. Run frozen HF Gemma 4 trunk forward.
  2. Extract the OUTPUT OF THE LAST DECODER LAYER, PRE-FINAL-NORM (shape (T, H)).
  3. Save (token_ids, hidden) to disk in a memory-mapped format.

CRITICAL: extraction must match what the Swift deployment feeds to the drafter.
Chunk 4 (SWAVerifyChunk4) returns `hidden_states` BEFORE `self.norm`, so
training data must be the same: output of layer[-1] pre-norm. Earlier versions
of this script used `hidden_states[-2]` which in HF Gemma 4 is the OUTPUT OF
LAYER 33 (penultimate) — a different tensor entirely. Using a forward hook on
the last decoder layer is the simplest way to get the exact chunk-4 state.

The training loop later consumes these cached pairs without touching the
trunk. This makes training ~100× faster than trunk-in-the-loop training,
and lets us keep the trunk on a single GPU while multi-GPU-training the
tiny MTP modules on a separate GPU.

Output format (per shard):
  {shard_idx}.tokens.bin   — int32 (N, T)  packed
  {shard_idx}.hidden.bin   — fp16 (N, T, H) packed
  {shard_idx}.meta.json    — shapes + stats

Target: 5M tokens on A100 in under 3 hours (Gate 1.5).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def iter_hf_dataset(name_or_path: str, streaming: bool = True, max_samples: int = None):
    """Yield text samples from an HF dataset. Supports streaming for large sets."""
    from datasets import load_dataset
    if name_or_path.startswith("shard:"):
        # Shard spec: "shard:dataset_name:split:start:end"
        parts = name_or_path.split(":")
        ds = load_dataset(parts[1], split=parts[2], streaming=streaming)
        start, end = int(parts[3]), int(parts[4])
        count = 0
        for i, ex in enumerate(ds):
            if i < start: continue
            if i >= end: break
            yield ex
            count += 1
            if max_samples and count >= max_samples: break
        return

    # Common chat/code datasets. Open (no gating) unless noted.
    dataset_presets = {
        # Guaranteed-open (good for first runs)
        "wikitext":    ("Salesforce/wikitext", "train", ["text"]),  # tiny, always works
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "train", ["text"]),  # high quality EN
        "oasst1":      ("OpenAssistant/oasst1", "train", ["text"]),  # EN chat
        "codealpaca":  ("sahil2801/CodeAlpaca-20k", "train", ["instruction", "input", "output"]),
        "github-code": ("codeparrot/github-code-clean", "train", ["code"]),  # open code (Apache 2.0 subset)
        "codesearch":  ("code_search_net", "train", ["whole_func_string"]),  # open, all langs
        # Gated — require HF auth + term acceptance. Skip if user hasn't agreed.
        "stack-small": ("bigcode/the-stack-smol", "train", ["content"]),
        "lmsys-chat":  ("lmsys/lmsys-chat-1m", "train", ["conversation"]),
        "c4-en":       ("allenai/c4", "train", ["text"]),  # needs subset 'en'
        "c4-ja":       ("allenai/c4", "train", ["text"]),  # needs subset 'ja'
        "sharegpt-ja": ("philschmid/sharegpt-raw", "train", ["conversations"]),
    }
    if name_or_path in dataset_presets:
        repo, split, text_keys = dataset_presets[name_or_path]
        kwargs = {"streaming": streaming}
        if name_or_path == "c4-en":
            kwargs["name"] = "en"
        if name_or_path == "c4-ja":
            kwargs["name"] = "ja"
        if name_or_path == "wikitext":
            kwargs["name"] = "wikitext-103-raw-v1"
        if name_or_path == "github-code":
            # codeparrot/github-code-clean has config by language; use python
            kwargs["name"] = "python-all"
        if name_or_path == "codesearch":
            kwargs["name"] = "python"
        if name_or_path == "oasst1":
            pass
        ds = load_dataset(repo, split=split, **kwargs)
        count = 0
        for ex in ds:
            # Collect text from configured fields
            text_parts = []
            for key in text_keys:
                val = ex.get(key)
                if isinstance(val, str):
                    text_parts.append(val)
                elif isinstance(val, list):
                    # conversation format
                    for turn in val:
                        if isinstance(turn, dict):
                            text_parts.append(turn.get("content") or turn.get("value") or "")
                        elif isinstance(turn, str):
                            text_parts.append(turn)
            text = "\n".join(t for t in text_parts if t)
            if text:
                yield {"text": text}
                count += 1
                if max_samples and count >= max_samples: break
        return

    # Fallback: raw path or HF repo
    ds = load_dataset(name_or_path, split="train", streaming=streaming)
    for ex in ds:
        yield ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str,
                    default=os.path.expanduser(
                        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
                        "4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
                    help="HF Gemma 4 checkpoint directory")
    ap.add_argument("--dataset", type=str, nargs="+",
                    default=["lmsys-chat", "stack-small"],
                    help="Dataset preset name(s) or HF repo path(s). Multiple = concat.")
    ap.add_argument("--samples-per-dataset", type=int, default=500,
                    help="Max samples per dataset (for 5M token target across 2 datasets).")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--output-dir", type=str, default="./output/mtp_train_cache")
    ap.add_argument("--shard-size", type=int, default=256,
                    help="Sequences per shard on disk.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-append to existing shards if present — continue numbering.
    import glob as _glob
    existing = sorted(_glob.glob(str(out_dir / "shard_*.tokens.npy")))
    starting_shard_idx = len(existing)
    if starting_shard_idx > 0:
        print(f"Append mode: found {starting_shard_idx} existing shards; "
              f"new shards will start at shard_{starting_shard_idx:04d}")

    print(f"Device: {args.device}  Dtype: {args.dtype}")
    print(f"Loading HF Gemma 4 from {args.hf_dir}...")
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        args.hf_dir, torch_dtype=dtype, device_map=args.device,
    ).eval()
    lm = hf.model.language_model
    H = lm.config.hidden_size

    # Register a hook on the last decoder layer to capture its PRE-FINAL-NORM
    # output — this is what SWAVerifyChunk4 returns as `hidden_states_out` at
    # inference. hf.output_hidden_states[-2] is L33 (different layer); do NOT use it.
    _last_hidden_capture = {}
    def _capture_last_hidden(_mod, _inp, outp):
        h = outp[0] if isinstance(outp, tuple) else outp
        _last_hidden_capture["h"] = h
    lm.layers[-1].register_forward_hook(_capture_last_hidden)

    total_tokens = 0
    shard_idx = starting_shard_idx
    buf_tokens = []
    buf_hiddens = []

    def flush():
        nonlocal shard_idx, buf_tokens, buf_hiddens
        if not buf_tokens: return
        tokens = np.stack(buf_tokens).astype(np.int32)
        hiddens = np.stack(buf_hiddens).astype(np.float16)  # always store fp16 to save disk
        np.save(out_dir / f"shard_{shard_idx:04d}.tokens.npy", tokens)
        np.save(out_dir / f"shard_{shard_idx:04d}.hidden.npy", hiddens)
        meta = {
            "shard_idx": shard_idx,
            "num_sequences": tokens.shape[0],
            "seq_len": tokens.shape[1],
            "hidden_size": hiddens.shape[-1],
        }
        with open(out_dir / f"shard_{shard_idx:04d}.meta.json", "w") as f:
            json.dump(meta, f)
        print(f"  Flushed shard {shard_idx}: {tokens.shape} tokens, "
              f"{hiddens.shape} hiddens ({hiddens.nbytes / 1e6:.1f} MB)")
        shard_idx += 1
        buf_tokens = []
        buf_hiddens = []

    print(f"\nDatasets: {args.dataset}")
    # Per-dataset processing: tokenize → chunk → trunk forward → save shards.
    # Crashes in dataset N don't lose datasets 0..N-1 (already flushed to disk).
    bos = tok.bos_token_id
    grand_total_tokens = 0
    grand_total_chunks = 0

    for ds_name in args.dataset:
        print(f"\n=== Dataset: {ds_name} ===")
        print(f"  Streaming {ds_name} (max {args.samples_per_dataset} samples)...")
        ds_tokens = [bos] if bos is not None else []
        ds_samples = 0
        try:
            for ex in iter_hf_dataset(ds_name, streaming=True, max_samples=args.samples_per_dataset):
                text = ex.get("text") or ex.get("content", "")
                if not text:
                    continue
                ids = tok(text, add_special_tokens=False).input_ids
                ds_tokens.extend(ids)
                ds_samples += 1
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            print(f"  Skipping; previous datasets' shards are safely on disk.")
            continue

        print(f"  Collected: {ds_samples} samples, {len(ds_tokens):,} tokens")

        # Chunk this dataset's tokens into seq_len windows
        ds_chunks = []
        for i in range(0, len(ds_tokens) - args.seq_len + 1, args.seq_len):
            ds_chunks.append(ds_tokens[i:i + args.seq_len])
        print(f"  {len(ds_chunks)} chunks of {args.seq_len}")

        if not ds_chunks:
            print(f"  (too few tokens to form even one chunk; skipping)")
            continue

        # Trunk forward + save per-dataset
        with torch.inference_mode():
            for i, chunk in enumerate(ds_chunks):
                ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
                # Hook captures last-layer pre-norm hidden; no need for output_hidden_states.
                lm(input_ids=ids, use_cache=False)
                last_hidden = _last_hidden_capture["h"][0].to(torch.float16).cpu().numpy()
                buf_tokens.append(np.array(chunk, dtype=np.int32))
                buf_hiddens.append(last_hidden)
                total_tokens += len(chunk)

                if len(buf_tokens) >= args.shard_size:
                    flush()

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i+1}/{len(ds_chunks)} chunks of {ds_name}, "
                          f"{total_tokens:,} total tokens")

        # Flush any remaining buffer for this dataset so its shards are
        # independent from the next dataset's (crash-safety).
        flush()
        grand_total_tokens += len(ds_tokens)
        grand_total_chunks += len(ds_chunks)

    print(f"\nDONE. Total tokens: {total_tokens:,} across {shard_idx} shards.")
    print(f"Storage: {sum(f.stat().st_size for f in out_dir.rglob('*')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
