#!/usr/bin/env python3
"""Quick W2 vs INT4 quality smoke test on Mac.

Loads both chunk sets, runs autoregressive generation with real embeddings
from the HF model, and prints side-by-side output for visual comparison.

Usage:
    python smoke_w2_quality.py \
        --w2-dir /tmp/w2-8k \
        --int4-dir output/all_chunks_8k \
        --steps 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import coremltools as ct
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

EMB_DIR = os.path.join(ROOT, "output/iphone_8k")
HF_DIR = os.path.join(ROOT, "output/gemma4-e2b-final/hf_model")

# Architecture constants
CTX = 8192
W = 512
HIDDEN = 1536
NLAYERS = 35
PLD = 256
MAX_HD = 512
TOTAL_PLD = NLAYERS * PLD
VOCAB = 262144
EMBED_SCALE = 39.191835403442383  # sqrt(hidden_size) for Gemma 4


def load_embeddings():
    """Load INT8 quantized embedding + per-layer embedding from disk."""
    embed_q8 = np.fromfile(os.path.join(EMB_DIR, "embed_tokens_q8.bin"), dtype=np.int8)
    embed_q8 = embed_q8.reshape(VOCAB, HIDDEN)
    embed_scales = np.fromfile(os.path.join(EMB_DIR, "embed_tokens_scales.bin"), dtype=np.float16)

    pl_q8 = np.fromfile(os.path.join(EMB_DIR, "embed_tokens_per_layer_q8.bin"), dtype=np.int8)
    pl_q8 = pl_q8.reshape(VOCAB, TOTAL_PLD)
    pl_scales = np.fromfile(os.path.join(EMB_DIR, "embed_tokens_per_layer_scales.bin"), dtype=np.float16)

    return embed_q8, embed_scales, pl_q8, pl_scales


def dequant_embed(q8, scales, token_id):
    """Dequantize INT8 embedding for a single token."""
    row = q8[token_id].astype(np.float32)
    s = float(scales[token_id])
    out = (row * (s / 127.0) * EMBED_SCALE).astype(np.float16)
    return out.reshape(1, 1, -1)


def dequant_perlayer(pl_q8, pl_scales, token_id):
    """Dequantize per-layer embedding."""
    row = pl_q8[token_id].astype(np.float32)
    s = float(pl_scales[token_id])
    out = (row * (s / 127.0) * EMBED_SCALE).astype(np.float16)
    return out.reshape(1, 1, -1)


def load_rope():
    """Load precomputed RoPE tables."""
    cos_f = np.load(os.path.join(EMB_DIR, "cos_full.npy"))   # (8192, 512)
    sin_f = np.load(os.path.join(EMB_DIR, "sin_full.npy"))
    cos_s = np.load(os.path.join(EMB_DIR, "cos_sliding.npy"))  # (8192, 256)
    sin_s = np.load(os.path.join(EMB_DIR, "sin_sliding.npy"))
    return cos_s, sin_s, cos_f, sin_f


def load_chunks(chunks_dir, cu):
    """Load 4 CoreML chunks."""
    chunks = []
    for i in range(1, 5):
        p = os.path.join(chunks_dir, f"chunk{i}.mlpackage")
        if not os.path.exists(p):
            p = os.path.join(chunks_dir, f"chunk{i}.mlmodelc")
        t = time.time()
        m = ct.models.MLModel(p, compute_units=cu)
        print(f"  chunk{i}: {time.time()-t:.1f}s")
        chunks.append(m)
    return chunks


def generate(chunks, embed_q8, embed_scales, pl_q8, pl_scales,
             cos_s_tbl, sin_s_tbl, cos_f_tbl, sin_f_tbl,
             prompt_ids, max_steps):
    """Autoregressive generation using 4 CoreML chunks."""
    c1, c2, c3, c4 = chunks

    # KV cache
    kSliding1 = np.zeros((7, 1, W, MAX_HD), dtype=np.float16)
    vSliding1 = np.zeros((7, 1, W, MAX_HD), dtype=np.float16)
    kFull1 = np.zeros((1, 1, CTX, MAX_HD), dtype=np.float16)
    vFull1 = np.zeros((1, 1, CTX, MAX_HD), dtype=np.float16)
    kSliding2 = np.zeros((5, 1, W, MAX_HD), dtype=np.float16)
    vSliding2 = np.zeros((5, 1, W, MAX_HD), dtype=np.float16)
    kFull2 = np.zeros((2, 1, CTX, MAX_HD), dtype=np.float16)
    vFull2 = np.zeros((2, 1, CTX, MAX_HD), dtype=np.float16)

    output_ids = list(prompt_ids)
    next_token = prompt_ids[-1]

    for step in range(len(prompt_ids) - 1 + max_steps):
        if step < len(prompt_ids) - 1:
            # Prefill: feed prompt tokens one by one
            tok = prompt_ids[step]
        else:
            tok = next_token

        pos = step

        # Embeddings
        h_in = dequant_embed(embed_q8, embed_scales, tok)
        plr = dequant_perlayer(pl_q8, pl_scales, tok)

        # Masks
        mask_full = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask_full[0, 0, 0, :pos + 1] = 0
        mask_sliding = np.full((1, 1, 1, W), -65504.0, dtype=np.float16)
        valid = min(pos + 1, W)
        mask_sliding[0, 0, 0, W - valid:] = 0
        umask = np.zeros((1, 1, CTX, 1), dtype=np.float16)
        umask[0, 0, min(pos, CTX - 1), 0] = 1.0

        # RoPE
        cos_s = cos_s_tbl[pos].reshape(1, 1, 1, 256).astype(np.float16)
        sin_s = sin_s_tbl[pos].reshape(1, 1, 1, 256).astype(np.float16)
        cos_f = cos_f_tbl[pos].reshape(1, 1, 1, 512).astype(np.float16)
        sin_f = sin_f_tbl[pos].reshape(1, 1, 1, 512).astype(np.float16)

        # Chunk 1
        out1 = c1.predict({
            "hidden_states": h_in,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_raw": plr,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kSliding1, "V_sliding_in": vSliding1,
            "K_full_in": kFull1, "V_full_in": vFull1,
        })
        h1 = out1["hidden_states_out"]
        plc = out1["per_layer_combined_out"]
        kSliding1 = out1["K_sliding_out"]
        vSliding1 = out1["V_sliding_out"]
        kFull1 = out1["K_full_out"]
        vFull1 = out1["V_full_out"]

        # Chunk 2
        out2 = c2.predict({
            "hidden_states": h1,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kSliding2, "V_sliding_in": vSliding2,
            "K_full_in": kFull2, "V_full_in": vFull2,
        })
        h2 = out2["hidden_states_out"]
        kSliding2 = out2["K_sliding_out"]
        vSliding2 = out2["V_sliding_out"]
        kFull2 = out2["K_full_out"]
        vFull2 = out2["V_full_out"]
        kv13_k = out2["kv13_k"]
        kv13_v = out2["kv13_v"]
        kv14_k = out2["kv14_k"]
        kv14_v = out2["kv14_v"]

        shared = {
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "kv13_k": kv13_k, "kv13_v": kv13_v,
            "kv14_k": kv14_k, "kv14_v": kv14_v,
        }

        # Chunk 3
        d3 = dict(shared)
        d3["hidden_states"] = h2
        out3 = c3.predict(d3)
        h3 = out3["hidden_states_out"]

        # Chunk 4
        d4 = dict(shared)
        d4["hidden_states"] = h3
        out4 = c4.predict(d4)

        tid = out4["token_id"]
        next_token = int(tid.flat[0]) if hasattr(tid, 'flat') else int(tid)

        if step >= len(prompt_ids) - 1:
            output_ids.append(next_token)
            # EOS check
            if next_token in (1, 107):  # Gemma EOS / <end_of_turn>
                break

    return output_ids[len(prompt_ids):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2-dir", type=str, default="/tmp/w2-8k")
    parser.add_argument("--int4-dir", type=str, default=os.path.join(ROOT, "output/all_chunks_8k"))
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    print("Loading embeddings...")
    embed_q8, embed_scales, pl_q8, pl_scales = load_embeddings()

    print("Loading RoPE tables...")
    cos_s, sin_s, cos_f, sin_f = load_rope()

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_DIR)

    # Test prompts
    prompts = [
        "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nExplain photosynthesis in two sentences.<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWrite a haiku about the ocean.<end_of_turn>\n<start_of_turn>model\n",
    ]

    cu = ct.ComputeUnit.CPU_ONLY

    for label, chunks_dir in [("INT4", args.int4_dir), ("W2", args.w2_dir)]:
        print(f"\n{'='*60}")
        print(f"Loading {label} chunks from {chunks_dir}")
        print(f"{'='*60}")
        chunks = load_chunks(chunks_dir, cu)

        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            print(f"\nPrompt: {prompt.strip()[-60:]}")
            t0 = time.time()
            out_ids = generate(
                chunks, embed_q8, embed_scales, pl_q8, pl_scales,
                cos_s, sin_s, cos_f, sin_f,
                ids, args.steps,
            )
            dt = time.time() - t0
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print(f"[{label}] ({dt:.1f}s, {len(out_ids)} tok): {text}")


if __name__ == "__main__":
    raise SystemExit(main())
