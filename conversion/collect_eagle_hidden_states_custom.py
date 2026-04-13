#!/usr/bin/env python3
"""Collect hidden states from custom Gemma4Model for EAGLE-3 draft retraining.

This is the fix for Blocker 1 (EAGLE3_INTEGRATION_STATE.md): the original
collect_eagle_hidden_states.py used HF Gemma4ForConditionalGeneration as the
teacher, but the deployed target is our custom Gemma4Model with Conv2d-based
layers. The L34 hidden states differ by 4.4× in norm, causing ~0% acceptance
on device.

This script runs the same custom forward path that the CoreML chunks use,
producing hidden states that match what the device sees.

Usage (Colab, A100):
    !git clone -q https://github.com/john-rocky/CoreML-LLM.git
    %cd CoreML-LLM/conversion
    !pip install -q safetensors
    !python collect_eagle_hidden_states_custom.py \
        --corpus /content/drive/MyDrive/eagle_corpus.jsonl \
        --output /content/drive/MyDrive/eagle_draft/training_data_custom.pt \
        --num-samples 30000 --seq-len 512

Runtime: ~2-4h on A100 for 30k samples (sequential per-token forward).
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model, Gemma4Config
from models.gemma4_swa_chunks import _run_layer_swa, v_norm
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax


def forward_all(model, input_ids, position):
    """Run full 35-layer forward on a single token, returning pre-lm_head hidden state.

    This mirrors the exact computation path of chunk1→chunk2→chunk3→chunk4,
    but in a single sequential loop for simplicity.

    Args:
        model: Gemma4Model instance
        input_ids: (1, 1) int32 — single token
        position: int — current position in the sequence

    Returns:
        hidden_states: (1, 1, hidden_size) fp16 — pre-lm_head hidden
    """
    config = model.config
    ctx = config.context_length
    W = 512  # sliding window size
    max_hd = config.global_head_dim  # 512
    nlayers = config.num_hidden_layers

    # Embedding
    embed_scale = config.hidden_size ** 0.5
    hidden = model.embed_tokens(input_ids).to(MODEL_DTYPE) * embed_scale  # (1, 1, hidden)

    # Per-layer embedding
    pl_raw = model.embed_tokens_per_layer(input_ids).to(MODEL_DTYPE)  # (1, 1, 8960)
    pl_raw = pl_raw * (config.hidden_size_per_layer_input ** 0.5)
    # Project through model-level per-layer projection
    h_conv = hidden.permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, 1)
    pl_proj = model.per_layer_model_projection(h_conv.to(MODEL_DTYPE))  # (1, 8960, 1, 1)
    pl_proj = pl_proj.squeeze(2).permute(0, 2, 1)  # (1, 1, 8960)
    pl_proj = pl_proj * model.per_layer_model_projection_scale
    # Combine per-layer embeddings: norm(proj) + raw, then scale
    # Must match SWAChunk1._compute_ple / gemma4_wrapper.py exactly
    pld = config.hidden_size_per_layer_input  # 256
    pl_combined_parts = []
    for i in range(nlayers):
        s = i * pld
        e = s + pld
        raw_slice = pl_raw[:, :, s:e]
        proj_slice = pl_proj[:, :, s:e]
        proj_normed = model.per_layer_projection_norm(proj_slice)
        combined = (proj_normed + raw_slice) * model.per_layer_input_scale
        pl_combined_parts.append(combined)
    per_layer_combined = torch.cat(pl_combined_parts, dim=2)  # (1, 1, 8960)

    # RoPE for current position
    cos_s = model.cos_sliding[position].view(1, 1, 1, config.head_dim)
    sin_s = model.sin_sliding[position].view(1, 1, 1, config.head_dim)
    cos_f = model.cos_full[position].view(1, 1, 1, config.global_head_dim)
    sin_f = model.sin_full[position].view(1, 1, 1, config.global_head_dim)

    # Masks
    mask_full = torch.full((1, 1, 1, ctx), -65504.0, dtype=MODEL_DTYPE, device=hidden.device)
    mask_full[0, 0, 0, :position + 1] = 0
    mask_sliding = torch.full((1, 1, 1, W), -65504.0, dtype=MODEL_DTYPE, device=hidden.device)
    valid = min(position + 1, W)
    mask_sliding[0, 0, 0, W - valid:] = 0
    update_mask = torch.zeros((1, 1, ctx, 1), dtype=MODEL_DTYPE, device=hidden.device)
    update_mask[0, 0, min(position, ctx - 1), 0] = 1.0

    # KV cache (persistent across calls — stored on model)
    # _run_layer_swa expects 4 separate args: K_sliding, V_sliding, K_full, V_full
    # Each is (1, num_kv_heads, seq_dim, max_hd) per non-shared layer
    if not hasattr(model, '_kv_cache') or model._kv_cache is None:
        nkv = config.num_key_value_heads
        model._kv_cache = {}
        for i in range(nlayers):
            if not config.is_kv_shared(i):
                model._kv_cache[i] = {
                    'ks': torch.zeros(1, nkv, W, max_hd, dtype=MODEL_DTYPE, device=hidden.device),
                    'vs': torch.zeros(1, nkv, W, max_hd, dtype=MODEL_DTYPE, device=hidden.device),
                    'kf': torch.zeros(1, nkv, ctx, max_hd, dtype=MODEL_DTYPE, device=hidden.device),
                    'vf': torch.zeros(1, nkv, ctx, max_hd, dtype=MODEL_DTYPE, device=hidden.device),
                }
        model._kv13_k = None
        model._kv13_v = None
        model._kv14_k = None
        model._kv14_v = None

    # Run all layers
    for i in range(nlayers):
        is_kv_shared = config.is_kv_shared(i)

        if not is_kv_shared:
            c = model._kv_cache[i]
            ks, vs, kf, vf = c['ks'], c['vs'], c['kf'], c['vf']
        else:
            ks = vs = kf = vf = None

        result = _run_layer_swa(
            model.layers[i], i, hidden,
            cos_s, sin_s, cos_f, sin_f,
            mask_full, mask_sliding, update_mask,
            ks, vs, kf, vf,
            config, per_layer_combined,
            model._kv13_k, model._kv13_v, model._kv14_k, model._kv14_v,
        )
        hidden = result[0]

        if not is_kv_shared:
            model._kv_cache[i]['ks'] = result[1]
            model._kv_cache[i]['vs'] = result[2]
            model._kv_cache[i]['kf'] = result[3]
            model._kv_cache[i]['vf'] = result[4]
        model._kv13_k = result[5]
        model._kv13_v = result[6]
        model._kv14_k = result[7]
        model._kv14_v = result[8]

    # Final norm
    hidden = model.norm(hidden)
    return hidden  # (1, 1, hidden_size)


def reset_kv(model):
    model._kv_cache = None
    model._kv13_k = None
    model._kv13_v = None
    model._kv14_k = None
    model._kv14_v = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="Path to HF model (for weights + tokenizer). Auto-downloads if not set.")
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Get HF model path
    hf_dir = args.hf_dir
    if hf_dir is None:
        from huggingface_hub import snapshot_download
        print("Downloading google/gemma-4-E2B-it...")
        hf_dir = snapshot_download("google/gemma-4-E2B-it")

    print(f"Loading custom Gemma4Model from {hf_dir}...")
    model = Gemma4Model.from_pretrained(hf_dir, context_length=args.seq_len)
    model = model.half().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    hidden_size = model.config.hidden_size
    embed_scale = hidden_size ** 0.5
    # LM head weight for target token computation
    lm_head_weight = model.lm_head.weight.data.squeeze(-1).squeeze(-1).clone().cpu().half()  # Conv2d → (vocab, hidden)
    print(f"hidden_size={hidden_size}, embed_scale={embed_scale:.2f}, vocab={lm_head_weight.shape[0]}")

    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"  {len(texts)} sequences")

    # Collect hidden states
    all_h_in, all_e_in, all_h_tgt, all_tok_tgt = [], [], [], []
    num = min(args.num_samples, len(texts))
    collected = 0
    skipped = 0
    total_pairs = 0
    t0 = time.time()

    with torch.no_grad():
        for text in tqdm(texts[:num], desc="Collecting"):
            ids = tokenizer.encode(text, return_tensors="pt",
                                   truncation=True, max_length=args.seq_len)
            N = ids.shape[1]
            if N < 32:
                skipped += 1
                continue

            # Run token-by-token through custom model
            reset_kv(model)
            hiddens = []  # collect post-norm hidden for each position

            for t in range(N):
                tok = ids[:, t:t+1].to(device).int()
                h = forward_all(model, tok, t)  # (1, 1, hidden)
                hiddens.append(h[0, 0].cpu().half())  # (hidden,)

            hiddens = torch.stack(hiddens, dim=0)  # (N, hidden)
            embeds = model.embed_tokens(ids.to(device))[0].cpu().half() * embed_scale  # (N, hidden)

            # EAGLE pairs: (h[t], embed(tok[t+1])) → h[t+1]
            all_h_in.append(hiddens[:-1])
            all_e_in.append(embeds[1:])
            all_h_tgt.append(hiddens[1:])

            logits = F.linear(hiddens[1:].float(), lm_head_weight.float())
            all_tok_tgt.append(logits.argmax(dim=-1))

            total_pairs += N - 1
            collected += 1

            if collected % 500 == 0:
                elapsed = time.time() - t0
                rate = collected / elapsed
                eta = (num - collected) / rate if rate > 0 else 0
                print(f"  {collected}/{num}, {total_pairs:,} pairs, "
                      f"{rate:.1f} seq/s, ETA {eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone: {collected} seqs, {total_pairs:,} pairs in {elapsed:.0f}s")
    print(f"  Skipped {skipped} short")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Concatenate + split
    h_in = torch.cat(all_h_in, dim=0)
    e_in = torch.cat(all_e_in, dim=0)
    h_tgt = torch.cat(all_h_tgt, dim=0)
    tok_tgt = torch.cat(all_tok_tgt, dim=0)

    M = h_in.shape[0]
    split = int(M * (1 - args.test_split))
    perm = torch.randperm(M)

    save_dict = {
        "train_h_in": h_in[perm[:split]], "train_e_in": e_in[perm[:split]],
        "train_h_tgt": h_tgt[perm[:split]], "train_tok_tgt": tok_tgt[perm[:split]],
        "test_h_in": h_in[perm[split:]], "test_e_in": e_in[perm[split:]],
        "test_h_tgt": h_tgt[perm[split:]], "test_tok_tgt": tok_tgt[perm[split:]],
        "lm_head_weight": lm_head_weight,
        "embed_scale": embed_scale,
        "hidden_size": hidden_size,
        "meta": {
            "model_id": "custom_gemma4_model",
            "teacher": "Gemma4Model (Conv2d, same as CoreML chunks)",
            "num_sequences": collected,
            "seq_len": args.seq_len,
            "total_pairs": M,
            "train_pairs": split,
            "test_pairs": M - split,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(save_dict, args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"\nSaved: {args.output} ({size_gb:.2f} GB)")
    print(f"  Train: {split:,} / Test: {M-split:,}")
    print(f"\nNext: load this in train_eagle3_draft.ipynb (no target model needed).")


if __name__ == "__main__":
    raise SystemExit(main())
