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
    !git clone -q -b claude/eagle3-retrain-fast https://github.com/john-rocky/CoreML-LLM.git
    %cd CoreML-LLM/conversion
    !pip install -q safetensors
    !python collect_eagle_hidden_states_custom.py \
        --corpus /content/drive/MyDrive/eagle_corpus.jsonl \
        --output /content/drive/MyDrive/eagle_draft/training_data_custom.pt \
        --num-samples 30000 --seq-len 512 \
        --batch-size 4 --compile

Runtime on A100 40GB (30k samples, seq_len=512):
    --batch-size 1 (baseline)          : ~7h  (10 GB GPU RAM used)
    --batch-size 4                     : ~1.5–2h  (~30 GB)
    --batch-size 4 + --compile         : ~1–1.2h  (~30 GB, first call ~1 min compile)
Numerics: batching is bit-identical to B=1 modulo matmul reduction order
(<1e-5 per element), since causal attention prevents padded positions
[Ni..max_N-1] from affecting real positions [0..Ni-1] in any sample.
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time
import numpy as np
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
    device = input_ids.device

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

    # Masks — ensure all on same device as model
    mask_full = torch.full((1, 1, 1, ctx), -65504.0, dtype=MODEL_DTYPE, device=device)
    mask_full[0, 0, 0, :position + 1] = 0
    mask_sliding = torch.full((1, 1, 1, W), -65504.0, dtype=MODEL_DTYPE, device=device)
    valid = min(position + 1, W)
    mask_sliding[0, 0, 0, W - valid:] = 0
    update_mask = torch.zeros((1, 1, ctx, 1), dtype=MODEL_DTYPE, device=device)
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
                    'ks': torch.zeros(1, nkv, W, max_hd, dtype=MODEL_DTYPE, device=device),
                    'vs': torch.zeros(1, nkv, W, max_hd, dtype=MODEL_DTYPE, device=device),
                    'kf': torch.zeros(1, nkv, ctx, max_hd, dtype=MODEL_DTYPE, device=device),
                    'vf': torch.zeros(1, nkv, ctx, max_hd, dtype=MODEL_DTYPE, device=device),
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


def forward_batch(model, input_ids):
    """Run full 35-layer forward on B sequences, returning hidden states at ALL positions.

    No KV cache — uses full causal attention mask. Conv2d(1×1) naturally
    parallelizes over the sequence dimension (spatial axis).

    Supports sample batching: input_ids shape is (B, seq_len). Padding at
    positions [Ni..seq_len-1] for sample i is safe under causal attention —
    real positions [0..Ni-1] do not attend to padded positions, so hiddens
    at those positions are numerically identical to running B=1 with the
    same Ni-length input.

    Args:
        input_ids: (B, seq_len) int32/int64  — B samples padded to same seq_len

    Returns:
        hidden_states: (B, seq_len, hidden_size) fp16
        fusion_list:   [3 × (B, seq_len, hidden_size)] — L8, L17, L34 fusion hiddens
    """
    config = model.config
    device = input_ids.device
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    nlayers = config.num_hidden_layers

    # Embedding: (B, seq_len, hidden)
    embed_scale = config.hidden_size ** 0.5
    hidden = model.embed_tokens(input_ids).to(MODEL_DTYPE) * embed_scale

    # Per-layer embedding: same as forward_all but batched
    pl_raw = model.embed_tokens_per_layer(input_ids).to(MODEL_DTYPE)
    pl_raw = pl_raw * (config.hidden_size_per_layer_input ** 0.5)
    h_conv = hidden.permute(0, 2, 1).unsqueeze(2)  # (B, hidden, seq_len, 1) — Conv2d batch!
    pl_proj = model.per_layer_model_projection(h_conv.to(MODEL_DTYPE))  # (B, 8960, seq_len, 1)
    pl_proj = pl_proj.squeeze(2).permute(0, 2, 1)  # (B, seq_len, 8960)
    pl_proj = pl_proj * model.per_layer_model_projection_scale

    pld = config.hidden_size_per_layer_input
    pl_combined_parts = []
    for i in range(nlayers):
        s = i * pld
        e = s + pld
        proj_normed = model.per_layer_projection_norm(pl_proj[:, :, s:e])
        combined = (proj_normed + pl_raw[:, :, s:e]) * model.per_layer_input_scale
        pl_combined_parts.append(combined)
    per_layer_combined = torch.cat(pl_combined_parts, dim=2)

    # RoPE for all positions
    cos_s = model.cos_sliding[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin_s = model.sin_sliding[:seq_len].unsqueeze(0).unsqueeze(0)
    cos_f = model.cos_full[:seq_len].unsqueeze(0).unsqueeze(0)
    sin_f = model.sin_full[:seq_len].unsqueeze(0).unsqueeze(0)

    # Causal mask: (1, 1, seq_len, seq_len)
    causal_mask = torch.full((seq_len, seq_len), -65504.0, dtype=MODEL_DTYPE, device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    # Shared KV storage
    kv13_k = kv13_v = kv14_k = kv14_v = None

    # Collect fusion layer hiddens (EAGLE-3 needs L8, L17, L34)
    FUSION_LAYERS = [8, 17, 34]
    fusion_hiddens = {}

    for i in range(nlayers):
        is_full = config.is_full_attention(i)
        is_kv_shared = config.is_kv_shared(i)
        hd = config.get_head_dim(i)
        layer = model.layers[i]

        residual = hidden
        h = layer.input_layernorm(hidden)
        # Conv2d over (1, hidden, seq_len, 1)
        x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # (1, hidden, seq_len, 1) → Conv2d batches seq_len

        # Q: (B, num_heads*hd, seq_len, 1) → (B, num_heads, seq_len, hd)
        q = layer.self_attn["q_proj"](x)
        q = q.view(B, num_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        # Q norm: reshape to (B, num_heads*seq_len, hd), norm, reshape back
        q = layer.self_attn["q_norm"](q.reshape(B, num_heads * seq_len, hd))
        q = q.view(B, num_heads, seq_len, hd)

        if not is_kv_shared:
            k = layer.self_attn["k_proj"](x)
            k = k.view(B, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            k = layer.self_attn["k_norm"](k.reshape(B, num_kv_heads * seq_len, hd))
            k = k.view(B, num_kv_heads, seq_len, hd)

            v = layer.self_attn["v_proj"](x)
            v = v.view(B, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            v = v_norm(v)

            # RoPE
            if is_full:
                q, k = apply_rotary_pos_emb(q, k, cos_f, sin_f)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)

            # Store kv13/kv14
            if i == 13:
                kv13_k, kv13_v = k, v
            elif i == 14:
                kv14_k, kv14_v = k, v
        else:
            # RoPE on Q only
            if is_full:
                q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
                k, v = kv14_k, kv14_v
            else:
                q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)
                k, v = kv13_k, kv13_v

        # GQA expand
        K_expanded = k.repeat_interleave(n_rep, dim=1)
        V_expanded = v.repeat_interleave(n_rep, dim=1)

        # Attention: (1, heads, seq_len, hd) @ (1, heads, hd, seq_len) → (1, heads, seq_len, seq_len)
        # For sliding window layers, use causal mask (full seq causal, not windowed —
        # training data collection doesn't need exact sliding window behavior,
        # hidden states at position t only depend on 0..t regardless)
        attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
        attn_weights = attn_weights + causal_mask
        attn_weights = ane_softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V_expanded)

        # Output projection
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, seq_len, -1)
        attn_output = layer.self_attn["o_proj"](
            attn_output.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden = residual + attn_output

        # MLP
        residual = hidden
        h = layer.pre_feedforward_layernorm(hidden)
        x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (B, hidden, seq_len, 1)
        gate = layer.mlp["gate_proj"](x_mlp)
        up = layer.mlp["up_proj"](x_mlp)
        gate = F.gelu(gate, approximate="tanh")
        mlp_out = layer.mlp["down_proj"](gate * up)
        hidden = mlp_out.squeeze(2).permute(0, 2, 1)
        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = residual + hidden

        # Per-layer input
        residual_pl = hidden
        s_idx = i * pld
        e_idx = s_idx + pld
        per_layer_slice = per_layer_combined[:, :, s_idx:e_idx]
        hs_conv = hidden.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        gated = layer.per_layer_input_gate(hs_conv)
        gated = F.gelu(gated, approximate="tanh")
        per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)
        gated = gated * per_layer_slice_conv
        gated = layer.per_layer_projection(gated)
        gated = gated.squeeze(2).permute(0, 2, 1)
        hidden = layer.post_per_layer_input_norm(gated)
        hidden = residual_pl + hidden
        hidden = hidden * layer.layer_scalar.to(MODEL_DTYPE)

        # Record fusion layer hiddens for EAGLE-3
        if i in FUSION_LAYERS:
            fusion_hiddens[i] = hidden.clone()  # (B, seq_len, hidden)

    # Final norm
    hidden = model.norm(hidden)
    # Return: final hidden + fusion layer hiddens (B dimension preserved)
    fusion_list = [fusion_hiddens[l] for l in FUSION_LAYERS]  # list of (B, seq_len, hidden)
    return hidden, fusion_list  # (B, seq_len, hidden), [3 × (B, seq_len, hidden)]


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
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Samples per forward_batch call. A100 40GB fits 4 safely at "
                             "seq_len=512, fp16. Numerics are identical to B=1 under causal "
                             "attention with end-padding (padded positions are discarded).")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap forward_batch in torch.compile(mode=\"reduce-overhead\"). "
                             "Typically 1.3–1.8x on A100. First call pays compile cost (~1 min).")
    args = parser.parse_args()

    device = args.device

    # Let cuDNN benchmark Conv2d algorithms on first call per unique shape. With
    # fixed (B, seq_len) shape this runs once at warmup and caches the fastest
    # kernel per op. Observed empirically: Gemma4Model's Conv2d(1×1) layers on
    # A100 default to a 10× suboptimal algorithm without this flag, making the
    # forward compute-bound at ~0.2% utilization.
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    if not getattr(tokenizer, "is_fast", False):
        print("  WARN: slow (Python) tokenizer in use — install `tokenizers` for Rust fast path")

    hidden_size = model.config.hidden_size
    embed_scale = hidden_size ** 0.5
    # LM head weight for target token computation. Keep both a GPU-resident fp16
    # copy (for fast argmax during collection — vocab=262144 × hidden=1536 fp32
    # matmul on CPU is ~1s/sample, dominates wallclock) and a CPU fp16 copy (for
    # serialization into the output .pt).
    lm_head_weight_gpu = model.lm_head.weight.data.squeeze(-1).squeeze(-1).to(MODEL_DTYPE)  # (vocab, hidden) GPU
    lm_head_weight = lm_head_weight_gpu.detach().cpu().half()
    print(f"hidden_size={hidden_size}, embed_scale={embed_scale:.2f}, vocab={lm_head_weight.shape[0]}")

    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"  {len(texts)} sequences")

    # Collect hidden states + fusion layer hiddens via disk-backed memmap.
    # Rationale: at 30k samples × 512 tokens × 1536 hidden × fp16, the six 2D
    # tensors (h_in, e_in, h_tgt, 3× fusion) would need ~280 GB of CPU RAM if
    # held in lists. Colab high-RAM tops out at 80 GB, so accumulating in-
    # memory OOMs around 38% completion. Instead, allocate sparse memmap files
    # on disk sized to an upper bound (num_samples × (seq_len-1) pairs) and
    # write each batch's output directly at a cursor. Final disk usage is
    # bounded by actual pairs (shorter sequences shrink it); sparse allocation
    # means unwritten regions consume no disk.
    num = min(args.num_samples, len(texts))
    max_pairs = num * (args.seq_len - 1)

    output_abs = os.path.abspath(args.output)
    data_dir = output_abs[:-3] + ".data" if output_abs.endswith(".pt") else output_abs + ".data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"  Streaming memmap files under: {data_dir}")
    print(f"  Upper bound: {max_pairs:,} pairs × hidden={hidden_size} × fp16")
    print(f"    per-tensor cap on disk: {max_pairs * hidden_size * 2 / 1e9:.1f} GB")
    print(f"    7-tensor cap on disk:   {(max_pairs * hidden_size * 2 * 6 + max_pairs * 8) / 1e9:.1f} GB (sparse)")

    def _make_mm(name, shape, dtype):
        path = os.path.join(data_dir, name)
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape), path

    h_in_mm,    h_in_path    = _make_mm("h_in.dat",    (max_pairs, hidden_size), np.float16)
    e_in_mm,    e_in_path    = _make_mm("e_in.dat",    (max_pairs, hidden_size), np.float16)
    h_tgt_mm,   h_tgt_path   = _make_mm("h_tgt.dat",   (max_pairs, hidden_size), np.float16)
    tok_tgt_mm, tok_tgt_path = _make_mm("tok_tgt.dat", (max_pairs,),            np.int64)
    fusion_mm_list = []
    fusion_paths = {}
    for l in [8, 17, 34]:
        mm, p = _make_mm(f"fusion_L{l}.dat", (max_pairs, hidden_size), np.float16)
        fusion_mm_list.append(mm)
        fusion_paths[l] = p

    cursor = 0
    collected = 0
    skipped = 0
    total_pairs = 0
    t0 = time.time()

    # Verify model is on GPU
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  RoPE device: {model.cos_sliding.device}")

    # Optional torch.compile for the forward path. Keep a handle to both so we
    # can fall back if compilation itself fails at import time.
    forward_batch_fn = forward_batch
    if args.compile:
        print(f"  Compiling forward_batch with torch.compile (mode=reduce-overhead)...")
        forward_batch_fn = torch.compile(forward_batch, mode="reduce-overhead", dynamic=False)

    BATCH_SIZE = max(1, args.batch_size)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print(f"  Batch size: {BATCH_SIZE} (pad_id={pad_id})")

    def _flush_batch(buffer):
        """Process a buffer of (ids_1d_cpu, N): forward + write to memmap at cursor."""
        nonlocal collected, total_pairs, cursor
        if not buffer:
            return 0
        B = len(buffer)
        # With --compile we MUST pad to a FIXED shape or torch.compile re-compiles on
        # every batch whose max_N differs (~40s per recompile on A100, fatal). Without
        # --compile, tighter in-batch max_N saves a bit of wasted padding compute.
        if args.compile:
            PAD_TO = args.seq_len
        else:
            PAD_TO = max(N for _, N in buffer)
        batched = torch.full((B, PAD_TO), pad_id, dtype=torch.long, device=device)
        for bi, (ids_1d, N) in enumerate(buffer):
            batched[bi, :N] = ids_1d.to(device)

        # Forward
        hiddens_B, fusion_list_B = forward_batch_fn(model, batched)
        # hiddens_B: (B, max_N, hidden) ; fusion_list_B: [3 × (B, max_N, hidden)]
        embeds_B = model.embed_tokens(batched).to(MODEL_DTYPE) * embed_scale  # (B, max_N, hidden)

        # LM head argmax on GPU. MUST be fp32 — with vocab=262144 and
        # hidden=1536, fp16 matmul overflows to Inf/NaN in ~some output
        # columns, causing argmax to consistently land on token 0. Using
        # fp32 keeps it correct at ~3 GB extra transient memory per batch
        # (B=16, seq=512, vocab=262144 fp32 logits ≈ 8.5 GB).
        logits_B = F.linear(hiddens_B[:, 1:].float(), lm_head_weight_gpu.float())
        tok_tgt_B = logits_B.argmax(dim=-1)  # (B, max_N-1) int64

        # Single GPU→CPU sync for the whole batch (amortized)
        hiddens_B_cpu = hiddens_B.cpu().half()
        embeds_B_cpu = embeds_B.cpu().half()
        fusion_cpu = [f.cpu().half() for f in fusion_list_B]
        tok_tgt_B_cpu = tok_tgt_B.cpu()

        for bi, (_, N) in enumerate(buffer):
            n_pairs = N - 1
            if cursor + n_pairs > max_pairs:
                print(f"  WARN: memmap upper bound hit at cursor={cursor}, skipping remainder")
                break

            hiddens_np = hiddens_B_cpu[bi, :N].numpy()  # (N, hidden) fp16
            embeds_np  = embeds_B_cpu[bi, :N].numpy()

            h_in_mm[cursor:cursor + n_pairs]  = hiddens_np[:-1]
            e_in_mm[cursor:cursor + n_pairs]  = embeds_np[1:]
            h_tgt_mm[cursor:cursor + n_pairs] = hiddens_np[1:]
            tok_tgt_mm[cursor:cursor + n_pairs] = tok_tgt_B_cpu[bi, :n_pairs].numpy().astype(np.int64)
            for fi, fm in enumerate(fusion_mm_list):
                fm[cursor:cursor + n_pairs] = fusion_cpu[fi][bi, :n_pairs].numpy()

            cursor += n_pairs
            total_pairs += n_pairs
            collected += 1

            if collected % 500 == 0:
                elapsed = time.time() - t0
                rate = collected / elapsed
                eta = (num - collected) / rate if rate > 0 else 0
                print(f"  {collected}/{num}, {total_pairs:,} pairs, "
                      f"{rate:.1f} seq/s, ETA {eta/60:.0f}min")
        return B

    buffer = []
    with torch.no_grad():
        for text in tqdm(texts[:num], desc="Collecting"):
            ids = tokenizer.encode(text, return_tensors="pt",
                                   truncation=True, max_length=args.seq_len)
            N = ids.shape[1]
            if N < 32:
                skipped += 1
                continue
            buffer.append((ids[0], N))  # keep on CPU until flush

            if len(buffer) >= BATCH_SIZE:
                _flush_batch(buffer)
                buffer = []

        # Flush remainder (may be < BATCH_SIZE). Under --compile, a smaller B
        # triggers recompilation; skip the tail instead (≤ BATCH_SIZE-1 samples lost).
        if buffer:
            if args.compile and len(buffer) != BATCH_SIZE:
                print(f"  Skipping {len(buffer)} tail samples to avoid recompile under --compile")
            else:
                _flush_batch(buffer)
            buffer = []

    elapsed = time.time() - t0
    print(f"\nDone: {collected} seqs, {total_pairs:,} pairs in {elapsed:.0f}s")
    print(f"  Skipped {skipped} short")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Flush + trim memmap files to actual used size (cursor). The files were
    # allocated for upper-bound max_pairs; unused rows at the tail are
    # truncated off disk.
    M = cursor
    for mm in [h_in_mm, e_in_mm, h_tgt_mm, tok_tgt_mm, *fusion_mm_list]:
        mm.flush()
    del h_in_mm, e_in_mm, h_tgt_mm, tok_tgt_mm, fusion_mm_list
    gc.collect()

    # Truncate each .dat file to actual size
    def _truncate(path, row_bytes, rows):
        os.truncate(path, row_bytes * rows)

    _truncate(h_in_path,    hidden_size * 2, M)
    _truncate(e_in_path,    hidden_size * 2, M)
    _truncate(h_tgt_path,   hidden_size * 2, M)
    _truncate(tok_tgt_path, 8, M)
    for l in [8, 17, 34]:
        _truncate(fusion_paths[l], hidden_size * 2, M)

    # Random train/test split over M pairs. Index arrays are small (120 MB max),
    # safe to hold in RAM.
    split = int(M * (1 - args.test_split))
    perm = torch.randperm(M)
    train_idx = perm[:split].contiguous()
    test_idx  = perm[split:].contiguous()

    # Manifest .pt — tiny, only holds metadata and small tensors (indices + LM head).
    # The large tensors stay on disk as memmap files under `data_dir`. The trainer
    # side loads them via np.memmap and wraps in torch.from_numpy for fancy
    # indexing in the existing batch-index pipeline (no IterableDataset needed).
    manifest = {
        "format": "memmap-v1",
        "data_dir": data_dir,
        "total_pairs": M,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "shapes": {
            "h_in":      (M, hidden_size),
            "e_in":      (M, hidden_size),
            "h_tgt":     (M, hidden_size),
            "tok_tgt":   (M,),
            "fusion_L8":  (M, hidden_size),
            "fusion_L17": (M, hidden_size),
            "fusion_L34": (M, hidden_size),
        },
        "dtypes": {
            "h_in": "float16", "e_in": "float16", "h_tgt": "float16",
            "tok_tgt": "int64",
            "fusion_L8": "float16", "fusion_L17": "float16", "fusion_L34": "float16",
        },
        "fusion_layers": [8, 17, 34],
        "lm_head_weight": lm_head_weight,
        "embed_scale": embed_scale,
        "hidden_size": hidden_size,
        "meta": {
            "model_id": "custom_gemma4_model",
            "teacher": "Gemma4Model (Conv2d, same as CoreML chunks)",
            "fusion_layers": [8, 17, 34],
            "num_sequences": collected,
            "seq_len": args.seq_len,
            "total_pairs": M,
            "train_pairs": split,
            "test_pairs": M - split,
        },
    }
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    torch.save(manifest, args.output)

    manifest_mb = os.path.getsize(args.output) / 1e6
    data_bytes = sum(
        os.path.getsize(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
    )
    print(f"\nSaved manifest: {args.output} ({manifest_mb:.1f} MB — contains lm_head_weight + indices)")
    print(f"  Data dir:     {data_dir} ({data_bytes / 1e9:.1f} GB across 7 memmap files)")
    print(f"  Train: {split:,} / Test: {M-split:,} pairs")
    print(f"\nNext: run train_eagle3_standalone.py — it auto-detects format=memmap-v1 and")
    print(f"      loads the memmap tensors with constant CPU RAM usage.")


if __name__ == "__main__":
    raise SystemExit(main())
