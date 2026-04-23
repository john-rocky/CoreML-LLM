"""Qwen3-VL 4B text-backbone decode split into 6 body chunks + a tail.

Ships the text path of `Qwen/Qwen3-VL-4B-Instruct` on iPhone ANE.
Vision tower (`.model.visual`) is intentionally dropped — Phase 2 will
add it as a separate mlpackage + DeepStack injection.

Architecture:
  * 36 `Qwen3VLTextDecoderLayer` — plain GQA attention, NO hybrid SSM.
  * head_dim = 128, num_kv_heads = 8, num_heads = 32 (GQA 4:1)
  * q_norm / k_norm RMSNorm on Q and K before RoPE (Qwen3-style)
  * mRoPE with mrope_section [24,20,20] and rope_theta=5e6 — but for
    TEXT-ONLY input T=H=W=position, so the multimodal interleave
    collapses to standard full-dim 1D RoPE. Positional inputs to each
    chunk are precomputed `cos`/`sin` of shape (1, 1, head_dim) per
    step, matching the shape the decoder layer expects after
    `unsqueeze(1)`.
  * tie_word_embeddings = True → lm_head shares embed_tokens weight.

Layout on disk (mirrors v1.1.0 Qwen3.5 2B pattern):
  embed_weight.bin            — raw fp16, (vocab=151936, hidden=2560)
                                Swift mmaps it; per-step 4 KB memcpy of
                                one row replaces a CoreML gather chunk.
  chunk_0..5.mlpackage        — 6 layers each, INT8 palettized.
  chunk_head.mlpackage        — final_norm + lm_head, INT8 palettized.

Per-step decode chain:
  Swift embed lookup → chunk_0 → ... → chunk_5 → chunk_head → logits

Each chunk takes hidden_in + position + cos + sin + its 6 layers' KV
cache (k_cache_i, v_cache_i), emits hidden (or logits for head) +
updated KV caches. Same fp16 hidden handoff as the 2B pattern.
"""
from pathlib import Path
import argparse
import shutil
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_SEQ = 2048          # decode mlpackage max context
NUM_BODY_CHUNKS = 6     # 36 layers / 6 = 6 per chunk
LAYERS_PER_CHUNK = 6


def load_text_config() -> Qwen3VLTextConfig:
    full = Qwen3VLTextConfig.from_pretrained(MODEL_ID)
    return full


def load_text_backbone():
    """Load Qwen3-VL 4B in fp32 and return the text backbone + lm_head.
    The vision tower is dropped — we only pull `.model.language_model`
    and `.lm_head` out of the full ConditionalGeneration."""
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    return full.model.language_model, full.lm_head


# ---- RMSNorm + MLP --------------------------------------------------------

class DecodeRMSNorm(nn.Module):
    """Qwen3 RMSNorm (matches HF's Qwen3VLTextRMSNorm): x / sqrt(var + eps) * w.
    fp32 internals to avoid drift; cast back to input dtype at the end."""
    def __init__(self, eps, weight):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * self.w.float()
        return x.to(in_dtype)


class DecodeMLP(nn.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""
    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        self.gate_w = nn.Parameter(gate_w.detach().clone(), requires_grad=False)
        self.up_w = nn.Parameter(up_w.detach().clone(), requires_grad=False)
        self.down_w = nn.Parameter(down_w.detach().clone(), requires_grad=False)

    def forward(self, x):
        g = F.silu(F.linear(x, self.gate_w))
        u = F.linear(x, self.up_w)
        return F.linear(g * u, self.down_w)


def rotate_half(x):
    """Standard RoPE half-rotation: [x1, x2] → [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE to (q, k). cos/sin shape: (1, 1, head_dim).
    q, k shape: (1, num_heads, 1, head_dim)."""
    cos = cos.unsqueeze(1)  # → (1, 1, 1, head_dim) broadcast across heads
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class DecodeAttention(nn.Module):
    """Qwen3-VL text attention for the per-step decode path.

    Builds on the HF `Qwen3VLTextAttention` forward:
      q = q_norm(q_proj(x))   # per-head RMSNorm on Q
      k = k_norm(k_proj(x))
      v =          v_proj(x)
      q, k = apply_rope(q, k, cos, sin)       # full head_dim=128 RoPE
      k_cache' = where(positions == position, k, k_cache)   # scatter-free
      v_cache' = where(positions == position, v, v_cache)
      attn = softmax(Q @ K_cache'^T / √d + causal_mask) @ V_cache'
      out = o_proj(attn)
    """
    def __init__(self, cfg, hf_layer, max_seq):
        super().__init__()
        attn = hf_layer.self_attn
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads
        self.max_seq = max_seq
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj_w = nn.Parameter(attn.q_proj.weight.detach().clone(),
                                      requires_grad=False)
        self.k_proj_w = nn.Parameter(attn.k_proj.weight.detach().clone(),
                                      requires_grad=False)
        self.v_proj_w = nn.Parameter(attn.v_proj.weight.detach().clone(),
                                      requires_grad=False)
        self.o_proj_w = nn.Parameter(attn.o_proj.weight.detach().clone(),
                                      requires_grad=False)
        # Qwen3 adds RMSNorm on per-head Q and K before RoPE
        self.q_norm = DecodeRMSNorm(cfg.rms_norm_eps, attn.q_norm.weight)
        self.k_norm = DecodeRMSNorm(cfg.rms_norm_eps, attn.k_norm.weight)

        # Scatter-free cache update: compare range to position, then select.
        # Register a constant `positions` buffer of shape (1, 1, max_seq, 1)
        # so the compare broadcasts against k_cache / v_cache.
        self.register_buffer(
            "positions",
            torch.arange(max_seq, dtype=torch.float32).view(1, 1, max_seq, 1),
            persistent=False,
        )
        # Causal mask: additive, fp32. Positions > current are -inf.
        # Shape: (1, 1, 1, max_seq). Built per-step from position.
        # We compute it inline inside forward to keep it position-dependent.

    def forward(self, hidden, position, cos, sin, k_cache, v_cache):
        # hidden: (1, 1, hidden_size)
        # position: (1,) — scalar step index as float
        # cos, sin: (1, 1, head_dim)
        # k_cache, v_cache: (1, num_kv_heads, max_seq, head_dim) fp16
        B = 1
        S = 1
        H = self.num_heads
        HKV = self.num_kv_heads
        D = self.head_dim

        q = F.linear(hidden, self.q_proj_w).view(B, S, H, D).transpose(1, 2)    # (1, H, 1, D)
        k = F.linear(hidden, self.k_proj_w).view(B, S, HKV, D).transpose(1, 2)  # (1, HKV, 1, D)
        v = F.linear(hidden, self.v_proj_w).view(B, S, HKV, D).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        # Scatter-free KV cache update.
        # positions: (1, 1, max_seq, 1); position: (1,) → (1, 1, 1, 1) after view
        # mask: (1, 1, max_seq, 1) bool → broadcast to full k_cache/v_cache shape.
        pos = position.view(1, 1, 1, 1)
        mask = self.positions.eq(pos)  # fp32 == broadcast → bool (1, 1, max_seq, 1)
        # k shape (1, HKV, 1, D) → broadcast to (1, HKV, max_seq, D) with mask
        k_new = torch.where(mask, k.expand(-1, -1, self.max_seq, -1), k_cache)
        v_new = torch.where(mask, v.expand(-1, -1, self.max_seq, -1), v_cache)

        # Repeat K/V across query-head groups for GQA.
        # (1, HKV, max_seq, D) → (1, H, max_seq, D) via repeat_interleave.
        k_rep = k_new.repeat_interleave(self.num_heads_per_kv, dim=1)
        v_rep = v_new.repeat_interleave(self.num_heads_per_kv, dim=1)

        # Attention scores: (1, H, 1, D) @ (1, H, D, max_seq) → (1, H, 1, max_seq)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) * self.scale
        # Causal mask: positions > current must be -inf.
        causal = (self.positions.view(1, 1, 1, self.max_seq) > pos).to(scores.dtype) * -1e4
        scores = scores + causal
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)  # (1, H, 1, D)
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)
        out = F.linear(out, self.o_proj_w)
        return out, k_new, v_new


class DecoderDecodeLayer(nn.Module):
    """One Qwen3-VL text decoder layer in decode form: pre-attn norm →
    self-attn (with KV cache update) → residual → pre-mlp norm → MLP
    → residual."""
    def __init__(self, cfg, hf_layer, max_seq):
        super().__init__()
        self.input_layernorm = DecodeRMSNorm(cfg.rms_norm_eps, hf_layer.input_layernorm.weight)
        self.post_attn_layernorm = DecodeRMSNorm(
            cfg.rms_norm_eps, hf_layer.post_attention_layernorm.weight)
        self.attn = DecodeAttention(cfg, hf_layer, max_seq)
        self.mlp = DecodeMLP(
            hf_layer.mlp.gate_proj.weight,
            hf_layer.mlp.up_proj.weight,
            hf_layer.mlp.down_proj.weight,
        )

    def forward(self, hidden, position, cos, sin, k_cache, v_cache):
        residual = hidden
        h = self.input_layernorm(hidden)
        attn_out, k_new, v_new = self.attn(h, position, cos, sin, k_cache, v_cache)
        hidden = residual + attn_out
        residual = hidden
        h = self.post_attn_layernorm(hidden)
        mlp_out = self.mlp(h)
        hidden = residual + mlp_out
        return hidden, k_new, v_new


# ---- Chunk modules --------------------------------------------------------

class DecodeChunkBody(nn.Module):
    """Body chunk: hidden_in → layers [start, end) → hidden, with
    per-layer KV cache as I/O tensors."""
    def __init__(self, cfg, hf_layers, start, end, max_seq):
        super().__init__()
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([
            DecoderDecodeLayer(cfg, hf_layers[i], max_seq)
            for i in range(start, end)
        ])

    def forward(self, hidden_in, position, cos, sin, *kv_states):
        hidden = hidden_in
        new_states = []
        for local_i, layer in enumerate(self.layers):
            k = kv_states[2 * local_i]
            v = kv_states[2 * local_i + 1]
            hidden, k_new, v_new = layer(hidden, position, cos, sin, k, v)
            new_states.append(k_new)
            new_states.append(v_new)
        return (hidden, *new_states)


class DecodeChunkHead(nn.Module):
    """Tail chunk: hidden → final_norm → lm_head → logits. Stateless."""
    def __init__(self, cfg, hf_text_model, lm_head, tie_word_embeddings):
        super().__init__()
        self.final_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_text_model.norm.weight)
        # tie_word_embeddings=True → use embed weight as lm_head weight
        lm_w = (hf_text_model.embed_tokens.weight
                if tie_word_embeddings else lm_head.weight)
        self.lm_head_w = nn.Parameter(lm_w.detach().clone(), requires_grad=False)

    def forward(self, hidden_in):
        hidden = self.final_norm(hidden_in)
        return F.linear(hidden, self.lm_head_w)


# ---- embed sidecar export -------------------------------------------------

def export_embed_fp16(embed_weight: torch.Tensor, out_path: Path) -> None:
    """Save `embed_tokens.weight` as contiguous fp16 row-major for Swift
    mmap. Same pattern as Qwen3.5 2B v1.1.0 — avoids CoreML gather
    dequantizing the embed into resident memory."""
    w = embed_weight.detach().to(torch.float16).contiguous()
    vocab, hidden = w.shape
    print(f"\n--- export embed_weight.bin ({vocab} × {hidden} fp16) ---")
    buf = w.cpu().numpy().astype(np.float16).tobytes()
    out_path.write_bytes(buf)
    mb = len(buf) / 1e6
    print(f"  wrote {out_path.name} ({mb:.0f} MB)")


# ---- CoreML convert + palettize + audit -----------------------------------

def _audit_ane(out_path: Path) -> float:
    reloaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    dev = Counter()
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = ("const" if (a is None and op.operator_name == "const")
                 else (a.preferred_compute_device.__class__.__name__ if a else "unknown"))
            dev[d] += 1
    total = sum(dev.values())
    compute = total - dev.get("const", 0)
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    pct = 100 * ane / compute if compute else 0.0
    print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%)")
    return pct


def _kv_shape(cfg, max_seq):
    return (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)


def convert_body(chunk, cfg, start_layer, end_layer, max_seq, out_path):
    print(f"\n--- convert body chunk layers [{start_layer}, {end_layer}) ---")
    head_dim = cfg.head_dim

    example = [torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float32)]
    example.append(torch.zeros(1, dtype=torch.float32))                 # position
    example.append(torch.zeros(1, 1, head_dim, dtype=torch.float32))    # cos
    example.append(torch.zeros(1, 1, head_dim, dtype=torch.float32))    # sin
    for _ in range(start_layer, end_layer):
        example.append(torch.zeros(*_kv_shape(cfg, max_seq)))  # k
        example.append(torch.zeros(*_kv_shape(cfg, max_seq)))  # v

    t0 = time.time()
    traced = torch.jit.trace(chunk, tuple(example), strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, 1, head_dim), dtype=np.float16),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    for i in range(start_layer, end_layer):
        ct_inputs.append(ct.TensorType(
            name=f"k_{i}", shape=_kv_shape(cfg, max_seq), dtype=np.float16))
        ct_inputs.append(ct.TensorType(
            name=f"v_{i}", shape=_kv_shape(cfg, max_seq), dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_k_{i}", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_v_{i}", dtype=np.float16))

    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def convert_head(chunk, cfg, out_path):
    print(f"\n--- convert tail (final_norm + lm_head) ---")
    example = (torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float32),)
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")
    ct_inputs = [ct.TensorType(
        name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16)]
    ct_outputs = [ct.TensorType(name="logits", dtype=np.float32)]
    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def palettize_pkg(fp16_pkg: Path, out_pkg: Path, nbits: int):
    print(f"\n--- palettize INT{nbits}: {fp16_pkg.name} → {out_pkg.name} ---")
    m_in = ct.models.MLModel(str(fp16_pkg))
    op_cfg = OpPalettizerConfig(mode="kmeans", nbits=nbits, granularity="per_tensor")
    opt_cfg = OptimizationConfig(global_config=op_cfg)
    t0 = time.time()
    m_out = palettize_weights(m_in, opt_cfg)
    print(f"  palettize done in {time.time()-t0:.1f}s")
    m_out.save(str(out_pkg))
    src_mb = sum(f.stat().st_size for f in fp16_pkg.rglob('*') if f.is_file()) / 1e6
    dst_mb = sum(f.stat().st_size for f in out_pkg.rglob('*') if f.is_file()) / 1e6
    print(f"  bundle: {src_mb:.0f} MB (fp16) → {dst_mb:.0f} MB (int{nbits}) "
          f"[{100*dst_mb/src_mb:.1f}%]")
    _audit_ane(out_pkg)


# ---- main -----------------------------------------------------------------

EMBED_BIN_NAME = "embed_weight.bin"
BODY_CHUNK_NAMES = [f"chunk_{i}" for i in range(NUM_BODY_CHUNKS)]
HEAD_CHUNK_NAME = "chunk_head"


def _body_boundaries(num_layers, num_chunks):
    assert num_layers % num_chunks == 0, \
        f"num_layers={num_layers} not divisible by num_chunks={num_chunks}"
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--num-chunks", type=int, default=NUM_BODY_CHUNKS)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_4b_decode_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print("loading Qwen3-VL 4B text backbone (fp32)...")
    t0 = time.time()
    cfg = load_text_config()
    print(f"  text cfg: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"num_kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim} "
          f"tie_word_embeddings={cfg.tie_word_embeddings}")
    text_model, lm_head = load_text_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"(vision tower dropped, {sum(p.numel() for p in text_model.parameters())/1e9:.2f}B "
          f"text params)")

    # Export embed sidecar + build head chunk module before we start
    # trimming per-chunk torch modules, so we can delete the HF model
    # sooner.
    export_embed_fp16(text_model.embed_tokens.weight, chunks_dir / EMBED_BIN_NAME)
    head_module = DecodeChunkHead(cfg, text_model, lm_head,
                                   cfg.tie_word_embeddings).eval().float()

    boundaries = _body_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}")

    # Body modules (each slices a range out of text_model.layers).
    body_modules = []
    for start, end in boundaries:
        m = DecodeChunkBody(cfg, text_model.layers, start, end, args.max_seq)
        body_modules.append(m.eval().float())
    del text_model, lm_head  # free ~16 GB fp32 weights

    # Convert body chunks
    for ci, ((start, end), mod, name) in enumerate(
        zip(boundaries, body_modules, BODY_CHUNK_NAMES[:args.num_chunks])
    ):
        fp16_path = fp16_dir / f"{name}.mlpackage"
        final_path = chunks_dir / f"{name}.mlpackage"
        convert_body(mod, cfg, start, end, args.max_seq, fp16_path)
        if args.nbits == 0:
            shutil.move(str(fp16_path), str(final_path))
        else:
            palettize_pkg(fp16_path, final_path, args.nbits)

    # Convert head
    fp16_head = fp16_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
    final_head = chunks_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
    convert_head(head_module, cfg, fp16_head)
    if args.nbits == 0:
        shutil.move(str(fp16_head), str(final_head))
    else:
        palettize_pkg(fp16_head, final_head, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\n✓ shipping artifacts under {chunks_dir}")
    for p in sorted(chunks_dir.iterdir()):
        if p.is_file():
            size = p.stat().st_size / 1e6
        else:
            size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
