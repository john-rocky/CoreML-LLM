"""Qwen3-VL 4B text-decode converter — ANE-OPTIMIZED variant.

Drop-in replacement for `build_qwen3_vl_4b_text_decode_chunks.py` that
applies the Gemma 4 ANE recipe (proven 11→31 tok/s speedup):

  1. **Conv2dLinear** for ALL projections (q/k/v/o/gate/up/down/lm_head)
     — ANE executes Conv2d ~3× faster than MatMul.
  2. **ANERMSNorm** for input_layernorm / post_attention_layernorm /
     final_norm / q_norm / k_norm — `cat([x,-x])` → LayerNorm → slice
     replaces RMSNorm with ANE's optimized LayerNorm kernel.
  3. **repeat_kv_ane** instead of `torch.repeat_interleave` — pure
     reshape+repeat+view, ANE-resident.
  4. **InModelArgmax** in chunk_head — returns `next_token: int32 (1,1)`
     instead of fp32 logits (1, 1, 151936). Removes the 600 KB
     ANE→Swift transfer per step.

Hidden layout convention: keep tensors in **`(B, hidden, 1, S)` Conv2d
form** throughout each chunk. Permutes happen only at chunk boundaries
(Swift sees fp16 `(1, 1, hidden)` like the v1 converter so the embed
mmap and `hidden_in`/`hidden` MLMultiArrays don't change shape).

Output paths land under `qwen3_vl_4b_decode_chunks/` so the existing
Swift loader picks them up unchanged.
"""
from pathlib import Path
import argparse
import shutil
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import (
    MODEL_DTYPE, ANERMSNorm, Conv2dLinear, InModelArgmax,
    apply_rotary_pos_emb, repeat_kv_ane,
)


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_SEQ = 2048
NUM_BODY_CHUNKS = 6
LAYERS_PER_CHUNK = 6


def load_text_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig.from_pretrained(MODEL_ID)


def load_text_backbone():
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    return full.model.language_model, full.lm_head


# ---- ANE-form decoder layer ------------------------------------------------

def _conv_from_linear(lin: nn.Linear) -> Conv2dLinear:
    """Build Conv2dLinear from an nn.Linear, copying weights (out, in) →
    (out, in, 1, 1)."""
    c = Conv2dLinear(lin.in_features, lin.out_features,
                      bias=lin.bias is not None, dtype=MODEL_DTYPE)
    c.conv.weight.data = lin.weight.data.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
    if lin.bias is not None:
        c.conv.bias.data = lin.bias.data.detach().to(MODEL_DTYPE)
    return c


def _norm_from_hf(weight: torch.Tensor, eps: float, hidden: int) -> ANERMSNorm:
    """Build ANERMSNorm from an HF RMSNorm weight tensor."""
    n = ANERMSNorm(hidden, eps=eps)
    n.weight.data = weight.detach().to(MODEL_DTYPE).clone()
    return n


class ANEDecoderLayer(nn.Module):
    """One Qwen3-VL text decoder layer in ANE-friendly form.

    Input/output: hidden in **Conv2d layout** (1, hidden, 1, 1) — caller
    permutes once at chunk entry and once at chunk exit, NOT per layer.

    Per-step:
      pre_norm(input_layernorm) → q/k/v Conv2d projections (in conv form)
        → reshape to (1, H, 1, D) for attention
        → q_norm/k_norm (ANERMSNorm) → RoPE
        → KV cache scatter-free update
        → repeat_kv_ane for GQA → attention scores → ane_softmax → matmul
        → o_proj Conv2d → residual
        → post_attn_layernorm → SwiGLU MLP (Conv2d gate/up/down) → residual
    """
    def __init__(self, cfg, hf_layer, max_seq):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.scale = 1.0 / (self.head_dim ** 0.5)

        attn = hf_layer.self_attn
        self.q_proj = _conv_from_linear(attn.q_proj)
        self.k_proj = _conv_from_linear(attn.k_proj)
        self.v_proj = _conv_from_linear(attn.v_proj)
        self.o_proj = _conv_from_linear(attn.o_proj)
        self.q_norm = _norm_from_hf(attn.q_norm.weight, cfg.rms_norm_eps, self.head_dim)
        self.k_norm = _norm_from_hf(attn.k_norm.weight, cfg.rms_norm_eps, self.head_dim)

        self.input_layernorm = _norm_from_hf(
            hf_layer.input_layernorm.weight, cfg.rms_norm_eps, self.hidden_size)
        self.post_attn_layernorm = _norm_from_hf(
            hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps, self.hidden_size)

        self.gate_proj = _conv_from_linear(hf_layer.mlp.gate_proj)
        self.up_proj = _conv_from_linear(hf_layer.mlp.up_proj)
        self.down_proj = _conv_from_linear(hf_layer.mlp.down_proj)

        self.register_buffer(
            "positions",
            torch.arange(max_seq, dtype=torch.float32).view(1, 1, max_seq, 1),
            persistent=False,
        )

    def _norm_in_conv_form(self, x_conv: torch.Tensor, norm: ANERMSNorm
                           ) -> torch.Tensor:
        """Apply ANERMSNorm to a (1, hidden, 1, 1) Conv2d-layout tensor.
        ANERMSNorm expects (..., hidden), so we view-out and view-in."""
        # (1, hidden, 1, 1) → (1, 1, hidden) → norm → (1, hidden, 1, 1)
        x = x_conv.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)
        x = norm(x)
        return x.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)

    def forward(self, hidden_conv, position, cos, sin, k_cache, v_cache):
        """
        hidden_conv: (1, hidden, 1, 1) fp16  — Conv2d layout
        position:    (1,)                fp32 scalar step index
        cos, sin:    (1, 1, head_dim)    fp16 RoPE for this position
        k_cache, v_cache: (1, num_kv_heads, max_seq, head_dim) fp16
        """
        residual = hidden_conv

        # --- input_layernorm (ANERMSNorm) ---
        h_conv = self._norm_in_conv_form(hidden_conv, self.input_layernorm)

        # --- attention projections (Conv2d) ---
        # q_proj output: (1, num_heads * head_dim, 1, 1)
        q = self.q_proj.forward_conv(h_conv)
        k = self.k_proj.forward_conv(h_conv)
        v = self.v_proj.forward_conv(h_conv)

        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim

        # Reshape to (1, num_heads, 1, head_dim) for attention math.
        # Conv output (1, H*D, 1, 1) → (1, H, D, 1) → (1, H, 1, D)
        q = q.view(1, H, D, 1).permute(0, 1, 3, 2)
        k = k.view(1, HKV, D, 1).permute(0, 1, 3, 2)
        v = v.view(1, HKV, D, 1).permute(0, 1, 3, 2)

        # Q/K norm + RoPE. cos/sin shape (1, 1, D) → broadcast (1, 1, 1, D)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos_b = cos.unsqueeze(1)  # (1, 1, 1, D)
        sin_b = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos_b, sin_b)

        # Scatter-free KV cache update (per Qwen3.5-2B v1.1.0 pattern).
        pos = position.view(1, 1, 1, 1)
        mask = self.positions.eq(pos)  # (1, 1, max_seq, 1) bool
        k_new = torch.where(mask, k.expand(-1, -1, self.max_seq, -1), k_cache)
        v_new = torch.where(mask, v.expand(-1, -1, self.max_seq, -1), v_cache)

        # GQA repeat (ANE-friendly reshape+repeat+view, no repeat_interleave)
        k_rep = repeat_kv_ane(k_new, self.num_heads_per_kv, HKV, self.max_seq, D)
        v_rep = repeat_kv_ane(v_new, self.num_heads_per_kv, HKV, self.max_seq, D)

        # Attention: (1, H, 1, D) @ (1, H, D, max_seq) → (1, H, 1, max_seq)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) * self.scale
        causal = (self.positions.view(1, 1, 1, self.max_seq) > pos).to(scores.dtype) * -1e4
        scores = scores + causal
        # ane_softmax: pure max/sub/exp/sum/div — avoids the softmax op on ANE
        from ane_ops import ane_softmax
        attn = ane_softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)  # (1, H, 1, D)

        # Back to Conv2d layout for o_proj: (1, H, 1, D) → (1, H*D, 1, 1)
        out = out.permute(0, 1, 3, 2).reshape(1, H * D, 1, 1)
        attn_out_conv = self.o_proj.forward_conv(out)
        hidden_conv = residual + attn_out_conv

        residual = hidden_conv
        h_conv = self._norm_in_conv_form(hidden_conv, self.post_attn_layernorm)
        gate = F.silu(self.gate_proj.forward_conv(h_conv))
        up = self.up_proj.forward_conv(h_conv)
        mlp_out = self.down_proj.forward_conv(gate * up)
        hidden_conv = residual + mlp_out

        return hidden_conv, k_new, v_new


# ---- Chunk modules --------------------------------------------------------

class ANEBodyChunk(nn.Module):
    """6-layer body chunk. Keeps hidden in Conv2d form internally; takes
    Swift-side `hidden_in` (1, 1, hidden) as input (1 permute at entry)
    and emits `hidden` (1, 1, hidden) for the next chunk (1 permute at
    exit). Net per-chunk overhead: 2 permutes vs 36 in the v1 converter."""
    def __init__(self, cfg, hf_layers, start, end, max_seq):
        super().__init__()
        self.start = start
        self.end = end
        self.hidden_size = cfg.hidden_size
        self.layers = nn.ModuleList([
            ANEDecoderLayer(cfg, hf_layers[i], max_seq)
            for i in range(start, end)
        ])

    def forward(self, hidden_in, position, cos, sin, *kv_states):
        # (1, 1, hidden) → (1, hidden, 1, 1) Conv2d form
        h = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        new_states = []
        for local_i, layer in enumerate(self.layers):
            k = kv_states[2 * local_i]
            v = kv_states[2 * local_i + 1]
            h, k_new, v_new = layer(h, position, cos, sin, k, v)
            new_states.append(k_new); new_states.append(v_new)
        # Conv2d → (1, 1, hidden) for chunk hand-off
        h_out = h.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)
        return (h_out, *new_states)


class ANEHeadChunk(nn.Module):
    """Tail: final_norm + lm_head + InModelArgmax. Returns the int32 next
    token directly so the per-step ANE→Swift transfer drops from
    ~600 KB (vocab fp32) to 4 bytes (single int32)."""
    def __init__(self, cfg, hf_text_model, lm_head, tie_word_embeddings):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.final_norm = _norm_from_hf(
            hf_text_model.norm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        # tie_word_embeddings: head.weight = embed_tokens.weight; clone
        # before wrapping so the embed export keeps its original
        # (already-cast) tensor.
        if tie_word_embeddings:
            lm_w = hf_text_model.embed_tokens.weight
        else:
            lm_w = lm_head.weight
        # Build Conv2d for the lm_head — vocab is huge (151936) but
        # Conv2d with kernel=1 is still the fastest path on ANE.
        self.lm_head = Conv2dLinear(cfg.hidden_size, cfg.vocab_size,
                                     bias=False, dtype=MODEL_DTYPE)
        self.lm_head.conv.weight.data = lm_w.detach().to(MODEL_DTYPE) \
            .unsqueeze(-1).unsqueeze(-1)
        self.argmax = InModelArgmax()

    def forward(self, hidden_in):
        # (1, 1, hidden) → Conv2d form
        h_conv = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        # final_norm in Conv2d-friendly view-out/view-in pattern
        h_seq = h_conv.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)
        h_seq = self.final_norm(h_seq)
        h_conv = h_seq.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        # lm_head Conv2d → (1, vocab, 1, 1) → (1, 1, vocab)
        logits_conv = self.lm_head.forward_conv(h_conv)
        logits = logits_conv.squeeze(-1).squeeze(-1).unsqueeze(1)  # (1, 1, vocab)
        token_id, _token_logit = self.argmax(logits)
        # token_id shape from argmax(dim=-1): (1, 1) int64; cast to int32
        return token_id.to(torch.int32)


# ---- Embed sidecar ---------------------------------------------------------

def export_embed_fp16(embed_weight: torch.Tensor, out_path: Path) -> None:
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
    print(f"\n--- convert ANE body chunk layers [{start_layer}, {end_layer}) ---")
    head_dim = cfg.head_dim

    # Trace inputs in MODEL_DTYPE (fp16) — Conv2d weights are fp16, and
    # Conv2d refuses mixed precision in PyTorch. position stays fp32
    # because it's compared against a fp32 positions buffer.
    example = [torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE)]
    example.append(torch.zeros(1, dtype=torch.float32))
    example.append(torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE))
    example.append(torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE))
    for _ in range(start_layer, end_layer):
        example.append(torch.zeros(*_kv_shape(cfg, max_seq), dtype=MODEL_DTYPE))
        example.append(torch.zeros(*_kv_shape(cfg, max_seq), dtype=MODEL_DTYPE))

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
    print(f"\n--- convert ANE head (final_norm + lm_head + in-graph argmax) ---")
    example = (torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE),)
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")
    ct_inputs = [ct.TensorType(
        name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16)]
    ct_outputs = [ct.TensorType(name="next_token", dtype=np.int32)]
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
HEAD_CHUNK_NAME = "chunk_head"


def _body_boundaries(num_layers, num_chunks):
    assert num_layers % num_chunks == 0
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--num-chunks", type=int, default=NUM_BODY_CHUNKS,
                    help="number of body chunks (default 6 → 6 layers each; "
                         "try 3 for fewer per-step dispatches)")
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
          f"num_kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    text_model, lm_head = load_text_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s")

    export_embed_fp16(text_model.embed_tokens.weight, chunks_dir / EMBED_BIN_NAME)
    head_module = ANEHeadChunk(cfg, text_model, lm_head,
                                cfg.tie_word_embeddings).eval().to(MODEL_DTYPE)

    boundaries = _body_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}")

    body_modules = []
    for start, end in boundaries:
        m = ANEBodyChunk(cfg, text_model.layers, start, end, args.max_seq)
        body_modules.append(m.eval().to(MODEL_DTYPE))
    del text_model, lm_head

    body_names = [f"chunk_{i}" for i in range(args.num_chunks)]
    for ci, ((start, end), mod, name) in enumerate(
        zip(boundaries, body_modules, body_names)
    ):
        fp16_path = fp16_dir / f"{name}.mlpackage"
        final_path = chunks_dir / f"{name}.mlpackage"
        convert_body(mod, cfg, start, end, args.max_seq, fp16_path)
        if args.nbits == 0:
            shutil.move(str(fp16_path), str(final_path))
        else:
            palettize_pkg(fp16_path, final_path, args.nbits)

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
