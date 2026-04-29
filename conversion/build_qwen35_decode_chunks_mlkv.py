"""Qwen3.5 decode chunks with KV-only MLState (single ct.StateType per
chunk). SSM state stays as conventional input/output tensors.

Bundle name suffix: `_mlkv` so the new path doesn't collide with the
multi-state `_mls` build.

Usage:
  python build_qwen35_decode_chunks_mlkv.py --out-dir /tmp/qwen35_0_8b_mlkv
  python build_qwen35_decode_chunks_mlkv.py --out-dir /tmp/qwen35_2b_mlkv \\
      --model-id Qwen/Qwen3.5-2B \\
      --bundle-name qwen3_5_2b_decode_chunks_mlkv
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

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)
from transformers import AutoModelForCausalLM, AutoConfig, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE, Conv2dLinear
from qwen3_5_decode_layer_mlkv import (
    MLKVDecoderLayer, _norm_from_hf, is_full_attn,
)
from test_qwen3_5_full_decode_trace import MAX_SEQ


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_BUNDLE_NAME = "qwen3_5_0_8b_decode_chunks_mlkv"
DEFAULT_NUM_CHUNKS = 4
EMBED_BIN_NAME = "embed_weight.bin"
CHUNK_NAMES = ["chunk_a", "chunk_b", "chunk_c", "chunk_d",
               "chunk_e", "chunk_f"]


def load_text_config(model_id: str) -> Qwen3_5TextConfig:
    full_cfg = AutoConfig.from_pretrained(model_id)
    text_dict = (full_cfg.text_config.to_dict()
                 if hasattr(full_cfg, "text_config") else full_cfg.to_dict())
    return Qwen3_5TextConfig.from_dict(text_dict)


def load_text_backbone(model_id: str):
    full = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    return full


def export_embed_fp16(hf_model, out_path: Path) -> None:
    w = hf_model.model.embed_tokens.weight.detach().to(torch.float16).contiguous()
    vocab, hidden = w.shape
    print(f"\n--- export {out_path.name} ({vocab} × {hidden} fp16) ---")
    out_path.write_bytes(w.cpu().numpy().astype(np.float16).tobytes())
    print(f"  wrote {out_path.name} ({(vocab*hidden*2)/1e6:.0f} MB)")


def _ssm_state_shapes(cfg):
    conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 \
        + cfg.linear_value_head_dim * cfg.linear_num_value_heads
    a = (1, conv_dim, cfg.linear_conv_kernel_dim)
    b = (1, cfg.linear_num_value_heads,
         cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    return a, b


def _kv_state_shape(cfg, l_full: int, max_seq: int):
    return (2 * l_full, cfg.num_key_value_heads, max_seq, cfg.head_dim)


class MLKVBodyChunk(nn.Module):
    """Body chunk owning a single kv_cache MLState. SSM state per-layer
    flows through input/output tensors (conv_state_X, rec_state_X).
    """
    def __init__(self, cfg, hf_model, start: int, end: int, max_seq: int):
        super().__init__()
        self.start = start
        self.end = end
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq

        layers = []
        full_count = 0
        self.layer_types = []  # per-position type for ordered I/O
        self.lin_layer_indices = []  # absolute layer ids that are SSM
        self.full_layer_indices = []  # absolute layer ids that are full
        for i in range(start, end):
            full = is_full_attn(i)
            self.layer_types.append("full" if full else "lin")
            if full:
                self.full_layer_indices.append(i)
                layers.append(MLKVDecoderLayer(cfg, hf_model.model.layers[i],
                                                max_seq,
                                                full_idx_in_chunk=full_count))
                full_count += 1
            else:
                self.lin_layer_indices.append(i)
                layers.append(MLKVDecoderLayer(cfg, hf_model.model.layers[i],
                                                max_seq))
        self.layers = nn.ModuleList(layers)
        self.full_count = full_count

        # Single MLState for KV cache.
        kv_shape = _kv_state_shape(cfg, max(full_count, 1), max_seq)
        self.register_buffer("kv_cache", torch.zeros(*kv_shape, dtype=MODEL_DTYPE))

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos,
                *ssm_states):
        """SSM states ordered: for each lin layer in order, (conv_state_i, rec_state_i)."""
        h = hidden_in
        new_ssm_outs = []
        ssm_iter = iter(ssm_states)
        for ltype, layer in zip(self.layer_types, self.layers):
            if ltype == "full":
                h = layer.forward_full(h, cos, sin, causal_mask, current_pos,
                                        self.kv_cache)
            else:
                conv_s = next(ssm_iter)
                rec_s = next(ssm_iter)
                h, new_conv, new_rec = layer.forward_lin(h, conv_s, rec_s)
                new_ssm_outs.append(new_conv)
                new_ssm_outs.append(new_rec)
        return (h, *new_ssm_outs)


TOPK_OUT = 2048   # top-K candidates emitted to Swift for sampling.
                  # K=40 was enough on Mac (correct top-1 in top-40)
                  # but iPhone A18 ANE's fp16 reduction places the
                  # right token outside top-40 sometimes (Japanese
                  # short prompts loop because the correct head token
                  # ranks ~50-1000). 2048 is well within ANE topk
                  # capacity (~10 KB transfer/step) and gives Swift
                  # enough candidates to fp32-rerank correctly.


class MLKVTailChunk(MLKVBodyChunk):
    """Body + final_norm + Conv2d lm_head, FULL fp16 logits (no topk).
    All ANE-resident — Swift handles rep_penalty + argmax in fp32 over
    the full vocab. Removes in-graph topk because rep_penalty applied
    only over top-K (40-2048) couldn't kill iPhone A18 loops where the
    correct fresh token is outside ANE's biased top-K. Full-vocab
    rep_penalty solves this since "demote ALL recent tokens" reaches
    every candidate, not just top-K.
    """
    def __init__(self, cfg, hf_model, start: int, max_seq: int):
        super().__init__(cfg, hf_model, start, cfg.num_hidden_layers, max_seq)
        self.final_norm = _norm_from_hf(
            hf_model.model.norm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        lm_w = (hf_model.model.embed_tokens.weight
                if cfg.tie_word_embeddings else hf_model.lm_head.weight)
        self.lm_head = Conv2dLinear(cfg.hidden_size, cfg.vocab_size,
                                     bias=False, dtype=MODEL_DTYPE)
        self.lm_head.conv.weight.data = (
            lm_w.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos,
                *ssm_states):
        body_out = super().forward(hidden_in, cos, sin, causal_mask,
                                    current_pos, *ssm_states)
        h = body_out[0]
        ssm_outs = body_out[1:]
        h = self.final_norm(h)
        logits = self.lm_head(h)                    # (1, 1, V) fp16
        return (logits, *ssm_outs)


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


def convert_chunk(chunk, cfg, max_seq, out_path: Path, kind: str):
    """Convert MLKV chunk (kind='body' or 'tail').

    body: emits hidden + SSM state outputs.
    tail: emits top_indices + top_values + normed_hidden + SSM state
          outputs (Swift uses normed_hidden + top_indices for fp32
          re-rank against the mmap'd embed sidecar to fix iPhone fp16
          tie issues — keeps the head fused on ANE for full speed).
    """
    print(f"\n--- convert MLKV {kind} chunk layers "
          f"[{chunk.start}, {chunk.end}) ---")
    conv_shape, rec_shape = _ssm_state_shapes(cfg)

    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_t, sin_t = rot(dummy, pos_ids)

    example = [
        torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE),
        cos_t.to(MODEL_DTYPE),
        sin_t.to(MODEL_DTYPE),
        torch.zeros(1, 1, 1, max_seq, dtype=MODEL_DTYPE),
        torch.zeros(1, dtype=torch.int32),
    ]
    for _ in chunk.lin_layer_indices:
        example.append(torch.zeros(*conv_shape, dtype=MODEL_DTYPE))
        example.append(torch.zeros(*rec_shape, dtype=MODEL_DTYPE))

    t0 = time.time()
    traced = torch.jit.trace(chunk, tuple(example), strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16),
        ct.TensorType(name="cos", shape=cos_t.shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=sin_t.shape, dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, max_seq), dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
    ]
    if kind == "tail":
        # Full fp16 logits — Swift handles rep_penalty + argmax in fp32.
        ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    else:
        ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    for i in chunk.lin_layer_indices:
        ct_inputs.append(ct.TensorType(
            name=f"conv_state_{i}", shape=conv_shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(
            name=f"rec_state_{i}", shape=rec_shape, dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_conv_state_{i}", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_rec_state_{i}", dtype=np.float16))

    kv_shape = _kv_state_shape(cfg, max(chunk.full_count, 1), max_seq)
    ct_states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
                     name="kv_cache"),
    ]

    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def convert_body_chunk(chunk, cfg, max_seq, out_path: Path):
    convert_chunk(chunk, cfg, max_seq, out_path, kind="body")


def convert_tail_chunk(chunk, cfg, max_seq, out_path: Path):
    convert_chunk(chunk, cfg, max_seq, out_path, kind="tail")


def convert_head_chunk(chunk, cfg, out_path: Path, mode: str = "fp32"):
    """Stateless head: hidden_in (1, 1, H) → top_indices, top_values.

    mode = "fp32"  : full fp32 graph (1GB), correct top-1, slowest GPU.
    mode = "mixed" : fp16 default + fp32 ONLY for conv (lm_head matmul);
                    final_norm/topk run fp16. Aim: keep correctness via
                    fp32 reduction in the matmul while saving GPU
                    cycles on small ops.
    mode = "fp16"  : full fp16 graph (~250MB INT8). Fast but breaks
                    fp16 ties → garbage output. Diagnostic only.

    Why we need fp32 at all: the lm_head matmul reduction over
    248320-vocab has many tokens whose fp32 logits land in the same
    fp16 bucket (e.g. "**." vs "-mark" both ~25.41). Splitting
    body+head into two Core ML models causes coremltools to re-order
    the matmul vs. the OLD fused chunk_d, breaking ties differently →
    wrong top-1 → garbage output (Paris-mark-mark-mark...). fp32
    reduction in the matmul breaks ties correctly.
    """
    from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
    if mode == "fp16":
        precision = ct.precision.FLOAT16
    elif mode == "fp32":
        precision = ct.precision.FLOAT32
    elif mode == "mixed":
        # Keep fp32 for the matmul AND the topk that reads it — if
        # topk's input gets cast back to fp16, we lose the precision
        # we just paid for and tie-break breaks again. transpose
        # between conv (fp32) and topk also needs to stay fp32 to
        # avoid intermediate cast.
        def keep_lm_path_fp32(op):
            if op.op_type in ("conv", "topk", "transpose"):
                return False
            return True
        precision = FP16ComputePrecision(op_selector=keep_lm_path_fp32)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    print(f"\n--- convert MLKV head (final_norm + lm_head + topk[k=40]) ---")
    example = (torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE),)
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")
    # Input always declared as fp16 so Swift can pass the body's fp16
    # output directly. CoreML auto-inserts fp16→fp32 cast at the
    # boundary when compute_precision=FLOAT32.
    ct_inputs = [ct.TensorType(
        name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16)]
    ct_outputs = [
        ct.TensorType(name="top_indices", dtype=np.int32),
        ct.TensorType(name="top_values",  dtype=np.float16),
    ]
    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def palettize_chunk(fp16_pkg: Path, out_pkg: Path, nbits: int):
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


def _chunk_boundaries(num_layers: int, num_chunks: int):
    assert num_layers % num_chunks == 0
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    ap.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / args.bundle_name
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_id} (fp32)...")
    t0 = time.time()
    cfg = load_text_config(args.model_id)
    print(f"  text config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"vocab={cfg.vocab_size} tie_word_embeddings={cfg.tie_word_embeddings}")
    hf = load_text_backbone(args.model_id)
    print(f"  loaded in {time.time()-t0:.1f}s as {type(hf).__name__}")

    boundaries = _chunk_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  chunk boundaries: {boundaries}")

    export_embed_fp16(hf, chunks_dir / EMBED_BIN_NAME)

    # 4 ANE body chunks — no in-graph final_norm/lm_head/topk. iPhone
    # A18 ANE's fp16 final_norm + lm_head matmul produces a biased
    # output that places the correct token outside even top-2048; the
    # only working strategy is to do final_norm + lm_head matmul in
    # Swift fp32 against the mmap'd embed sidecar (= lm_head weight
    # under tied embeddings). Mac's separate fp32-head split path
    # confirmed this works at clean quality. We dump the final_norm
    # weight as a tiny sidecar so Swift can run RMSNorm in fp32.
    modules = []
    for ci, (start, end) in enumerate(boundaries):
        if ci == len(boundaries) - 1:
            # Last chunk: body + final_norm + lm_head, emit FULL fp16
            # logits (no in-graph topk). Swift handles rep_penalty +
            # argmax in fp32 over the full vocab.
            m = MLKVTailChunk(cfg, hf, start, args.max_seq)
        else:
            m = MLKVBodyChunk(cfg, hf, start, end, args.max_seq)
        modules.append(m.eval().to(MODEL_DTYPE))
    # Dump final_norm weight (1024 fp16 = ~2 KB) as a sidecar.
    fn_w = hf.model.norm.weight.detach().to(torch.float16).numpy()
    (chunks_dir / "final_norm.bin").write_bytes(fn_w.tobytes())
    print(f"  dumped final_norm.bin ({fn_w.nbytes} bytes, eps={cfg.rms_norm_eps})")
    # Also dump rms_norm_eps so Swift uses the right epsilon.
    import json as _json
    (chunks_dir / "head_meta.json").write_text(_json.dumps({
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.hidden_size),
    }))
    del hf

    chunk_names = CHUNK_NAMES[:args.num_chunks]
    for ci, (torch_m, name) in enumerate(zip(modules, chunk_names)):
        is_tail = ci == len(boundaries) - 1
        fp16_path = fp16_dir / f"{name}.mlpackage"
        final_path = chunks_dir / f"{name}.mlpackage"
        if is_tail:
            convert_tail_chunk(torch_m, cfg, args.max_seq, fp16_path)
        else:
            convert_body_chunk(torch_m, cfg, args.max_seq, fp16_path)
        if args.nbits == 0:
            shutil.move(str(fp16_path), str(final_path))
        else:
            palettize_chunk(fp16_path, final_path, args.nbits)

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
