#!/usr/bin/env python3
"""Phase B redux — build mlpackages for the real Swift-side dispatch probe.

Phase B's Python predict() numbers were overhead-contaminated. This
script builds the artifacts the Swift harness (`moe-dispatch-probe`)
will measure properly:

  1. single_expert.mlpackage   — one SwiGLU expert (2048→1408→2048)
  2. multifunction_N.mlpackage — N expert functions in one mlpackage,
     selectable by MLModelConfiguration.functionName

Both use random fp16 weights — the dispatch-latency test does not
depend on weight values, only on graph shape and ANE placement.

Usage:
  pyenv shell lama-cml
  python conversion/phase_b_redux_build.py --out-dir /tmp/moe_probe --n-funcs 16
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import Conv2dLinear, ANERMSNorm, MODEL_DTYPE


def palettize_int4(mlmodel: "ct.models.MLModel") -> "ct.models.MLModel":
    """Apply 4-bit palettization — production deployment uses this; all
    dispatch tests should measure it because it cuts weight-read
    bandwidth ~4× vs fp16."""
    cfg = OptimizationConfig(
        global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
    return palettize_weights(mlmodel, cfg)


HIDDEN = 2048
EXPERT_INTER = 1408
SHARED_INTER = 5632
NUM_EXPERTS = 60
TOP_K = 4


class SingleExpert(nn.Module):
    """SwiGLU FFN expert, Conv2dLinear for ANE. Input/output (1,H,1,1)."""
    def __init__(self, hidden: int = HIDDEN, inter: int = EXPERT_INTER):
        super().__init__()
        self.gate_proj = Conv2dLinear(hidden, inter, bias=False)
        self.up_proj = Conv2dLinear(hidden, inter, bias=False)
        self.down_proj = Conv2dLinear(inter, hidden, bias=False)
        # Random fp16 weights
        for p in self.parameters():
            p.data = (torch.randn_like(p.data) * 0.02).to(MODEL_DTYPE)

    def forward(self, x_bc1t: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward_conv(x_bc1t)
        up = self.up_proj.forward_conv(x_bc1t)
        intermediate = F.silu(gate) * up
        return self.down_proj.forward_conv(intermediate)


class LayerGather(nn.Module):
    """One layer's routed-expert dispatch: 60 experts as in-graph
    constants, runtime gather to select TOP_K. Inputs: x + topk_idx +
    topk_weights. This is the design that, IF fast on GPU in Swift,
    lets us fuse a whole layer (and chunk 6 layers) into one mlpackage.
    """
    def __init__(self):
        super().__init__()
        self.gate_up_all = nn.Parameter(
            (torch.randn(NUM_EXPERTS, 2 * EXPERT_INTER, HIDDEN) * 0.02).to(MODEL_DTYPE))
        self.down_all = nn.Parameter(
            (torch.randn(NUM_EXPERTS, HIDDEN, EXPERT_INTER) * 0.02).to(MODEL_DTYPE))

    def forward(self, x_bc1t, topk_idx, topk_weights):
        x_flat = x_bc1t.reshape(1, HIDDEN)
        gu = self.gate_up_all.index_select(0, topk_idx)   # (K, 2*inter, H)
        down = self.down_all.index_select(0, topk_idx)    # (K, H, inter)
        gate_up = torch.einsum("kih,bh->kbi", gu, x_flat)  # (K, 1, 2*inter)
        gate, up = gate_up.chunk(2, dim=-1)
        inter = F.silu(gate) * up                          # (K, 1, inter)
        down_out = torch.einsum("kbi,khi->kbh", inter, down)  # (K, 1, H)
        weighted = (down_out * topk_weights.view(-1, 1, 1)).sum(dim=0)
        return weighted.view(1, HIDDEN, 1, 1)


class DenseBackboneChunk(nn.Module):
    """N fused layers of the STATIC Qwen MoE backbone: per layer = norm
    + Q/K/V/O attention matmuls + decode-style attention reduction +
    norm + router matmul + shared-expert SwiGLU + shared gate. Routed
    experts are NOT here — they're the dynamic part.

    Random weights; the point is the compute-graph SHAPE (matmul sizes,
    op count) for a representative dispatch-latency measurement. Decode
    is T=1 so attention is a rank-1 update; we approximate the score
    path with a fixed-size matmul against a small KV window so the
    op-count is realistic without needing a real KV cache.
    """
    def __init__(self, n_layers: int = 6, kv_window: int = 256):
        super().__init__()
        self.n_layers = n_layers
        self.kv_window = kv_window
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.q = nn.ModuleList(); self.k = nn.ModuleList()
        self.v = nn.ModuleList(); self.o = nn.ModuleList()
        self.router = nn.ModuleList()
        self.shared_gate = nn.ModuleList()
        self.sh_gate = nn.ModuleList(); self.sh_up = nn.ModuleList()
        self.sh_down = nn.ModuleList()
        for _ in range(n_layers):
            self.norms1.append(ANERMSNorm(HIDDEN))
            self.norms2.append(ANERMSNorm(HIDDEN))
            self.q.append(Conv2dLinear(HIDDEN, HIDDEN, bias=False))
            self.k.append(Conv2dLinear(HIDDEN, HIDDEN, bias=False))
            self.v.append(Conv2dLinear(HIDDEN, HIDDEN, bias=False))
            self.o.append(Conv2dLinear(HIDDEN, HIDDEN, bias=False))
            self.router.append(Conv2dLinear(HIDDEN, NUM_EXPERTS, bias=False))
            self.shared_gate.append(Conv2dLinear(HIDDEN, 1, bias=False))
            self.sh_gate.append(Conv2dLinear(HIDDEN, SHARED_INTER, bias=False))
            self.sh_up.append(Conv2dLinear(HIDDEN, SHARED_INTER, bias=False))
            self.sh_down.append(Conv2dLinear(SHARED_INTER, HIDDEN, bias=False))
        for p in self.parameters():
            if p.dim() > 1:
                p.data = (torch.randn_like(p.data) * 0.02).to(MODEL_DTYPE)

    def forward(self, x_bc1t, kv_window_tensor):
        # x_bc1t: (1, H, 1, 1); kv_window_tensor: (1, H, 1, kv_window)
        # stand-in "past keys/values" — fixed-size, representative.
        h = x_bc1t
        for i in range(self.n_layers):
            # --- attention ---
            n1 = self.norms1[i].forward if hasattr(self.norms1[i], "forward") else None
            # ANERMSNorm expects (..., H); reshape from conv layout.
            x_flat = h.reshape(1, HIDDEN)
            normed = self.norms1[i](x_flat).reshape(1, HIDDEN, 1, 1)
            q = self.q[i].forward_conv(normed)   # (1,H,1,1)
            k = self.k[i].forward_conv(normed)
            v = self.v[i].forward_conv(normed)
            # decode attention: score q against kv_window keys, weight values.
            # approximate: q (1,H,1,1) -> (1,1,H); kv (1,H,1,W) -> (1,H,W)
            q3 = q.reshape(1, 1, HIDDEN)
            kvw = kv_window_tensor.reshape(1, HIDDEN, self.kv_window)
            scores = torch.bmm(q3, kvw)                       # (1,1,W)
            probs = F.softmax(scores, dim=-1)
            ctx = torch.bmm(probs, kvw.transpose(1, 2))        # (1,1,H)
            ctx4 = ctx.reshape(1, HIDDEN, 1, 1)
            attn_out = self.o[i].forward_conv(ctx4)
            h = h + attn_out
            # --- MoE static part: router + shared expert ---
            x_flat2 = h.reshape(1, HIDDEN)
            normed2 = self.norms2[i](x_flat2).reshape(1, HIDDEN, 1, 1)
            _ = self.router[i].forward_conv(normed2)           # router logits
            sg = self.sh_gate[i].forward_conv(normed2)
            su = self.sh_up[i].forward_conv(normed2)
            shared = self.sh_down[i].forward_conv(F.silu(sg) * su)
            gate_scalar = torch.sigmoid(
                self.shared_gate[i].forward_conv(normed2))
            h = h + gate_scalar * shared
        return h


def convert_one(module: nn.Module, out_path: str, int4: bool = False) -> str:
    example = torch.zeros(1, HIDDEN, 1, 1, dtype=MODEL_DTYPE)
    traced = torch.jit.trace(module.eval().to(MODEL_DTYPE), example)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name="x_bc1t",
                              shape=(1, HIDDEN, 1, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="y_bc1t", dtype=np.float16)],
    )
    if int4:
        mlmodel = palettize_int4(mlmodel)
    mlmodel.save(out_path)
    return out_path


def convert_layer_gather(out_path: str, int4: bool = False) -> str:
    """Convert the LayerGather module (3 inputs: x, topk_idx, topk_weights)."""
    m = LayerGather().eval().to(MODEL_DTYPE)
    x_ex = torch.zeros(1, HIDDEN, 1, 1, dtype=MODEL_DTYPE)
    idx_ex = torch.arange(TOP_K, dtype=torch.int32)
    w_ex = torch.full((TOP_K,), 0.25, dtype=MODEL_DTYPE)
    traced = torch.jit.trace(m, (x_ex, idx_ex, w_ex))
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="x_bc1t", shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType(name="topk_idx", shape=(TOP_K,), dtype=np.int32),
            ct.TensorType(name="topk_weights", shape=(TOP_K,), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="y_bc1t", dtype=np.float16)],
    )
    if int4:
        mlmodel = palettize_int4(mlmodel)
    mlmodel.save(out_path)
    return out_path


def convert_dense_backbone(out_path: str, n_layers: int,
                           kv_window: int = 256, int4: bool = False) -> str:
    """Convert the DenseBackboneChunk (2 inputs: x, kv_window_tensor)."""
    m = DenseBackboneChunk(n_layers=n_layers, kv_window=kv_window)
    m = m.eval().to(MODEL_DTYPE)
    x_ex = torch.zeros(1, HIDDEN, 1, 1, dtype=MODEL_DTYPE)
    kv_ex = torch.zeros(1, HIDDEN, 1, kv_window, dtype=MODEL_DTYPE)
    traced = torch.jit.trace(m, (x_ex, kv_ex))
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="x_bc1t", shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType(name="kv_window", shape=(1, HIDDEN, 1, kv_window),
                          dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="y_bc1t", dtype=np.float16)],
    )
    if int4:
        mlmodel = palettize_int4(mlmodel)
    mlmodel.save(out_path)
    return out_path


def build_multifunction(out_dir: str, n_funcs: int, int4: bool = False) -> str:
    """Build N single-expert mlpackages then merge into one multifunction."""
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    tag = "_int4" if int4 else ""
    tmp_dir = os.path.join(out_dir, f"_mf_parts{tag}")
    os.makedirs(tmp_dir, exist_ok=True)
    desc = MultiFunctionDescriptor()
    print(f"[mf{tag}] converting {n_funcs} expert functions...")
    t0 = time.time()
    for i in range(n_funcs):
        part = os.path.join(tmp_dir, f"expert_{i}.mlpackage")
        convert_one(SingleExpert(), part, int4=int4)
        desc.add_function(part, src_function_name="main",
                          target_function_name=f"expert_{i}")
        if (i + 1) % 8 == 0:
            print(f"[mf{tag}]   {i+1}/{n_funcs} done ({time.time()-t0:.1f}s)")
    desc.default_function_name = "expert_0"
    mf_path = os.path.join(out_dir, f"multifunction_{n_funcs}{tag}.mlpackage")
    if os.path.exists(mf_path):
        shutil.rmtree(mf_path)
    save_multifunction(desc, mf_path)
    print(f"[mf{tag}] saved {mf_path} ({time.time()-t0:.1f}s total)")
    shutil.rmtree(tmp_dir)
    return mf_path


def compile_to_mlmodelc(mlpackage_path: str) -> str:
    """Compile mlpackage -> mlmodelc next to it."""
    out = mlpackage_path.replace(".mlpackage", ".mlmodelc")
    if os.path.exists(out):
        shutil.rmtree(out)
    compiled = ct.models.MLModel(mlpackage_path).get_compiled_model_path()
    shutil.copytree(compiled, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/tmp/moe_probe")
    p.add_argument("--n-funcs", type=int, default=16,
                   help="number of expert functions in the multifunction model")
    p.add_argument("--dense-layers", type=int, default=6,
                   help="layers per dense-backbone chunk")
    p.add_argument("--skip-multifunction", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Phase B redux build → {args.out_dir} ===")

    # 1. Single expert
    print(f"[single] building single_expert.mlpackage")
    t0 = time.time()
    single_path = os.path.join(args.out_dir, "single_expert.mlpackage")
    convert_one(SingleExpert(), single_path)
    print(f"[single] done in {time.time()-t0:.1f}s")

    # 2. Multifunction — fp16 + INT4 (ANE can't gather but CAN run
    #    function-selected INT4 experts; this is the all-ANE path).
    if not args.skip_multifunction:
        for is_int4 in (False, True):
            try:
                build_multifunction(args.out_dir, args.n_funcs, int4=is_int4)
            except Exception as e:
                print(f"[mf] ERROR building multifunction (int4={is_int4}): {e}")
                import traceback
                traceback.print_exc()

    # 3. Layer-gather (60 experts as constants + runtime gather, 1 layer)
    #    fp16 + INT4 — production uses INT4, all dispatch tests need it.
    for tag, is_int4 in [("", False), ("_int4", True)]:
        print(f"[lg{tag}] building layer_gather{tag}.mlpackage")
        t0 = time.time()
        try:
            convert_layer_gather(
                os.path.join(args.out_dir, f"layer_gather{tag}.mlpackage"),
                int4=is_int4)
            print(f"[lg{tag}] done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[lg{tag}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # 4. Dense backbone chunk. Build BOTH 1-layer (the correct per-layer
    #    cost, since routed experts interleave and break 6-layer fusion)
    #    AND the requested N-layer (kept for the chunked-static analysis).
    for nl in sorted({1, args.dense_layers}):
        for tag, is_int4 in [("", False), ("_int4", True)]:
            print(f"[db{tag}] building dense_backbone_{nl}L{tag}.mlpackage")
            t0 = time.time()
            try:
                convert_dense_backbone(
                    os.path.join(args.out_dir, f"dense_backbone_{nl}L{tag}.mlpackage"),
                    nl, int4=is_int4)
                print(f"[db{tag}] done in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"[db{tag}] ERROR: {e}")
                import traceback
                traceback.print_exc()

    # 5. Single expert INT4 (fp16 already built above)
    print(f"[single_int4] building single_expert_int4.mlpackage")
    t0 = time.time()
    try:
        convert_one(SingleExpert(),
                    os.path.join(args.out_dir, "single_expert_int4.mlpackage"),
                    int4=True)
        print(f"[single_int4] done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[single_int4] ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Report sizes
    print(f"\n=== Artifacts ===")
    for entry in sorted(os.listdir(args.out_dir)):
        full = os.path.join(args.out_dir, entry)
        if os.path.isdir(full):
            size = sum(f.stat().st_size for f in Path(full).rglob('*') if f.is_file())
            print(f"  {entry}: {size/1e6:.1f} MB")
    print(f"\nNext: swift run -c release moe-dispatch-probe {args.out_dir}")


if __name__ == "__main__":
    main()
