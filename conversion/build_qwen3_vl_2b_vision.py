"""Qwen3-VL 2B vision encoder converter.

Traces `Qwen3VLVisionModel` at a fixed input resolution (448×448 RGB,
temporal=1 so padded to 2 for the temporal patch) and emits a
CoreML mlpackage that produces:
  - hidden:      fp16 (1, 196, 2048)  — final vision tokens after
                 spatial_merge (28×28 patches → 14×14 after merge = 196)
  - deepstack_0: fp16 (1, 784, 2048) — tap at vision layer 5 + merger
  - deepstack_1: fp16 (1, 784, 2048) — tap at vision layer 11 + merger
  - deepstack_2: fp16 (1, 784, 2048) — tap at vision layer 17 + merger

Simplifications vs HF vision model:
  - Single image, single temporal slice → cu_seqlens path degenerates
    to full attention over seq_len=784. Skip the packed-batch cu_seqlens
    split entirely.
  - grid_thw fixed → pre-compute pos_embed + rotary as constants.

Vision features later land in Swift, which pushes them into the text
chain via chunk_0's extended input set (deepstack inject at text layers
0/1/2 per HF impl). Phase 2a = this converter. Phase 2b = chunk_0
rewrite with deepstack inputs. Phase 2c = Swift prefill + UI.
"""
from pathlib import Path
import argparse
import math
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
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed, Qwen3VLVisionBlock, Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).parent))


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
# Input image resolution (square). 448 → 28×28 = 784 patches at
# spatial=16. Low enough that the vision attention (seq=784) fits on
# ANE, high enough to preserve useful detail. Matches Qwen3-VL's
# recommended image preprocessing resolution for single-image chat.
IMAGE_SIZE = 448


def load_vision_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig.from_pretrained(MODEL_ID)


def load_vision_backbone():
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    return full.model.visual


# ---- fixed-grid vision model ---------------------------------------------

class FixedGridVisionModel(nn.Module):
    """Qwen3VLVisionModel with grid_thw hardcoded to (1, H_patches, W_patches)
    so all position / rotary tensors are static constants and the
    cu_seqlens-driven packed-batch attention simplifies to full
    attention over the single image's patch sequence."""

    def __init__(self, hf_vision, image_size: int):
        super().__init__()
        cfg = hf_vision.config
        self.config = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        assert image_size % cfg.patch_size == 0, \
            f"image_size {image_size} must be divisible by patch_size {cfg.patch_size}"
        self.grid_h = image_size // cfg.patch_size
        self.grid_w = image_size // cfg.patch_size
        self.seq_len = self.grid_h * self.grid_w
        self.deepstack_visual_indexes = cfg.deepstack_visual_indexes

        # Reuse HF modules as-is — their weights are what we want
        self.patch_embed = hf_vision.patch_embed
        self.pos_embed = hf_vision.pos_embed
        self.num_grid_per_side = hf_vision.num_grid_per_side
        self.blocks = hf_vision.blocks
        self.merger = hf_vision.merger
        self.deepstack_merger_list = hf_vision.deepstack_merger_list

        # Precompute pos_embed weights for this grid — replaces
        # fast_pos_embed_interpolate which has data-dependent control flow.
        pos = self._precompute_pos_embed(hf_vision)
        self.register_buffer("pos_embed_fixed", pos, persistent=False)

        # Precompute rotary emb (cos, sin) as constants. rot_pos_emb's
        # output for grid (T=1, H, W) is a fixed tensor we can bake in.
        cos, sin = self._precompute_rotary(hf_vision)
        self.register_buffer("rot_cos", cos, persistent=False)
        self.register_buffer("rot_sin", sin, persistent=False)

    @torch.no_grad()
    def _precompute_pos_embed(self, hf_vision):
        """Run HF's fast_pos_embed_interpolate once to get fixed weights."""
        grid_thw = torch.tensor([[1, self.grid_h, self.grid_w]], dtype=torch.long)
        return hf_vision.fast_pos_embed_interpolate(grid_thw)

    @torch.no_grad()
    def _precompute_rotary(self, hf_vision):
        grid_thw = torch.tensor([[1, self.grid_h, self.grid_w]], dtype=torch.long)
        rotary_pos_emb = hf_vision.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(self.seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, pixel_values):
        """
        pixel_values: `(num_patches=784, C*T_p*P*P=1536)` fp16 — the
        same layout HF's `Qwen2VLImageProcessor` produces.

        Earlier revisions tried two other input shapes:
          (a) Raw `(3, 2, 448, 448)` fed directly to
              `patch_embed`. The internal `.view(-1, C, T_p, P, P)`
              only extracts valid patches when the input is already
              patchified; on raw pixel layout the first 1536 sequential
              bytes become "patch 0", collapsing spatial structure
              (cos_sim ≈ 0.47 vs HF on a checker image).
          (b) In-graph patchify from raw `(3, 2, 448, 448)` via a
              rank-10 `view + permute(0,1,4,7,5,8,3,2,6,9)`. Correct
              numerically, but the A18 Pro / iOS 26 ANE compiler
              faults during `MLModel` init with EXC_BAD_ACCESS on that
              reshape pattern (Mac Studio ANE handles it fine; the
              iPhone compiler does not).

        Fix: keep the input as the pre-patchified `(784, 1536)`
        tensor. The Swift preprocessor does the HF patchify permutation
        on CPU (~1 ms, 1.2 M fp16 elements per frame) and hands this
        tensor to the Core ML model. The only reshape left inside the
        model is `patch_embed`'s own rank-5 `.view`, which ANE
        compiles cleanly.
        """
        # hidden_states enters as (784, 1536); patch_embed reshapes to
        # (784, 3, 2, 16, 16) internally and emits (784, 1024).
        hidden_states = self.patch_embed(pixel_values)   # (seq, hidden)
        hidden_states = hidden_states + self.pos_embed_fixed

        # cu_seqlens for a single image = [0, seq_len] — attention
        # spans the whole sequence.
        cu_seqlens = torch.tensor([0, self.seq_len], dtype=torch.int32,
                                   device=hidden_states.device)

        position_embeddings = (self.rot_cos, self.rot_sin)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                ds = self.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(ds)

        merged = self.merger(hidden_states)  # (merged_seq, out_hidden)
        return (merged, *deepstack_features)


# ---- convert + audit ------------------------------------------------------

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
    cpu = dev.get("MLCPUComputeDevice", 0)
    gpu = dev.get("MLGPUComputeDevice", 0)
    pct = 100 * ane / compute if compute else 0.0
    print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%) — CPU={cpu} GPU={gpu}")
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    bundle_dir = out_root / "qwen3_vl_2b_vision"
    fp16_dir = out_root / "_fp16_intermediate"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Qwen3-VL 2B vision (fp32, image_size={args.image_size})...")
    t0 = time.time()
    vision = load_vision_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s")
    cfg = vision.config
    print(f"  vision cfg: depth={cfg.depth} hidden={cfg.hidden_size} "
          f"out_hidden={cfg.out_hidden_size} patch={cfg.patch_size} "
          f"deepstack_indexes={cfg.deepstack_visual_indexes}")

    model = FixedGridVisionModel(vision, args.image_size).eval().float()
    del vision
    print(f"  grid: {model.grid_h}×{model.grid_w} = {model.seq_len} patches "
          f"→ spatial_merge={model.spatial_merge_size} "
          f"→ {model.seq_len // (model.spatial_merge_size ** 2)} vision tokens")

    # Input: pre-patchified (num_patches, C*T_p*P*P) — matches HF's
    # Qwen2VLImageProcessor output exactly. num_patches = (grid_t=1)
    # × grid_h × grid_w = 1 × 28 × 28 = 784; patch_flat =
    # 3 × 2 × 16 × 16 = 1536.
    patch_flat = 3 * 2 * model.patch_size * model.patch_size
    num_patches = model.grid_h * model.grid_w  # grid_t=1 for single image
    example = torch.zeros(num_patches, patch_flat, dtype=torch.float32)
    t0 = time.time()
    traced = torch.jit.trace(model, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    fp16_path = fp16_dir / "vision.mlpackage"
    final_path = bundle_dir / "vision.mlpackage"

    ct_inputs = [ct.TensorType(
        name="pixel_values",
        shape=(num_patches, patch_flat),
        dtype=np.float16,
    )]
    # Merger output: (seq_len / spatial_merge², out_hidden)
    # DeepStack outputs: same shape as merger output (they use the same
    # patch-merger architecture internally).
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="deepstack_0", dtype=np.float16),
        ct.TensorType(name="deepstack_1", dtype=np.float16),
        ct.TensorType(name="deepstack_2", dtype=np.float16),
    ]
    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(fp16_path))
    size_mb = sum(f.stat().st_size for f in fp16_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved fp16 {fp16_path.name} ({size_mb:.0f} MB)")
    _audit_ane(fp16_path)

    if args.nbits == 0:
        shutil.move(str(fp16_path), str(final_path))
    else:
        print(f"\n--- palettize INT{args.nbits} ---")
        m_in = ct.models.MLModel(str(fp16_path))
        op_cfg = OpPalettizerConfig(mode="kmeans", nbits=args.nbits,
                                     granularity="per_tensor")
        opt_cfg = OptimizationConfig(global_config=op_cfg)
        t0 = time.time()
        m_out = palettize_weights(m_in, opt_cfg)
        print(f"  palettize done in {time.time()-t0:.1f}s")
        m_out.save(str(final_path))
        src_mb = sum(f.stat().st_size for f in fp16_path.rglob('*') if f.is_file()) / 1e6
        dst_mb = sum(f.stat().st_size for f in final_path.rglob('*') if f.is_file()) / 1e6
        print(f"  bundle: {src_mb:.0f} MB (fp16) → {dst_mb:.0f} MB (int{args.nbits}) "
              f"[{100*dst_mb/src_mb:.1f}%]")
        _audit_ane(final_path)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\n✓ shipping artifact: {final_path}")


if __name__ == "__main__":
    main()
