"""Gemma 4 Vision Encoder conversion for CoreML.

Contains three paths:

1. `save_vision_weights` — dumps the still-image encoder weights as an
   npz. Legacy fallback.

2. `convert_video_vision_to_coreml` — traces the vision tower at
   video-grade resolution (max_soft_tokens=70 → 64 real tokens) and
   converts it to `vision_video.mlpackage`. Output shape is
   `(1, 64, hidden_size)`. Runs on GPU today.

3. `convert_still_image_vision_ane_to_coreml` — still-image encoder at
   square 48×48 fixed grid (=2304 patches, 256 soft tokens), static
   attention mask, static pooling, input pre-patchified in Swift.
   Targets ANE; meant to replace the prebuilt `vision.mlpackage`.

The video path monkey-patches two things so coremltools 9.0 can handle
the vision tower:

- The encoder `forward` method is swapped for a version that builds a
  static additive attention mask from `attention_mask`, so
  `torch.jit.trace` does not trip over `create_bidirectional_mask`.

- Two coremltools frontend handlers (`_cast` and `clamp`) are patched
  to (a) extract scalars from 1-elt numpy constants and (b) preserve
  integer dtype across one-sided `clamp` calls — otherwise the
  downstream `one_hot` rejects float indices.

Both coremltools patches are best-effort and become no-ops if the
internals change — the caller can still fall back to the standalone
`conversion/phase2/trace_video_vision.py` script for debugging.
"""

from __future__ import annotations

import os
import shutil
import types

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Legacy still-image path
# ---------------------------------------------------------------------------

def extract_image_features(hf_model, processor, image) -> torch.Tensor:
    """Run the HF still-image vision pipeline and return projected features."""
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "describe"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    image_position_ids = inputs.get("image_position_ids")

    with torch.no_grad():
        vision_out = hf_model.model.get_image_features(
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            return_dict=True,
        )
        image_features = vision_out.pooler_output

    return image_features


def save_vision_weights(hf_model, output_dir: str) -> None:
    """Save still-image vision tower weights as an npz for PyTorch fallback."""
    vision_dir = os.path.join(output_dir, "vision_weights")
    os.makedirs(vision_dir, exist_ok=True)

    vision_model = hf_model.model.vision_tower
    embed_vision = hf_model.model.embed_vision

    state = {}
    for name, param in vision_model.named_parameters():
        state[f"vision_tower.{name}"] = param.data.cpu().half().numpy()
    for name, param in embed_vision.named_parameters():
        state[f"embed_vision.{name}"] = param.data.cpu().half().numpy()

    np.savez_compressed(os.path.join(vision_dir, "vision_weights.npz"), **state)
    print(f"  Saved {len(state)} vision weight tensors to {vision_dir}/")


# ---------------------------------------------------------------------------
# Video-grade vision path (Phase 2)
# ---------------------------------------------------------------------------

# Matches processor_config.json's video_processor settings for E2B:
#   max_soft_tokens = 70  → 64 real tokens on a square 384×384 frame
#   patch_size      = 16  → 24 patches per side, 576 used + 54 padding = 630
VIDEO_NUM_PATCHES = 630
VIDEO_PATCH_DIM = 16 * 16 * 3
VIDEO_TOKENS_PER_FRAME = 64


def _patch_coremltools_for_video_convert() -> None:
    """Idempotently install two coremltools frontend patches."""
    from coremltools.converters.mil.frontend.torch import ops as _ct_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY as _REG,
    )
    from coremltools.converters.mil.mil import Builder as _mb
    from coremltools.converters.mil.mil import types as _ct_types

    if getattr(_ct_ops, "_gemma4_video_patched", False):
        return

    # Patch 1 — `_cast` on a 1-elt (but not 0-dim) numpy const.
    def _patched_cast(context, node, dtype, dtype_name):
        inputs = _ct_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray):
                val = val.item() if val.size == 1 else val.reshape(()).item()
            if not isinstance(val, dtype):
                res = _mb.const(val=dtype(val), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x2 = _mb.squeeze(x=x, name=node.name + "_item")
            res = _mb.cast(x=x2, dtype=dtype_name, name=node.name)
        else:
            res = _mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _ct_ops._cast = _patched_cast

    # Patch 2 — one-sided `clamp` on an int tensor preserves dtype.
    _orig_clamp = _ct_ops.clamp

    def _patched_clamp(context, node):
        inputs = _ct_ops._get_inputs(context, node, expected=[1, 2, 3])
        x = inputs[0]
        min_in = inputs[1] if len(inputs) > 1 else None
        max_in = inputs[2] if len(inputs) > 2 else None
        if (min_in is None or max_in is None) and not _ct_types.is_float(x.dtype):
            res = x
            if max_in is not None:
                res = _mb.minimum(x=res, y=max_in)
            if min_in is not None:
                res = _mb.maximum(x=res, y=min_in, name=node.name)
            else:
                res = _mb.identity(x=res, name=node.name)
            context.add(res, node.name)
            return
        _orig_clamp(context, node)

    _ct_ops.clamp = _patched_clamp
    # The dispatcher looks up handlers in the registry by op name, not via
    # module attribute, so the registered entries need to be overwritten too.
    _REG.set_func_by_name(_patched_clamp, "clamp")
    _REG.set_func_by_name(_patched_clamp, "clip")

    _ct_ops._gemma4_video_patched = True


def _patch_vision_encoder_forward(hf_model) -> None:
    """Swap the inner vision encoder forward with a static-mask version."""
    encoder = hf_model.model.vision_tower.encoder
    if getattr(encoder, "_gemma4_video_forward_patched", False):
        return

    def _static_mask_forward(self, inputs_embeds, attention_mask,
                              pixel_position_ids=None, **kwargs):
        am = attention_mask.to(inputs_embeds.dtype)
        row = am.unsqueeze(1).unsqueeze(2)  # (B,1,1,S) key mask
        col = am.unsqueeze(1).unsqueeze(3)  # (B,1,S,1) query mask
        allow = row * col
        mask = (1.0 - allow) * torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                **kwargs,
            )
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    encoder.forward = types.MethodType(_static_mask_forward, encoder)
    encoder._gemma4_video_forward_patched = True


class _VideoVisionWrapper(nn.Module):
    """(1,630,768) fp32 + (1,630,2) int32 → (1, 64, hidden_size) fp16."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model.model

    def forward(self, pixel_values, pixel_position_ids):
        out = self.model.get_image_features(
            pixel_values=pixel_values.to(torch.float16),
            image_position_ids=pixel_position_ids.long(),
            return_dict=True,
        )
        return out.pooler_output.unsqueeze(0)


def _synthetic_video_inputs():
    """Zero frame + 24×24 patch grid (576 valid, 54 padding) for tracing."""
    pv = torch.zeros(1, VIDEO_NUM_PATCHES, VIDEO_PATCH_DIM, dtype=torch.float32)
    pid = torch.full((1, VIDEO_NUM_PATCHES, 2), -1, dtype=torch.int32)
    k = 0
    for py in range(24):
        for px in range(24):
            pid[0, k, 0] = px
            pid[0, k, 1] = py
            k += 1
    return pv, pid


def convert_video_vision_to_coreml(
    hf_model,
    output_dir: str,
    *,
    filename: str = "vision_video.mlpackage",
    run_parity_check: bool = True,
) -> str:
    """Trace + convert the video-grade vision encoder to CoreML.

    Returns the absolute path to the produced mlpackage.
    """
    _patch_coremltools_for_video_convert()
    _patch_vision_encoder_forward(hf_model)

    wrapper = _VideoVisionWrapper(hf_model).eval()
    pv, pid = _synthetic_video_inputs()

    print("  Running reference forward...")
    with torch.no_grad():
        ref = wrapper(pv, pid)
    print(f"    ref shape: {tuple(ref.shape)}")

    print("  Tracing vision tower (patched static mask)...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (pv, pid), check_trace=False, strict=False,
        )

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="pixel_values",
                          shape=(1, VIDEO_NUM_PATCHES, VIDEO_PATCH_DIM),
                          dtype=np.float32),
            ct.TensorType(name="pixel_position_ids",
                          shape=(1, VIDEO_NUM_PATCHES, 2),
                          dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
    )

    out_path = os.path.join(output_dir, filename)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path) for f in fns
    ) / 1024 / 1024
    print(f"  Saved {out_path} ({size_mb:.1f} MB)")

    if run_parity_check:
        print("  Parity check vs HF forward...")
        cm = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        res = cm.predict({
            "pixel_values": pv.numpy(),
            "pixel_position_ids": pid.numpy().astype(np.int32),
        })
        pred = res["image_features"].astype(np.float32)
        ref_np = ref.cpu().to(torch.float32).numpy()
        cos = float((ref_np * pred).sum() /
                    (np.linalg.norm(ref_np) * np.linalg.norm(pred) + 1e-9))
        max_abs = float(np.abs(ref_np - pred).max())
        print(f"    cosine={cos:.4f}  max_abs_diff={max_abs:.4f}")
        if cos < 0.98:
            raise RuntimeError(
                f"video vision parity check failed: cosine={cos:.4f} < 0.98"
            )

    return out_path


# ---------------------------------------------------------------------------
# Still-image ANE path — square fixed 48×48 grid, static mask, static pool
# ---------------------------------------------------------------------------

# 48×48 patches. patch_size=16 → 768×768 input canvas.
# k=3 average pooling over the 48×48 grid → 16×16 = 256 soft tokens.
# No padding (no -1 positions), so the pooler's one_hot / masked_fill
# degenerate to constants that fold at convert time.
STILL_GRID_SIDE = 48
STILL_NUM_PATCHES = STILL_GRID_SIDE * STILL_GRID_SIDE  # 2304
STILL_PATCH_DIM = 16 * 16 * 3                           # 768
STILL_POOL_K = 3
STILL_OUTPUT_TOKENS = (STILL_GRID_SIDE // STILL_POOL_K) ** 2  # 256


def _patch_embed_vision_rmsnorm_eps(hf_model) -> None:
    """Replace the vision pre-projection RMSNorm with an ANE-friendly form.

    The stock `embedding_pre_projection_norm` is a zero-scale RMSNorm
    with eps=1e-6 reduced in fp32 (`hidden_states.float()`). After
    coremltools' FLOAT16 compute_precision collapses the fp32 cast,
    iOS ANE computes `mean(x²)` in fp16 and flushes any subnormal
    accumulator to zero, then `rsqrt(0) = Inf` blows a handful of
    output tokens to ~1000× their true magnitude. Direction is still
    correct (per-token cos ≈ 1.0) but the outlier magnitudes corrupt
    every downstream softmax in the LLM.

    Fix: use the `layer_norm([x, -x])` identity — `cat([x, -x])` has
    mean 0 so `LayerNorm` == RMSNorm on the first half. ANE's native
    layer_norm kernel handles the reduction + rsqrt correctly in fp16
    without the subnormal trap, and it fuses into a single op.
    """
    import torch.nn.functional as F
    norm = hf_model.model.embed_vision.embedding_pre_projection_norm
    if getattr(norm, "_gemma4_ane_eps_patched", False):
        return
    hidden_size = norm.dim if hasattr(norm, "dim") else None
    # Discover hidden size from module params / first-forward probe if
    # not exposed — all Gemma4RMSNorm instances we patch here have
    # .dim (or we can pull it from config).
    if hidden_size is None:
        hidden_size = hf_model.model.vision_tower.config.hidden_size

    # fp16-safe eps that is NOT subnormal (min fp16 normal ≈ 6.1e-5).
    ane_eps = 1e-5

    def _ane_forward(self, x):
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None, bias=None,
            eps=float(ane_eps),
        )
        half, _ = torch.chunk(normed, 2, dim=-1)
        if getattr(self, "with_scale", False) and getattr(self, "weight", None) is not None:
            half = half * self.weight.to(half.dtype)
        return half

    norm.forward = types.MethodType(_ane_forward, norm)
    norm._gemma4_ane_eps_patched = True


def _patch_vision_tower_skip_strip(hf_model) -> None:
    """Bypass `hidden_states[pooler_mask]` strip-padding in the vision tower.

    The stock forward does a boolean-index gather at the end so the
    output dim matches the number of non-padding soft tokens. With our
    fixed full-grid input there *are* no padding positions, so the
    strip is a no-op — but the data-dependent gather traces into a
    `gather_nd` that iOS ANE silently miscompiles (output comes back
    as all zeros at predict time, even though Mac CPU/GPU produces
    correct features). Replacing the strip with identity drops that
    op entirely and keeps the output shape static.
    """
    vision = hf_model.model.vision_tower
    if getattr(vision, "_gemma4_ane_skip_strip_patched", False):
        return

    _orig_forward = vision.forward

    def _static_shape_forward(self, pixel_values, pixel_position_ids, **kwargs):
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size ** 2)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions)
        output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
            **kwargs,
        )

        hidden_states, _pooler_mask = self.pooler(
            hidden_states=output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        # SKIP: hidden_states = hidden_states[pooler_mask]
        # (caller guarantees no padding positions — full square grid)

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    vision.forward = types.MethodType(_static_shape_forward, vision)
    vision._gemma4_ane_skip_strip_patched = True
    # Keep the unbound original so future callers can restore it.
    vision._gemma4_orig_forward = _orig_forward


class _StillImageVisionWrapper(nn.Module):
    """(1, 2304, 768) fp16 + (1, 2304, 2) int32 → (1, 256, hidden) fp16.

    Square-fixed grid, no padding. Feeding always-valid positions (no
    -1 entries) lets the pooler's `masked_fill` and one_hot paths
    collapse to constants that fold into the compiled graph.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model.model

    def forward(self, pixel_values, pixel_position_ids):
        out = self.model.get_image_features(
            pixel_values=pixel_values.to(torch.float16),
            image_position_ids=pixel_position_ids.long(),
            return_dict=True,
        )
        # With the ANE strip-bypass patch, pooler_output is already
        # (1, output_tokens, hidden). Without it (stock HF), the tensor
        # is (output_tokens, hidden) because `hidden_states[pooler_mask]`
        # drops the batch dim; reshape back to canonical 3-D.
        x = out.pooler_output
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x


def _synthetic_still_inputs():
    """Zero frame + full 48×48 patch grid (2304 valid, 0 padding)."""
    pv = torch.zeros(1, STILL_NUM_PATCHES, STILL_PATCH_DIM, dtype=torch.float32)
    pid = torch.zeros(1, STILL_NUM_PATCHES, 2, dtype=torch.int32)
    k = 0
    for py in range(STILL_GRID_SIDE):
        for px in range(STILL_GRID_SIDE):
            pid[0, k, 0] = px
            pid[0, k, 1] = py
            k += 1
    return pv, pid


def _audit_ane_placement(mlpackage_path: str) -> float:
    """Return ANE % of compute ops for a saved mlpackage. Mac-only."""
    from collections import Counter
    try:
        reloaded = ct.models.MLModel(
            mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        compiled = reloaded.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"    ANE audit skipped: {e}")
        return -1.0
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


def convert_still_image_vision_ane_to_coreml(
    hf_model,
    output_dir: str,
    *,
    filename: str = "vision.ane.mlpackage",
    run_parity_check: bool = True,
) -> str:
    """Trace + convert the still-image vision encoder to an ANE-targeted CoreML.

    Square 48×48 fixed grid → 256 soft tokens. Swift-side preprocess
    must resize the input to 768×768 and patchify to (2304, 768).

    Returns the absolute path to the produced mlpackage.
    """
    _patch_coremltools_for_video_convert()
    _patch_vision_encoder_forward(hf_model)
    _patch_vision_tower_skip_strip(hf_model)
    _patch_embed_vision_rmsnorm_eps(hf_model)

    wrapper = _StillImageVisionWrapper(hf_model).eval()
    pv, pid = _synthetic_still_inputs()

    print(f"  Still-image grid: {STILL_GRID_SIDE}×{STILL_GRID_SIDE} = "
          f"{STILL_NUM_PATCHES} patches → pool k={STILL_POOL_K} → "
          f"{STILL_OUTPUT_TOKENS} soft tokens")

    print("  Running reference forward (fp32 → fp16 inside wrapper)...")
    with torch.no_grad():
        ref = wrapper(pv, pid)
    print(f"    ref shape: {tuple(ref.shape)}")

    print("  Tracing vision tower (patched static mask)...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (pv, pid), check_trace=False, strict=False,
        )

    print("  Converting to CoreML (target ANE)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="pixel_values",
                          shape=(1, STILL_NUM_PATCHES, STILL_PATCH_DIM),
                          dtype=np.float16),
            ct.TensorType(name="pixel_position_ids",
                          shape=(1, STILL_NUM_PATCHES, 2),
                          dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
    )

    out_path = os.path.join(output_dir, filename)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path) for f in fns
    ) / 1024 / 1024
    print(f"  Saved {out_path} ({size_mb:.1f} MB)")

    ane_pct = _audit_ane_placement(out_path)
    if 0 <= ane_pct < 80:
        print(f"  WARNING: ANE placement {ane_pct:.1f}% < 80% — iterate "
              f"before deploying (likely ops to hoist: pooler one_hot, "
              f"masked_fill, rotary_emb cos/sin from pixel_position_ids).")

    if run_parity_check:
        print("  Parity check vs HF forward...")
        # Use CPU_AND_GPU for parity reference — avoids ANE fp16 quant
        # noise polluting the numerical check; we'll measure ANE drift
        # separately on-device.
        cm = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        res = cm.predict({
            "pixel_values": pv.numpy().astype(np.float16),
            "pixel_position_ids": pid.numpy().astype(np.int32),
        })
        pred = res["image_features"].astype(np.float32)
        ref_np = ref.cpu().to(torch.float32).numpy()
        cos = float((ref_np * pred).sum() /
                    (np.linalg.norm(ref_np) * np.linalg.norm(pred) + 1e-9))
        max_abs = float(np.abs(ref_np - pred).max())
        print(f"    cosine={cos:.4f}  max_abs_diff={max_abs:.4f}")
        if cos < 0.98:
            raise RuntimeError(
                f"still-image vision parity check failed: cosine={cos:.4f} < 0.98"
            )

    return out_path
