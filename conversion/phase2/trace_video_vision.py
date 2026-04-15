"""Trace the Gemma 4 vision tower for video-grade inputs (64 soft tokens).

Strategy: monkey-patch the vision encoder's forward to use a
pre-computable static attention mask so torch.jit.trace doesn't hit
the dynamic `create_bidirectional_mask` path. Then hand the traced
graph to coremltools.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, shutil, types
import torch, torch.nn as nn
import numpy as np
import coremltools as ct
from transformers import Gemma4ForConditionalGeneration

# ── Patch coremltools _cast to handle 1-elt (non-0-dim) numpy arrays.
# Without this, `aten::Int` on a shape-(1,) constant raises
# `TypeError: only 0-dimensional arrays can be converted to Python scalars`
# at MIL import time (ops.py:3048). Fix: flatten via .item() before casting.
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil.mil import Builder as _mb

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

# ── Patch `clamp` so that a one-sided clamp (e.g. `.clamp(min=0)`) on an
# int tensor does not get silently promoted to fp32 via the float-typed
# +/- inf default bound. Preserving int dtype is required because the
# tensor feeds directly into `one_hot` which only accepts int indices.
from coremltools.converters.mil.mil import types as _ct_types

def _patched_clamp(context, node):
    inputs = _ct_ops._get_inputs(context, node, expected=[1, 2, 3])
    x = inputs[0]
    min_in = inputs[1] if len(inputs) > 1 else None
    max_in = inputs[2] if len(inputs) > 2 else None
    # One-sided on int → min/max path preserves dtype.
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
    # Fall back to the original implementation.
    _orig_clamp(context, node)

_orig_clamp = _ct_ops.clamp
_ct_ops.clamp = _patched_clamp
# The dispatcher looks up handlers in the registry by op name, not via
# module attributes. Overwrite the registered entry for both `clamp` and
# its alias `clip` so the patch actually gets called at convert time.
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY as _REG,
)
_REG.set_func_by_name(_patched_clamp, "clamp")
_REG.set_func_by_name(_patched_clamp, "clip")

MODEL = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E2B-it")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision_video.mlpackage"

NUM_PATCHES = 630
PATCH_DIM = 16 * 16 * 3

print("loading HF model (CPU, fp16)...")
hf = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cpu")
hf.eval()


def patched_vision_forward(self, inputs_embeds, attention_mask,
                             pixel_position_ids=None, **kwargs):
    """Replacement forward that skips create_bidirectional_mask.

    Vision encoder is bidirectional self-attention; we just need a
    (B, 1, S, S) mask where padded positions are -inf. Build it from
    `attention_mask` (B, S) which the caller already provides.
    """
    B, S, _ = inputs_embeds.shape
    # additive 4D mask: 0.0 at valid, -inf at pad
    am = attention_mask.to(inputs_embeds.dtype)
    row = am.unsqueeze(1).unsqueeze(2)          # (B,1,1,S)  — key mask
    col = am.unsqueeze(1).unsqueeze(3)          # (B,1,S,1)  — query mask (redundant but keeps math symmetric)
    allow = row * col                            # (B,1,S,S)
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

# Gemma4 vision tower's inner `encoder` is the module with the broken forward.
encoder = hf.model.vision_tower.encoder
encoder.forward = types.MethodType(patched_vision_forward, encoder)


class VideoVisionWrapper(nn.Module):
    """(1,630,768) fp32 + (1,630,2) int32 → (1, 64, 1536) fp16."""
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


wrapper = VideoVisionWrapper(hf).eval()

pv = torch.zeros(1, NUM_PATCHES, PATCH_DIM, dtype=torch.float32)
pid = torch.full((1, NUM_PATCHES, 2), -1, dtype=torch.int32)
k = 0
for py in range(24):
    for px in range(24):
        pid[0, k, 0] = px
        pid[0, k, 1] = py
        k += 1

print("reference forward...")
with torch.no_grad():
    ref = wrapper(pv, pid)
print(f"  ref shape: {tuple(ref.shape)}")

print("tracing (with patched vision encoder)...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (pv, pid),
                              check_trace=False, strict=False)

print("converting via coremltools (jit)...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="pixel_values", shape=(1, NUM_PATCHES, PATCH_DIM),
                      dtype=np.float32),
        ct.TensorType(name="pixel_position_ids", shape=(1, NUM_PATCHES, 2),
                      dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
mlmodel.save(OUT)
size = sum(os.path.getsize(os.path.join(dp, f))
           for dp, _, fns in os.walk(OUT) for f in fns) / 1024 / 1024
print(f"saved {OUT} ({size:.1f} MB)")

# Parity check
print("\nparity check...")
cm = ct.models.MLModel(OUT, compute_units=ct.ComputeUnit.CPU_AND_GPU)
res = cm.predict({"pixel_values": pv.numpy(),
                  "pixel_position_ids": pid.numpy().astype(np.int32)})
pred = res["image_features"]
ref_np = ref.cpu().to(torch.float32).numpy()
pred_np = pred.astype(np.float32)
cos = (ref_np * pred_np).sum() / (
    np.linalg.norm(ref_np) * np.linalg.norm(pred_np) + 1e-9)
print(f"  cosine similarity vs. HF forward: {cos:.4f}")
print(f"  max abs diff: {np.abs(ref_np - pred_np).max():.4f}")
