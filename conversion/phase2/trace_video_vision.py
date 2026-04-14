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

MODEL = "/Users/daisukemajima/Documents/Models/gemma4-e2b/hf_model"
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
