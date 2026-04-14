"""Probe the Gemma 4 vision tower to understand how to produce 70-soft-token output."""
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import Gemma4ForConditionalGeneration, Gemma4Processor

MODEL = "/Users/daisukemajima/Documents/Models/gemma4-e2b/hf_model"

print("loading model (CPU, fp16)...")
m = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cpu")
m.eval()

print("\n=== vision_tower top-level modules ===")
for name, mod in m.model.vision_tower.named_children():
    print(f"  {name:20s} {type(mod).__name__}")

print("\n=== embed_vision ===")
for name, mod in m.model.embed_vision.named_children():
    print(f"  {name:20s} {type(mod).__name__}")

print("\n=== vision_config ===")
vc = m.model.vision_tower.config
for k in ["default_output_length", "patch_size", "pooling_kernel_size",
          "hidden_size", "position_embedding_size", "max_position_embeddings"]:
    print(f"  {k}: {getattr(vc, k, '?')}")

# Try a small forward pass: 384×384 square → 24×24 patches = 576 real,
# needs to fit in 630 for video budget.
print("\n=== dummy forward: 384×384 (24×24 patches) ===")
num_patches = 630   # video budget cap
patch_dim = 16 * 16 * 3
pv = torch.zeros(1, num_patches, patch_dim, dtype=torch.float16)
pid = torch.full((1, num_patches, 2), -1, dtype=torch.long)
# Real 24×24 grid
k = 0
for py in range(24):
    for px in range(24):
        pid[0, k, 0] = px
        pid[0, k, 1] = py
        k += 1
print(f"  real patches: {k} / {num_patches}")

with torch.no_grad():
    out = m.model.get_image_features(
        pixel_values=pv,
        image_position_ids=pid,
        return_dict=True,
    )
    feat = out.pooler_output
print(f"  pooler_output shape: {tuple(feat.shape)}")
# After embed_vision.embedding_projection we get (N, 1536)
proj = m.model.embed_vision.embedding_projection(feat)
print(f"  after projection:    {tuple(proj.shape)}")
