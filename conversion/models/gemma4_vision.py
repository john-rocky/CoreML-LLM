"""Gemma 4 Vision Encoder conversion for CoreML.

Converts the vision tower + pooler + embed_vision projection
into a single CoreML model.

Input:  pixel_values (1, num_patches, patch_dim) — e.g. (1, 2520, 768)
        image_position_ids (1, num_patches, 2) — 2D patch coordinates
Output: image_features (1, num_output_tokens, hidden_size) — e.g. (1, 280, 1536)
"""

from __future__ import annotations

import os
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from ane_ops import MODEL_DTYPE


def extract_image_features(hf_model, processor, image) -> torch.Tensor:
    """Extract image features using the HuggingFace model's vision tower.

    This runs the full HF vision pipeline (patch embed → encoder → pooler → projection)
    and returns the projected features ready to be injected into the text decoder.

    Args:
        hf_model: Gemma4ForConditionalGeneration model
        processor: Gemma4Processor
        image: PIL Image

    Returns:
        image_features: (num_tokens, hidden_size) tensor — e.g. (280, 1536)
    """
    # Process image to get pixel_values and position_ids
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
        image_features = vision_out.pooler_output  # (num_tokens, hidden_size)

    return image_features


def save_vision_weights(hf_model, output_dir: str) -> None:
    """Save vision tower weights for potential future CoreML conversion.

    The vision encoder has complex dynamic masking that makes torch.jit.trace
    difficult. For now, we save the weights and use PyTorch for vision encoding.
    CoreML conversion of the vision tower is a future enhancement.
    """
    import numpy as np

    vision_dir = os.path.join(output_dir, "vision_weights")
    os.makedirs(vision_dir, exist_ok=True)

    vision_model = hf_model.model.vision_tower
    embed_vision = hf_model.model.embed_vision

    # Save key components
    state = {}
    for name, param in vision_model.named_parameters():
        state[f"vision_tower.{name}"] = param.data.cpu().half().numpy()
    for name, param in embed_vision.named_parameters():
        state[f"embed_vision.{name}"] = param.data.cpu().half().numpy()

    np.savez_compressed(os.path.join(vision_dir, "vision_weights.npz"), **state)
    print(f"  Saved {len(state)} vision weight tensors to {vision_dir}/")
