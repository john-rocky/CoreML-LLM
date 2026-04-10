#!/usr/bin/env python3
"""Convert Gemma 4 E2B audio tower to CoreML.

The audio tower is a 12-layer Conformer encoder + output_proj + embed_audio.
Input: mel spectrogram (1, T, 128) float32
Output: audio_features (1, num_tokens, 1536) float16

The number of output tokens depends on audio duration:
  ~25 tokens per second of audio (40ms per token).
  2 sec → 50 tokens, 30 sec → 750 tokens.

Usage:
    python convert_audio.py --output ./output/audio
"""
import argparse
import os
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import Gemma4ForConditionalGeneration


class AudioTowerWrapper(nn.Module):
    """Wraps audio_tower + embed_audio for CoreML export.

    The HF Conformer uses dynamic masking (create_bidirectional_mask) that
    prevents torch.jit.trace. We save the weights and run the pipeline
    in two steps:
    1. Extract audio features using HF model in Python (feature extraction)
    2. The CoreML model just wraps embed_audio (RMSNorm + Linear 1536→1536)

    For the full pipeline, audio_tower runs in Python during conversion to
    pre-extract features. On iPhone, we'll need to either:
    a) Convert audio_tower layer-by-layer (like text decoder chunks), or
    b) Run mel → audio_tower in a separate CoreML model with fixed input size.
    """
    def __init__(self, model):
        super().__init__()
        # Only the embed projection (lightweight, traceable)
        self.embed_norm = model.model.embed_audio.embedding_pre_projection_norm
        self.embed_proj = model.model.embed_audio.embedding_projection

    def forward(self, audio_tower_output):
        # audio_tower_output: (1, num_tokens, 1536) from audio tower
        h = self.embed_norm(audio_tower_output)
        h = self.embed_proj(h)
        return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="./output/gemma4-e2b-final/hf_model")
    parser.add_argument("--output", type=str, default="./output/audio")
    parser.add_argument("--max-frames", type=int, default=199,
                        help="Max mel frames (199 ≈ 2 sec). Use 3000 for 30 sec.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading HF model...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu")
    model.eval()

    wrapper = AudioTowerWrapper(model).eval().half()

    T = args.max_frames  # mel spectrogram time frames
    sample_features = torch.zeros(1, T, 128, dtype=torch.float16)
    sample_mask = torch.ones(1, T, dtype=torch.bool)

    # The Conformer encoder has dynamic masking (create_bidirectional_mask)
    # that prevents torch.jit.trace. Same issue as the vision encoder.
    #
    # Strategy: save audio_tower weights as numpy for potential future
    # CoreML conversion (layer-by-layer, like text decoder). For now,
    # the audio tower runs in Python on Mac for feature extraction.
    # On iPhone, we'll need to re-implement the Conformer for CoreML.

    # Step 1: Save audio tower weights
    print("Saving audio tower weights...")
    audio_weights = {}
    for name, param in model.model.audio_tower.named_parameters():
        audio_weights[f"audio_tower.{name}"] = param.data.cpu().half().numpy()
    for name, param in model.model.embed_audio.named_parameters():
        audio_weights[f"embed_audio.{name}"] = param.data.cpu().half().numpy()
    np.savez_compressed(os.path.join(args.output, "audio_weights.npz"), **audio_weights)
    print(f"  Saved {len(audio_weights)} weight tensors")

    # Step 2: Test extraction with real audio
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(args.model_path)
    t_arr = np.linspace(0, 2.0, sr := 16000 * 2)
    test_audio = (np.sin(2 * np.pi * 440 * t_arr) * 0.5).astype(np.float32)
    features = fe([test_audio], sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        out = model.model.audio_tower(
            features["input_features"].to(torch.float16),
            attention_mask=features["input_features_mask"])
        h = out.last_hidden_state
        h = wrapper.embed_norm(h)
        h = wrapper.embed_proj(h)
        print(f"  Test: 2 sec audio → {h.shape[1]} tokens, shape {h.shape}")
        np.save(os.path.join(args.output, "test_audio_features.npy"),
                h.cpu().float().numpy())

    # Step 3: Save feature extractor config for Swift mel computation
    import json
    fe_config = {
        "sampling_rate": 16000,
        "feature_size": 128,
        "frame_length": 320,
        "hop_length": 160,
        "fft_length": 512,
        "mel_floor": 1e-5,
        "min_frequency": 0,
        "max_frequency": 8000,
        "log_offset": 0.001,
        "preemphasis": 0.97,
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "eoa_token_id": 258883,
        "ms_per_token": 40,
        "max_tokens": 750,
    }
    with open(os.path.join(args.output, "audio_config.json"), "w") as f:
        json.dump(fe_config, f, indent=2)
    print(f"  Saved audio_config.json")

    print(f"\nAudio conversion complete!")
    print(f"  Weights:  {args.output}/audio_weights.npz")
    print(f"  Config:   {args.output}/audio_config.json")
    print(f"  Test:     {args.output}/test_audio_features.npy")
    print(f"\nNOTE: The Conformer encoder cannot be traced for CoreML due to")
    print(f"dynamic masking. Next step: re-implement the 12-layer Conformer")
    print(f"in a trace-friendly way (like we did for the text decoder).")


if __name__ == "__main__":
    main()
