"""Standalone repro for the video-grade vision encoder convert.

The real implementation lives in
`conversion/models/gemma4_vision.py::convert_video_vision_to_coreml`.
This script just loads the HF weights and hands them off so the
conversion pipeline can be exercised outside the main CLI.

Usage:
    python conversion/phase2/trace_video_vision.py [OUT_DIR]

Env vars:
    GEMMA4_MODEL   HF repo id or local path (default: google/gemma-4-E2B-it)
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import Gemma4ForConditionalGeneration

# Make `from models...` imports resolve regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from models.gemma4_vision import convert_video_vision_to_coreml  # noqa: E402


def main() -> None:
    model = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E2B-it")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading HF model from {model}...")
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        model, dtype=torch.float16, device_map="cpu",
    )
    hf.eval()

    convert_video_vision_to_coreml(hf, out_dir)


if __name__ == "__main__":
    main()
