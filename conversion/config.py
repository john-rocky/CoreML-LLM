"""Model registry for CoreML-LLM conversion pipeline.

Each entry defines the HuggingFace repo ID and architecture module
used by convert.py to download and convert models.
"""

from dataclasses import dataclass


@dataclass
class ConversionConfig:
    """Configuration for a model conversion job."""

    hf_repo: str
    architecture: str  # Module name in conversion/models/
    default_context_length: int = 2048
    max_context_length: int = 32768
    description: str = ""


MODEL_REGISTRY: dict[str, ConversionConfig] = {
    "qwen2.5-0.5b": ConversionConfig(
        hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
        architecture="qwen2",
        default_context_length=2048,
        max_context_length=32768,
        description="Qwen2.5 0.5B Instruct - smallest, fastest pipeline validation",
    ),
    "qwen2.5-1.5b": ConversionConfig(
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        architecture="qwen2",
        default_context_length=2048,
        max_context_length=32768,
        description="Qwen2.5 1.5B Instruct - good quality/size balance",
    ),
    "qwen2.5-3b": ConversionConfig(
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        default_context_length=2048,
        max_context_length=32768,
        description="Qwen2.5 3B Instruct - highest quality Qwen2 for mobile",
    ),
    "gemma4-e2b": ConversionConfig(
        hf_repo="google/gemma-4-E2B-it",
        architecture="gemma4",
        default_context_length=512,
        max_context_length=131072,
        description="Gemma 4 E2B Instruct - Google's smallest Gemma 4 text decoder",
    ),
}


def list_models() -> None:
    """Print available models."""
    print("Available models:")
    print("-" * 60)
    for name, cfg in MODEL_REGISTRY.items():
        print(f"  {name:20s}  {cfg.hf_repo}")
        if cfg.description:
            print(f"  {'':20s}  {cfg.description}")
        print()
