#!/usr/bin/env python3
"""Run our custom Gemma4Model (the one we trace + convert to CoreML) on the
same input as HF, compare per-chunk hidden states.

If our PyTorch model already differs from HF, the bug is in `gemma4.py`.
If our PyTorch model matches HF but the compiled CoreML chunk differs,
the bug is in coremltools emission / quantization.
"""
from __future__ import annotations
import sys, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gemma4 import Gemma4Model

HF_DIR = "/Users/majimadaisuke/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"


def _diff(label, hf, our):
    a = hf.float().flatten()
    b = our.float().flatten()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
    diff = (a - b).abs()
    print(f"{label:<35} cos={cos:.6f}  |hf|={a.norm():.2f}  |our|={b.norm():.2f}  "
          f"max_diff={diff.max():.4f}  mean_diff={diff.mean():.5f}")


def main():
    # HF reference
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    text_model_hf = target.model.language_model
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    # Our custom model
    if not os.path.isdir(HF_DIR):
        print(f"ERROR: HF dir not found at {HF_DIR}")
        sys.exit(1)
    our_model = Gemma4Model.from_pretrained(HF_DIR, context_length=64)
    our_model.eval().to(torch.float32)
    print(f"loaded our Gemma4Model: {our_model}")
    print(f"our config full_partial_rotary_factor = {our_model.config.full_partial_rotary_factor}")
    print(f"our config sliding_rope_theta = {our_model.config.sliding_rope_theta}")
    print(f"our config full_rope_theta = {our_model.config.full_rope_theta}")

    # Forward both on a 1-token input.
    ids = torch.tensor([[2]], dtype=torch.long)  # BOS

    # HF forward.
    with torch.no_grad():
        out_hf = text_model_hf(input_ids=ids, output_hidden_states=True,
                               use_cache=False, return_shared_kv_states=True)
    hs_hf = out_hf.hidden_states  # tuple of (1, 1, hidden)
    print(f"\nHF forward: {len(hs_hf)} hidden_states")
    print(f"  hs[0]  (embed):    norm={hs_hf[0].norm():.3f}")
    print(f"  hs[1]  (post-L0):  norm={hs_hf[1].norm():.3f}")
    print(f"  hs[8]  (post-L7):  norm={hs_hf[8].norm():.3f}")
    print(f"  hs[15] (post-L14): norm={hs_hf[15].norm():.3f}")
    print(f"  hs[-1] (post-norm): norm={hs_hf[-1].norm():.3f}")

    # Our model forward — we need to drive its internal layer chain.
    # Build a SIMPLIFIED forward that mirrors HF's: embed → 35 layers → norm.
    print(f"\nour model has: embed_tokens, layers ({len(our_model.layers)}), norm")
    embed = our_model.embed_tokens(ids).float()
    print(f"  our embed (raw): norm={embed.norm():.3f}")
    # HF scales embed by sqrt(hidden) inside Gemma4ScaledWordEmbedding.
    # Swift runtime applies the same scale via EmbeddingLookup. Apply manually.
    embed_scaled = embed * (our_model.config.hidden_size ** 0.5)
    print(f"  our embed (*sqrt(hidden)): norm={embed_scaled.norm():.3f}")
    _diff("embed *sqrt(hidden)", hs_hf[0], embed_scaled)

    # Now run through the layers manually, mirroring HF's flow.
    # Build per_layer_inputs from get_per_layer_inputs.
    per_layer_inputs_hf = text_model_hf.get_per_layer_inputs(ids, embed_scaled).float()
    if per_layer_inputs_hf.ndim == 4:
        per_layer_inputs_flat = per_layer_inputs_hf.reshape(
            per_layer_inputs_hf.shape[0], per_layer_inputs_hf.shape[1], -1)
    else:
        per_layer_inputs_flat = per_layer_inputs_hf
    print(f"  per_layer_inputs (HF): norm={per_layer_inputs_flat.norm():.3f}")

    # Compare to our embed_per_layer scaled version.
    # Our model has self.embed_per_layer that's nn.Embedding without explicit scale.
    pl_emb_ours = our_model.embed_tokens_per_layer(ids).float()
    pl_emb_ours_scaled = pl_emb_ours * (our_model.config.hidden_size_per_layer_input ** 0.5)
    print(f"  per_layer_inputs (ours scaled): norm={pl_emb_ours_scaled.norm():.3f}")
    if pl_emb_ours_scaled.shape != per_layer_inputs_flat.shape:
        print(f"  shape mismatch: ours {tuple(pl_emb_ours_scaled.shape)} vs HF {tuple(per_layer_inputs_flat.shape)}")
    else:
        _diff("per_layer_raw *sqrt(ple)", per_layer_inputs_flat, pl_emb_ours_scaled)


if __name__ == "__main__":
    main()
