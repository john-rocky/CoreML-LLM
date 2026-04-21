#!/usr/bin/env python3
"""Direct test: run Google's TFLite MTP drafter on a real prompt with real
target KV cache, and measure acceptance rate vs HF Gemma 4.

This is the DECISIVE test — if even the raw TFLite drafter doesn't match
target argmax, then MTP Path A is fundamentally broken (wrong architecture,
wrong training, or wrong layer-assumption). If TFLite gets high acceptance,
then our PyTorch reimplementation has a bug we need to find.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F

HF_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
    "4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
)
TFL_PATH = "/Users/majimadaisuke/Downloads/CoreML-LLM/output/mtp_probe/section_9.tflite"

# TFLite expected K/V shapes and their INT8 dequantization scales (from metadata dump)
CTX = 32003
KV13_K_SCALE = 0.005955  # scale for INT8 → fp32
KV13_V_SCALE = 0.047244
KV14_K_SCALE = 0.001091
KV14_V_SCALE = 0.017857


def fp_to_int8(arr_fp, scale):
    """Quantize fp values to INT8 with given scale (zp=0)."""
    q = np.round(arr_fp / scale).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def int8_to_fp(arr_int8, scale):
    return arr_int8.astype(np.float32) * scale


def main():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from ai_edge_litert.interpreter import Interpreter

    print(f"Loading HF Gemma 4...")
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        HF_DIR, torch_dtype=torch.float32,
    ).eval()
    lm = hf.model.language_model
    embed_table = lm.embed_tokens.weight.to(torch.float32)

    print(f"Loading TFLite drafter...")
    interp = Interpreter(model_path=TFL_PATH)
    interp.allocate_tensors()
    runner = interp.get_signature_runner('mtp_drafter')

    prompts = [
        "The capital of France is",
        "光合成とは",
        "def fibonacci(n):",
        "Once upon a time, there was",
        "The weather today is",
    ]

    total_match = 0
    total_cases = 0

    for prompt in prompts:
        print(f"\n{'='*70}\nPROMPT: {prompt!r}")
        ids = tok(prompt, return_tensors="pt").input_ids
        N = ids.shape[1]

        with torch.no_grad():
            out = lm(input_ids=ids, output_hidden_states=True, use_cache=True)

        # Target's natural next token
        last_logits = F.linear(out.last_hidden_state[:, -1:, :], embed_table)
        if hasattr(lm.config, "final_logit_softcapping") and lm.config.final_logit_softcapping:
            cap = lm.config.final_logit_softcapping
            last_logits = torch.tanh(last_logits / cap) * cap
        natural_next = last_logits.argmax(-1).item()
        print(f"  tokens: {ids[0].tolist()}")
        print(f"  target natural next: {natural_next} ({tok.decode([natural_next])!r})")

        # L34 raw hidden state (before final norm). hidden_states[-1] is post-norm.
        l34_raw = out.hidden_states[-2][:, -1:, :].float()
        print(f"  L34 raw hidden norm: {l34_raw.norm().item():.3f}")

        # Extract KV at L13 and L14 from HF cache
        pkv = out.past_key_values
        if hasattr(pkv, "layers"):
            k13, v13 = pkv.layers[13].keys, pkv.layers[13].values
            k14, v14 = pkv.layers[14].keys, pkv.layers[14].values
        else:
            k13, v13 = pkv[13]
            k14, v14 = pkv[14]

        # K shape from HF: (1, num_kv_heads=1, seq, hd)
        seq = k13.shape[2]
        # TFLite expects:
        #   K: (1, 1, ctx=32003, hd) — LEFT-ALIGNED (absolute positions)
        #   V: (1, 1, hd, ctx) — transposed, LEFT-ALIGNED
        # Initialize with zeros, populate positions 0..seq-1

        kv13_k_fp = np.zeros((1, 1, CTX, 256), dtype=np.float32)
        kv13_v_fp = np.zeros((1, 1, 256, CTX), dtype=np.float32)
        kv14_k_fp = np.zeros((1, 1, CTX, 512), dtype=np.float32)
        kv14_v_fp = np.zeros((1, 1, 512, CTX), dtype=np.float32)

        kv13_k_fp[:, :, :seq, :] = k13.numpy()
        kv13_v_fp[:, :, :, :seq] = v13.transpose(-1, -2).numpy()
        kv14_k_fp[:, :, :seq, :] = k14.numpy()
        kv14_v_fp[:, :, :, :seq] = v14.transpose(-1, -2).numpy()

        # Quantize to INT8
        kv13_k_int8 = fp_to_int8(kv13_k_fp, KV13_K_SCALE)
        kv13_v_int8 = fp_to_int8(kv13_v_fp, KV13_V_SCALE)
        kv14_k_int8 = fp_to_int8(kv14_k_fp, KV14_K_SCALE)
        kv14_v_int8 = fp_to_int8(kv14_v_fp, KV14_V_SCALE)

        # Build mask: positions 0..seq-1 valid, rest masked
        mask = np.zeros((1, 1, 1, CTX), dtype=np.bool_)
        mask[:, :, :, :seq] = True
        # Actually — next token will be at position seq, so mask through seq-1 only
        # But try both variants

        # Build activations: concat(embed(natural_next), L34_hidden)
        embed_raw = embed_table[natural_next].unsqueeze(0).unsqueeze(0).float().detach().numpy()
        l34_np = l34_raw.numpy()

        activations = np.concatenate([embed_raw, l34_np], axis=-1).astype(np.float32)
        input_pos = np.array([seq], dtype=np.int32)  # position of next token
        param_tensor = np.zeros((1, 1, 1, 7), dtype=np.int32)
        param_tensor[0, 0, 0, 0] = seq

        # Run TFLite drafter
        try:
            out_tfl = runner(
                activations=activations,
                input_pos=input_pos,
                kv_cache_k_13=kv13_k_int8,
                kv_cache_v_13=kv13_v_int8,
                kv_cache_k_14=kv14_k_int8,
                kv_cache_v_14=kv14_v_int8,
                mask=mask,
                param_tensor=param_tensor,
            )
        except Exception as e:
            print(f"  TFLite ERROR: {e}")
            continue

        logits = out_tfl['logits']
        top5 = np.argsort(-logits[0, 0])[:5].tolist()
        is_match = (top5[0] == natural_next)
        in_top5 = natural_next in top5
        print(f"  TFLite top5: {top5}")
        print(f"  match argmax: {'YES' if is_match else 'no'}   in top5: {'YES' if in_top5 else 'no'}")

        total_match += 1 if is_match else 0
        total_cases += 1

    print(f"\n{'='*70}\n=== SUMMARY ===")
    print(f"TFLite drafter argmax match rate: {total_match}/{total_cases}")
    if total_match >= total_cases * 0.5:
        print("→ TFLite WORKS. Our PyTorch reimplementation has a bug.")
    elif total_match == 0:
        print("→ TFLite also 0%. Drafter artifact itself may not be correct for E2B-it.")
        print("  (Check: is section_9.tflite the right MTP drafter?")
        print("   Is HF model the right checkpoint? Are KV cache scales right?)")
    else:
        print("→ TFLite partial. Inherent drafter approximation.")


if __name__ == "__main__":
    main()
