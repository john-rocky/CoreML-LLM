"""Empirical mask/KV-layout verification for production Swift drafter.

Loads PRODUCTION right-aligned KV dumped via MTP_CHUNK_DUMP. Runs HF target on
the same prompt to get reference shared_kv. Then:
  REF top-1   = HF drafter(target's shared_kv at pos)        [reference]
  PROD top-1  = CoreML drafter(production right-aligned KV)  [our path]

If REF == PROD (or top-5 overlap > 80 %), the right-aligned mask + KV layout is
behaving correctly under INT4 quantization noise — mask is NOT the bottleneck.
If they diverge sharply (e.g. < 20 % top-5 overlap), there IS a layout/mask bug.

This is the empirical sibling of compare_drafter_hf_vs_coreml.py — that one used
HF KV converted to CoreML format and ignored production. This one uses real
production KV from a Swift run.
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coremltools as ct


PROMPT_TRANSLATE = (
    "Translate the following 10 sentences from English to French.\n"
    "1. Hello, how are you today?\n2. The cat is on the table.\n"
    "3. I would like a cup of coffee.\n4. Where is the train station?\n"
    "5. We are going to the beach tomorrow.\n6. She reads books every "
    "evening.\n7. They live in a small village.\n8. The weather is "
    "very nice.\n9. Could you help me, please?\n10. I love this song."
)


def cosine(a, b):
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/tmp/coreml_kv_dump_translate")
    ap.add_argument("--target", default="google/gemma-4-E2B-it")
    ap.add_argument("--assistant", default="google/gemma-4-E2B-it-assistant")
    ap.add_argument("--coreml-drafter",
                    default="/tmp/mtp_drafter_lse2/mtp_drafter.mlpackage")
    ap.add_argument("--prompt", default="translate")
    args = ap.parse_args()

    info = open(f"{args.dump_dir}/info.txt").read()
    print(f"production dump info: {info.strip()}")
    pos = int(info.split("pos=")[1].split()[0])
    next_id = int(info.split("nextID=")[1].split()[0])
    n_written = pos  # number of positions filled in cache (0..pos-1)
    print(f"  pos={pos}, n_written={n_written}, nextID={next_id}")

    # ----- load production KV -----
    kv13_k_prod = np.frombuffer(
        open(f"{args.dump_dir}/kv13_k.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 512, 256)
    kv13_v_prod = np.frombuffer(
        open(f"{args.dump_dir}/kv13_v.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 512, 256)
    kv14_k_prod = np.frombuffer(
        open(f"{args.dump_dir}/kv14_k.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 2048, 512)
    kv14_v_prod = np.frombuffer(
        open(f"{args.dump_dir}/kv14_v.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 2048, 512)
    h4_prod = np.frombuffer(
        open(f"{args.dump_dir}/h4_postnorm.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 1536)
    embed_prod = np.frombuffer(
        open(f"{args.dump_dir}/embed_input.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 1536)
    print(f"  kv13_k_prod: {kv13_k_prod.shape}, kv14_k_prod: {kv14_k_prod.shape}")
    print(f"  h4_postnorm_prod: {h4_prod.shape}, embed_input_prod: {embed_prod.shape}")

    # ----- HF reference target run -----
    print("\n=== HF reference run ===")
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16).eval().to("mps")

    msgs = [{"role": "user", "content": PROMPT_TRANSLATE}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("mps")
    L = ids["input_ids"].shape[1]
    print(f"  HF prompt_len={L} (production pos was {pos})")

    # Production pos = n_written = prompt_len + 1 (bootstrap token). The
    # bootstrap token id is in info.txt as nextID. Build input_ids =
    # prompt + [nextID] so HF target's KV covers the SAME positions as
    # production cache (0..n_written-1 = 0..pos-1).
    if n_written == L + 1:
        boot = torch.tensor([[next_id]], device="mps", dtype=ids["input_ids"].dtype)
        use_ids = torch.cat([ids["input_ids"], boot], dim=1)
        print(f"  HF input = prompt({L}) + bootstrap nextID({next_id}) "
              f"→ {use_ids.shape[1]} tokens")
    else:
        use_len = min(n_written, L)
        use_ids = ids["input_ids"][:, :use_len]
        print(f"  WARN: production n_written={n_written} != prompt+1 ({L+1}); "
              f"using prompt[:{use_len}]")

    with torch.no_grad():
        out = target.model.language_model(
            input_ids=use_ids, return_shared_kv_states=True,
            output_hidden_states=False)
    hf_swa_k = out.shared_kv_states["sliding_attention"][0]  # (1, h, use_len, 256)
    hf_swa_v = out.shared_kv_states["sliding_attention"][1]
    hf_full_k = out.shared_kv_states["full_attention"][0]
    hf_full_v = out.shared_kv_states["full_attention"][1]
    hf_h_post_norm = out.last_hidden_state[:, -1:]  # (1, 1, 1536)

    print(f"  HF swa_k: {hf_swa_k.shape}, full_k: {hf_full_k.shape}")
    print(f"  HF h_last: {hf_h_post_norm.shape}")

    # ----- KV cosine sanity. Production stores n_written positions in the
    # right side of W=512 (slots [W-n_written..W-1]). HF stores positions
    # 0..n_written-1 in slots [0..n_written-1]. So compare production
    # slots [W-n_written..W-1] to HF slots [0..n_written-1].
    W_swa = 512
    L_use = min(n_written, hf_swa_k.shape[2])
    # production slot for HF position i = (W - n_written) + i
    start_slot = W_swa - n_written
    coreml_kv13_k_used = kv13_k_prod[0, 0, start_slot:start_slot + L_use]
    hf_swa_k_used = hf_swa_k[0, 0, :L_use].detach().float().cpu().numpy()
    print(f"  KV cosine production vs HF (first {L_use} positions; "
          f"prod slots [{start_slot}..{start_slot+L_use-1}] vs HF [0..{L_use-1}]): "
          f"{cosine(coreml_kv13_k_used, hf_swa_k_used):.4f}")
    # per-position spot check
    for i in [0, L_use//2, L_use-1]:
        c = cosine(coreml_kv13_k_used[i], hf_swa_k_used[i])
        print(f"    pos {i}: cos = {c:.4f}")
    # CRITICAL: which HF position best matches production's LAST slot?
    # If production slot W-1 holds something other than position n_written-1,
    # we have an off-by-one or cache-shift bug.
    last_slot = kv13_k_prod[0, 0, W_swa - 1]
    print(f"\n  Last-slot scan (production slot {W_swa-1} vs HF pos i):")
    full_hf = hf_swa_k[0, 0].detach().float().cpu().numpy()
    best = (-1.0, -1)
    for hf_pos in range(max(0, n_written - 5), n_written):
        c = cosine(last_slot, full_hf[hf_pos])
        marker = "  <-- best so far" if c > best[0] else ""
        if c > best[0]:
            best = (c, hf_pos)
        print(f"    HF pos {hf_pos}: cos = {c:.4f}{marker}")
    print(f"  best HF match: pos {best[1]} (cos {best[0]:.4f})")
    cos_h = cosine(h4_prod, hf_h_post_norm.detach().float().cpu().numpy())
    print(f"  h_post_norm (production at pos {n_written-1} vs HF h_last): "
          f"cos = {cos_h:.4f}")

    # ----- HF drafter REFERENCE top-1 -----
    print("\n=== HF drafter on HF reference KV ===")
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16).eval().to("mps")
    target_embed = target.model.language_model.embed_tokens

    # In HF flow at this point: predict position n_written from h_last.
    # last_token_id = the token at position n_written-1 (= the bootstrap
    # token, = production nextID).
    last_token_id = use_ids[:, -1:]
    print(f"  HF last_token_id={int(last_token_id.item())} "
          f"(production nextID={next_id}) — should match")

    emb_hf = target_embed(last_token_id)  # already scaled
    inputs_embeds_hf = torch.cat([emb_hf, hf_h_post_norm], dim=-1)
    pos_ids = torch.tensor([[n_written - 1]], dtype=torch.long, device="mps")
    with torch.no_grad():
        d_out = assistant(
            inputs_embeds=inputs_embeds_hf,
            position_ids=pos_ids,
            shared_kv_states={
                k: (v[0][:, :, :n_written, :], v[1][:, :, :n_written, :])
                for k, v in out.shared_kv_states.items()
            },
            use_cache=False,
        )
    hf_top1 = int(d_out.logits[0, -1, :].argmax().item())
    hf_top5 = torch.topk(d_out.logits[0, -1, :], k=5)
    hf_top5_ids = hf_top5.indices.tolist()
    print(f"  HF reference top1={hf_top1}  top5={hf_top5_ids}")

    # ----- CoreML drafter on PRODUCTION right-aligned KV -----
    print("\n=== CoreML drafter on PRODUCTION right-aligned KV ===")
    coreml_drafter = ct.models.MLModel(
        args.coreml_drafter, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # production KV is exactly what Swift passes:
    #   kv13_k: (1,1,512,256) right-aligned
    #   kv13_v: (1,1,512,256) right-aligned (Swift transposes before drafter call)
    #   kv14_k: (1,1,2048,512) left-aligned
    #   kv14_v: (1,1,2048,512) left-aligned (Swift transposes before drafter call)
    # Apply same transposes Swift does.
    kv13_v_t = kv13_v_prod.transpose(0, 1, 3, 2)  # (1,1,256,512)
    kv14_v_t = kv14_v_prod.transpose(0, 1, 3, 2)  # (1,1,512,2048)

    # RoPE: production Swift uses constant draftPos = pos - 1 = n_written-1
    # via lookupCos/SinSWA/Full.
    rope_pos = n_written - 1
    rot_full = target.model.language_model.rotary_emb
    cos_full_full, sin_full_full = rot_full(
        torch.zeros(1, 1, 1536, device="mps", dtype=torch.bfloat16),
        torch.tensor([[rope_pos]], device="mps"),
        "full_attention",
    )
    cos_swa_full, sin_swa_full = rot_full(
        torch.zeros(1, 1, 1536, device="mps", dtype=torch.bfloat16),
        torch.tensor([[rope_pos]], device="mps"),
        "sliding_attention",
    )
    cos_full_np = cos_full_full[0, :, :cos_full_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)
    sin_full_np = sin_full_full[0, :, :sin_full_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)
    cos_swa_np = cos_swa_full[0, :, :cos_swa_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)
    sin_swa_np = sin_swa_full[0, :, :sin_swa_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)

    # Production mask: maskPos = pos - 1 = n_written-1
    # makeSlidingCausalMask: valid = min(maskPos+1, W) = min(n_written, W)
    #                       start = W - valid; mp[i] = 0 if i >= start else -inf
    valid = min(n_written, W_swa)
    start = W_swa - valid
    mask_swa_np = np.full((1, 1, 1, W_swa), -65500.0, dtype=np.float16)
    mask_swa_np[0, 0, 0, start:] = 0.0
    # makeCausalMask(maskPos, ctx): mp[i] = 0 if i <= maskPos else -inf
    ctx = 2048
    maskPos = n_written - 1
    mask_full_np = np.full((1, 1, 1, ctx), -65500.0, dtype=np.float16)
    mask_full_np[0, 0, 0, :maskPos + 1] = 0.0

    out_dict = coreml_drafter.predict({
        "embed_token": embed_prod,
        "proj_act": h4_prod,
        "kv13_k": kv13_k_prod,
        "kv13_v": kv13_v_t,
        "kv14_k": kv14_k_prod,
        "kv14_v": kv14_v_t,
        "cos_swa": cos_swa_np,
        "sin_swa": sin_swa_np,
        "cos_full": cos_full_np,
        "sin_full": sin_full_np,
        "mask_swa": mask_swa_np,
        "mask_full": mask_full_np,
    })
    prod_top1 = int(out_dict["top_k_indices"][0])
    prod_top5 = [int(x) for x in out_dict["top_k_indices"][:5]]
    print(f"  PROD top1={prod_top1}  top5={prod_top5}")

    # ===== ISOLATION TEST: HF reference KV + production-shaped mask =====
    # If we feed HF's clean reference KV/h_last to CoreML drafter using
    # the SAME mask shape we use in production, the output should match
    # HF reference top-1. If yes -> mask is definitively NOT the bug.
    # If no  -> mask IS the bug (or drafter mlpackage itself is wrong).
    print("\n=== ISOLATION: HF reference KV + production-built mask ===")
    # Pad HF KV to W (right-aligned, mirroring production layout).
    pad_swa_k = np.zeros((1, 1, W_swa, 256), dtype=np.float16)
    pad_swa_v = np.zeros((1, 1, W_swa, 256), dtype=np.float16)
    pad_full_k = np.zeros((1, 1, ctx, 512), dtype=np.float16)
    pad_full_v = np.zeros((1, 1, ctx, 512), dtype=np.float16)
    hf_swa_k_np = hf_swa_k[0, 0].detach().float().cpu().numpy().astype(np.float16)
    hf_swa_v_np = hf_swa_v[0, 0].detach().float().cpu().numpy().astype(np.float16)
    hf_full_k_np = hf_full_k[0, 0].detach().float().cpu().numpy().astype(np.float16)
    hf_full_v_np = hf_full_v[0, 0].detach().float().cpu().numpy().astype(np.float16)
    n_hf = hf_swa_k_np.shape[0]
    pad_swa_k[0, 0, W_swa - n_hf:] = hf_swa_k_np
    pad_swa_v[0, 0, W_swa - n_hf:] = hf_swa_v_np
    pad_full_k[0, 0, :n_hf] = hf_full_k_np
    pad_full_v[0, 0, :n_hf] = hf_full_v_np
    pad_swa_v_t = pad_swa_v.transpose(0, 1, 3, 2)
    pad_full_v_t = pad_full_v.transpose(0, 1, 3, 2)
    h_hf_np = hf_h_post_norm.detach().float().cpu().numpy().astype(np.float16)
    emb_hf_np = emb_hf.detach().float().cpu().numpy().astype(np.float16)
    out_iso = coreml_drafter.predict({
        "embed_token": emb_hf_np,
        "proj_act": h_hf_np,
        "kv13_k": pad_swa_k, "kv13_v": pad_swa_v_t,
        "kv14_k": pad_full_k, "kv14_v": pad_full_v_t,
        "cos_swa": cos_swa_np, "sin_swa": sin_swa_np,
        "cos_full": cos_full_np, "sin_full": sin_full_np,
        "mask_swa": mask_swa_np, "mask_full": mask_full_np,
    })
    iso_top1 = int(out_iso["top_k_indices"][0])
    iso_top5 = [int(x) for x in out_iso["top_k_indices"][:5]]
    print(f"  ISO top1={iso_top1}  top5={iso_top5}")
    print(f"  HF reference top1={hf_top1}")
    iso_overlap = len(set(iso_top5) & set(hf_top5_ids))
    print(f"  ISO==HF top1?  {iso_top1 == hf_top1};  top-5 overlap {iso_overlap}/5")

    print("\n=== VERDICT ===")
    overlap = len(set(prod_top5) & set(hf_top5_ids))
    print(f"  HF top1 == PROD top1?  {hf_top1 == prod_top1}")
    print(f"  top-5 overlap: {overlap}/5")
    print(f"  HF top5: {hf_top5_ids}")
    print(f"  PROD top5: {prod_top5}")
    if hf_top1 == prod_top1 and overlap >= 4:
        print("  → mask + RoPE + KV layout PROVE CORRECT under INT4 noise.")
    elif overlap >= 3:
        print("  → mostly correct; INT4 noise visible but no layout/mask bug.")
    else:
        print("  → divergence > INT4 noise; investigate layout/mask/RoPE.")


if __name__ == "__main__":
    main()
