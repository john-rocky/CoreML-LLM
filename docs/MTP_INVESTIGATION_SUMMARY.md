# MTP investigation — one-page summary for parent session

Date: 2026-04-17
Inputs: `docs/MTP_INVESTIGATION_HF.md`, `docs/MTP_INVESTIGATION_LITERT.md`, `docs/MTP_INVESTIGATION_PATH_A_AUTOPSY.md`

## Confidence movement

**Starting confidence:** 50% (full A100 retrain worth the $300–1000 / 5–10 days)
**New confidence:** **~20%** — direction **DOWN**

## Three findings that moved the number

1. **HF has no MTP weights, no config flag, no code path.** Inspected all 2011 safetensors tensors in `gemma4-e2b-final/hf_model/model.safetensors`; zero matched `mtp|draft|speculative|multi_token|extra_head`. `transformers` library has no MTP modules in either `gemma3n/` or `gemma4/`. Google engineer publicly confirmed exclusion is deliberate. MTP exists **only** inside the `.litertlm` container.

2. **LiteRT's MTP training recipe is not public** — searched `google-ai-edge/LiteRT-LM`, `ai-edge-torch`, Gemma 4 blog, HF blog, arXiv 2505.00232 (ML Drift). Weights are public (Section 9 of `.litertlm`, already extracted, 44.3 MB, 4-layer Q-only transformer, hidden=256, FFN=2048). Architecture is **disjoint** from HF base (HF hidden=1536, FFN=6144) — it is a separate small model, not a head grafted onto HF.

3. **Path A's 0% wasn't a bug — it was quantization distribution drift, now numerically proven.** fp32 forward on Mac Studio: top-1 = 3.1%, top-5 = 6.2%, **cosine 0.9935** with HF's softcapped logits, correct-token median rank = 2,499/262,144. The weights are learned and nearly correct; the argmax is systematically shifted because the drafter was trained against LiteRT's W4A8 quantized main LLM. Tokenizer is 100% identical (ruled out). Extraction code is clean (one cosmetic debug-print bug, no functional bug).

## Why confidence dropped more than a single factor would suggest

A full retrain at A100 × 5–10 days was priced assuming recipe *discoverability* would close most of the variance. Findings invert that: weights exist but architecture is disjoint and tiny (4-layer, hidden=256), recipe is hidden, and — critically — even a **successful** retrain landing in the 30–50% acceptance range doesn't clear the **~77% iPhone break-even** set by item 11c (verify-vs-decode fp16 drift) in `PRIORITY_ROADMAP.md`. MTP is gated behind an unrelated fix, and the upper bound on MTP's contribution is small compared to the Metal-GPU path (LiteRT's 56.5 tok/s is a GPU number; our Metal stack can close most of that gap on its own).

## Recommended action

**撤退 from full retrain. Do NOT spend $300–1000 / 5–10 A100 days now.**

Order of operations instead:

1. **Close item 11c first** (verify-vs-decode fp16 drift). Until break-even drops below ~55%, no MTP configuration, trained or extracted, produces a net speedup on device.
2. **Prioritize the Metal-LLM Phase 3 work** (docs/metal_llm_impl memory). LiteRT's 56.5 tok/s is structurally a GPU number — Metal is where the gap is actually closeable. Expected uplift is ~25 tok/s vs MTP's projected ~5 tok/s on top.
3. **Revisit MTP as head-only fine-tune (~1 A100 day, not 5–10)** *after* 11c closes. Path A's extracted weights are on disk, validated as structurally healthy; fine-tune `mtp_post_proj + lm_head` against fp targets on 1–5M token pairs. Target: 30–50% acceptance. Drop full-retrain from the roadmap.

## What changed vs prior belief

- "Google MTP trained for W4A8" → confirmed with numeric evidence (cosine 0.9935 not 0%, rank-median 2499), elevated from hypothesis to fact.
- "Path A failed because of tensor/tokenizer bug" → ruled out cleanly.
- "Full retrain could unlock 60%+ acceptance" → still plausible but now gated behind 11c, so the economic value drops independent of the training probability.
- "MTP is on the critical path to beating LiteRT 56.5 tok/s" → **false**. LiteRT's number comes from GPU; MTP is an on-top optimization. Metal path is the critical path.

## Stop-condition mapping

- Task 1: **stop-cond-1** fired (HF has no MTP).
- Task 2: **stop-cond-2 partial** (weights public, recipe not public).
- Task 3: **"other"** — distribution mismatch, numerically localized.

Combined effect: retreat, not ablation, not retrain.
