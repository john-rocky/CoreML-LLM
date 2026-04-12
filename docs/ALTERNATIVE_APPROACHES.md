# Alternative Approaches — Beyond Speculative Decoding

This is a decision log for alternative speedup directions we considered beyond the
current EAGLE-3 + W8A8 path. Kept for posterity so future iteration has the full
option set on record.

**Primary scope: optimize WITHIN Gemma 4 E2B as the target model.** Techniques
that replace or distill away from Gemma 4 are genuine alternatives but belong
to separate product tracks (a "Turbo" SKU, an "ANE-native" SKU, etc.), not the
main CoreML-LLM library recipe. They are kept in this doc for transparency but
ranked separately below.

Status legend: **S** = picked, **A** = viable next, **B** = parked, **C** = deferred, **D** = rejected.

## Scope split

### Inside Gemma 4 scope (main product)
These are weight-surgery + fine-tune + runtime tricks that preserve Gemma 4 E2B
as the model of record. They compose cleanly with the existing chunk+prefill
pipeline and EAGLE-3 draft.
- #2 Sliding-only (QLoRA recovery)
- #3 MQA on full-attention (**SELECTED**, implemented)
- MLA retrofit (MHA2MLA-style, separate doc)
- StreamingLLM+QLoRA recovery (for ctx>2K quality)
- DuoAttention / KIVI / W8A8 / EAGLE-3 / Q-batch / pre-alloc — all in `docs/SPEED_8K.md`

### Outside Gemma 4 scope (separate product tracks)
These change the model of record. Listed for completeness but they are new
products, not optimizations of the current one.
- #1 Distill Gemma 4 → 1B (needs ~$500-1k of A100 time)
- #5a Distill Gemma 4 → smaller pretrained student (Qwen 0.5B, Llama 3.2 1B, SmolLM2 1.7B; ~$200-1k)
- #5b Take existing small model + ANE surgery + QLoRA (~$20-50)
- #5 From-scratch ANE-native model (~$30-50k, research-grant tier, not individually feasible)
- #4 Cascade router (second model needed, can be any pretrained small)

---

## #1. Distill Gemma 4 E2B (2.7B) → 1B
Ship a smaller "Turbo" SKU trained to mimic Gemma 4 E2B outputs.

| Axis | Value |
|---|---|
| Status | **A** — parked, revisit after current stack lands |
| Cost | H100 × 1-2 weeks, 5-10 B tokens self-distillation |
| Expected decode cost | ~3× faster per token |
| Projected tok/s (iPhone 17 Pro) | 2K: 90-100, 8K: 45-55 (solo) |
| Compound with EAGLE-3 | 2K: 180+, 8K: 90+ |
| Risk | Quality drop (especially multilingual + code) |
| Differentiation | "1B at Gemma-4 quality" — strong vs Apple Foundation (3B) |
| Why parked | Locks us to a specific distilled model; less reusable as a library recipe |

---

## #2. Sliding-only — drop all 7 full-attention layers
Replace full-attention with sliding window in L4/L9/L14/L19/L24/L29/L34. All 35 layers become W=512 (Gemma 3 architecture).

| Axis | Value |
|---|---|
| Status | **B** — fallback if MQA under-performs |
| Cost | QLoRA 1-2 days on A100 |
| Expected decode cost | ctx-invariant at sliding-layer speed |
| Projected tok/s | 2K: 31 (unchanged), 8K: **31** (from 15) |
| Compound with EAGLE-3 | 2K: 60, 8K: 60 (both equal, the point) |
| Risk | Long-context retrieval quality degrades (needle-in-haystack) |
| Differentiation | Matches Apple's own 3B design direction (5 local + 1 global) |
| Why parked | WFA quality-regression evidence suggests sliding-only will also regress; QLoRA may not fully recover |

---

## #3. MQA for full-attention layers (num_kv 2 → 1) — **SELECTED**
Collapse the 2 KV heads in full-attention layers to 1 via weight averaging, then QLoRA-recover.

| Axis | Value |
|---|---|
| Status | **S** — executing now |
| Cost | Weight surgery (minutes) + optional QLoRA (hours on A100) |
| Expected decode cost | 8K full-attn bandwidth halved → +40% on 8K path |
| Projected tok/s | 2K: 32 (small), 8K: **20-22** (from 15) |
| Compound with EAGLE-3 | 8K: 44-48, +DuoAttention: 60+, +W8A8: 80+ |
| Risk | Quality loss without QLoRA; QLoRA recovery well-documented for GQA→MQA |
| ANE compat | ✅ Static shape change only (num_kv in chunks) |
| Why picked | Smallest-touch change with meaningful wins; fully composable with everything else |

---

## #4. Cascade router (small model for easy tokens, big for hard)
Run a 300M token predictor first; on high-confidence tokens, commit from small model; otherwise fall to Gemma 4.

| Axis | Value |
|---|---|
| Status | **C** — deferred to v0.7+ |
| Cost | 300M model training + routing threshold tuning |
| Expected tok/s | 3-5× average speedup depending on prompt difficulty |
| Risk | Lossy (small model's tokens accepted directly) — hard to QA |
| ANE compat | ✅ Two models co-resident on ANE is fine |
| Why deferred | Breaks "lossless" contract; LongBench gates harder to reason about |

---

## #5. From-scratch ANE-native model — **not individually feasible**
Design architecture from ANE constraints upward: head_dim=128, all sliding, Conv2d projections, no QK-norm (cat-trick RMSNorm native).

| Axis | Value |
|---|---|
| Status | **D** — rejected at our budget; documented as ideal for institutional backing |
| Cost | 1T tokens on 8×H100 ≈ **$30-50k cloud** or 1 month of a shared A100 cluster |
| Realistic alternative | #5a / #5b below |
| Why rejected | Individual-project budget cannot absorb a ~$30k pretraining run. Re-open only with a research grant, sponsorship, or Google/HF compute credits. |

## #5a. Distill Gemma 4 → 1B (separate product: "CoreML-LLM Turbo")
Teacher = Gemma 4 E2B; Student = 1B (new-initialized or smaller pretrained).

| Axis | Value |
|---|---|
| Status | **C** — deferred; "Turbo" SKU, not main library |
| Cost | 5-10B tokens self-distill on A100 ≈ **$200-1k** |
| Expected tok/s | 2K: 90-100, 8K: 45-55 solo; with EAGLE-3 × 2× |
| Risk | Quality drop on multilingual + code |
| Why deferred | Main product scope is Gemma 4 itself |

## #5b. Existing small model + ANE surgery (separate product)
Take Gemma 3 270M / Llama 3.2 1B / Qwen 2.5 0.5B, apply MQA + sliding-only + ANE-friendly ops, QLoRA-stabilize.

| Axis | Value |
|---|---|
| Status | **C** — deferred; alternative SKU |
| Cost | A100 × 1-2 days ≈ **$20-50** |
| Expected tok/s | 8K: 60+ solo (smaller base, lower cap) |
| Risk | Base capability lower than Gemma 4 |
| Why deferred | Not the primary scope |

---

## Decision rationale (2026-04-12)

Current stack ranked by composable speedup at pure ANE execution:

```
baseline            15 tok/s @ 8K
+ pre-alloc         16   (trivial, from Swift)
+ Q-batch KV-share  17   (~40 LoC)
+ MQA (#3)          22   (QLoRA hours)           ← NEW
+ W8A8              31   (calibration, in flight)
+ DuoAttention      46   (head identification + chunk surgery)
+ EAGLE-3           92   (in training, verify lossless)
```

Realistic ANE overhead correction (×0.65) → **~60 tok/s @ 8K** under pure ANE.

**Logic**:
- #3 (MQA) is picked because it is the smallest-touch change that compounds with everything we have in flight. No speculative-decoding dependence, no W8A8 dependence.
- #1 and #5 are left for later because they commit to a specific model or training pipeline — less reusable as a general CoreML-LLM library recipe.
- #2 is in reserve if MQA quality holds but LongBench still suffers on long contexts.
- #4 is architecturally attractive but lossy — deferred until lossless options are exhausted.

---

## What "Gemma-4 ceiling" means

The list of **things to exhaust inside Gemma 4 scope** before entertaining
separate product tracks (#1/#5a/#5b). All within individual-project budget:

- [ ] #3 MQA + QLoRA recovery ← implementation pushed, not yet executed
- [ ] #2 Sliding-only with QLoRA recovery (fallback if MQA is insufficient)
- [ ] W8A8 proper calibration (in flight, bench session)
- [ ] DuoAttention head cap (Tier-2, head identification ready)
- [ ] EAGLE-3 deployed on iPhone (in training + conversion pipeline ready)
- [ ] Pre-alloc masks + KV-share Q-batch in ChunkedEngine (bench session)
- [ ] Prefill on A19 Pro GPU tensor cores (independent TTFT win)
- [ ] StreamingLLM+QLoRA for true 8K quality recovery
- [ ] MLA retrofit via MHA2MLA (v0.6+ architectural upgrade, still within Gemma 4)

**Once every item above is shipped-or-rejected-with-data**, further speed
requires *leaving* Gemma 4 scope (#5a distill, #5b small-model swap), which is
a separate product decision — not the main library's roadmap.

From-scratch (#5) remains rejected at our budget unless institutional compute
becomes available.
