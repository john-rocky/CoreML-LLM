# Priority Roadmap — Unified ranking of all speed candidates

Consolidated from 6 docs, ~52 candidates. Ranked by **ROI** (expected gain
÷ effort), filtered by **lossless** and **ANE-compatible**. Rejected items
listed at bottom with reason.

**Baseline**: 14.5 tok/s @ ctx=8192, iPhone 17 Pro (measured 2026-04-13).
**Goal**: 50+ tok/s @ 8K, lossless, pure ANE decode.

**Key reframing** (from Orion arXiv 2603.06728): ANE runs at 0.07% of peak
19 TFLOPS. INT8 == FP16 throughput. matmul ~3× slower than Conv1×1.
**Bottleneck is dispatch overhead, not compute or bandwidth.**

---

## Phase 0 — Diagnostics & zero-risk micro-opts (< 1 day each)

Run these **now**, before any speculative or conversion work. They either
(a) price the rest of the roadmap or (b) are free throughput.

| # | What | Gain | Effort | Source |
|---|---|---|---|---|
| ~~**0a**~~ | ~~**MLComputePlan audit**~~ | **DONE (2026-04-13)** — chunk1-3 = 100% ANE. chunk4 = 8 CPU ops in InModelArgmax tail (cost ~0.003, ~1-3% of step). No hidden CPU/GPU fallback on compute ops. **Dispatch-overhead hypothesis confirmed.** | — | V2 §G2, EXPERIMENTS.md |
| **0b** | **ANE pipeline prewarming** — 4× dummy predictions at load | first-token fix | 10 LoC Swift | V3 §B4 |
| **0c** | **exp2 softmax** — replace torch.exp → torch.exp2 (ANE native) | 0–5% free | 2 LoC + reconvert | V3 §B1 |

0a result: no compute-op fallback to fix. The 5–25% "if fallback found"
scenario did not materialize. Bottleneck is confirmed as 4× per-step
dispatch overhead, not per-op device placement. **MLState is the lever.**

---

## Phase 1 — Training-free, no EAGLE-3 dependency (1–2 weeks)

These can ship **today**. No draft model needed.

| Priority | What | Gain (standalone) | Effort | Lossless | Source |
|---|---|---|---|---|---|
| ~~**1**~~ | ~~**W2A16 palettization**~~ | **REJECTED (2026-04-13)** — W2/W3 post-training = gibberish. QAT required. | — | — | FUND §3 |
| ~~**2**~~ | ~~**MLState stateful KV**~~ | **REJECTED (2026-04-13)** — `coreml_update_state` fails on ANE (error -14) on both Mac and iPhone. GPU-only. Apple's own on-device LLMs use stateless I/O. | — | — | FUND §2 |
| **3** | **MLP tile reshape (B,C,8,8)** — vision ANE trick | up to ×1.5 on FFN | 1 day + reconvert | exact | V3 §B2 |
| **4** | **GQA broadcast matmul** — drop repeat_interleave | ×1.05–1.15 | 1 day + reconvert | exact | V2 §G3 |
| **5** | **KV-share Q-batching** — stack L19/24/29/34 Q | ×1.08 | ~40 LoC | yes | SPEED P2.2 |
| **5b** | **exp2 softmax** — ANE native exp2 instruction | 0–5% | 2 LoC + reconvert | exact | V3 §B1 |

Items 3, 4, 5, 5b can be batched in a **single reconversion pass** (PR #17 ready).

**Phase 1 stack (conservative)**: 14.5 × 1.2 × 1.10 × 1.08 × 0.65
≈ **14 tok/s** (micro-opts alone are modest without MLState/W2).
**Real path to 50+ tok/s now depends entirely on Phase 2 (speculative decoding).**

---

## Phase 2 — EAGLE-3 speculative decoding (THE critical path)

EAGLE-3 is now the **only** path to 50+ tok/s. SuffixDecoding measured
T1=18% on device — too low to be primary. Q=K verifier is built for
EAGLE-3 first; other draft sources can reuse it later.

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **6** | **EAGLE-3 retrain** (custom Gemma4Model target) | fixes Blocker 1 | Colab 2-4h | EAGLE3_STATE |
| **7** | **Q=K multi-function verifier** — prerequisite | enabling | 2 days | V2 §G1 |
| **8** | **In-model top-K** — replace argmax with topk(8) | enabling | 0.5 day + reconvert | V2 §G5 |
| **9** | **KV direct-write in commitAccepted** | fixes Blocker 2 | 1-2 days Swift | EAGLE3_STATE |
| **10** | **Sequoia** — optimal tree topology for verifier | +20–33% | 1 day offline DP | V3 §A4 |
| **11** | **Traversal Verification** — better acceptance | +10–20% | 0.5 day Swift | V3 §A5 |

**EAGLE-3 target**: acc0 ≥ 50% against custom target → Q=K verify →
KV direct-write → **40-60 tok/s @ 2K, 25-35 tok/s @ 8K**.

### Demoted to auxiliary (build after EAGLE-3 lands)

| What | Measured | Role |
|---|---|---|
| **SuffixDecoding** | T1=18%, hit=48% after 4 turns | Auxiliary: use when tree has high-confidence match, fall back to EAGLE-3 |
| **SSSD / Token Recycling** | Not measured | Same tier as SuffixDecoding |

---

## Phase 3 — Post-EAGLE-3 optimizations

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **12** | **SuffixDecoding as auxiliary** — high-confidence tree hits | +10-30% on repetitive workloads | 1 day wiring | FUND §1 |
| **13** | **Staged Speculative** — chunk1-2 as stage-1 draft | ×1.3–1.8 | 2 days | V3 §A6 |
| **14** | **Mirror SD** — NPU+GPU parallel (EAGLE-3 successor) | +30% over EAGLE-3 | 2–3 days | UNEXP §B |

SuffixDecoding reuses EAGLE-3's Q=K verifier. Use tree when count > threshold,
fall back to EAGLE-3 on miss.

---

## Phase 4 — Attention & KV optimization (long-context quality)

| Priority | What | Gain | Effort | Lossless | Source |
|---|---|---|---|---|---|
| **17** | **DuoAttention** — retrieval vs streaming heads | ×1.50 | hours offline + chunk surgery | yes | SPEED A1 |
| **18** | **SparQ Attention** — fixed top-r sparse attention | ~16× BW on full-attn | 2 days + reconvert | near (~0.1%) | V3 §D1 |
| **19** | **Cascading KV Cache** — training-free 8K quality | 8K ≈ 2K cost | 2–3 days | near | UNEXP §C |
| **20** | **TransMLA** — post-training MLA retrofit | ×10.6 @ 93% KV | 2–3 days + QLoRA | near | V3 §D3 |

---

## Phase 5 — UX & deployment

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **21** | **GPU prefill** — A19 Pro tensor cores, TTFT only | TTFT 13s → 5s | 1–2 days | UNEXP §A |
| **22** | **Prefix KV caching** — persistent on disk | TTFT 4–35× on hit | 1 day Swift | UNEXP §E |
| **23** | **Vocab pruning** — 262k → ~50k | −1.7 GB download | 1 day | UNEXP §D |
| **24** | **Mixed-bit palettization** — per-layer {1,2,4,6,8} | better q/size vs uniform | 1–2 days | V3 §C1 |

---

## Projected throughput path (updated 2026-04-13)

```
Baseline (measured)                    14.9 tok/s @ 8K

Phase 0: diagnostics (DONE)
  MLComputePlan audit                  no fallback ops found
  exp2 softmax (PR #17, unmeasured)    ×1.03  → 15.3

Phase 1: micro-opts only (W2/MLState rejected)
  + MLP tile reshape                   ×1.10  → 16.9
  + GQA broadcast + Q-batch            ×1.08  → 18.2

Phase 2: EAGLE-3 speculative (THE critical path)
  + EAGLE-3 retrain (acc0 ≥ 50%)      ×2.0   → 36.4
  + Q=K verifier + KV direct-write     ×1.3   → 47.3
  + Sequoia tree optimization          ×1.15  → 54.4

ANE overhead correction                ×0.85  → 46 tok/s
```

Conservative (EAGLE-3 acc0=40%, no Sequoia): **~30 tok/s @ 8K**.
Median (acc0=55%, Q=K, Sequoia): **~46 tok/s @ 8K**.
Upper bound (acc0=70%+, all optimizations): **~60 tok/s @ 8K**.

**SuffixDecoding (measured T1=18%) adds ~10-15% on top of EAGLE-3
for repetitive workloads only. Not in the critical path.**

---

## Rejected (confirmed dead-ends)

| What | Why | Date |
|---|---|---|
| W2A16/W3A16 post-training palettization | Complete gibberish. QAT required for sub-4-bit. | 2026-04-13 |
| MLState stateful KV | `coreml_update_state` → error -14 on ANE (Mac + iPhone). GPU-only. | 2026-04-13 |
| W8A8 (coremltools activation quant) | `ANECCompile() FAILED` on iPhone 17 Pro | 2026-04-13 |
| INT8 KV cache | ANE dequantizes to FP16 before compute. 0 wall-clock gain. | 2026-04-12 |
| Naive WFA (windowed full attention) | Quality regression past window. | 2026-04-10 |
| Medusa (untrained) | 1.3% accuracy on Gemma 4. | 2026-04-08 |
| Self-speculative (LayerSkip, no pretraining) | 0% accuracy without training. | 2026-04-08 |
| From-scratch ANE-native model | $30-50k budget. Rejected at individual scale. | 2026-04-07 |
| SDPA fusion | Incompatible with Gemma 4 QK-norm scale=1.0. | 2025 |

---

## How to read this with the other docs

- **SPEED_8K.md**: original roadmap. Items here are re-prioritized in the
  table above; some demoted (W8A8 → rejected), some kept (EAGLE-3, DuoAttention).
- **ALTERNATIVE_APPROACHES.md**: model-level alternatives (#1-#5b). Outside
  Gemma 4 scope; not in this roadmap.
- **UNEXPLORED_APPROACHES.md**: Round 1 (A-F). GPU prefill, Mirror SD,
  Cascading KV, vocab pruning, prefix KV, MIL graph optim → absorbed above.
- **UNEXPLORED_APPROACHES_V2.md**: Round 2 (G1-G5). Runtime mechanics →
  absorbed above (G1=Phase 2 item 6, G2=Phase 0a, etc.).
- **FUNDAMENTAL_UNTRIED.md**: Round 3 (§0-§4). SuffixDecoding, MLState,
  W2A16, LayerSkip → absorbed above as Phase 1-2 core items.
- **UNEXPLORED_APPROACHES_V3.md**: Round 4 (A1-D3). Speculative sweep,
  ANE micro-opts → absorbed above.

This doc supersedes individual priority sections in all of the above.
The candidate details (how they work, sources, risks) remain in their
original docs.
