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
| **0a** | **MLComputePlan audit** — print per-op device for each chunk | diagnostic (0–25%) | 0.5 day Swift | V2 §G2 |
| **0b** | **ANE pipeline prewarming** — 4× dummy predictions at load | first-token fix | 10 LoC Swift | V3 §B4 |
| **0c** | **exp2 softmax** — replace torch.exp → torch.exp2 (ANE native) | 0–5% free | 2 LoC + reconvert | V3 §B1 |

After 0a: if fallback ops found, fix them before proceeding. This alone
could deliver 5–25%.

---

## Phase 1 — Training-free, no EAGLE-3 dependency (1–2 weeks)

These can ship **today**. No draft model needed.

| Priority | What | Gain (standalone) | Effort | Lossless | Source |
|---|---|---|---|---|---|
| **1** | **W2A16 palettization** (Apple's recipe, not W8A8) | ×1.4–2.0 | 2 days | near (quality gate) | FUND §3 |
| **2** | **MLState stateful KV** — re-evaluate on iOS 26 | ×1.3–2.0 or 0 | 3–4 days | yes | FUND §2 |
| **3** | **MLP tile reshape (B,C,8,8)** — vision ANE trick | up to ×1.5 on FFN | 1 day + reconvert | exact | V3 §B2 |
| **4** | **GQA broadcast matmul** — drop repeat_interleave | ×1.05–1.15 | 1 day + reconvert | exact | V2 §G3 |
| **5** | **KV-share Q-batching** — stack L19/24/29/34 Q | ×1.08 | ~40 LoC | yes | SPEED P2.2 |

Items 1, 3, 4, 5 can be batched in a **single reconversion pass**.

**Phase 1 stack (conservative)**: 14.5 × 1.4 × 1.3 × 1.2 × 1.08 × 0.65
≈ **22 tok/s**. With MLState upper bound: **35 tok/s**.

---

## Phase 2 — Training-free speculative (EAGLE-3 不要, 1–2 weeks)

All produce candidate continuations consumed by one shared **Q=K
verifier** (which Phase 2 also builds as prerequisite).

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **6** | **Q=K multi-function verifier** — prerequisite for all spec | enabling | 2 days | V2 §G1 |
| **7** | **In-model top-K** — replace argmax with topk(8) | enabling | 0.5 day + reconvert | V2 §G5 |
| **8** | **SuffixDecoding** — CPU-only draft from session history | ×1.8–3.0 (chat) | 2–3 days Swift | FUND §1 |
| **9** | **SSSD** — trie n-gram cache, lighter than SuffixDecoding | ×1.5–2.9 | 2 days Swift | V3 §A2 |
| **10** | **Token Recycling** — adjacency-matrix draft recycling | ×1.3–1.6 | 2 days Swift | V3 §A3 |
| **11** | **Sequoia** — optimal tree topology for the verifier | +20–33% on any above | 1 day offline DP | V3 §A4 |
| **12** | **Traversal Verification** — better token acceptance | +10–20% accepted | 0.5 day Swift | V3 §A5 |

Items 8-10 are **interchangeable draft sources** feeding the same verifier
(items 6-7). Union their candidates per step for maximum coverage. Items
11-12 are **pure algorithmic boosts** on top of any speculative method.

**Phase 2 stack**: Phase 1 result × SuffixDecoding 2.0× × Sequoia 1.25×
= **55–88 tok/s** (crosses the 50 tok/s goal without EAGLE-3).

---

## Phase 3 — EAGLE-3 lands (in training, adds ~2×)

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **13** | **EAGLE-3 deploy** | ×2.0 | conversion + bench | SPEED P1 |
| **14** | **Sequoia + Traversal applied to EAGLE-3 tree** | +25–40% | 1 day | V3 §A4/A5 |
| **15** | **Staged Speculative** — chunk1-2 as stage-1 draft | ×1.3–1.8 | 2 days | V3 §A6 |
| **16** | **Mirror SD** — NPU+GPU parallel (EAGLE-3 successor) | +30% over EAGLE-3 | 2–3 days | UNEXP §B |

EAGLE-3 composes with Phase 2 speculative: use SuffixDecoding when tree
has high confidence, fall back to EAGLE-3 otherwise.

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

## Projected throughput path

```
Baseline (measured)                    14.5 tok/s @ 8K

Phase 0: diagnostics + micro-opt
  + exp2 softmax                       ×1.03  → 14.9
  + MLComputePlan fixes (if any)       ×1.10  → 16.4

Phase 1: training-free, no draft
  + W2A16 palettization                ×1.5   → 24.6
  + MLState (if iOS 26 clears)         ×1.4   → 34.5
  + MLP tile reshape                   ×1.2   → 41.4
  + GQA broadcast + Q-batch            ×1.10  → 45.5

Phase 2: training-free speculative
  + SuffixDecoding + SSSD union        ×2.0   → 91.0
  + Sequoia + Traversal Verify         ×1.25  → 113.8

ANE overhead correction                ×0.65  → 74 tok/s

Phase 3: EAGLE-3 (replaces/compounds speculative)
  Likely ceiling with all stacked      → 80-120 tok/s @ 8K
```

Conservative (MLState 0×, W2 low, Suffix 1.5×): **~35 tok/s**.
Median (everything lands at midpoint): **~74 tok/s**.
Upper bound: **~120 tok/s**.

---

## Rejected (confirmed dead-ends)

| What | Why | Date |
|---|---|---|
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
