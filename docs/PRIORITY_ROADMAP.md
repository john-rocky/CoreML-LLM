# Priority Roadmap — Unified ranking of all speed candidates

Consolidated from 11 docs, ~70 candidates. Ranked by **ROI** (expected gain
÷ effort), filtered by **lossless** and **ANE-compatible**. Rejected items
listed at bottom with reason.

**Baseline**: 14.5 tok/s @ ctx=8192, iPhone 17 Pro (measured 2026-04-13).
**Goal**: 50+ tok/s @ 8K, lossless, pure ANE decode.

**Key reframing** (from Orion arXiv 2603.06728): ANE runs at 0.07% of peak
19 TFLOPS. INT8 == FP16 throughput. matmul ~3× slower than Conv1×1.
**Bottleneck is dispatch overhead, not compute or bandwidth.**

**2026-04-14 update:** MTP Path A extraction complete (Google's 44 MB drafter,
`output/mtp_probe/section_10.tflite`). L34 parity blocker **eliminated** as false
alarm (indexing artifact; our Gemma4Model matches HF at all 35 layers, rel
diff < 1e-5). Phase 2 now has **two parallel paths**: MTP (faster-to-ship,
no retraining) and EAGLE-3 (retrain required, Blocker 1 root cause now
understood — training-corpus KV-sharing mismatch, not a forward bug).

**2026-04-14 late:** MTP Path A end-to-end integration complete and deployed
to iPhone 17 Pro. **acc0 = 0 %** across all prompts — drafts never match
target. Runtime plumbing (verify chunks, write-through KV, per-step RoPE,
bootstrap, carry state) all verified working. Root cause is the 0.82 cosine
TFLite→PyTorch parity gap — the drafter's extraction quality is
insufficient to hit target's argmax. Details: `docs/MTP_INTEGRATION_RESULTS.md`.
**Recommendation: fallback to EAGLE-3 retrain (Track B) or self-trained MTP
heads (Path C in MTP_PATH_A_FINDINGS §7).**

---

## Phase 0 — Diagnostics & zero-risk micro-opts (< 1 day each)

Run these **now**, before any speculative or conversion work. They either
(a) price the rest of the roadmap or (b) are free throughput.

| # | What | Gain | Effort | Source |
|---|---|---|---|---|
| ~~**0a**~~ | ~~**MLComputePlan audit**~~ | **DONE (2026-04-13)** — chunk1-3 = 100% ANE. chunk4 = 8 CPU ops in InModelArgmax tail (~1-3% of step). Dispatch-overhead hypothesis confirmed. | — | V2 §G2, EXPERIMENTS.md |
| **0b** | **ANE pipeline prewarming** — 4× dummy predictions at load | first-token fix | 10 LoC Swift | V3 §B4 |
| **0c** | **exp2 softmax** — replace torch.exp → torch.exp2 (ANE native) | 0–5% free | 2 LoC + reconvert | V3 §B1 |
| **0d** | **Prefill bypass** — skip chunk3+4 for N-1 prompt tokens (Q-only layers never produce KV) | TTFT −47% | 0.5 day Swift | ANE_SURVEY §1 |
| **0e** | **Output buffer pooling** — reuse MLMultiArray via NSCache | −1–2 ms/step | 0.5 day Swift | ANE_SURVEY |
| **0f** | **Ping-pong buffer audit** — 2-deep sync between consecutive chunks (ANE async safety) | correctness | 0.5 day | ANE_SURVEY (ANEMLL) |
| **0g** | **SRAM 32 MB working-set check** — tune prefillN per chunk to avoid 30% cliff | avoid 30% drop | 0.5 day analysis | SOURCES (Orion) |

0a result: no compute-op fallback to fix. Bottleneck is 4× per-step
dispatch overhead, not per-op device placement.

---

## Phase 1 — Training-free micro-opts & refinements (1–2 weeks)

Ship without draft model. Most batchable in a single reconversion pass.

| Priority | What | Gain | Effort | Lossless | Source |
|---|---|---|---|---|---|
| ~~**1**~~ | ~~**W2A16 palettization**~~ | **REJECTED** — W2/W3 post-training = gibberish. QAT required. | — | — | FUND §3 |
| ~~**2**~~ | ~~**MLState stateful KV**~~ | **REJECTED** — `coreml_update_state` error -14 on ANE (Mac+iPhone). GPU-only. | — | — | FUND §2 |
| **3** | **MLP tile reshape (B,C,8,8)** — fill ANE spatial PEs | up to ×1.5 on FFN | 1 day + reconvert | exact | V3 §B2 |
| **4** | **GQA broadcast matmul** — drop repeat_interleave | ×1.05–1.15 | 1 day + reconvert | exact | V2 §G3 |
| **5** | **KV-share Q-batching** — stack L19/24/29/34 Q | ×1.08 | ~40 LoC | yes | SPEED P2.2 |
| **5b** | **exp2 softmax** — ANE native exp2 instruction | 0–5% | 2 LoC + reconvert | exact | V3 §B1 |
| **5c** | **KV INT8 (TurboQuant)** — INT8 KV + fp16 residual window (128–256 recent) | 50% KV mem, long-ctx | 2–3 days | near (~0.1%) | SOURCES |
| **5d** | **Mixed-bit palettization (IQ style)** — embed/lm_head 6-bit, FFN 2–4-bit | 20–30% size or +quality | 1–2 days | near | SOURCES, LITERT_CONTAINER |
| **5e** | **SDPA fusion re-test** — iOS 18 native fused op may have fixed QK-norm scale=1.0 | 5–10% attn if fused | 1 day + reconvert | exact (if works) | ANE_SURVEY §4b |
| **5f** | **Draft & Verify (training-free self-spec)** — exit at L14 → LM head | ×1.5–2.0 | 2 days | yes | V5 |
| **5g** | **Kangaroo (lightweight adapter self-spec)** | ×1.68 | 2 days | yes | V5 |

Items 3, 4, 5, 5b batchable in **single reconversion pass (PR #17 ready)**.

**Phase 1 stack (conservative)**: 14.5 × 1.10 × 1.08 × 1.05 ≈ **18 tok/s** (micro-opts
alone). Real path to 50+ requires Phase 2.

---

## Phase 2 — Speculative decoding (THE critical path)

**Two parallel tracks. MTP is primary (no training, 3–4 days). EAGLE-3 is
fallback (needs Colab retrain, 3–5 days).** Q=K verifier + KV direct-write
serve both.

### Track A — MTP Path A (primary, fastest-to-ship)

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **6A** | **PyTorch reimpl of Google drafter** — extract weights from TFLite | prereq | 1–2 days | MTP_PATH_A §4.2 |
| **7A** | **CoreML conversion (INT4, ~6–10 MB)** — mirrors build_eagle3.py | prereq | 1 day | MTP_PATH_A §4.3 |
| **8A** | **Swift `MtpDraftSource` wiring** — feeds on L34 tap (already exists) | prereq | 0.5 day | MTP_PATH_A §4.4 |
| **9A** | **On-device bench** — acceptance vs baseline | validation | 0.5 day | — |

Expected: **60–80% acceptance** (Google trained against reference fwd, no
distribution mismatch). **×1.3–1.6 direct**. Drafter 44 MB vs EAGLE-3 188 MB.
Primary unknown: `activations=(1,1,3072)` = 2×hidden — concat of `[hidden, ?]`
where `?` is either `embed(next_token)` or prev `projected_activations`.
Determined at start of 6A.

### Track B — EAGLE-3 retrain (fallback)

Blocker 1 root cause reframed (2026-04-14): training collected HF hidden
states with `use_cache=False` (no KV-sharing L15+); inference has KV-sharing
active. Retrain with `use_cache=True` traces should fix acceptance.

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **6B** | **EAGLE-3 retrain** with correct forward-mode traces | fixes Blocker 1 | Colab 2–4 h | EAGLE3_STATE |
| **6C** | **DistillSpec** divergence-aware training (JSD/FKL/RKL) | +10–45% on retrained EAGLE-3 | +2–3 days GPU | SOURCES |

### Shared infrastructure (both tracks need these)

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **7** | **Q=K multi-function verifier** | enabling | 2 days | V2 §G1 |
| **8** | **In-model top-K** — argmax → topk(8) | enabling | 0.5 day + reconvert | V2 §G5 |
| **9** | **KV direct-write in commitAccepted** | fixes Blocker 2 | 1–2 days Swift | EAGLE3_STATE |
| **10** | **Sequoia (Y-tree) optimal topology** | +15–33% (e.g. 36→41.6 tok/s) | 1–2 days offline DP | V3 §A4, ANE_SURVEY |
| **11** | **Traversal Verification** | +10–20% | 0.5 day Swift | V3 §A5 |
| **11b** | **Verify chunks T=4** — Google uses T=3+1; extend our T=3 | K=4 capability | 0.5 day + reconvert | LITERT_CONTAINER |

### Track C — Zero-training auxiliaries (ship alongside A or B)

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **12** | **Prompt Lookup Decoding** — ~20 LoC n-gram retrieval | ×2.4 (summaries/QA), 0× on chat | 0.5 day | SOURCES |
| **13** | **Cross-vocabulary SD (Qwen 2.5 0.5B drafter)** — already bundled | ×1.5–2.5 | 3–4 days | V5 |
| **14** | **SuffixDecoding** (measured T1=18%, hit=48% after 4 turns) | +10–30% repetitive workloads | 1 day wiring | FUND §1 |

**EAGLE-3 training targets (if Track B chosen)**: acc0 ≥ 50% → Q=K →
KV direct-write → **40–60 tok/s @ 2K, 25–35 tok/s @ 8K**.

---

## Phase 3 — Post-speculative optimizations

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **15** | **Staged Speculative** — chunk1-2 as stage-1 draft | ×1.3–1.8 | 2 days | V3 §A6 |
| **16** | **Mirror SD** — NPU+GPU parallel | +30% over primary drafter | 2–3 days | UNEXP §B |
| **17** | **DISCO (dynamic speculation control)** — adaptive K | +10% free atop any spec | 1 day CPU-side | V5 |
| **18** | **SAM-Decoding (suffix automaton)** — O(n) space vs SuffixDecoding O(n²) | workload-dep | 2 days | V5 |
| **19** | **Lookahead Decoding (Jacobi trie)** — code-heavy workloads | ×2.66–6.26 code / ×1.5–2 chat | 3 days | V5 |
| **20** | **Speculative Streaming** — minimal-param n-gram | ×1.8–3.1 | 2–3 days | V5 |
| **21** | **ReDrafter RNN** — Apple's 1-layer drafter, 5–10 MB | ×2.3 | 3–4 days | ANE_SURVEY |

---

## Phase 4 — Attention & KV optimization (long-context quality)

| Priority | What | Gain | Effort | Lossless | Source |
|---|---|---|---|---|---|
| **22** | **DuoAttention** — retrieval vs streaming heads | ×1.50 | hours offline + chunk surgery | yes | SPEED A1 |
| **23** | **SparQ Attention** — fixed top-r sparse attention | ~16× BW on full-attn | 2 days + reconvert | near (~0.1%) | V3 §D1 |
| **24** | **Cascading KV Cache** — training-free 8K quality | 8K ≈ 2K cost | 2–3 days | near | UNEXP §C |
| **25** | **TransMLA** — post-training MLA retrofit | ×10.6 @ 93% KV | 2–3 days + QLoRA | near | V3 §D3 |
| **26** | **Residual/Lazy KV allocation** — PagedAttention concept adapted | memory on 8K | 1–2 days | yes | SOURCES |

---

## Phase 5 — UX & deployment

| Priority | What | Gain | Effort | Source |
|---|---|---|---|---|
| **27** | **GPU prefill** — A19 Pro tensor cores, TTFT only | TTFT 13s → 5s | 1–2 days | UNEXP §A |
| **28** | **Prefix KV caching** — persistent on disk | TTFT 4–35× on hit | 1 day Swift | UNEXP §E |
| **29** | **System Prompt KV cache (Ollama pattern)** — resume turns 2+ | TTFT ×2–5 on turn 2+ | 0.5 day | SOURCES |
| **30** | **Vocab pruning** — 262k → ~50k | −1.7 GB download | 1 day | UNEXP §D |
| **31** | **Multi-function weight dedup** (MultiFunctionDescriptor) | −50% disk on multi-fn | 1 day | ANE_SURVEY |

---

## Projected throughput path (updated 2026-04-14)

```
Baseline (measured)                    14.5 tok/s @ 8K

Phase 0: diagnostics + micro-opts
  MLComputePlan audit (DONE)           no fallback ops found
  exp2 softmax (PR #17, unmeasured)    ×1.03  → 14.9
  Output pooling + prewarm             ×1.03  → 15.4
  Prefill bypass                       TTFT only, decode unchanged

Phase 1: reconversion pass (PR #17 + follow-ups)
  + MLP tile reshape                   ×1.10  → 16.9
  + GQA broadcast + Q-batch            ×1.08  → 18.3
  + KV INT8 (TurboQuant)               ×1.05  → 19.2  (mostly memory, small decode gain)

Phase 2A: MTP Path A (primary)
  + MTP drafter @ 65% acc              ×1.45  → 27.8
  + Q=K verifier + KV direct-write     ×1.30  → 36.2
  + Y-tree (Sequoia) topology          ×1.15  → 41.6

  Alternate Phase 2B: EAGLE-3 retrained
  + EAGLE-3 @ 50% acc                  ×2.0   → 36.4
  + Q=K + KV direct-write              ×1.3   → 47.3
  + Sequoia                            ×1.15  → 54.4

ANE overhead correction                ×0.85  → 35–46 tok/s
```

Conservative (MTP acc=55%, no Y-tree): **~28 tok/s @ 8K**.
Median (MTP acc=65% + Q=K + Y-tree): **~41 tok/s @ 8K**.
Upper bound (EAGLE-3 retrain + all optimizations): **~54 tok/s @ 8K**.

**Auxiliaries (Prompt Lookup, SuffixDecoding, DISCO) add 10–30% on top
for repetitive / long-prompt workloads.**

---

## Rejected (confirmed dead-ends)

| What | Why | Date |
|---|---|---|
| W2A16/W3A16 post-training palettization | Complete gibberish. QAT required for sub-4-bit. | 2026-04-13 |
| MLState stateful KV | `coreml_update_state` → error -14 on ANE (Mac + iPhone). GPU-only. | 2026-04-13 |
| W8A8 (coremltools activation quant) | `ANECCompile() FAILED` on iPhone 17 Pro | 2026-04-13 |
| INT8 KV cache (naive, no residual) | ANE dequantizes to FP16 before compute. 0 wall-clock gain. (TurboQuant w/ residual is different — kept as 5c.) | 2026-04-12 |
| Naive WFA (windowed full attention) | Quality regression past window. | 2026-04-10 |
| Medusa (untrained) | 1.3% accuracy on Gemma 4. | 2026-04-08 |
| Self-speculative (LayerSkip, no pretraining) | 0% accuracy without training. | 2026-04-08 |
| From-scratch ANE-native model | $30-50k budget. Rejected at individual scale. | 2026-04-07 |
| ~~SDPA fusion~~ | ~~Incompatible with Gemma 4 QK-norm scale=1.0.~~ **RECONSIDER** as 5e — iOS 18 fused SDPA may resolve this. | 2025 / re-open 2026-04-14 |

### Status corrections (2026-04-14)

- **EAGLE-3 Blocker 1 (L34 parity)**: was a FALSE ALARM — indexing artifact
  in comparison harness. `output_hidden_states[35]` is post-norm, not L34
  raw. Our forward matches HF at all 35 layers (rel diff < 1e-5). Root
  cause of draft/target mismatch is training-time KV-sharing corpus
  (HF `use_cache=False` ≠ inference `use_cache=True`), not a forward bug.
- **EAGLE-3 Blocker 2 (KV direct-write)** remains critical. Current commit
  path re-runs T=1 decode per accepted token. Verify chunks v2 exist but
  not deployed.
- **MTP Path A** extraction DONE. `output/mtp_probe/section_10.tflite`,
  44.3 MB. Ready for PyTorch reimplementation.

---

## How to read this with the other docs

- **SPEED_8K.md**: original roadmap. Items here are re-prioritized in the
  table above; some demoted (W8A8 → rejected), some kept (EAGLE-3, DuoAttention).
- **ALTERNATIVE_APPROACHES.md**: model-level alternatives. Outside Gemma 4 scope.
- **UNEXPLORED_APPROACHES.md** (Rounds 1): GPU prefill, Mirror SD, Cascading KV,
  vocab pruning, prefix KV, MIL graph optim → absorbed.
- **UNEXPLORED_APPROACHES_V2.md** (Round 2): Runtime mechanics (G1–G5) → absorbed.
- **UNEXPLORED_APPROACHES_V3.md** (Round 4): Speculative sweep, ANE micro-opts → absorbed.
- **UNEXPLORED_APPROACHES_V5.md** (Round 5): MTP/Draft&Verify/Kangaroo/
  Cross-vocab/DISCO/SAM/Lookahead/Streaming → absorbed (Phase 2/3).
- **UNEXPLORED_SOURCES.md**: gap analysis — KV INT8 TurboQuant, IQ mixed-bit,
  DistillSpec, Prompt Lookup, SRAM tuning, Residual alloc, system-prompt KV → absorbed.
- **FUNDAMENTAL_UNTRIED.md** (Round 3): SuffixDecoding, MLState, W2A16, LayerSkip → absorbed.
- **ANE_OPTIMIZATION_SURVEY.md**: Apple AFM prefill bypass, ANEMLL ping-pong,
  ExecuTorch output pooling, Sequoia Y-tree, ReDrafter, multi-fn weight dedup,
  iOS 18 SDPA fusion re-test → absorbed (Phase 0/1/2/3/5).
- **LITERT_RUNTIME_ANALYSIS.md** / **LITERT_CONTAINER_ANALYSIS.md**: Google's
  runtime / container format — Verify T=4, mixed INT8/INT4 per layer → absorbed.
- **MTP_PATH_A_FINDINGS.md**: Phase 2 Track A critical path. See for I/O
  contract, PyTorch reimpl plan, parity gates.
- **EAGLE3_INTEGRATION_STATE.md**: Phase 2 Track B state. Blocker 1 reframed,
  Blocker 2 remains.

This doc supersedes individual priority sections in all of the above.
Candidate details (how they work, sources, risks) remain in their origin docs.
