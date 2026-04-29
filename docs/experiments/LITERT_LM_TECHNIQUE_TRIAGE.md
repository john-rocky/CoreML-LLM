# LiteRT-LM technique triage — what's tried, what's still alive

**Date:** 2026-04-25
**Sources:**
- Fresh source-read of `~/Downloads/LiteRT-LM` main branch (8 files, 3 sub-agent passes)
- Audit of all 75 docs in this repo (1 sub-agent pass)
- Cross-referenced against `HANDOFF.md` (2026-04-18), `SURVIVING_HYPOTHESES.md`,
  `PRIORITY_ROADMAP.md`, `CHUNK_CONSOLIDATION_BENCH.md`, `LITERT_LM_ARCH_VERIFIED.md`,
  `CPU_BOTTLENECK_INVESTIGATION.md`.

**Baseline being moved:** 31.4 tok/s @ 2K iPhone 17 Pro (drafters OFF).
14.5 tok/s @ 8K. Per `HANDOFF.md`: hard ceiling on the current chunk graph.

---

## 0. TL;DR

A fresh deep-read of LiteRT-LM produced 7 candidate techniques. Cross-checking
against this repo's existing docs:

- **5 of 7 already dead or low-ROI for our ANE path** (dual-bank KV =
  IOSurface lock; multi-prefill-sig + pending-token + nth_element +
  Apple Metal compile flags = either not the bottleneck or GPU-only).
- **1 trapped behind 11c** (MTP/EAGLE-3 — same wall every drafter has hit).
- **1 partially-explored, never benchmarked end-to-end** (sampler-driven
  pipelining — documented in `LITERT_RUNTIME_ANALYSIS.md` Tier-A A1 +
  V6-1/V6-2 hints, no shipping evidence).

**Net new actionable items: 1 small.** Plus 2 confirmations / re-pricings:

1. **Sampler-driven pipelining (V6 A1)** — overlap CPU sampling with
   the next ANE dispatch. Estimated +1–2 tok/s. ~0.5–1 day Swift.
   See §3.1.
2. **Confirm SRAM 32 MB working-set cliff is not active** —
   `PRIORITY_ROADMAP.md` item 0g listed as Phase 0 diagnostic but never
   measured on device per the audit. ~0.5 day Instruments. See §3.2.
3. **Re-price item 11c with a cheap dual-buffer-via-COPY workaround
   for verify only** (not for the steady-state KV cache). ANE may
   accept this where the IOSurface-lock path didn't. See §3.3 — this
   is *speculative*, propose for discussion only.

The dominant lever remains **item 11c → drafter retrain**. Nothing in
LiteRT-LM's source unblocks 11c. Their MTP works because their MTP is
trained against their W4A8 quantized target and their verify is a
greedy-exact-match on Metal GPU; both pre-conditions are unavailable to
us.

---

## 1. Status table — 7 LiteRT-LM techniques

| # | Technique | LiteRT-LM ref | Status here | Why |
|---|---|---|---|---|
| 1 | Multiple prefill signatures + work-group split | `litert_compiled_model_executor_utils.cc:240` `GetOptimizedPrefillWorkGroups`; `llm_executor_settings.h:133` `prefill_batch_sizes` | **NOT WORTH PURSUING** | Prefill is <0.3 % of decode-step time per `BASELINE_SPEED_AUDIT`. Ceiling gain ≈ 0.1 tok/s. The 4-chunk chunk consolidation bench (`CHUNK_CONSOLIDATION_BENCH.md`, 2026-04-17) directly measured "halving dispatches saved 0.8 ms not 4.6 ms — per-layer compute dominates, not dispatch". Prefill chunking has even lower ROI than decode chunking. |
| 2 | Pending input token (last token held back) | `llm_litert_compiled_model_executor.cc:599`, `cc:1130-1148` | **NOT WORTH PURSUING** | Architectural complexity for 1–2 % savings on a path that's already <0.3 % of the step. Would touch `ChunkedEngine`, prefill state machine, and decode entry. No measurable payoff on a decode-bound device. |
| 3 | Dual-bank KV cache (kv_cache_buffers_1_/2_ pointer swap) | `llm_litert_compiled_model_executor.h:277`, `kv_cache.h:65-76` | **TRIED — STRUCTURALLY DEAD** | Per `CPU_BOTTLENECK_INVESTIGATION.md`: `LLM_DOUBLE_BUFFER_KV` measured **16× slowdown** on iPhone, removed. iOS IOSurface-backed `MLMultiArray` locks once used as model input → cannot be reused as output backing. Apple-side limit, not fixable in our pipeline. Documented dead so the path isn't re-attempted. (LiteRT-LM avoids this because its KV lives in Metal-managed memory, not IOSurface.) |
| 4 | Sampler-handles-input pipelining | `llm_litert_compiled_model_executor.cc:1283-1337` `SetSamplerInputHandling`, `SwapSamplerInputTensors`; settings `sampler_handles_input = true` | **PARTIALLY DOCUMENTED, NOT SHIPPED** | `LITERT_RUNTIME_ANALYSIS.md` Appendix Tier-A A1 has it as "+5 % gain estimate, 0.5 day Swift". `UNEXPLORED_APPROACHES_V6.md` V6-1 / V6-2 cover the underlying iOS hints. No PR or bench evidence found. **Action item §3.1.** |
| 5 | CPU sampling micro-opts (nth_element top-K, top-P early break, top-K-only softmax) | `runtime/components/sampling_cpu_util.cc:35-250` | **NOT WORTH PURSUING** | CPU sampling is 3 % of step time per `CPU_BOTTLENECK_INVESTIGATION.md` (cpu_active=3.0 ms vs ANE_wait=68.7 ms). Halving sampling time = 1.5 % of step ≈ 0.5 tok/s ceiling. Vocab is 262 k but logits are computed on ANE (single-pass), so partial sort doesn't avoid memory reads. Re-evaluate only if a Metal port (Phase 3) ever happens. |
| 6 | MTP / Medusa speculative decoding (drafter + verify, [embedding ⊕ activation]) | `llm_litert_mtp_drafter.cc:313-490` | **TRAPPED BEHIND 11c** | Six drafter variants tried (EAGLE-3 retrain, MTP Path A, MTP Path C, Cross-vocab Qwen, Prompt Lookup, SuffixDecoding); all blocked by **item 11c verify-protocol contamination** per `HANDOFF.md`. Live acc rates 0–38 % vs 60–80 % needed. LiteRT-LM's MTP works because (a) drafter is trained against their W4A8 target (we can't replicate the recipe — `MTP_PATH_A_FINDINGS`), and (b) verify runs in one Metal command buffer (we run on chunked ANE with the contamination semantic). **Their source contains no fix that maps to our blocker.** §3.3 is a long-shot alternative framing. |
| 7 | Apple-specific GPU compile options (`SetPreferTextureWeights(false)`, `SetUseMetalArgumentBuffers(true)`, `MAP_PRIVATE`+`MADV_DONTNEED`) | `llm_executor_settings_utils.cc:88-93`; `memory_mapped_file_posix.cc:117` | **N/A FOR ANE PATH** | These are `ml_drift::metal` flags. `LITERT_LM_ARCH_VERIFIED.md` §1 confirmed: LiteRT-LM uses `Backend::GPU` exclusively on iOS. No ANE backend exists in their source. Only relevant if `METAL_PORT_REFERENCE` becomes the critical path — currently deprioritised. |

---

## 2. What this confirms about the current strategy

The fresh LiteRT-LM read gives **independent corroboration** of `HANDOFF.md`'s
"32 tok/s is the hard ceiling on the current chunk graph" framing.

LiteRT-LM's headline 56 tok/s decomposes (using their own settings flags
+ benchmark cards) as:

```
~20 tok/s         GPU base decode (Metal, ml_drift, no MTP)
× 2.8x            MTP K=3 @ ~65 % accept, single-command-buffer verify
+ ~5 % overhead   for sampler-on-GPU + KV-share + sliding-512 housekeeping
≈ 56 tok/s
```

Our path:

```
31 tok/s          ANE base decode (4 chunks, 99.78 % ANE placement)
× 1.0x            MTP/EAGLE blocked by 11c → 0–17 % live accept
≈ 31 tok/s
```

**We start with a faster base** (the 0.07 % ANE peak utilisation is enough
to beat their GPU-base on per-token compute). The full gap is in the
2.8x multiplier, and the multiplier is gated entirely on 11c. No flag,
buffer trick, or kernel option in their source moves this. The
`SURVIVING_HYPOTHESES.md` framing remains correct.

What does **not** survive the LiteRT-LM cross-check:

- Any hypothesis that "we're missing a CoreML compile flag" — we're not.
  Their compile flags are `ml_drift::metal::Environment` only, not
  CoreML, and they're orthogonal.
- Any hypothesis that "we're missing an architectural trick on the
  ANE/decode side" — we're not. Their architectural tricks (KV share,
  sliding-512, separate signatures, dual-bank, single-buffer with offset
  param) are either already implemented, structurally rejected by ANE,
  or worth <5 % each.

---

## 3. Action plan — three items, ranked

### 3.1 Sampler-driven pipelining (V6-A1) — SHIP

**Status entering action plan:** Identified in `LITERT_RUNTIME_ANALYSIS.md`
Appendix A1 + `UNEXPLORED_APPROACHES_V6.md` V6-1, V6-2; no shipping evidence
in repo or PR list.

**What it is.** During the current ANE decode dispatch, prepare the
*next* step's `position` and `mask` tensors on the CPU in parallel.
After the current dispatch returns logits, the only remaining CPU work
is argmax(token) and embedding lookup before re-dispatching.

**Current flow (3 ms CPU + 65 ms ANE serial, plus dispatch overhead):**

```
ANE chunk1..4 (~65 ms) → CPU argmax (~1 ms) → CPU prep position/mask (~2 ms)
  → CPU embedding (~0.2 ms) → next ANE dispatch
```

**Proposed flow:**

```
[ANE chunk1..4 (~65 ms)]  ‖  [CPU prep next position+mask (~2 ms, in parallel)]
  → CPU argmax (~1 ms) → CPU embedding (~0.2 ms) → next ANE dispatch
```

**Why this is the only LiteRT-LM-derived item with positive ROI.**
Position is deterministic (`current_step + 1`); attention mask is
deterministic for the next step; both can be computed during the
current dispatch. Only the embedding lookup depends on the argmax
result.

**Implementation sketch (`Sources/CoreMLLLM/ChunkedEngine.swift`):**

1. Wrap the per-step input prep in a `Task { ... }` that runs alongside
   `prediction(from:)`. Position and mask have no data dependency on
   the current step's output.
2. Add `MLModelConfiguration.optimizationHints.reshapeFrequency = .infrequent`
   per V6-1. Pure runtime hint, fixed shapes everywhere → confirmed safe.
3. Optional: pre-allocate the next-step `MLDictionaryFeatureProvider` with
   the prepared tensors so re-dispatch is `prediction(from: prepared)`
   not a hot-path build.

**Expected gain.** 1–3 ms per step, i.e. ~32 ms → ~30 ms = **~33 tok/s
@ 2K**, +1.6 tok/s. At 8K, the 2 ms saving against ~70 ms step ≈ 3 %,
i.e. 14.5 → ~14.9 tok/s.

**Effort.** 0.5–1 day Swift. No model surgery, no reconvert, no
quality risk.

**Risk.** Minimal — if the parallel prep finishes after the dispatch
returns, the worst case is "no overlap", same wall clock as today. If
the parallel prep introduces a Swift concurrency cost > 2 ms, revert.

**Acceptance / test plan.**

- iPhone 17 Pro 2K decode bench before/after; expect +1–2 tok/s.
- `COMPUTE_PLAN_AUDIT=1` to confirm ANE placement is unchanged.
- Smoke prompts from `Sources/accept-rate-bench/Prompts.swift` chat /
  code / qa / summary — output bit-exact (this is purely scheduling).

---

### 3.2 SRAM 32 MB working-set audit — CONFIRM-OR-EXIT

**Status entering action plan:** `PRIORITY_ROADMAP.md` lists this as
Phase 0 item **0g** ("SRAM 32 MB working-set check — tune prefillN per
chunk to avoid 30% cliff", source: SOURCES Orion). Audit found no
follow-through measurement on device.

**What it is.** Orion (arXiv 2603.06728) measured a ~30 % throughput
cliff when ANE per-chunk working set exceeds 32 MB. If our chunks (or
specifically chunk2 with full-attn KV at 8K) cross this, we have a
hidden 30 % multiplier.

**Why it's worth a half-day.** Per the chunk consolidation bench numbers,
chunk2 is ~20.7 ms while chunk1 is ~12.3 ms — a 70 % asymmetry. Some of
this is the full-attn cost difference, but the 32 MB check is an
independent diagnostic: if we're below the cliff, ignore; if we're
above, re-tune.

**Implementation sketch.**

1. Instruments → System Trace → Neural Engine track. Measure per-chunk
   resident set during a steady-state decode at 2K and 8K.
2. If chunk2 (or chunk3 with kv13/kv14 reads at 8K) crosses 32 MB,
   flag for follow-up.
3. If under 32 MB everywhere, close this item permanently and remove 0g
   from the roadmap.

**Expected outcome.** Most likely no signal (per-chunk weights are large
but ANE pages them; KV reads stream). But the 0.5-day cost is small
enough that we should retire the item rather than keep it open.

**Risk.** None — measurement only.

---

### 3.3 11c framing — IOSurface-bypass dual buffer for verify only

**Status:** *Speculative*. Propose for discussion before commit.

**Background.** Item 11c blocker is "verify writes drafter proposals
into KV at positions P+1..P+K-1 *before* acceptance is decided".
`SURVIVING_HYPOTHESES.md` S1 ("Dual-KV verify protocol") explored
this; downgraded because IOSurface lock kills the pointer-swap
implementation per `LLM_DOUBLE_BUFFER_KV`.

**The reframe.** The IOSurface lock is on **persistent** input/output
buffers. Verify runs once per burst (not per step) and produces
**transient** KV writes. There may be an implementation path where:

1. Steady-state decode KV cache stays in the current single
   IOSurface-backed `MLMultiArray` layout — **unchanged**.
2. Verify chunks output KV slices `(1, K, hidden_dim)` to a **new
   non-IOSurface MLMultiArray** (host-memory backed, not zero-copy
   but small — K=3 × 256 hidden × fp16 ≈ 1.5 KB per slice).
3. After acceptance, Swift writes only `accepted_count` of those
   slices into the persistent IOSurface KV cache via standard
   `dynamic_update_slice` semantics.

**Why this might thread the needle that `LLM_DOUBLE_BUFFER_KV` couldn't.**
The 16× slowdown was caused by allocating fresh IOSurfaces per step.
Here, the per-step path stays unchanged; only the verify path (already
slower than decode by design) gains a small host-memory copy.

**Why it might still fail.** CoreML's `dynamic_update_slice` on an
IOSurface-backed output may itself trigger the lock. Verify-chunk
re-export with KV-as-output rather than KV-as-mutated is the actual
expensive part — multi-week per the existing `KICKOFF_11C` plan.

**Decision.** Don't commit time on this until the existing `KICKOFF_11C`
Phase 0 audit lands — its findings will tell us whether the per-K-position
KV-output redesign is feasible at all. If it is, this dual-buffer-via-copy
variant becomes a candidate **implementation choice** within the existing
11c plan, not an alternative to 11c.

**Expected gain (if 11c lands AND drafter retrain hits 50 %+ live).**
Per `PRIORITY_ROADMAP.md` Phase 2A projection, ~36 tok/s @ 8K with MTP
K=3 + verify chunks + Y-tree topology, scaling to ~41–47 tok/s with
EAGLE-3. At 2K the projection is correspondingly higher; conservatively
**~45 tok/s @ 2K** if 11c closes cleanly and we hit 50 % live accept.

**Effort.** 11c itself is ~3 weeks (Python verify-chunk re-export + Mac
fp32/fp16 parity + Swift rewrite + iPhone test). The dual-buffer-via-copy
implementation choice within 11c adds ~2–3 days if chosen.

---

## 4. What the LiteRT-LM read explicitly does *not* unblock

Listing these so they're not re-discovered later:

- **No CoreML compile flag we're missing.** LiteRT-LM has zero CoreML
  code paths. Their flags are `ml_drift::metal` (`SetUseMetalArgumentBuffers`,
  `SetPreferTextureWeights`, etc.) and apply only if a Metal port is
  pursued. `LITERT_LM_ARCH_VERIFIED.md` already concluded this; the
  fresh read agrees.

- **No ANE-friendly variant of MLState in their code.** Their
  "single-buffer cache + offset param tensor" pattern (their
  `gpu_optimized_single_buffer_cache_`, `runtime/.../utils.cc:311-331`)
  is functionally what MLState would give us, but the underlying
  primitive (one buffer, mutated by both kernel reads and writes) is
  what ANE rejects with error -14 per `HANDOFF.md`. No workaround in
  their source.

- **No drafter recipe.** Their drafter is in
  `output/mtp_probe/section_10.tflite` (already extracted, already
  failed at the artifact level — `MTP_INTEGRATION_RESULTS.md` §5).
  The training pipeline lives in their internal Google tooling and is
  not in `ai-edge-torch`. Confirmed by both this audit and
  `LITERT_RUNTIME_ANALYSIS.md` §C1.

- **No fused SDPA we can lift.** Their attention is **not** SDPA-fused —
  they use a transposed-V layout and manual matmul subgraphs
  (`runtime_bmm.impl_*`). `LITERT_RUNTIME_ANALYSIS.md` §A3 already
  documented this. Roadmap item 5e (iOS 18 fused SDPA re-test) is
  independent of LiteRT-LM and remains worth checking on its own.

- **No alternative to verify-protocol redesign.** Their verify on GPU
  doesn't have the contamination problem **because Metal command-buffer
  semantics are different from ANE's**, not because of any algorithm
  trick we can port. Whether on GPU or ANE, the underlying issue is:
  if the model writes K KV positions and we want to roll back the
  rejected ones, we need either (a) per-position KV outputs that we
  selectively write back, or (b) a cheap KV snapshot/restore. Both
  require model surgery. LiteRT-LM uses (a) implicitly — their
  `dynamic_update_slice` ops sit inside one Metal command buffer where
  the rejected positions just stay un-read, masked off by attention
  bounds. We need the explicit version on ANE.

---

## 5. Numbers

| Lever | Effort | Gain @ 2K | Gain @ 8K | Net change | Source |
|---|---|---|---|---|---|
| Sampler-driven pipelining (§3.1) | 0.5–1 day | +1–2 tok/s | +0.4 tok/s | 31 → ~33 / 14.5 → ~15 | LiteRT `cc:1283-1337` + V6-1/V6-2 |
| SRAM 32 MB audit (§3.2) | 0.5 day | 0–9 tok/s if cliff is active | 0–4 tok/s | conditional | Orion arXiv 2603.06728 + roadmap 0g |
| 11c + EAGLE-3 retrain (§3.3 + existing) | 3+ weeks | +14 tok/s | +13 tok/s | 31 → ~45 / 14.5 → ~28 | `KICKOFF_11C.md`, `PHASE_B_DECISION.md` |
| Chunk consolidation 4 → 2 (already benched) | 0 (done) | +0.7 tok/s | n/a | shipped | `CHUNK_CONSOLIDATION_BENCH.md` 2026-04-17 |
| Multi-prefill-sig, pending-token, dual-bank, top-K micro-opt, Apple Metal flags (LiteRT-LM 1, 2, 3, 5, 7) | — | — | — | rejected by audit | this doc §1 |

The honest 2-week-out projection is **~33 tok/s @ 2K** (sampler
pipelining only). The honest 6-week-out projection is **~45 tok/s @ 2K**
if 11c lands and a drafter retrain hits 50 % live accept; everything
else from the LiteRT-LM read costs more time than it returns.

---

## 6. Open questions for the next session

1. Confirm sampler-driven pipelining isn't already partially landed
   somewhere in `ChunkedEngine.swift` — the audit found no PR but the
   codebase is large. If `optimizationHints.reshapeFrequency` is
   already set, §3.1's effort drops to ~2 hours.
2. SRAM cliff measurement: who has the iPhone 17 Pro + Instruments
   environment ready for the 0.5-day audit?
3. 11c Phase 0 audit (per `KICKOFF_11C.md`) — is it scheduled? §3.3's
   dual-buffer-via-copy implementation choice should be raised at the
   Phase 0 review, not before.

---

## 7. Pointers (LiteRT-LM source)

For the curious, exact file:line refs from the deep-read:

| Concept | LiteRT-LM file | Lines |
|---|---|---|
| Multi-prefill-signature dispatch | `runtime/executor/litert_compiled_model_executor_utils.cc` | 240–270 (`GetOptimizedPrefillWorkGroups`) |
| Pending input token | `runtime/executor/llm_litert_compiled_model_executor.cc` | 599–735, 1130–1148 |
| Dual-bank KV swap | `runtime/executor/llm_litert_compiled_model_executor.h` | 277–280; `kv_cache.h:65-76` |
| Single-buffer + param-tensor offset | `runtime/executor/litert_compiled_model_executor_utils.cc` | 311–331 (`FillSingleBufferCacheParamTensor`) |
| Sampler-driven pipelining | `runtime/executor/llm_litert_compiled_model_executor.cc` | 1283–1337 (`SetSamplerInputHandling`, `SwapSamplerInputTensors`) |
| MTP draft loop | `runtime/executor/llm_litert_mtp_drafter.cc` | 313–490 |
| Apple Metal compile flags | `runtime/executor/llm_executor_settings_utils.cc` | 88–93 |
| iOS mmap policy | `runtime/util/memory_mapped_file_posix.cc` | 100–125 |
| Settings (all knobs) | `runtime/executor/llm_executor_settings.h` | 38–292 |
| All Apple-relevant compile opts in one place | `runtime/executor/llm_executor_settings_utils.cc` | 88–93, 197–212, 234–246 |

The Metal sampler `top_k_metal_sampler.{h,mm}` and Swift entry
points are **on dev branches, not on main** (e.g. `origin/litert_lm_pr_886490076`,
`origin/litert_lm_pr_905266111`). Main only ships the prebuilt dylib
at `prebuilt/ios_arm64/libLiteRtTopKMetalSampler.dylib`. They're not
relevant for our ANE path but documented here for completeness.
