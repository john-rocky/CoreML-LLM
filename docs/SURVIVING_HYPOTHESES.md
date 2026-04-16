# Surviving hypotheses — post full-docs audit + 5-agent research sweep

**Date:** 2026-04-16
**Basis:** All 44 docs read, cross-referenced against 5 parallel research
agents covering LiteRT-LM source, llama.cpp/ANEMLL/MLX, 30+ papers
(2025-2026), Apple WWDC/coremltools, and codebase chunk architecture.

---

## The framing that changes everything

**Our base decode (31 tok/s) is FASTER than Google's base decode (~20 tok/s).**

LiteRT-LM's 56 tok/s = base ~20 tok/s × MTP speculative 2.8× boost.
Source: back-calculated from HuggingFace benchmark card; CPU-only 25 tok/s
(no MTP) is consistent. ML Drift GPU paper (CVPR 2025 Workshop) confirms
GPU decode is memory-bound and slower per-token than ANE.

The entire speed gap is speculative decoding. If we get a drafter with
60 %+ acceptance, theoretical ceiling is 31 × 2.8 = **87 tok/s**.

---

## Hypothesis S1: Dual-KV verify protocol

**Status:** NOT REFUTED by any existing doc. Genuinely new approach.

### What it is

Maintain two KV buffer sets: **clean** (committed-only) and
**speculative** (verify writes here). After acceptance, copy only
accepted tokens' KV from speculative to clean. Rejected residue stays
in speculative and is overwritten next cycle.

### Why existing docs don't cover this

All docs (PHASE_B_DECISION, PHASE_C_TIGHTENING, NEXT_SESSION_C0) frame
the fix as either:
- (a) Output-space tolerance — tested, insufficient
- (b) Delayed KV write-through — "multi-week redesign" of verify chunks

Neither considers a **dual-buffer approach**. The dual-KV protocol
doesn't delay writes — it writes freely to a SEPARATE buffer. No model
reconversion needed; it's a Swift-side buffer management change.

### Paper basis

- **QuantSpec** (ICML 2025, **Apple**): self-speculative with separate
  4-bit draft KV and full-precision target KV. Draft contamination
  cannot reach target KV by construction.
- **SpecPV** (arXiv 2512.02337): partial verification with periodic
  KV refresh to eliminate accumulated errors.

### What it fixes (and what it might not)

**Fixes:** Across-cycle contamination — rejected tokens' KV residue
from cycle N-1 polluting cycle N's position 0 verification. This is
likely the dominant factor in the chain bench regression (code: oracle
2.63 → chain 1.01).

**Uncertain:** Within-cycle contamination — position P+2 seeing
drafter proposal d1's KV at P+1 during a single verify call. However,
standard chain-acceptance handles this: if d1 is rejected at P+1,
we stop and don't look at P+2. If d1 is accepted (d1 == target),
then the KV at P+1 is correct. So within-cycle contamination only
matters for accepted positions, where it's clean by definition.

**The logic:** In the current chain bench, after each verify cycle,
rejected positions' KV stays in the persistent cache. The NEXT
cycle's verify reads this contaminated cache at position 0, causing
even the FIRST acceptance check to be affected by stale drafter
residue. Dual-KV prevents this by ensuring the persistent cache only
contains committed-token KV.

### How to test

Cheapest test: add KV snapshot/restore around `verifyCandidates` in
the chain bench. If chain-mode E[tok/burst] rises toward oracle
numbers, the hypothesis is confirmed.

```swift
// Before verify
let kvSnapshot = snapshotKVBuffers()
// Run verify (writes contaminated KV to persistent buffers)
let argmax = try engine.verifyCandidates(tokens, startPosition: pos)
// Restore clean KV
restoreKVBuffers(from: kvSnapshot)
// Write back only accepted positions
for i in 0..<acceptedCount {
    try engine.predictStep(tokenID: Int(committed[i]), position: pos + i)
}
```

If E[tok/burst] rises from 1.01 to >2.0 on code, the hypothesis
stands. If it stays at ~1.0, the contamination is within-cycle and
dual-KV doesn't help.

### Confidence: MEDIUM-HIGH

The logic is sound: across-cycle contamination is the only mechanism
that can affect position 0 of the next cycle, which is where the
chain bench shows the largest drop. The dual-KV approach is well-
grounded in QuantSpec (Apple's own paper). The test is cheap (<1 day).

### If confirmed → unblocks

ALL speculative decoding: MTP, EAGLE-3 retrain, Cross-vocab, PL,
SuffixDecoding. The drafters themselves are not broken (oracle numbers
prove the proposals are good). The verify contamination is the sole
blocker.

---

## Hypothesis S2: Chunk consolidation (4 → 3 dispatches)

**Status:** NOT REFUTED. Prototype exists but was never device-tested.

### What it is

Merge chunk3 (L15-24) + chunk4 (L25-34 + lm_head) into one chunk.
Both are KV-shared SWA layers with no full-attention complexity.
Result: 3 dispatches per token instead of 4.

### Why not yet tried

- `gemma4_swa_merged.py` merges chunk2+3 (17 layers), NOT chunk3+4
- The 15-layer ANE compiler threshold was treated as gospel
- chunk3+4 merge = 20 layers + lm_head, exceeds the threshold

### Why it might work now

- The 15-layer threshold was empirically discovered on earlier iOS
  versions. iOS 26 / coremltools 9 may have raised it.
- chunk3+4 layers are ALL KV-shared (no own K/V projections). Each
  layer is simpler than chunk1/2 layers. Layer complexity may matter
  more than count.
- ANEMLL's empirical limit is 950 MB per chunk. chunk3 (210 MB) +
  chunk4 (503 MB) = 713 MB, well under.
- Orion proved dispatch overhead is ~2.3 ms per round-trip. Removing
  one dispatch = ~13 ms saved per token.

### Expected gain

- 4 dispatches at ~13 ms = ~51 ms → 3 dispatches ≈ ~38 ms
- 31 × (51/38) ≈ **41 tok/s** at 2K (conservative)
- Composes with speculative decoding: if S1 works + S2 works →
  41 × 2.8 = **115 tok/s theoretical**

### How to test

1. Write `gemma4_swa_chunks_3way.py` merging chunk3+4
2. Convert and compile with coremltools 9 / iOS 26 target
3. If ANE compiler hangs → dead. If it compiles → bench on device.

### Confidence: MEDIUM

The compiler threshold is real but may have moved. The cost of testing
is 1-2 days conversion + bench. If it fails, we lose nothing.

---

## Hypothesis A1: GPU prefill (item 27)

**Status:** CONFIRMED viable by multiple papers. iPhone measurement pending.

PR #86 already implements `GPU_PREFILL=1`. Mac measurement showed 9.7×
slower (expected — Mac GPU lacks tensor cores), decode regression none.

Paper support: HeteroLLM (SOSP 2025), Hybe (ISCA 2025),
hybrid-ane-mlx-bench (AtomGradient 2026) all validate GPU-prefill +
NPU-decode as the optimal split for mobile.

Expected: TTFT 13 s → ~1 s on iPhone A19 Pro.

**Confidence: HIGH** — just needs iPhone measurement.

---

## Hypothesis A2: Runtime hints (reshapeFrequency + outputBackings)

**Status:** NOT REFUTED. Identified in V6-1/V6-2 but never implemented.

- `reshapeFrequency = .infrequent`: skip per-call shape-trace. ~0.5 ms/step.
- `outputBackings`: zero-copy output via pre-allocated IOSurface. ~0.5-1 ms/step.
- MLComputePlan warm-pool reuse: ~0.8 ms first-call savings.

Combined: ~1-2 tok/s improvement. Essentially free.

**Confidence: HIGH** — documented Apple APIs, zero risk.

---

## Hypothesis B1: Prefill bypass (skip chunk3+4 for N-1 prompt tokens)

**Status:** NOT REFUTED by existing measurements. PR #33's 6× regression
was a DIFFERENT change (full prefill bypass including decode-path
changes). The pure prefill-only bypass (chunk3+4 skip for prompt tokens
except the last) has never been tested.

Apple's own AFM tech report documents this exact optimization:
"because Block 2 does not produce any keys or values, the prefill
stage is able to bypass all of its computation."

Expected: TTFT -47% (chunk3+4 = 47% of prefill time). Composes with
GPU prefill for even faster TTFT.

**Confidence: HIGH** — Apple uses this in production.

---

## Hypothesis B2: SDPA fusion re-test (iOS 26 + coremltools 8.3)

**Status:** Previous test was on iOS 18 and showed incompatibility with
Gemma 4's QK-norm scale=1.0. coremltools 8.3 introduced
`scaled_dot_product_attention_sliced_q` pass with reported 34% ANE
speedup and 45% memory reduction. ExecuTorch registers `coreml::sdpa`
to preserve fusion. Worth retesting on iOS 26.

**Confidence: LOW-MEDIUM** — may or may not have fixed the scale=1.0 issue.

---

## What is NOT on this list (confirmed dead)

| Approach | Evidence |
|---|---|
| LayerSkip at L14 | 0/60 match, measured 2026-04-16 |
| MLState on ANE | error -14, confirmed still broken on iOS 26 |
| W8A8 | ANECCompile FAILED |
| W2/W3 post-training | gibberish without QAT |
| conversion-side graph opts | coremltools 9 already applies; no effect |
| D1b chunk pipelining | -24% regression, structural c3→c4 data dep |
| MTP Path A weights | trained against LiteRT's W4A8, incompatible |
| ANE parallel dispatch | driver serialises all submissions (PR #75) |
| Mirror SD | blocked by item 11c + ANE serialisation |

---

## Hypothesis S0: GPU verify + ANE decode (heterogeneous speculation)

**Status:** NOT REFUTED. Genuinely new architecture not discussed in any
existing doc. Highest theoretical ceiling of all hypotheses.

### The core insight

Every failed speculation attempt assumed verify runs on ANE. ANE's
4-chunk serial dispatch makes verify K=3 cost ~2× decode — too
expensive for the math to close. **Move verify to GPU where it's a
single monolithic dispatch.**

### Why the math works

```
ANE decode (4 chunks serial):        52 ms/token
GPU verify K=3 (monolithic, 1 dispatch): ~60 ms (estimated)

cycle = 60 ms for E[tok/burst] tokens
break-even = 60 / 52 = 1.15

oracle code PL-n3:   2.94 / 1.15 = 2.56× → 79 tok/s
oracle chat CV:      2.31 / 1.15 = 2.01× → 62 tok/s
oracle summary:      3.26 / 1.15 = 2.83× → 88 tok/s
```

Break-even is 1.15. Every drafter clears it comfortably. Compare to
ANE verify where break-even is 2.0-3.0 (most drafters fail).

### Why GPU verify doesn't have the 4-chunk overhead

GPU has no 15-layer compiler threshold. All 35 layers compile into
one `.mlpackage` with `computeUnits = .cpuAndGPU`. One dispatch
processes K=3 tokens with GPU batch parallelism. No serial chunk
round-trips.

### KV handoff: zero-copy via unified memory

Apple Silicon's unified memory means ANE and GPU share the same
address space. IOSurface-backed MLMultiArray enables:

1. ANE decode writes KV to IOSurface → GPU reads same IOSurface
2. GPU verify writes accepted KV → ANE reads same IOSurface
3. Zero memcpy, zero DMA — same physical pages.

### Within-cycle contamination

GPU verify has the same within-cycle contamination as ANE verify
(position P+2 sees d1's KV at P+1). But the chain-acceptance logic
handles this: if d1 is rejected, we stop before checking P+2.
Contamination only occurs at accepted positions where d_k == t_k,
meaning the KV is clean by definition.

The across-cycle contamination (rejected-token KV residue from
previous cycles) can be addressed by either:
- KV snapshot/restore around GPU verify (cheap at ~0.3ms for 60MB)
- OR accepting the residue and re-running one decode_q1 for the
  bonus token (52ms), still well within budget

### What's needed

| Item | Work | Days |
|---|---|---|
| Monolithic GPU verify model | Convert 35-layer Gemma 4 as single mlpackage, `.cpuAndGPU` | 2-3 |
| KV I/O contract alignment | Ensure monolithic verify's KV layout matches 4-chunk decode's layout | 1 |
| Swift orchestration | ANE decode → drafter → GPU verify → accept → KV writeback | 2-3 |
| iPhone measurement | GPU verify latency on A19 Pro tensor cores | 1 trip |

Total: ~7-10 days if the monolithic model compiles.

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Memory: 2 model sets (~2GB) on 8GB iPhone | jetsam | Measure `phys_footprint`; consider weight sharing |
| Monolithic 35-layer GPU compilation fails | blocker | Try 2-chunk GPU variant as fallback |
| iPhone A19 GPU slower than estimated | math breaks | Mac test showed GPU slow, but Mac lacks tensor cores |
| KV layout mismatch between chunked and monolithic | incorrect output | Validate with per-layer cosine similarity |

### Why no existing doc refutes this

- PR #75 "ANE serialises submissions" — irrelevant. This is ANE+GPU,
  not ANE+ANE.
- PHASE_B_DECISION "verify-chunk write-through" — assumes ANE verify.
  GPU verify is a different execution path.
- Mirror SD "blocked by 11c + ANE serialisation" — Mirror SD puts
  the DRAFTER on GPU and verify on ANE. This is the REVERSE: decode
  on ANE, verify on GPU.

### Confidence: MEDIUM

The math is compelling (break-even 1.15 vs ANE's 2.0-3.0). The risk
is implementation complexity + monolithic compilation feasibility.
Highest ceiling of any hypothesis: **62-88 tok/s** at oracle rates.

---

## Revised S1 status: Dual-KV alone is insufficient

The earlier S1 (Dual-KV) analysis was updated after deeper
investigation. The dominant contamination source is **within-cycle**
(v3 argmax → v4 chain drop), not across-cycle. Dual-KV only fixes
across-cycle. Additionally, even with oracle rates, ANE verify's
2.0× cost ratio means break-even requires E[tok/burst] > 3.0,
which most drafters don't clear.

Dual-KV remains useful as a COMPONENT of the GPU verify architecture
(addressing across-cycle contamination for the bonus token), but is
NOT a standalone fix. Confidence downgraded from MEDIUM-HIGH to LOW
as a standalone hypothesis.

---

## Recommended execution order (revised)

| # | Hypothesis | Test cost | If works | Ceiling |
|---|---|---|---|---|
| 1 | **A2 Runtime hints** | 0.5 day | Free +1-2 tok/s | 33 tok/s |
| 2 | **A1 GPU prefill** (iPhone test) | 0 (PR ready) | TTFT 13s → ~1s | — |
| 3 | **B1 Prefill bypass** | 0.5 day | TTFT -47% | — |
| 4 | **S2 Chunk 3+4 merge** | 1-2 days | ~41 tok/s | 41 tok/s |
| 5 | **S0 GPU verify** | 7-10 days | 62-88 tok/s | 88 tok/s |
| 6 | **B2 SDPA re-test** | 1 day | +5-10% | ~34 tok/s |

S0 (GPU verify) is now the highest-ceiling item. Items 1-3 are quick
wins to ship immediately. S2 is a stepping stone. S0 is the moon shot
that could decisively beat LiteRT-LM.

---

## Sources (new, not previously in docs)

- QuantSpec (ICML 2025, Apple): arXiv 2502.10424
- SpecPV: arXiv 2512.02337
- ML Drift / LiteRT-LM GPU engine: arXiv 2505.00232 (CVPR 2025 Workshop)
- Orion ANE reverse-engineering: arXiv 2603.06728
- HeteroLLM: arXiv 2501.14794 (SOSP 2025)
- Hybe GPU-NPU hybrid: ISCA 2025
- hybrid-ane-mlx-bench: github.com/AtomGradient/hybrid-ane-mlx-bench
- sd.npu context-aligned drafting: arXiv 2510.15312
- coremltools 8.3 release notes: SDPA sliced_q pass
- LiteRT-LM HuggingFace benchmark: litert-community/gemma-4-E2B-it-litert-lm
