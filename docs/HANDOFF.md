# Next-session handoff

**Last updated:** 2026-04-16 (full docs audit + coremltools 9 findings).

---

## 1-minute summary

CoreML-LLM ships Gemma 4 E2B at **31 tok/s decode on iPhone 17 Pro**
(ANE, ~1 W, ~1 GB). That decode rate is the **hard ANE ceiling** on
the current chunk graph — every avenue to raise it has been tried and
failed. The sole remaining high-impact lever is **TTFT** (13 s → ~1 s
via GPU prefill). That is the current focus.

---

## What was tried and why it failed

### Speculative decoding — ALL paths dead

| Drafter | Result | Root cause |
|---|---|---|
| EAGLE-3 (trained) | 0 % acc on device | Trained against HF `use_cache=False` hidden states; inference uses `use_cache=True` → distribution mismatch |
| MTP Path A (Google TFLite extraction) | 0 % acc | Drafter trained against LiteRT W4A8 target, incompatible with our HF fp target |
| MTP Path C (self-trained K=2) | 16 tok/s (baseline 31) | acc0=17 % live, below 40 % break-even |
| Cross-vocab Qwen 2.5 0.5B | 1.8 tok/s on iPhone | 10× below projection; Qwen forward too slow on ANE |
| Prompt Lookup (n=2, n=3) | Net regression 3/4 categories | Chain-mode acceptance below break-even |
| SuffixDecoding | T1=18 % | Workload-dependent; insufficient for primary drafter |
| Union of all above | 15–21 tok/s (baseline 32) | All components regress; defaults OFF on main |

**Common blocker across all drafters:** verify writes drafter proposals
into the KV cache at positions P+1..P+K-1 **before** acceptance is
decided. Subsequent target argmaxes condition on contaminated cache.
This is structural — PR #72 proved it's semantic (not fp16 numerical).
Fixing it requires either K=1 verify (zero speedup) or fundamentally
better drafters (60 %+ acc) so contamination rarely triggers. Our best
drafter hits 38 %. Google's works because their MTP is 60-80 % against
their own quantized target.

### Decode-side non-speculative — all dead

| Approach | Result | Why |
|---|---|---|
| D1b chunk pipelining (ANE+GPU split) | −24 % all categories | c3→c4 strict data dep; only ~1 µs overlap window |
| ANE parallelism (PR #75) | 0 % gain | ANE driver serialises all submissions |
| INTEGRATED_ROADMAP conversion opts (softmax swap, FusedQKV, NCHW, etc.) | No effect | coremltools 9 already applies these passes internally |
| PR #17 micro-opts (MLP tile, GQA broadcast, exp2) | 5.5× slower | Ops fell off ANE fast path |
| MLState stateful KV | error −14 | ANE compiler rejects `coreml_update_state`; GPU-only. Confirmed still broken on iOS 26. |
| W8A8 activation quant | ANECCompile() FAILED | iPhone ANE compiler rejects quantize/dequantize MIL ops |
| W2/W3 post-training palettization | Complete gibberish | Requires QAT (multi-day GPU), not post-training |
| LayerSkip at L14 | 0/60 match (0.0%) | chunk3 (L15-24) skip produces random tokens; L14 hidden is useless without refinement. Measured 2026-04-16 via `LAYERSKIP_PROBE=1`. |

### Why 32 tok/s is the hard ceiling

The bottleneck is **4 serial ANE dispatches per token** at ~13 ms each
(~51 ms/step). ANE peak utilisation is 0.07 % — compute and bandwidth
are not the constraint. Dispatch count is. The only ways past this are:

1. Fewer chunks (2-chunk variant prototyped but ANE compiler stability
   concerns; being explored in a separate session)
2. Speculative decoding (all paths blocked, see above)
3. A future Apple OS/API change

None of these are on the current critical path.

---

## What is alive

### GPU prefill via MLX-Swift (item 27) — **CURRENT FOCUS**

| Axis | Current | Projected |
|---|---|---|
| TTFT @ 512 prompt | ~13 s (ANE prefill) | **~1 s** (GPU tensor cores) |
| Decode tok/s | 31 | 31 (unchanged — decode stays on ANE) |
| Power during prefill | ~1 W | Brief GPU spike, then back to ~1 W |

Prefill is compute-bound (512 × 1536 × 1536 × 35 layers ≈ 42 GFLOPS).
A19 Pro GPU tensor cores at 50 % utilisation = 3.75 TFLOPS → ~11 ms.
ANE does ~66 ms. 6× speedup on prefill path alone.

Implementation: compile prefill chunks with `.cpuAndGPU`, decode chunks
stay on `.cpuAndNeuralEngine`. Swift fast-switches based on batch size.
Weights shared — no double download.

Effort: 7–10 days. No training, no calibration.

**Mac measurement (2026-04-16, PR #86):** GPU prefill on Mac Studio
was 9.7× slower than ANE (278 ms → 2697 ms). This is expected — Mac
GPU goes through Metal/CoreML overhead without tensor cores. The
target is iPhone A19 Pro GPU with tensor cores (3.75 TFLOPS). Decode
regression: none (33.0 → 32.9 tok/s). iPhone measurement pending.

### Video multimodal — Phase 2

PR #84 (video placeholder token fix) is OPEN. Phase 2 native encoder
conversion is documented in `docs/VIDEO_PHASE2_CONTINUATION.md` but
not started.

---

## Open PRs

| PR | Status | Action |
|---|---|---|
| #83 | OPEN | PR-1 safe baseline — **close** (coremltools 9 renders it moot) |
| #84 | OPEN | Video placeholder token fix — review and merge if correct |
| #85 | OPEN | PR-2 softmax swap — **close** (coremltools 9 renders it moot) |
| #79 | OPEN | D1b pipelining negative result — keep as documentation |
| #76 | OPEN | Track A tolerance — blocked; no longer on critical path |
| #33 | DRAFT | 0d prefill bypass — 6× regression, unresolved |

---

## Key numbers

- iPhone 17 Pro 2K baseline: **31.4 tok/s** decode, **~154 tok/s** prefill
- Per-chunk decode: c1=5.9 c2=6.8 c3=8.1 c4=10.4 ms
- ANE placement: 99.78 % (7,294/7,310 ops)
- Memory: ~1 GB `phys_footprint`
- LiteRT-LM iOS: 56 tok/s (Metal GPU, 3–5 W) — different axis, not a target

---

## Docs landscape (for future sessions)

44 docs exist. Most are **historical investigation records**, not
active plans. The authoritative docs are:

| Doc | What it is |
|---|---|
| **This file** | Start here |
| `MOBILE_2K_COMPETITIVE_PLAN.md` | Value prop (power + TTFT + 32 tok/s) |
| `BASELINE_SPEED_AUDIT.md` | Per-chunk cost breakdown |
| `SESSION_STATE.md` | PR/branch state (stale after 4/15; this file is newer) |

Everything else is historical context. Key investigation chains:

- **Spec decoding refutation:** PHASE_A5 → PHASE_B_LIVE_ACCEPT_RATE_GAP → V3_ARGMAX → V4_CHAIN → PHASE_C_TIGHTENING → PHASE_B_DECISION
- **Pipelining refutation:** BASELINE_SPEED_AUDIT → PR #79 (branch docs)
- **Conversion opts refutation:** INTEGRATED_ROADMAP (coremltools 9 already applies; no effect — verbal report 2026-04-16)
- **MTP/EAGLE refutation:** MTP_PATH_A_FINDINGS → MTP_INTEGRATION_RESULTS → MTP_PATH_C_FINDINGS → EAGLE3_INTEGRATION_STATE

### Known stale/contradictory docs

| Doc | Issue |
|---|---|
| INTEGRATED_ROADMAP.md | Claims 50-65 tok/s via conversion opts; **invalidated** (coremltools 9 already applies them) |
| ALTERNATIVE_APPROACHES.md | Uses wrong hidden_dim=2560 (actual: 1536); stale projections |
| RESEARCH.md | Stale numbers (28 tok/s, LiteRT ~30 tok/s) |
| CONVERSION.md | Claims MLState is "Shipping" — it's rejected |
| EAGLE3_DEPLOY.md | Claims 55-70 tok/s target — EAGLE-3 is dead |
| FUNDAMENTAL_UNTRIED.md | Optimistic on MLState and W2A16 — both rejected |
| ROUTE_B_EXECUTION_PLAN.md | 70-85 tok/s target — invalidated by PR #62 |
| UNEXPLORED_APPROACHES_V5.md | MTP as "S-tier" — all MTP paths failed |

These docs are kept for historical context but **do not plan work on
top of their numbers**.
