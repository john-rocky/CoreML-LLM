# Session 2026-04-24 followup — v1.2.0 shipped, next candidates

Continuation of `SESSION_2026_04_24.md`. Everything listed as "#0 — Ship
N=1024" in that file landed as v1.2.0.

## Shipped 2026-04-24

| Tag / commit | Change |
|---|---|
| `a878c44` | `writeSlidingFromPrefill` realLen > W fix (right-aligned SWA cache) |
| `a878c44` | `PREFILL_N = 1024` (`conversion/models/gemma4_prefill_chunks.py`) |
| `14a9965` | `LLM_DEFER_PREFILL` default-on, opt-out with `=0` |
| `b7bed2e` | `ModelDownloader.swift` → HF branch `n1024` for gemma4-e2b |
| HF `n1024` | N=1024 prefill metadata (4 chunks × 4 files) on `mlboydaisuke/gemma-4-E2B-coreml` |
| GH release | `v1.2.0 — N=1024 prefill + faster load for Gemma 4 E2B` |

Verified on iPhone 17 Pro: multi-turn chat at prefill lengths 442 / 620 /
715 tokens, all decoded through EOS. The >W cases exercise the fix path.

## Observed during test — **not shipped, still open**

### F1. UI freeze after turn 3 EOS
After the 3rd turn's decode loop breaks on `nid=106` (EOS, normal exit),
the ChatView UI stops responding. Swift inference stack completes cleanly
(last log line: `[HangDbg] loop-top tokenCount=40 nid=106 pos=755`). Not
in `CoreMLLLM` / `ChunkedEngine`. AutoLayout warnings + `Unexpected
missing conversation type, fallback to message` noise throughout session
— likely ChatView state-machine issue on multi-turn wrap-up.

Entry point: `ChatView.swift` stream termination + SwiftUI
`AsyncThrowingStream` continuation finish handling.

## Next E2B decode candidates (ranked, Metal + fine-tune + drafter all off the table)

Current floor: 31 tok/s decode @ 2K on iPhone 17 Pro.
Ceiling under current constraints: ~40 tok/s (ROUND7_FINDINGS.md:254-262).

### [A] Runtime hints — ~0.5–1 day, no risk
- **V6-1** `optimizationHints.reshapeFrequency = .infrequent` (iOS 18.2+)
  — skips per-call shape-trace on fixed-shape models. ~0.5 ms/step.
- **V6-2** `MLComputePlan` warm-pool reuse — avoid plan rebuild on first
  prediction after load. ~0.8 ms first-call.

Source: `docs/UNEXPLORED_APPROACHES_V6.md` §V6-1, §V6-2.

### [B] Decode chunk pipelining — ~1 week, **+10–20% decode**
Currently chunks 1→2→3→4 run serial with CPU prep/copy between each.
Async-dispatch chunk N+1 prep during chunk N compute so ANE stays
saturated. Expected 31 → 35–37 tok/s.

Source: `docs/BASELINE_SPEED_AUDIT.md:59-64`,
`docs/ANE_ONLY_LEVERS.md` item B.

### [C] Quantization refinement — ~3 days, Mac reconvert
- **V6-3** coremltools 8 `granularity="per_block", block_size=32`
  re-palettization. Quality ≥ current INT4 per-channel.
- **V6-6** SpinQuant / QuaRot Hadamard rotation before palettization.
  Composes with V6-3.
- **V6-10** per-head-scaled INT8 KV (custom CoreML op). Halves KV RAM,
  helps long-context decode.

Source: `docs/UNEXPLORED_APPROACHES_V6.md` §V6-3, §V6-6, §V6-10.

### [D] Architecture post-training — days–week, Mac work
- **R7-1 COMPACT** — joint vocab + FFN pruning. ★ top pick (Gemma family
  tested per paper; static-graph clean). 3–5 days. +2–4 tok/s.
- **R7-4 SCAP** — per-layer statistical sparsity audit. 2 days. +1–2 tok/s.
  Pair with R7-1.
- **R7-3 LaRoSA** — layerwise rotated top-k sparsity. Needs V6-6 rotation
  first. 4–5 days. +1–2 tok/s.
- **R7-2 R-Sparse** — rank-aware GeGLU activation sparsity. Highest infra
  risk, go last. 4–6 days.

Aggregate (R7-1 + R7-3 + R7-4 all landing): 31 → 35–37 tok/s @ 2K,
15 → 17–19 tok/s @ 8K.

Source: `docs/ROUND7_FINDINGS.md` §Proposed execution order.

### [E] TTFT / RAM (not decode speed)
- **Bundle size trim** — per-layer embeddings precision audit (current
  3.1GB vs Gallery 2.58GB). Affects load time / RAM, not TTFT or decode.
- **PrefixCache** — already ship opt-in (`LLM_PREFIX_CACHE=1`). Default-on
  reverted (d471a7f) because math favored only narrow regime. Fine for
  fixed system prompts on new chat sessions.

## Anti-list (carry forward + updates)

Dead / do not re-evaluate unless premise changes:

- **Multi-length prefill variants** — iPhone ANE realLen-aware, padding
  free (2026-04-23 kill, 557e71b revert)
- **Default-on PrefixCache** — math doesn't favor broad deployment
  (2026-04-23 d471a7f revert)
- **Merging `feat/litert-perf-adoptions` wholesale** — shelved
- **K=V alias on E2B (#7 remainder)** — `attention_k_eq_v=false` structural
  (2026-04-24)
- **Shipping PREFILL_N > W without the SWA write fix** — crashes (fixed
  in a878c44)
- **All drafter-based speculative paths on E2B** — live accept rate
  exhausted 2026-04-22: EAGLE-3 HASS, MTP, LayerSkip, GliDe, Clover-2,
  Harmony-Decoding. Path A confirmed dead after 29 rounds.
- **Chunk consolidation 4→2** — +1 tok/s only, refutes dispatch-overhead
  theory (`project_chunk_consolidation_dead.md`)
- **MLState** — prior attempt failed; current IOSurface MLMultiArray path
  is the fixed baseline

## Beyond the 40 tok/s ANE ceiling

Only route: **Metal Phase 3** port. Blueprint in
`llama.cpp/src/models/gemma4-iswa.cpp` + FlashAttention kernel
`f32_dk256_dv256` matching our head-dim. Reference doc:
`docs/METAL_PORT_REFERENCE.md`. LiteRT's 56.5 tok/s on iPhone 17 Pro is
Metal GPU + MTP (source-confirmed 2026-04-22 in
`docs/LITERT_LM_ARCH_VERIFIED.md`).

## Suggested execution order next session

Pick exactly one of:

1. **[A]** (hints only) — day-long quick win before the big moves.
2. **[B]** (chunk pipelining) — biggest single ANE-only decode win.
3. **[D] R7-1 COMPACT** — architectural, Mac-side heavy but sticks.
4. **F1** (UI freeze) — user-facing polish, orthogonal to decode speed.
5. **Gemma 4 E4B batched prefill port** — different surface; E2B
   template straight ports, 10× TTFT on E4B long prompts.
