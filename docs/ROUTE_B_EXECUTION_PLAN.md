# Route B execution plan — drafter-free path to 70+ tok/s @ 2K

> ⚠️ **Projected tok/s numbers in this doc (70–85 tok/s, per-task
> speedups) are invalidated by PR #62 (2026-04-15).** The projections
> were derived from the oracle-replay bench (`accept-rate-bench-v2.json`)
> which over-claims live accept rates by 3–9×. Union on Mac measured
> 15–21 tok/s vs baseline 32 — a regression, not a +50 %.
>
> Task scaffolding (I1/I2/I3/T2–T4) remains broadly useful, but the
> ROI ordering and tok/s targets tied to A5 numbers are wrong. Re-read
> after task #2 (target-argmax bench) produces live-equivalent data.
> See `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` for the full finding.

**Purpose:** detailed, self-contained execution plan for the
drafter-free parallel track described in
`docs/MOBILE_2K_COMPETITIVE_PLAN.md`. Any engineer (or agent) can
pick this up without prior session context and ship item-by-item.

**Current baseline:** 31.4 tok/s @ 2K on iPhone 17 Pro (A19 Pro).
**Target:** ~~70–85 tok/s~~ TBD — under reassessment after PR #62.
Previous target of 70–85 tok/s assumed A5 drafter accept rates that
did not hold in live decoding.

**Why Route B exists:** Route A (MTP / HASS / EAGLE-3 retrain)
needs A100-class GPU time and is a parallel session. Route B uses
only pre-trained drafters or prompt-based heuristics, so it can
progress independently. ~~If training stalls — still delivers
~85 tok/s on its own.~~ The "Route B alone" ceiling is being
recomputed with live-equivalent accept rates.

---

## Context the executor needs (≤ 5 minutes)

1. `Sources/CoreMLLLM/SpeculativeLoop.swift` defines a
   `SpeculativeTarget` protocol with three methods:
   `lastHiddenMulti(at:)`, `verifyCandidates(_:K:)`, `commitAccepted(_:)`.
2. `Sources/CoreMLLLM/ChunkedEngine.swift` conforms to
   `SpeculativeTarget` and already implements:
   - `verifyCandidates(tokens: [Int32], startPosition: Int) -> [Int32]`
     (the Q=K verify path over multi-function `verify_qK` chunks)
   - `predictStep(tokenID: Int, position: Int) -> Int` (single-token
     decode used between bursts)
   - `currentPosition: Int` that must be advanced after every
     commit
3. `Sources/CoreMLLLM/PromptLookupDraft.swift` (merged in PR #36)
   exposes `PromptLookupDraft.propose(history:ngramSize:maxDraftLen:)`
   as a static pure function — the algorithm is ready, only the
   decode-loop integration is missing.
4. The main generation loop is `CoreMLLLM.swift:generate`, around
   line 415 (`// Decode loop with tok/s tracking`). Each iteration
   calls `engine.predictStep(tokenID:position:)` then advances
   `engine.currentPosition`.
5. MTP wiring at `Sources/CoreMLLLM/MtpSpeculativeEngine.swift`
   already shows how to plug a drafter into the loop via
   `drawBurst(...) -> [Int32]`. Route B drafters should be inserted
   at the same seam — see the `specEngine` branch in `CoreMLLLM.swift`.

---

## Ordered task list

Each task has: **what**, **where**, **interface**, **test**, and
**done signal**. Tasks are independent unless a Depends row calls
it out.

### Task 1 — Prompt Lookup Decoding wiring
**Size:** 0.5 day. **No external deps.** **Best starting point.**

| | |
|---|---|
| What | Route the running token history through `PromptLookupDraft.propose` once per decode step, pass any proposals through `engine.verifyCandidates`, commit accepted prefix via `engine.currentPosition += accepted.count`. Skip cleanly when accept ratio < threshold. |
| Where | `Sources/CoreMLLLM/CoreMLLLM.swift` — inside the decode loop, near line 430 where `specEngine` is already branched. Introduce a `PromptLookupLoop` class (similar shape to `MtpSpeculativeEngine`) that wraps the existing algorithm in a `SpeculativeTarget`-consuming API. |
| Interface | ```swift<br>final class PromptLookupLoop {<br>    let ngramSize: Int<br>    let maxDraftLen: Int<br>    private(set) var rollingAccept: Double = 1.0<br>    func drawAndVerify(target: SpeculativeTarget, history: [Int32], at position: Int) throws -> [Int32]<br>    func reset()<br>}<br>``` |
| Test | Fixed prompt that quotes a string, e.g. `"Complete: the quick brown fox jumps over the lazy dog. Please repeat: the quick brown"`. Expect ×1.5+ decode tok/s on that prompt. On free-form chat (`"What is a transformer?"`) expect ≈ 1.0× (no regression). |
| Done signal | PR with commit titled `feat: Prompt Lookup Decoding runtime wiring (Phase 2C item 12)`. bit-exact sampling preserved (`temperature = 0`). On-device bench shows ×1.3+ on prompt-quoting prompts, no regression elsewhere. |

### Task 2 — SuffixDecoding (suffix tree over session history)
**Size:** 1 day. **Depends on Task 1** (same loop integration point; share acceptance plumbing).

| | |
|---|---|
| What | Maintain a suffix tree of every token ever emitted in the current session. Per decode step, query the tree for the longest suffix of the running tail and propose children as drafts. Same verify path as Task 1. |
| Where | New file `Sources/CoreMLLLM/SuffixDecodingDraft.swift`. Integrate in the same `drawAndVerify` style. |
| Interface | ```swift<br>final class SuffixDecodingDraft {<br>    func propose(historyTail: ArraySlice<Int32>, maxDraftLen: Int) -> [Int32]<br>    func ingest(_ tokens: [Int32])<br>    func reset()<br>}<br>``` |
| Test | Multi-turn session where the model self-quotes a previous turn. Hit rate should reach ~48% after 4 turns (per our prior measurement in `docs/EXPERIMENTS.md`). |
| Done signal | PR `feat: SuffixDecoding runtime` — branch probably `feature/suffix-decoding-impl` (see `origin/claude/suffix-decoding-impl` for a prior attempt to reference). |

### Task 3 — Cross-vocabulary SD with Qwen 2.5 0.5B
**Size:** 3–4 days. **Independent of Task 1 / 2** but shares the `drawAndVerify` seam. This is where Route B gets its main speculative drafter.

| | |
|---|---|
| What | Qwen 2.5 0.5B is already in `ModelDownloader.ModelInfo.defaults` as a text-only model. Load it as an MLModel, run ~K steps of drafting on CPU/GPU, translate each Qwen token to Gemma token via a runtime vocab-mapping table (use the shared BPE prefix where it matches; fall back to detokenize→retokenize on miss), then verify through `engine.verifyCandidates`. |
| Where | New `Sources/CoreMLLLM/CrossVocabDraft.swift`. Vocab map loaded from `hf_model/tokenizer.json` at both ends; a 50 k × 4-byte table suffices (Qwen vocab ~151 k, Gemma ~262 k). |
| Interface | ```swift<br>final class CrossVocabDraft {<br>    init(drafterURL: URL, vocabMap: VocabMap) throws<br>    func propose(tokenContext: [Int32], K: Int) throws -> [Int32]<br>}<br>``` |
| Test | Accept rate ≥ 40% on general chat. Qwen 0.5B paraphrases are close enough to Gemma for short continuations. |
| Done signal | PR with measured accept-rate histogram attached; decode tok/s ≥ ×1.5 on prompts with modest novelty. |

### Task 4 — Union-of-drafters (compose 1+2+3)
**Size:** 3 days. **Depends on Tasks 1, 2, 3.**

| | |
|---|---|
| What | Per burst, run all three drafters (Prompt Lookup, Suffix, Qwen) in parallel, keep the longest accepted prefix across all, send to the single verify pass. Cost: 1 extra verify call per miss; benefit: each source covers a different workload. |
| Where | New `Sources/CoreMLLLM/DrafterUnion.swift` that owns the three sub-drafters and gates by rolling accept ratio. |
| Interface | `drawAndVerify` on the union; sub-drafters already share the same interface shape. |
| Test | Mixed workload: half code/summarization, half free chat. Expected +15–25% over best single drafter. |
| Done signal | Bench table comparing single vs union on both workloads, attached to the PR. |

### Task 5 — Async ANE dispatch infrastructure (I1)
**Size:** 4–6 days. **Load-bearing** for Tasks 6 and 7.

| | |
|---|---|
| What | Let `ChunkedEngine` queue multiple `MLModel.prediction(from:)` calls so chunk N+1 of step S+1 can start before chunk N of step S has returned, provided the dependency graph (KV read/write) is respected. Use `DispatchQueue` + `CheckedContinuation` per chunk, or the newer `MLModel.asyncPrediction(from:)` if API matches. Add ping-pong IOSurface buffers between consecutive chunks so no single buffer is written and read simultaneously. |
| Where | `Sources/CoreMLLLM/ChunkedEngine.swift` — introduce an internal `DispatchSchedule` struct that holds in-flight tasks. `MtpSpeculativeEngine` and the new `DrafterUnion` must become aware. |
| Interface | `predictStepAsync(tokenID:position:) async throws -> Int` variant, plus a slot pool pattern for KV ping-pong. |
| Test | Stress test: 512 consecutive decode steps with random positions; verify no race, no stale KV, bit-exact output vs synchronous path. |
| Done signal | A soak test passes; profiler shows overlap between chunk dispatch calls. |

### Task 6 — Mirror SD (GPU drafter concurrent with ANE verify) (T2)
**Size:** 3–4 days. **Depends on Task 5 (I1) and Task 3 (Qwen drafter, or any drafter compiled for GPU).**

| | |
|---|---|
| What | Compile the drafter with `MLModelConfiguration.computeUnits = .cpuAndGPU` so it runs on Metal tensor cores rather than ANE. While ANE is busy on `verifyCandidates` for burst k, fire the GPU drafter for burst k+1 in the background. Use `IOSurface`-backed tensor handoff so no staging copy crosses CPU memory. |
| Where | New `Sources/CoreMLLLM/MirrorSpeculativeLoop.swift` scaffolding already exists; wire it up with the real drafter. |
| Interface | Same `drawAndVerify` API, just internally asynchronous. |
| Test | Decode tok/s vs serial drafter. Expect +30% minimum; more if verify duration ≈ drafter duration. |
| Done signal | PR with measured overlap ratio (drafter-wall-time / verify-wall-time) and resulting tok/s delta. |

### Task 7 — Staged chunk pipelining (T4)
**Size:** 5–7 days. **Depends on Task 5 (I1).**

| | |
|---|---|
| What | Step N+1's chunk1 starts as soon as step N's chunk2 returns `kv13`/`kv14` (the only cross-step dependency chunks 3/4 need is already captured once chunk2 finishes). That overlaps roughly one chunk's latency per step. |
| Where | `Sources/CoreMLLLM/ChunkedEngine.swift` — rewrite `predictStep` into a state machine that releases downstream chunks as soon as their inputs become available. |
| Interface | No public API change; `predictStep` remains single-entry and returns the same `Int` token id. |
| Test | Throughput test under sustained decode, confirm ≈ +25% tok/s; latency-per-token unchanged or slightly higher, which is acceptable. |
| Done signal | PR with before/after profile showing chunk overlap. |

### Task 8 — Runtime hints (V6-1, V6-2)
**Size:** 1.5 days combined. **Independent; safe to land first.**

| | |
|---|---|
| What | Set `MLModelConfiguration.optimizationHints.reshapeFrequency = .infrequent` (iOS 18.2+) and materialise `MLComputePlan` once per chunk at load, reusing across `prediction(from:)` calls. |
| Where | `ChunkedEngine.load` around line 86–140. |
| Test | Decode tok/s should tick up 1–2% steadily. No correctness change. |
| Done signal | PR `feat: iOS 18 runtime hints (V6-1, V6-2)`. |

### Task 9 — Blockwise-32 W4 palettization (V6-3)
**Size:** 1 day + reconvert. **Quality headroom, not speed.**

| | |
|---|---|
| What | In `conversion/build_*.py`, set `OpLinearQuantizerConfig(granularity="per_block", block_size=32)`. Reconvert all four chunks. |
| Where | `conversion/build_flash.py` and/or `conversion/build_chunks.py`. Rebuild chunks, re-push staging. |
| Test | Quality eval — PPL should be same-or-lower than current per-channel INT4. |
| Done signal | Reconverted chunks + PR with quality table. Not merge-blocker on tok/s; lands in the background. |

---

## Recommended ordering

```
(ship immediately, non-blocking)
├── Task 8 (runtime hints)      [+1–2%]
├── Task 9 (blockwise-32)       [quality headroom, may unlock W2 later]
│
(minimum viable Route B — order matters)
├── Task 1 (Prompt Lookup)      [+1.3× on code/QA]
├── Task 2 (SuffixDecoding)     [+1.1–1.3× on multi-turn]
├── Task 3 (Cross-vocab Qwen)   [+1.5–1.8× on chat]   ←── main speculative driver
├── Task 4 (Union)              [+1.15×]
│                               → ≈ 70 tok/s on mixed workload
│
(infrastructure for further compounding)
├── Task 5 (async dispatch I1)  [enabler]
├── Task 6 (Mirror SD)          [+1.3×]
├── Task 7 (staged pipelining)  [+1.25×]
│                               → ≈ 110 tok/s ceiling on code-heavy mix
```

Route A (MTP / HASS) can drop in at **any point after Task 5** as
another source feeding into `DrafterUnion`, lifting the ceiling
further without reordering the above.

---

## Interaction with Route A

Route A (MTP training, user's other session) produces a trained
drafter that also consumes `ChunkedEngine.verifyCandidates`. When
both routes land:

1. Route A's drafter is the highest-accept source (55–65%).
2. Route B's Prompt Lookup / Suffix / Qwen sources cover
   workloads where Route A's trained drafter under-performs
   (long quotes, exact self-citations).
3. The union gate (Task 4) prefers the longest accepted prefix
   across all four, so Route B is strictly additive — it never
   hurts when Route A wins a burst.

Do not block Route B progress on Route A's training.

---

## Files the executor may touch

- Safe to modify:
  - `Sources/CoreMLLLM/CoreMLLLM.swift` (generation loop)
  - `Sources/CoreMLLLM/ChunkedEngine.swift` (conforms to
    `SpeculativeTarget`, hosts verify+commit)
  - New files under `Sources/CoreMLLLM/` for each drafter
  - `Examples/CoreMLLLMChat/CoreMLLLMChat/ChatView.swift` only if a
    toggle is needed (don't bike-shed UI)

- Do not touch without coordination:
  - `Sources/CoreMLLLM/MtpSpeculativeEngine.swift`,
    `MtpDraftSource.swift` — owned by Route A session
  - `conversion/` scripts except Task 9's palettization change
  - `Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift` beyond
    minimal glue

---

## Testing / bench template

For every PR:

1. Fixed prompt set (code / chat / summary — 3 per category).
2. Record `[Profile]` lines before and after; keep 5 steady-state
   samples.
3. Verify bit-exact output on the first 16 tokens with
   `temperature = 0`. Any regression is a merge blocker regardless
   of tok/s.
4. Acceptance histogram (how many of K drafts accepted) in PR body.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Async dispatch (Task 5) exposes ANE ordering bugs | high | Extensive soak test before Tasks 6/7 land |
| Qwen drafter vocab map has edge cases (multi-byte BPE) | medium | Fallback to detokenize→retokenize on misses |
| Prompt Lookup accepts stale drafts after position overflow | low | Reset per prompt; rolling accept gate already planned |
| Staged pipelining wins less than expected (bottleneck shifts) | medium | Keep serial path as fallback switch |

---

## When Route B is "done"

- ≥ 70 tok/s measured @ 2K on iPhone 17 Pro for mixed chat
- ≥ 85 tok/s on code-heavy workload
- No bit-exact regression with `temperature = 0`
- Feature flag exposed so the old serial path is a toggle away
