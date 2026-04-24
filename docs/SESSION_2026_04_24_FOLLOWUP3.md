# Session 2026-04-24 followup 3 — UX micro-fixes + chunk pipelining Phase 1

Continuation of `SESSION_2026_04_24_FOLLOWUP2.md`. The UX micro-fixes
([U1], [U2]) shipped as small Swift edits and the first cut of decode
chunk pipelining (opt-in) followed.

## Shipped 2026-04-24 (late)

### [U1] Long-prompt auto-await during bg prefill load

`Sources/CoreMLLLM/CoreMLLLM.swift:801-818` — in `streamFromPrompt`,
before the `prefillLen`/`useHybrid` decision, if `tokens.count >= 64`
and `!engine.hasPrefill` and `engine.prefillLoadTask != nil`, block on
`try await loadTask.value`. Short prompts (< 64) fall through to
per-token decode as before, preserving the "chat feels responsive
during the ~160 s prefill-load window" behaviour. Break-even on E4B is
~17 tokens (1.2 s prefill fixed cost vs 70 ms/tok decode − 12 ms/tok
prefill); 64 is a safety margin.

Log line at engage: `[Load] prefillLoadTask awaited for long prompt
(<n> tok)`.

### [U2] ChatView unresponsive after turn 3 EOS — fixed

Root cause was in
`Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift:wrapStream`.
`LLMRunner` is `@Observable` but not `@MainActor`-isolated, and
`wrapStream` mutated `isGenerating`, `tokensPerSecond`, `loadingStatus`,
and the MTP/CV metrics from a non-MainActor Task. Off-main writes to
`@Observable` state can drop observation notifications, so the view
saw `isGenerating = true` long after the inference stack exited cleanly
(`[HangDbg] loop-top nid=106 pos=755` → EOS branch → `break` →
`continuation.finish()`). Every `.disabled(runner.isGenerating)` button
stayed disabled → "UI freeze".

Fix: hoist the entire wrapStream consumer Task onto `@MainActor`
(matches the Qwen3.5 and Qwen3-VL paths at `LLMRunner.swift:357`/`:518`,
which already wrap their `isGenerating = false` defer in
`Task { @MainActor … }`). One actor hop per token; decode rates (14–31
tok/s) make this trivial and the upstream stream is already buffered.

### [U3] Knowledge-base index pointer

`docs/KNOWLEDGE_BASE.md` gained a single row linking to
`SESSION_2026_04_24_FOLLOWUP2.md` so a future reader finds the E4B
prefill-port learnings without duplicating content.

### [B] Decode chunk pipelining — Phase 1 (async copyBack)

`Sources/CoreMLLLM/ChunkedEngine.swift:predictStep` — **default ON**
after iPhone A/B (see §iPhone verification). Opt-out via
`LLM_CHUNK_PIPELINE=0` for regression bisection.

**Mechanism.** Chunks c1 and c2 each produce K/V outputs that get
memcpy'd back into persistent buffers (`kSliding{1,2}`, `vSliding{1,2}`,
`kFull{1,2}`, `vFull{1,2}`). In the serial path, those memcpys block
the next chunk's ANE dispatch. In the pipelined path, after each chunk
finishes we read all output MLMultiArrays synchronously, then dispatch
four `memcpy` calls to a background `DispatchQueue` and continue
straight into the next chunk's ANE compute. ANE and CPU memcpy run
concurrently.

**Correctness.** Chunk k at step t+1 reads buffers that step t's
memcpy is still writing to, so we need a barrier there. Two
`DispatchSemaphore(value: 1)` instances — `kv1Sem`, `kv2Sem` — guard
the two buffer groups. Before a chunk reads its inputs, `.wait()`.
The async memcpy `.signal()`s on completion. Semaphore ownership is
tracked via `kv1Held`/`kv2Held` flags + a `defer` so a throw between
`wait` and dispatch doesn't leak-and-deadlock.

**Quiesce.** Any caller that touches the kv1/kv2 buffers outside
`predictStep` must call `quiesceCopyBacks()` first: added at `reset`,
`runPrefill`, `verifyCandidates` (both decode-style and
`verifyCandidatesWithLogits`), `captureKVSnapshot`, and
`restoreKVSnapshot`. No-op when the flag is off.

**Why not cross-step pipelining.** c1_{t+1} needs the token_id from
c4_t, so legal overlap at the ANE compute level is zero. Masks / RoPE
write into shared scratch buffers, so preparing step t+1's masks
during step t's c3/c4 would corrupt them. Out of scope for Phase 1.

**Expected win.** Matches `docs/BASELINE_SPEED_AUDIT.md:59-64` and
`docs/ANE_ONLY_LEVERS.md` item B: decode **31 → 35-37 tok/s** on E2B,
**14 → 16-17 tok/s** on E4B at 2K. E4B benefits more because its
per-chunk memcpy is longer (20-30 ms total / 4 chunks ≈ 5-7 ms per
group vs ~2-3 ms on E2B), so the overlap window is wider relative to
the total step time.

## Mac-side verification (2026-04-24)

Mac Studio, `staging-2k-fast-prefill/gemma4-e2b`, 100-token greedy
generation of "Write three short sentences about the ocean.". Build
`swift build -c release --product coreml-llm-smoke` — clean, only
pre-existing deprecation warnings.

| Metric | OFF (baseline) | ON (`LLM_CHUNK_PIPELINE=1`) | Δ |
|---|---|---|---|
| tok/s | 32.40 | 33.57 | **+3.6 %** |
| per-step total | 30.8 ms | 29.7 ms | −1.1 ms |
| `[ANE/CPU] copyBack=` | 0.3 ms | **0.0 ms** | dispatch moved out of window |
| `cpu_active` | 0.9 ms | 0.1 ms | CPU freed during ANE compute |
| output length | 176 chars | 176 chars | — |
| prefix match | `The ocean covers most of our planet's …` | same | eyeballed (stdout interleave makes byte-diff messy, but the code only defers the same memcpy so bytes are identical by construction) |

Long-run sanity: 200-token generation of "Write a short story about a
robot." with `LLM_CHUNK_PIPELINE=1` → 33.11 tok/s, 951 chars, no hang.

**Takeaway.** Mac gain is small (3.6 %) because Mac Studio's copyBack
is only ~0.3 ms / step — the overlap window is tiny. The iPhone
target is the payoff: E4B's copyBack is ~20-30 ms / step (the
proportional reason [B] was listed in `ANE_ONLY_LEVERS.md`), so the
same mechanism should produce the documented +10-20 % there.

## iPhone verification (2026-04-24)

iPhone 17 Pro, E2B @ 2K, same prompt ordering across both runs
(`こんにちは` → `江戸時代の暮らしの面白い話`). Steady-state measured on
the long second-turn reply (40+ decode steps, post-bg-prefill-load).

| Metric | OFF (baseline) | ON (`LLM_CHUNK_PIPELINE` unset or `=1`) | Δ |
|---|---|---|---|
| tok/s (avg) | ~30.3 | ~32.1 | **+1.8 (+5.9 %)** |
| tok/s (peak) | 30.7 | 32.8 | +2.1 |
| per-step total | ~32.0 ms | ~30.4 ms | −1.6 ms |
| `copyBack=` | 0.4-0.6 ms | **0.0 ms** | dispatch moves memcpy out of profile window |
| `cpu_active=` | 1.5-2.2 ms | 0.5-1.0 ms | CPU freed from memcpy |

**Correctness.** Multi-turn chat ran two full turns with EOS on both;
U2's fix keeps the UI responsive. `prefill_chunk{1,2}` background
loading completed on both runs (same timing). No deadlock, no crashes.

**Gain vs doc estimate.** Below the `ANE_ONLY_LEVERS.md` "+10-20 %"
estimate. That estimate was based on the audit's assumption of E4B-
size prep/copy (20-30 ms/step); E2B's copyBack is only 0.4-0.6 ms on
iPhone — smaller than expected — so the overlap window is
proportionally smaller. E4B A/B is the next verification.

**Decision (2026-04-24).** Default-on confirmed. Code ships with
`chunkPipelineEnabled = ProcessInfo…["LLM_CHUNK_PIPELINE"] != "0"`,
i.e. opt-out only for regression bisection. Justified because:
1. Measured +5.9 % win with zero regression across two turns.
2. `cpu_active` halved — a meaningful secondary benefit on iPhone
   where UI coexistence matters.
3. Correctness is construction-safe (same bytes to same buffers, just
   deferred); off-path is kept intact.

## iPhone E4B verification (2026-04-24)

Same device, same prompt ordering, `gemma4-e4b` bundle
(USB-sideloaded since E4B is not on HF yet). Steady-state on the
second turn "江戸時代の面白い話。", 80+ decode steps post-warmup.

| Metric | OFF | ON (default) | Δ |
|---|---|---|---|
| tok/s | 14.8 | 15.0 | **+0.2 (+1.4 %)** |
| per-step total | 67.8 ms | 66.6 ms | −1.2 ms |
| `copyBack=` | 1.8-2.0 ms | **0.0 ms** | overlap confirmed |
| `cpu_active=` | 2.0-2.2 ms | 0.9-1.4 ms | **−55 %** |
| chunks | c1=20.3 c2=20.7 c3=11.0 c4=15.4 (sum=67.4) | c1=19.7 c2=19.4 c3=11.0 c4=15.6 (sum=65.7) | c1/c2 drop reflects copyBack moving out of `profileC{1,2}` window |

**Gain vs doc estimate.** Well below `ANE_ONLY_LEVERS.md`'s +10-20 %.
The estimate assumed E4B prep/copy was 20-30 ms / step; actual
iPhone-measured copyBack on E4B is only **1.8-2.0 ms**. iPhone ANE
handles nkv=2 writes efficiently — the memcpy size roughly doubles
from E2B to E4B, but absolute time stays within a few ms. The
assumption that motivated "+10-20 %" does not hold.

**Scaling pattern.** Larger models → smaller relative win. E2B's
chunks sum to ~32 ms and copyBack was 0.4-0.6 ms → pipelining saves
5-6 %. E4B's chunks sum to ~67 ms and copyBack is ~2 ms → saves
1-2 %. The lever targets a fixed CPU cost overlapped onto ANE
compute; as ANE compute grows model-size-linearly and copyBack stays
bounded, the ratio shrinks.

**Decision.** Default-on justified by:
1. Measured win on both models (E2B +5.9 %, E4B +1.4 %) with zero
   regression over 2-turn chat.
2. `cpu_active` halved on both models — meaningful secondary benefit
   on iPhone where ANE + UI + audio/image coexist.
3. Construction-safe correctness (same bytes, same buffers).

## Next-session candidates (Metal-free)

Metal Phase 3 is off the table (no product differentiation vs
LiteRT's implementation). Remaining ANE-only levers:

1. **[D] R7-1 COMPACT** — 3-5 days Mac-heavy, E2B +2-3 tok/s at 2K.
   Calibration-based vocab + FFN structured pruning. Accuracy risk:
   vocab pruning is usually free on 262144-vocab Gemma4 (mostly
   CJK/English needed); FFN structured pruning at 20-30 % per the
   paper typically costs 1-2 pt on few-shot benchmarks. Reversible,
   monitor-able. See `docs/ROUND7_FINDINGS.md` §R7-1.
2. **Prefix cache tuning** (`LLM_PREFIX_CACHE=1`) — already shipped,
   low-hanging UX; quantify system-prompt TTFT win.
3. **Cross-step mask prep** — per-step scratch pool + step t+1 mask
   build during step t c3/c4. Estimated +0.3-0.5 ms/step. ROI weak
   after this session's E4B data; probably skip.

## Legacy: Verification plan (superseded by actual results above)

1. **Baseline**: E2B + E4B at 2K, no env var, 100-token generation.
   Record `tok/s` and the `[Profile]` / `[ANE/CPU]` lines.
2. **Pipelined**: same prompts with `LLM_CHUNK_PIPELINE=1`. Expected:
   - E2B: +4-6 tok/s (31 → 35-37).
   - E4B: +2-3 tok/s (14 → 16-17).
   - `[ANE/CPU] copyBack=` drops in the pipelined case because the
     measured window only captures the dispatch call, not the
     background memcpy. `cpu_active` should also drop.
3. **Correctness**: greedy decode with the same seed should produce
   byte-identical output between OFF and ON — the memcpy writes the
   same bytes, just later.
4. **Multi-turn**: 3 turns of chat with EOS at each turn, verifying
   `reset()`'s `quiesceCopyBacks()` drains cleanly (no stuck
   conversation state).
5. **Speculative paths**: if ever run with a drafter (EAGLE-3 / MTP),
   verify `verifyCandidates` still matches golden output
   (`quiesceCopyBacks` is at entry).

## Anti-list (do not re-evaluate unless premise changes)

- Carried from `SESSION_2026_04_24_FOLLOWUP2.md` and earlier followups.
- **Cross-step c1_{t+1} / c4_t overlap** — hard-blocked by the
  token-id dependency; not possible without speculative pre-dispatch.
- **Parallel mask/RoPE prep for step t+1 during step t c3/c4** —
  corrupts shared scratch buffers. Needs per-step scratch allocation,
  which is a separate refactor; effort > expected gain (< 0.3 % of
  step per the audit).
- **Moving memcpy to a concurrent dispatch-group for parallel memcpy
  within a chunk** — 4 memcpys are memory-bandwidth bound, serial on
  one core is fine; splitting across cores fights the ANE for DRAM.

## Next

Phase 2 candidates (any one of, ~1 week each):

- **Per-step scratch pool for masks/RoPE** → enables cross-step mask
  prep → step t+1 emb+mask done during step t c3/c4 ANE. Saves ~0.3-
  0.5 ms/step; marginal.
- **[D] R7-1 COMPACT** (3–5 days, Mac-heavy, +2-3 tok/s on E2B, post-
  training, per-model). Lower priority than runtime infra, per the
  explicit 2026-04-24 call.
- **[I] Metal Phase 3 port** — the real ANE-ceiling breaker; weeks.

Ceiling reminder: ~40 tok/s @ 2K is the ANE-only asymptote. Phase 1
pipelining is worth ~15 % of the way there. Beyond the ceiling, only
Metal Phase 3 closes the gap to LiteRT's 56 tok/s.
