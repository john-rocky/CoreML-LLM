# iPhone ANE has realLen-aware prefill compute; Mac ANE doesn't

**Date:** 2026-04-23
**Source:** Mac Studio bench + iPhone 17 Pro measurements during the
multifunction-prefill-variants attempt (reverted, see
`docs/ANE_ONLY_LEVERS.md` and commit 557e71b).

## The finding in one line

The Gemma 4 prefill chunk graph (compiled for N=512 static batch) spends
compute roughly **proportional to `real_len`** on iPhone 17 Pro ANE, and
**proportional to `N` regardless of `real_len`** on Mac Studio ANE.

## Evidence

**Mac Studio (cpuAndNeuralEngine) — prefill_chunk1 at fixed N, 20 iter mean:**

| function | N   | real_len=N (all valid) | real_len=10 (rest masked -inf) |
|---------:|----:|----------------------:|-------------------------------:|
| prefill_b64  |  64 |  4.6 ms               | 4.5 ms                          |
| prefill_b128 | 128 |  9.1 ms               | 9.3 ms                          |
| prefill_b256 | 256 | 19.8 ms               | 19.9 ms                         |
| prefill_b512 | 512 | 46.0 ms               | 45.6 ms                         |

Ratio smallest/largest = 0.10x, matches compute ratio 64/512 = 0.125.
Mask sparsity does not help on Mac ANE. **Compute scales with N.**

**iPhone 17 Pro ANE (from ChunkedEngine [Prefill] logs, default b512
graph only, no multifunction involvement):**

| real_len | total prefill | tok/s | c1    | c2    | c3    | c4    |
|---------:|--------------:|------:|------:|------:|------:|------:|
|       10 | 75.6 ms       | 132   | 15.9  |  8.1  | 19.1  | 19.7  |
|      348 | 560.1 ms      | 621   | 52.1  | 49.1  | 88.0  | 87.8  |

If iPhone ANE ran constant-N compute like Mac, the 10-token and 348-token
rows would be nearly identical. They differ by ~7×, which tracks `real_len`
closely. **iPhone ANE effectively skips the masked-out portion of the
compute.**

## Why this matters — multifunction prefill variants don't help on iPhone

The multifunction approach assumed padding-to-N was the bottleneck for
short prompts: a 10-token prompt through a b512 graph "wastes" 500 tokens
of ANE compute. On Mac that's true (46 ms for 10 tokens vs ~4 ms if we
had a b16 variant). **On iPhone the padded positions are ~free** — the b512
graph does only ~realLen worth of work.

Measured on iPhone, 10-token prefill via b64 variant was 13.5 ms (c1)
vs 15.9 ms via default b512 — a 15% delta, not the 8× the work-ratio
would predict.

Meanwhile, loading 4 variants side-by-side *regressed* the default b512
path by 2.2× on a 348-token prompt (1250 ms vs 560 ms) — likely ANE
state / working-set contention across the 4 resident MLModel instances
per chunk (4 × 4 = 16 compiled programs fighting for ANE cache).

Net: iPhone multifunction was **negative** value. Reverted.

## Consequences for optimization priorities

### Things this closes off (iPhone target)

- **Multi-length prefill variants** (LiteRT-LM S1 pattern). iPhone already
  "does" what the optimization is supposed to deliver. Don't ship.
- **Short-prompt-specific TTFT optimizations** that work by reducing
  padded compute. Pointless on iPhone ANE.
- **Dynamic reshape of prefill N per prompt**. No meaningful gain.

### Things this opens up

- **Padding is free on iPhone** — we can ship a single prefill variant
  with generous N (e.g., 1024) without paying compute cost on short
  prompts. Only the few-MB weight footprint changes. Coverage of
  medium-length prompts (256-1024 tokens) improves without runtime cost.
- **Decode-side optimization takes priority** over prefill. iPhone
  decode at 30 tok/s is the steady-state bottleneck, and prefill
  already hits 621 tok/s on long prompts.

### Things this also invalidates — PrefixCache stays opt-in

Naively PrefixCache would look better after this finding — "every cached
token is a real ANE second saved." But the cache's restore path runs
**per-token decode** (33 ms/tok on iPhone) over the delta tokens, not
batched prefill. On iPhone the marginal prefill cost is ~1.4 ms/tok, so:

- Fresh prefill:  60 + 1.43 × realLen  ms
- Cache restore:  5 + 33 × delta  ms
- Break-even:     delta < (55 + 1.43 × realLen) / 33

| realLen | max delta for cache to win |
|---|---|
|   50 |   3.8 tok |
|  100 |   6.0 tok |
|  500 |    23 tok |
| 1000 |    45 tok |
| 2000 |    88 tok |

Ratio rule of thumb: matchLen/realLen ≥ 0.95 for iPhone ANE to win.
For short conversational chat (realLen 30-100), the delta is usually
bigger than this allows → cache loses. PrefixCache was briefly flipped
to default-on (commit fe0f4a9) and then reverted (commit d471a7f) once
this math was redone. It remains a useful opt-in for workloads with
very long prompts and tiny per-turn deltas (agent loops, long system
prompts with short user content), but is not a universal default.

Long-term fix: batched-prefill the delta tokens after restore instead
of per-token decode. Would change break-even from `33 × delta` to
`1.43 × delta` — cache would win for virtually any match >= 20% of
realLen. Requires a variable-start-position prefill graph.

### Open questions (not answered here)

- Does the sparsity effect hold for chunks 2/3/4 the same way it does
  for chunk1? (The aggregate numbers say yes — chunk3/chunk4 times
  scale with real_len too.)
- Does this generalize to other ANE workloads (decode, verify), or is
  it specific to the prefill causal-mask op? Decode has real_len=1
  always so the question doesn't apply directly.
- Is this a property of iPhone 17 Pro / A19 Pro ANE specifically, or
  earlier iPhone ANE versions as well? Unknown — no older-device test.
- Does the mask pattern have to be the `-inf` causal shape we currently
  emit, or does any zero-mask work? Only tested the causal form.

## Artifacts kept as research reference

- `conversion/spikes/multifunction_prefill_spike.py` — validated dedup works
- `conversion/build_prefill_multifunction.py` — full builder, produces
  variants as multifunction mlpackage. Useful if we ever target Mac or
  a future ANE without sparsity optimization.
- `conversion/benchmark_prefill_multifunction.py` — Mac-side timing bench
  that surfaced the Mac-vs-iPhone gap
- `scripts/compile_multifunction_prefill.sh`, `scripts/push_multifunction_prefill.sh`
- 75% hash-dedup finding still valid: spike showed `save_multifunction`
  dedups identical weights perfectly, so this is a usable technique on
  platforms where variant selection actually pays off.

## The reverted runtime pieces

- ChunkedEngine.swift S1 router (PrefillSet, pickPrefillSet,
  attachPrefillVariant) — reverted in commit 557e71b.
- `LLM_PREFILL_MULTIFUNCTION` env var — removed with the revert.
- `LLM_DEFER_PREFILL` stays, still useful independent of multifunction.
