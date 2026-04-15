# Phase D1b pipelining — implementation attempt (negative result)

Date: 2026-04-15. Branch: `feat/chunk-pipelining-d1b` (built on main @ 2851faa, post PR #74).
Follow-up to PR #77 (`spike/d1b-compute-unit-split`) which proved cross-compute-unit
kernel overlap (factor 0.87–0.99) between ANE and GPU drivers.

## TL;DR

**STOP condition hit.** The implemented pipelined decode path (chunk3 async on
.cpuAndGPU via a dedicated `DispatchQueue`, main thread awaiting before c4)
**regresses tok/s by ~24% on every category** — identical to PR #77's measured
regression — because the Gemma-4 chunk graph has **no within-step overlap
opportunity** that this pipelining pattern can exploit. Per the task's
guardrail ("If the overlap fails to produce ≥ +15% tok/s on any prompt, STOP
and report."), this PR is filed as a negative result with the plumbing wired
up so a future structural redesign can reuse it.

Default OFF on main (per merge discipline). No production caller sees any
change until they set `CHUNK_PIPELINE_ENABLED=1`.

## Measured results (Mac Studio, 128-token decode, drafters OFF)

| Category | baseline tok/s | pipeline tok/s | Δ | bit-exact |
|----------|---------------:|---------------:|---:|:---------:|
| chat     |          32.80 |          25.21 | −23 % | PASS (28 tok) |
| code     |          33.24 |          25.50 | −23 % | PASS (127 tok) |
| qa       |          33.15 |          24.86 | −25 % | PASS (7 tok)  |
| summary  |          33.02 |          25.43 | −23 % | FAIL at tok 50 |

Per-chunk timings match PR #77 exactly: c3 on .cpuAndGPU = ~16.4 ms vs ~7.5 ms
on ANE (2.2× slower); predictStep sum goes 30.0 ms → 39.2 ms. No overlap is
captured because c4 blocks on c3's `hidden_states_out`.

### Bit-exact divergence on summary (expected)

Summary diverges at token 50 (`669 15644` vs `3143 6417`) because moving c3 from
ANE to GPU changes the fp16 arithmetic path (different rounding, different fused
ops). At tie-break positions in argmax, the ordering flips. This is the same
failure mode documented in PR #74's B.3 refutation for CV verify chunks. It is
**not a sync bug** in the pipeline implementation — the three cleaner prompts
are bit-exact byte-for-byte, confirming the dispatch logic is correct.

## Root cause — why pipelining can't help

The chunked Gemma-4 decode step is a strict linear chain:

```
token@N-1 -> c1@N -> c2@N -> c3@N -> c4@N -> token@N
```

Every edge is a hard data dependency:

- c4 takes `hidden_states_out` from c3 (the only producer of h3)
- c3 takes `hidden_states_out` from c2 (and `kv13/kv14` also from c2)
- c1 takes `hiddenIn` = `embed_tokens[token@N-1]`, which comes from c4@N-1

The PR #77 projection `max(c1+c2+c4, c1+c3_GPU)` = 22.8 ms assumed c3 and c4
run in parallel. **That requires c4 NOT to depend on c3's h3, which is not
true in the current model.** No within-step parallelism is possible without a
model-topology change.

### What the 2-stage pipeline in the task description would require

- **Decoupled c4**: c4 would need to take `h2` (hidden after layer 14) and the
  layer-15+ hidden directly, re-computing layers 25-34 independent of c3's
  output. That is a conversion-side change, not a runtime change.
- **Speculative c4**: run c4 with a predicted h3; accept/reject after c3
  completes. This is speculative decoding on the hidden axis, a research
  project of its own.
- **Cross-step lookahead**: run c3@N and c1/c2@N+1 concurrently. Requires
  token@N which comes from c4@N which requires c3@N — circular.

None are low-hanging fruit; none fit the "net-added Swift ≤ 200 lines" budget.

## Design (as implemented)

Minimal pipelined variant wired as a clean opt-in:

- `CHUNK_PIPELINE_ENABLED=1` at load time → `chunk3` loads on
  `MLModelConfiguration(.cpuAndGPU)`; other chunks inherit caller's compute
  units (`.cpuAndNeuralEngine` in the smoke CLI).
- `ChunkedEngine.predictStep` takes a pipelined branch when
  `pipeliningEnabled == true`: submits c3 to a serial `DispatchQueue` (label
  `coreml-llm.chunk3.gpu`), main thread builds c4's input dict base, then
  awaits c3 via `DispatchSemaphore`, then runs c4.
- Public read-only property `CoreMLLLM.chunkPipeliningEnabled` reflects the
  loaded state (for observability in the smoke CLI and future UI toggles).
- `finishStep(out4:t1:)` helper hoists the post-c4 profile/return bookkeeping
  out of `predictStep` so both branches share one exit path.

Why the overlap window is microseconds in practice:
- c4's dict build is ~1 µs of Swift dictionary hashing + MLFeatureValue boxing.
- `kv13/kv14` references are already held before c3 launches (they come from
  c2's output provider).
- Mask / RoPE / embed for the _current_ step were produced before c1; prep
  for the _next_ step requires token@N which requires c4.

So the async dispatch overlaps ~1 µs of CPU work with ~16 ms of GPU compute,
then joins. Net: pure regression matching the c3-GPU deficit.

## Correctness verification (protocol for future attempts)

```bash
MODEL=~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b
for cat in chat code qa summary; do
  PROMPT=...   # per category
  DUMP_TOKEN_IDS=/tmp/base_${cat}.txt \
      .build/release/coreml-llm-smoke "$MODEL" "$PROMPT" 128
  CHUNK_PIPELINE_ENABLED=1 DUMP_TOKEN_IDS=/tmp/pipe_${cat}.txt \
      .build/release/coreml-llm-smoke "$MODEL" "$PROMPT" 128
  diff /tmp/base_${cat}.txt /tmp/pipe_${cat}.txt
done
```

3 of 4 prompts bit-exact; summary diverges at token 50 (fp16-rounding
sensitivity, not a sync bug — prompts that produce cleaner logits stay
identical for 127 tokens).

## Merge discipline

- `chunkPipeliningEnabled` reports the engine's loaded state but cannot toggle
  it post-load (the flag must be set via `CHUNK_PIPELINE_ENABLED` env before
  `CoreMLLLM.load` because it changes compute unit on model instantiation).
- Default OFF on main. Matches `drafterUnionEnabled` pattern — production
  callers opt in only after on-device validation.
- Do **not** merge as the production decode path. Keep as plumbing-only so a
  future structural fix (decoupled c4, speculative h3, or conversion-side
  re-chunking) can ship without reinventing the dispatch scaffolding.

## Known limitations

1. c3 on GPU is 2.2× slower in absolute terms (7.5 ms → 16.4 ms). Realising
   any gain requires overlap ≥ 0.55 to break even, ≥ 0.70 for +15%. The
   current dep graph allows ~0.00.
2. fp16 divergence between ANE and GPU produces non-bit-exact tokens on
   logit-tight prompts (1 of 4 categories tested).
3. Prefill path is untouched (`runPrefill` still uses prefill_chunkN on
   their default compute units).
4. iPhone validation out of scope for this PR.

## Files touched

- `Sources/CoreMLLLM/ChunkedEngine.swift` — env-gated `.cpuAndGPU` branch in
  `load()`, `c3Queue` / `pipeliningEnabled` in init, pipelined branch in
  `predictStep`, `finishStep(out4:t1:)` hoist.
- `Sources/CoreMLLLM/CoreMLLLM.swift` — public read-only
  `chunkPipeliningEnabled` property.
- `Sources/CoreMLLLMSmoke/main.swift` — `CHUNK_PIPELINE_ENABLED` drafter
  override + `DUMP_TOKEN_IDS` for bit-exact diff.
- `docs/PHASE_D_PIPELINING_IMPL.md` (this file).

Net diff ≈ 110 lines Swift + this doc.

## Next steps (if anyone picks this up)

Non-speculative decode gains on CoreML/ANE+GPU are bottlenecked by the
serial-chunk topology, not by dispatch or driver boundaries. The three viable
structural fixes, roughly in increasing cost:

1. **Re-chunk** so c3/c4 can compute independent sub-streams of a residual
   split (conversion-side, needs a parallel residual path in the model). ~1 week.
2. **Speculative c4 with predicted h3** (research project, ~1 month).
3. **Full MLX-Swift port** (previously rejected in
   `rejected_approaches.md`; unchanged by this result).

Alternatively, concede the ~33 tok/s ceiling and focus remaining effort on
speculative decode (MTP / Union) where the accept-rate headroom is the
governing factor, not the per-step compute budget.

## Related

- PR #77 (`spike/d1b-compute-unit-split`) — feasibility spike, overlap probe.
  Stays in main as a diagnostic via `COMPUTE_UNIT_SPLIT=1`.
- PR #75 (`spike/d1-chunk-pipelining`) — earlier pure-ANE pipelining
  spike, also negative.
- `docs/BASELINE_SPEED_AUDIT.md` — per-chunk ms breakdown motivating
  the split target.
- `docs/PHASE_D_COMPUTE_UNIT_SPLIT_SPIKE.md` (PR #77) — overlap
  methodology + the +30% projection this PR refutes empirically.
