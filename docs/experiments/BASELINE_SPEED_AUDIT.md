# Baseline speed audit — non-speculative decode (Track F)

Date: 2026-04-15. Branch: `docs/baseline-speed-audit`.
Scope: where a single decode step spends time when speculative drafters
are OFF. Motivates the C0-failure pivot choice.

## TL;DR

- Per decode step (~51.7 ms avg on Mac Studio) the cost splits as
  `c4 31 % > c2 25 % > c3 23 % > c1 21 %`. Embedding and mask build are
  < 0.3 % — effectively free.
- The four chunk predictions run **serially** on the same ANE context;
  that sequential chain is the wall-clock. No single chunk is a big
  lever on its own (30 % off c4 buys only +1.9 tok/s).
- **Top 2 candidates, ranked by gain / effort**:
  1. **Staged chunk pipelining (Phase D1)** — overlap c_k decode of
     step *t* with c_{k+1} decode of step *t-1*. Upper bound ≈ total /
     max(c_i) ≈ 51.7 / 16.3 ≈ **3.2×** (aspirational). Realistic 1.5–
     2.0× because MLMultiArray handoff + ANE serialisation cut into
     overlap. Effort: multi-day Swift; high payoff.
  2. **GPU prefill via MLX-Swift (Phase 5 item 27)** — TTFT-only win.
     Does not touch the decode numbers in this audit; kept as #2 because
     the audit confirms decode has no easy single-chunk lever, so UX
     gains (TTFT) are the next-best non-speculative win.
- Recommendation if C0 fails: **prototype staged pipelining first**.
  KV direct-write stays blocked on the same verify-chunk numerics as
  C0, so it is not a C0-independent pivot.

## Methodology

- Host: Mac Studio (macOS 25.0.0). Drafters default OFF. Release build
  of `coreml-llm-smoke`, 100 max tokens per prompt.
- Model: `~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b`.
- Prompts: one from each category in `Sources/accept-rate-bench/Prompts.swift`
  — `chat-define-transformer`, `code-complete-sum`,
  `qa-where-is-swift`, `sum-para-ane`.
- Data source: existing `[Profile]` log line
  (`Sources/CoreMLLLM/ChunkedEngine.swift` L508–L522). Each field is a
  cumulative running average since step 1; the final line in each log
  is the authoritative per-run mean. No code was modified for this
  audit.
- Logs: `/tmp/baseline-audit/{chat,code,qa,summary}.log`. Aggregated
  numbers in `/tmp/baseline-audit/summary.txt`.

### Caveat on absolute tok/s

Measured tok/s averages 19.3 on Mac Studio — below the 32 tok/s iPhone
17 Pro reference in `MOBILE_2K_COMPETITIVE_PLAN.md`. The 17 Pro number
is the target-hardware baseline; Mac Studio is just a profiling rig.
The per-chunk **shape** is what this audit relies on, and that shape
is architecture-stable (same 4-chunk graph, same ANE compute plan).

## Per-chunk cost table

All times in ms per decode step, cumulative mean over the 100-token
generation. `emb` and `mask` come from the same log line.

| Category | emb | mask | c1   | c2   | c3   | c4   | sum  | predict | total | tok/s |
|----------|-----|------|------|------|------|------|------|---------|-------|-------|
| chat     | 0.1 | 0.0  | 10.1 | 13.3 | 12.0 | 16.3 | 51.7 | 51.7    | 51.9  | 19.3  |
| code     | 0.0 | 0.0  | 10.9 | 14.5 | 10.9 | 15.1 | 51.4 | 51.4    | 51.4  | 19.4  |
| qa       | 0.0 | 0.0  | 10.7 | 11.1 | 11.8 | 16.4 | 50.0 | 50.0    | 50.0  | 20.0  |
| summary  | 0.0 | 0.0  | 12.3 | 12.1 | 12.3 | 16.6 | 53.3 | 53.3    | 53.3  | 18.8  |
| **AVG**  | 0.0 | 0.0  | 11.0 | 12.8 | 11.8 | 16.1 | 51.6 | 51.6    | 51.7  | 19.4  |

Share of step cost (AVG): c1 21.3 %, c2 24.8 %, c3 22.9 %, **c4 31.2 %**.

## Per-category variation

- Spread between categories is ≤ 6 % of total step time (50.0 ms qa →
  53.3 ms summary). Within noise / thermal for 3-run samples.
- **c4 dominates in every category** (15.1–17.2 ms). It hosts the
  decode head + output projection, which is input-invariant.
- `qa` is fastest (lighter c2, 11.1 ms) but the effect is small.
- No category-specific chunk blow-up was observed; the workload is
  uniformly c4-heavy with c2/c3 close behind.

## Headroom estimate

Simple arithmetic: reduce one chunk by 30 %, hold the others constant.

| Change          | New step | New tok/s | Delta       |
|-----------------|---------:|----------:|------------:|
| c1 −30 %        |  48.4 ms |     20.7  | +1.3 (+6.7 %) |
| c2 −30 %        |  47.8 ms |     20.9  | +1.5 (+7.8 %) |
| c3 −30 %        |  48.2 ms |     20.8  | +1.4 (+7.2 %) |
| c4 −30 %        |  46.9 ms |     21.3  | +1.9 (+9.8 %) |
| all 4 −30 %     |  36.2 ms |     27.6  | +8.3 (+43 %)  |
| perfect overlap |  16.1 ms |     62    | +43 (asymptote, unachievable) |

Take-aways:
- Single-chunk micro-optimisation wins are small. A 30 % c4 speed-up
  (e.g. better output-projection layout) is worth ≈ +10 %. Not
  worthless, but not path-to-56.
- The only decode-side lever with a realistic 1.5×+ ceiling is
  **overlap** of the sequential chain — i.e. staged pipelining.

## Recommendation (C0-failure pivot)

Rank (gain × 1 / cost):

1. **Staged chunk pipelining (Phase D1)**. Keeps correctness (same
   chunk math, different dispatch order), independent of the verify-
   chunk numerics that block C0 and KV direct-write. Realistic gain
   1.5–2.0× tok/s; effort multi-day Swift + correctness harness.
2. **GPU prefill via MLX-Swift (Phase 5 item 27)**. Does not help the
   decode numbers in this audit, but delivers the second-best UX win
   (TTFT 13 s → 5 s on 8K, per `PRIORITY_ROADMAP.md`). Cheaper than
   staged pipelining (1–2 days) but tok/s-neutral on decode.
3. KV direct-write: **not** a C0-independent pivot. The verify-chunk
   parity failure in `PHASE_B_V4_CHAIN_FINDINGS.md` also gates this
   path; picking it up pre-C0 would repeat C0's investigation.
4. Per-chunk micro-opts (c4 output projection, SDPA fusion retry):
   small and reconversion-bound. Keep on deck but not a pivot.

### One concrete follow-up

Land a **pipelining feasibility spike**: in a scratch branch, rewrite
`predictStep` as a 2-stage pipeline (produce step *t*'s c1 output
while the previous step's c2..c4 finish; commit the previous step
after c4). Measure the achieved overlap in `[Profile]` (c1 shadow vs
the c2..c4 tail). Target: ≥ 30 % wall-clock reduction on the same
four prompts before investing in a full 4-stage pipeline. Deliver as
a dedicated PR; no docs reshuffle needed.

## Related

- `docs/PHASE_B_DECISION.md` §"What this means for the 56 tok/s target"
  — the fork this audit serves.
- `docs/PRIORITY_ROADMAP.md` Phase 5 item 27 (GPU prefill), Phase D1
  (staged chunk pipelining in HANDOFF).
- `Sources/CoreMLLLM/ChunkedEngine.swift` L508–L522 — `[Profile]` line
  definition.
- Raw data: `/tmp/baseline-audit/*.log`, `/tmp/baseline-audit/summary.txt`
  (not committed — regenerate with `swift build -c release --product
  coreml-llm-smoke` and the commands in §Methodology).
