# Mobile 2K competitive plan — ANE-native value prop

**Status:** 2026-04-15 late (post D1b-impl retraction). Supersedes
the previous "beat LiteRT-LM at 56 tok/s" framing. See §"What
changed" for the history.

> **Retraction callout (2026-04-15 late).** An earlier revision of
> this doc claimed a **~43 tok/s projected decode ceiling** grounded
> in PR #77's compute-unit-split spike. **That projection is
> withdrawn.** PR #79 implemented the full 2-stage pipeline it
> required and measured a **24 % regression** across all 4 prompt
> categories (baseline 32.8–33.2 → pipelined 24.9–25.5 tok/s on Mac
> Studio), with a bit-exact failure on `summary` at token 50 due to
> fp16 rounding between ANE and GPU backends of chunk 3. The
> structural blocker is a strict `c3 → c4` data dependency: c4
> consumes c3's `hidden_states_out`, so the only within-step overlap
> window is a ~1 µs Swift dict-build against a ~16 ms GPU c3 — pure
> regression. The numerical claim collapses to the **measured
> 32 tok/s ANE decode ceiling**. Full analysis in PR #79 /
> `docs/PHASE_D_PIPELINING_IMPL.md` (branch `feat/chunk-pipelining-d1b`,
> HEAD 7c21c7b). Value prop below is updated accordingly.

**Goal:** ship the strongest *ANE-native* Gemma 4 E2B runtime on
iPhone at ctx=2048. The competitive axis is **not** raw decode tok/s —
LiteRT-LM wins that on Metal GPU — but the triad of **sustained
power draw, TTFT, and decode tok/s ceiling under ANE constraints.**

---

## Value proposition (one-liner)

> ANE-native LLM runtime for Apple Silicon. Sustained **~1 W** with
> **~1 s TTFT** (projected via item 27) on 512-token prompts and a
> **32 tok/s decode ceiling** (measured, current) — different
> competitive axis from LiteRT-LM's 56 tok/s at 3–5 W.

The 1 s TTFT is a **projection** conditional on item 27 shipping
(see §"Projection basis"). The 32 tok/s decode figure is the
**measured ANE ceiling on iPhone 17 Pro / Mac Studio** after PR #79
retired the 43 tok/s D1b pipelining projection.

---

## Competitive table

Gemma 4 E2B on iPhone 17 Pro, ctx=2048. Power numbers are rough order-
of-magnitude, not calibrated measurements.

| Axis | LiteRT-LM iOS (Metal GPU) | This repo (ANE, shipped today) | This repo (after item 27, projected) |
|---|---:|---:|---:|
| Decode tok/s @ 2K | **56** | **32** (measured ceiling) | **32** (no non-speculative decode lever left) |
| TTFT @ 512 prompt | ~1–2 s (Metal prefill) | ~13 s (ANE prefill) | **~1 s** (GPU prefill) |
| Sustained power draw | 3–5 W (GPU active) | **~1 W** (ANE) | ~1 W + brief GPU prefill spikes |
| Battery life @ continuous decode | baseline | **~3× baseline** | ~3× baseline |
| GPU / CPU availability for host app | low (GPU saturated) | **high** | high (GPU used only during prefill) |
| Background / always-on friendly | no (thermal + power) | **yes** | yes |
| Model-placement footprint | fp16 GPU weights | INT4 ANE weights | INT4 ANE weights |

On tok/s-for-tok/s we are **~42 % slower than LiteRT-LM** on decode
(32 vs 56). We do not claim parity, and we do not claim to close
more than a marginal fraction of that gap on the current chunk
graph — PR #79 showed the non-speculative decode-overlap lever is
structurally unavailable. We claim to win a different envelope.

The UX argument carries on that asymmetry: **32 tok/s is ~6× human
read speed (~5 tok/s)**, so decode is already faster than the user
can follow. Where LiteRT-LM wins 56-vs-32 is a throughput regime the
typical chat / assistant UX doesn't consume. The product wedge is
sustained ~1 W, ~1 s TTFT, and the GPU-free host envelope — not
decode parity.

---

## Why the ANE-native axis is a real product

Decode tok/s is not the only number a user feels. For a phone app:

- **Sustained power and thermals.** GPU-resident LLM inference at
  5 W is fine for a 10-second answer and brutal for a 10-minute chat.
  ANE at ~1 W sustains indefinitely without the phone heating up.
- **Background and multi-task behaviour.** GPU contention with game
  rendering, camera, video decode, or other ML (vision encoders,
  diffusion, Whisper) is a real deployment constraint. ANE-resident
  LLMs leave the GPU free.
- **Battery-life / energy-per-token.** At ~1 W vs ~4 W average, the
  same number of decoded tokens consumes ~25 % of the energy. For
  long-form agentic usage this dominates UX.
- **Privacy / local-first.** Shared with LiteRT-LM (both are on-device)
  but worth stating: this repo is a Swift package with no network
  dependency and a ~1 GB `phys_footprint` — shippable inside an iOS
  app without server infra.

None of these shows up in a tok/s bar chart. All of them matter for a
certain class of product (background assistant, long-running agent,
always-on notetaker, on-the-go with limited charge).

---

## Projection basis

### 32 tok/s decode ceiling (measured, current)

Mac Studio 128-token decode with drafters OFF (PR #79, 2026-04-15):
chat 32.80, code 33.24, qa 33.15, summary 33.02 tok/s. iPhone 17 Pro
measures 31.4 tok/s at 2 K under the same defaults. This is the
**current ANE decode ceiling** on the shipped chunk graph.

#### Why the earlier 43 tok/s projection was retracted

The 43 tok/s figure came from PR #77's compute-unit-split spike:
`max(c1+c2+c4_ANE, c1+c3_GPU)` ≈ 23 ms/step, grounded in a measured
0.87–0.99 kernel-overlap factor between ANE and GPU driver queues.
The projection assumed c3 and c4 could run in parallel. **PR #79
implemented the full 2-stage pipeline and refuted that assumption
empirically:**

- Every category regressed by ~24 % (see retraction callout at top).
- Root cause: c4 takes c3's `hidden_states_out` as input — a strict
  data dependency. The only overlap window between c3 and c4 within
  a step is the ~1 µs Swift dict-build, negligible against the
  ~16 ms GPU c3. The cross-step pipeline (c3 of step N+1 concurrent
  with c3 of step N) is similarly blocked because c3 of step N+1
  needs c4 of step N to emit the just-decoded token.
- PR #75 independently showed that adding ANE-only parallelism does
  not help because the ANE driver serialises all submissions.

Conclusion: on the current CoreML chunk graph there is **no
non-speculative decode overlap available**. Three future options
documented in PR #79's `docs/PHASE_D_PIPELINING_IMPL.md`
(decoupled c4, speculative h3, model re-chunking) all require
`conversion/`-side work — none are runtime-only.

### 1 s TTFT (prefill)

Comes from `UNEXPLORED_APPROACHES.md` §A (GPU prefill via MLX-Swift)
and the `PRIORITY_ROADMAP.md` Phase 5 item 27 estimate. Prefill is
compute-bound, GPU tensor cores are ~10× more efficient at that
workload than ANE is at 512-token batch prefill. ANE-only prefill
today is ~13 s on iPhone 17 Pro for 512 prompt tokens; MLX-Swift GPU
prefill at A19 Pro's tensor-core rates lands in the ~1 s region. GPU
is lit briefly during prefill and drops back to idle for decode on
ANE — this is why the sustained-power advantage is preserved even
though we use the GPU.

Item 27 has not been implemented. 1 s TTFT is a projection.

### Why speculative decoding is not in the ceiling

Multiple PRs in this session proved the CoreML/ANE speculative-
decoding path is blocked:

- **v3 (PR #65)** — ruled out decode_q1 vs verify_qK argmax drift.
- **v4 (PR #66)** — identified chain-mode gap, initially attributed
  to batched-fp16 ordering in verify_qK.
- **B.3 (PR #72)** — refuted the batched-fp16 hypothesis; K serial
  decode_q1 calls reproduced the gap. The real mechanism is
  **semantic**: verify writes drafter proposals into the KV cache
  *before* acceptance is decided, contaminating subsequent target
  argmaxes. See `docs/PHASE_C_TIGHTENING_FINDINGS.md`.
- **Track A tolerance (PR #76, 2026-04-15)** — output-space tolerance
  at the acceptance test did not recover meaningful accept rate; the
  chain contamination is load-bearing, not a numerical tightness
  issue.
- **ANE pipelining audit (PR #75)** — ANE driver serialises chunk
  submissions; Mirror-SD cost-hiding does not get the expected
  concurrency win even if the semantic bug were fixed.

The speculative path remains technically open via a multi-week
verify-protocol redesign (delayed KV write-through), but is not a
forecast input for this plan. The plan's ceiling does not require
speculation to land.

---

## What changed (why the rewrite)

The previous version of this doc set **"70–110 tok/s at 2K, i.e.
1.25–2× Google's iOS build"** as the goal. That framing is retired.

- The 56 tok/s LiteRT-LM number is on Metal GPU at 3–5 W. Matching
  it would require pivoting this repo to MLX-Swift / GPU decode,
  which is an explicitly rejected direction (the repo's reason for
  existing is ANE-native placement).
- Under ANE constraints, the measured decode ceiling is **32 tok/s**.
  An earlier revision of this doc projected ~43 tok/s via a PR #77
  compute-unit split and D1b pipelining; PR #79's full
  implementation empirically refuted that (see retraction callout
  and §"Projection basis" for the mechanism). 32 tok/s is ~58 % of
  LiteRT-LM's decode rate — a ~42 % gap, not ~20 %.
- The speculative-decoding route that was supposed to close the gap
  is blocked at the verify-chunk write-through layer (see above).
- Reframing on power + TTFT + tok/s-ceiling is **honest about what
  ANE can deliver** and **identifies where we actually win**: a
  different user segment (background-friendly, battery-sensitive,
  GPU-contended host apps). The decode-rate gap is real; the UX
  argument has to carry the pitch, not decode parity.

We do not claim parity on decode rate. We claim a better product on
a different axis triad.

---

## Execution — single tractable decode-adjacent path forward

Only one non-speculative decode-adjacent item remains after the D1b
refutation: **item 27 (GPU prefill)**, which targets the TTFT axis
rather than the decode axis. The decode axis is parked at the
measured 32 tok/s ceiling pending either a model re-conversion
(PR #79's three future options) or the multi-week verify-protocol
redesign (C0 option b).

| # | Item | Status | Axis unlocked |
|---|---|---|---|
| **A** | ~~**D1b chunk pipelining**~~ — **RETRACTED 2026-04-15** by PR #79 full-impl measurement (−24 % on all 4 categories). Structural blocker: `c3 → c4` data dependency. Three future options (decoupled c4 / speculative h3 / model re-chunking) require `conversion/` work. | refuted; plumbing kept OFF-by-default on `feat/chunk-pipelining-d1b` | — |
| **B** | **Item 27 GPU prefill via MLX-Swift** — offload the 512-token prefill batch to GPU tensor cores. | not started; elevated to **Phase C critical path** (was "stretch"). Sole remaining tractable decode-adjacent lever. | TTFT 13 s → ~1 s (projection) |

Item B is Swift-side work; no chunk reconversion required. It
preserves the ANE-resident decode story (decode stays on ANE; GPU
is used for prefill only).

### Explicitly out of scope

- Any pivot to MLX-Swift / GPU decode for steady-state inference.
  That's a different product; the user has rejected it.
- Further speculative-decoding wiring (items 14 / 14b) until / unless
  the verify-protocol redesign lands. Not on this plan's critical
  path.
- 8K context speedup — tracked separately in `docs/SPEED_8K.md`. The
  2K plan optimises the shipped config.
- Chasing LiteRT-LM on tok/s. Different envelope, different claim.

---

## Risks & honest limits

| Risk | Mitigation / note |
|---|---|
| ~~D1b pipelining hits an iOS-side dispatch quirk~~ | **N/A — D1b retracted 2026-04-15 by PR #79 full-impl measurement; structural, not an iOS quirk.** |
| GPU prefill (item 27) surfaces a Metal ↔ CoreML handoff cost that eats the TTFT win | First-pass measurement is cheap; if the handoff dominates we revise the TTFT claim. With D1b retracted, item 27 is the only remaining decode-adjacent projection in this doc. |
| Power-draw advantage is harder to measure than tok/s and easy to wave hands at | Add Instruments energy trace to the benchmark doc before leaning on it as a competitive claim. Treat the 3× battery-life number as an order-of-magnitude until measured. |
| 32 tok/s decode ceiling is a hard competitive floor — 42 % behind LiteRT-LM — and no tractable non-speculative lever remains | Pitch leans on power + TTFT + background-friendliness; decode parity is explicitly not claimed. If decode parity becomes a requirement, revisit PR #79's three future options (all need model re-conversion). |

---

## How this relates to other docs

- `docs/PRIORITY_ROADMAP.md` — the comprehensive menu; item 27 promoted
  to Phase C critical path here.
- `docs/HANDOFF.md` — next-session plan aligned with the two-item
  execution above.
- `docs/PHASE_B_DECISION.md` — why speculative decoding is no longer
  a forecast input.
- `docs/BASELINE_SPEED_AUDIT.md` — per-chunk cost numbers that inform
  the D1b ceiling estimate.
- `docs/PHASE_A5_DECISION.md` — historical; superseded (both the
  A5 union-of-drafters plan and the 56 tok/s parity framing it fed).
