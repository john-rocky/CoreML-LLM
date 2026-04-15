# Mobile 2K competitive plan — ANE-native value prop

**Status:** 2026-04-15. Supersedes the previous "beat LiteRT-LM at 56
tok/s" framing. See §"What changed" for the history.

**Goal:** ship the strongest *ANE-native* Gemma 4 E2B runtime on
iPhone at ctx=2048. The competitive axis is **not** raw decode tok/s —
LiteRT-LM wins that on Metal GPU — but the triad of **sustained
power draw, TTFT, and decode tok/s ceiling under ANE constraints.**

---

## Value proposition (one-liner)

> ANE-native LLM runtime for Apple Silicon. Target **~43 tok/s at
> ~1 W sustained**, with **~1 s TTFT** on 512-token prompts — different
> competitive axis from LiteRT-LM's 56 tok/s at 3–5 W.

Both the 43 tok/s figure and the 1 s TTFT are **projections, not
shipped numbers**. See §"Projection basis" for what grounds them and
what has to ship to realise each.

---

## Competitive table

Gemma 4 E2B on iPhone 17 Pro, ctx=2048. Power numbers are rough order-
of-magnitude, not calibrated measurements.

| Axis | LiteRT-LM iOS (Metal GPU) | This repo (ANE, shipped today) | This repo (after D1b + item 27, projected) |
|---|---:|---:|---:|
| Decode tok/s @ 2K | **56** | 31 | **~43** |
| TTFT @ 512 prompt | ~1–2 s (Metal prefill) | ~13 s (ANE prefill) | **~1 s** (GPU prefill) |
| Sustained power draw | 3–5 W (GPU active) | **~1 W** (ANE) | ~1 W + brief GPU prefill spikes |
| Battery life @ continuous decode | baseline | **~3× baseline** | ~3× baseline |
| GPU / CPU availability for host app | low (GPU saturated) | **high** | high (GPU used only during prefill) |
| Background / always-on friendly | no (thermal + power) | **yes** | yes |
| Model-placement footprint | fp16 GPU weights | INT4 ANE weights | INT4 ANE weights |

On tok/s-for-tok/s we are **~20 % slower than LiteRT-LM** even if
everything on the right-hand column ships. We do not claim to match
them on decode rate. We claim to win a different envelope.

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

### 43 tok/s ceiling (decode)

Comes from the Phase D compute-unit split spike (PR #77). With
chunk 3/4 placed on GPU and chunks 1/2 on ANE, the two compute units
overlap at factor 0.87–0.99 across drivers — the critical-path
estimate drops from 51.7 ms per step (fully serial, ~19 tok/s on Mac
Studio audit) to ~23 ms per step at full pipelining, which on iPhone
translates to ~43 tok/s from the measured 31 tok/s baseline. This
**requires shipping D1b chunk pipelining** (in flight on
`feat/chunk-pipelining-d1b`). Until D1b merges and measures on an
iPhone 17 Pro, 43 tok/s remains a projection.

What ANE-chunk pipelining *cannot* do: PR #75 showed the ANE driver
serialises all submissions at the driver level, so adding more ANE
parallelism on the same compute unit does not help. The win is only
available by splitting across ANE + GPU.

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
- Under ANE constraints, the decode ceiling is ~43 tok/s (PR #77
  split + D1b pipelining). That is ~77 % of LiteRT-LM's throughput.
- The speculative-decoding route that was supposed to close the gap
  is blocked at the verify-chunk write-through layer (see above).
- Reframing on power + TTFT + tok/s-ceiling is **honest about what
  ANE can deliver** and **identifies where we actually win**: a
  different user segment (background-friendly, battery-sensitive,
  GPU-contended host apps).

We do not claim parity on decode rate. We claim a better product on
a different axis triad.

---

## Execution — two tractable paths forward

Neither depends on the speculative-decoding work.

| # | Item | Status | Axis unlocked |
|---|---|---|---|
| **A** | **D1b chunk pipelining** — overlap chunk N+1 step *t* with chunk N step *t-1* across ANE+GPU split (PR #77 validated the split; D1b implements the full 4-stage pipeline). | in flight on `feat/chunk-pipelining-d1b` | decode tok/s 31 → ~43 (projection) |
| **B** | **Item 27 GPU prefill via MLX-Swift** — offload the 512-token prefill batch to GPU tensor cores. | not started; elevated to **Phase C critical path** (was "stretch"). | TTFT 13 s → ~1 s (projection) |

Both are Swift-side work; no chunk reconversion required. Both preserve
the ANE-resident decode story (decode stays on ANE; GPU is used for
prefill only in B, and only for chunks 3/4 in A).

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
| D1b pipelining hits an iOS-side dispatch quirk | Falls back to serial execution; decode stays at 31 tok/s baseline. We don't regress. |
| GPU prefill (item 27) surfaces a Metal ↔ CoreML handoff cost that eats the TTFT win | First-pass measurement is cheap; if the handoff dominates we ship only D1b and revise TTFT claim. |
| Power-draw advantage is harder to measure than tok/s and easy to wave hands at | Add Instruments energy trace to the benchmark doc before leaning on it as a competitive claim. Treat the 3× battery-life number as an order-of-magnitude until measured. |
| 43 tok/s projection is optimistic — PR #77 overlap was measured on Mac; iPhone ANE/GPU concurrency may differ | PR #77 measured 0.87–0.99 overlap across drivers; iPhone measurement is the gating exit criterion for the claim. |

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
