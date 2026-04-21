# D5 — Stateful KV on ANE: Re-evaluation for iOS 26 / coremltools 9.0

Date: 2026-04-15
Owner: Investigation dispatched from main project, focus on whether the
earlier MLState failure (error -14 on iOS 18 ANE) is worth a retry now that
iOS 26 and coremltools 9.0 have shipped.

---

## Executive verdict — STAY-REJECTED (with a narrow, low-cost confirmation probe)

After a second pass through Apple's 2025 documentation (coremltools 9.0
release notes, the stateful-models guide, WWDC25 Foundation Models material),
third-party implementations (ANEMLL 0.3.5, smpanaro/coreml-llm-cli,
Apple's own Llama 3.1 reference from Nov 2024), and every public
`ANEServicesError -14` / "failed to build execution plan" datapoint we
could find, there is **no positive evidence** that the `coreml_update_state`
MIL op is now schedulable on the Apple Neural Engine. Apple's 9.0 release
notes add iOS26/macOS26 deployment targets and an expanded state read/write
API, but say nothing about ANE support for stateful models. Apple's own
flagship on-device LLM reference (Llama 3.1, Nov 2024) ships with
`compute_units=CPU_AND_GPU` and **explicitly** explains why: GPU wins the
memory-bandwidth race for decode, and the stateful KV feature is presented
as a GPU optimization. No shipping open-source LLM (ANEMLL, coreml-llm-cli,
HF Mistral-CoreML demo) uses MLState with ANE. The one unresolved Apple
Developer Forums thread on the symptom (thread 810987) is ~1 year old,
still has zero Apple engineer replies, and only confirms a 32-alignment
symptom that our `build_stateful_padded.py` already ruled out as a
sufficient fix. Given that a retry costs a full conversion + device test
day and the expected upside from FUNDAMENTAL_UNTRIED's own median estimate
(×1.4 decode) is below the upside of the two alternative dispatch-amortization
paths we are already pursuing (chunk consolidation and speculative decoding),
**we stay rejected as the primary path**. A 1–2 hour confirmation probe on
iOS 26 device is permitted, only to avoid relying on stale 18.x evidence
indefinitely — but it does not justify any code redesign.

---

## 1. Previous attempt recap — what was tried and how it failed

Source of truth: `.claude/worktrees/agent-ad21e314/conversion/build_stateful.py`
and `build_stateful_padded.py`, plus the PyTorch modules
`models/gemma4_stateful_chunks.py` (192 lines) and
`models/gemma4_stateful_padded.py` (190 lines).

**Scope.** Chunk 2 only (Gemma 4 E2B, layers 8–14). Chunks 1/3/4 were left
alone because chunk2 is the hottest KV producer and the cheapest
isolation test. Chunk1 prefill is also a candidate but was gated on
chunk2 succeeding.

**StateType shapes declared.**

- `kv_sliding`: `(N_SLIDING*2, 1, W, max_hd) = (10, 1, 512, 512)` fp16
  — 5 sliding-window K slots + 5 V slots for L8–12.
- `kv_full`:    `(N_FULL*2, 1, CTX, max_hd) = (4, 1, 8192, 512)` fp16
  — 2 full-attn K + 2 V for L13–14.

The heads axis is `1` (num_kv_heads). The head-dim axis is the global 512
even though most layers use 256; zero-pad is added inside the forward.

Mutation pattern: `register_buffer` holders on the `nn.Module`, updated
by index-assignment (sliding: shift-and-append `cat`; full: mask-based
`torch.where`). coremltools lowers this via the default
`canonicalize_inplace_pattern` + `prefer_state_in_downstream` passes.

**Conversion settings.**

- `minimum_deployment_target=ct.target.iOS18`
- `compute_precision=FLOAT16`
- `compute_units=CPU_AND_NE`
- Post-convert INT4 palettization (per-grouped-channel, group 32),
  resulting in a ~134 MB mlpackage.
- The second variant, `build_stateful_padded.py`, additionally pads the
  KV heads axis `1 → 32` (`KV_HEADS_PAD`) to test the 32-alignment
  hypothesis.

**Result (iOS 18, Apr 2026-04-13 and -04-15).**

- Conversion: succeeds. Palettization: succeeds. Smoke test on
  `CPU_ONLY`: graph is runtime-valid, finite outputs in 49 ms.
- `CPU_AND_NE` load on Mac ANE: `ANEServicesError Domain = Code=-14`,
  "Failed to build the model execution plan".
- Same on iPhone 17 Pro.
- Padded variant: same -14 at save-time validation. Padding KV heads to
  a multiple of 32 is **not** sufficient.
- Recorded in `docs/EXPERIMENTS.md` §"MLState stateful KV cache –
  Rejected (2026-04-13)" and §"MLState + KV heads padded to 32 –
  Rejected (2026-04-15)".

**Pipeline pass check.** We do not override the pass pipeline; the
`common::canonicalize_inplace_pattern` + `common::prefer_state_in_downstream`
passes are in the default `_CLEANUP_PASSES` (ct 9.0 `pass_pipeline.py`
lines 125–145). `MIL_PASSES_ADDITIONAL.md` §7.3 notes a theoretical
ordering concern with `merge_consecutive_reshapes` running before
`canonicalize_inplace_pattern`, but in the default pipeline the inplace
canonicalizer is placed before the reshape merger, so that concern
does not apply to the current attempts. The -14 does **not** come from
a mis-ordered pass.

---

## 2. Error -14 reverse engineering

Sources: `ANEServicesErrorDomain` header traces in public Xcode
symbolication dumps; Apple Developer Forums thread 810987 ("ANE Error
with Stateful Model: 'Unable to build execution plan'"); coremltools
issues 2325 ("Failed to build the model execution plan using a model
architecture file") and 2519; our own logs.

**Symptom surface.** `-14` from `ANEServicesErrorDomain` consistently
surfaces as "Failed to build the model execution plan using a model
architecture file". Apple does not publish the enum, but the observed
pattern across issues is that this is the **ANE compiler's pre-execution
scheduling stage** — the MIL program was validated, the
`.mlmodelc/model.espresso.net` was written, but the ANE task-graph
builder rejected one or more operators when trying to lay them onto
ANE tiles. It is not a precision-loss error, not an int64 index error
per se, and not a missing-op error in the CPU interpreter sense. It is
specifically "no valid tile schedule."

**Known triggers in the wild.**

1. **Stateful ops (`coreml_update_state`, `read_state`, `write_state`).**
   Every public reproduction of `-14` with `states=...` maps back to
   this family of ops. coremltools issue 2325 is the canonical
   external repro; our `build_stateful.py` and `build_stateful_padded.py`
   are the second and third.
2. **Non-32-aligned state width.** Forums thread 810987 shows state
   shape `(1,3,480,270)` fails and `(1,3,480,256)` passes. Our padded
   variant disproves this as a **sufficient** fix — 32-aligning the
   heads axis did not help — but 32-alignment may still be a
   **necessary** one (our padded variant only padded heads, not the
   seq-dim; however seq dims of 512 and 8192 are already mod-32, so
   the remaining suspect is elsewhere).
3. **Flexible + fixed inputs mixed with state.** coremltools issue 2548
   shows MIL stateful programs breaking when both shape modes coexist.
   Not our case — our inputs are fully fixed.

**What -14 is almost certainly not.**

- Not `int64` state indices — we use fp16 everywhere, and 9.0 added
  int8 support as well.
- Not a weight-compression bug — reproduces at `nbits=0`.
- Not a precision mismatch — reproduces with `FLOAT16` only.

**Net.** -14 on our graph is the ANE task-graph builder refusing to
schedule `coreml_update_state`. No Apple engineer has ever
publicly confirmed the root cause; every external discussion is
unanswered. The 32-alignment workaround is partial at best and does
not cover our shapes.

---

## 3. iOS 26 / coremltools 9.0 evidence

### 3.1 coremltools 9.0 release notes (2025-11-10)

Checked via `github.com/apple/coremltools/releases` and the 9.0b1
(2025-07-28) preview notes. Verbatim on state: *"Ability to read and
write model state"*. Also: support for iOS26/macOS26/watchOS26/tvOS26
deployment targets, Python 3.13, PyTorch 2.7, and model input/output
with int8 dtype.

**What the notes do NOT say:**

- No mention of Neural Engine or ANE being a supported backend for
  stateful models.
- No mention of a `-14` fix, tile-scheduler update, or state-op
  lowering change for ANE.
- No mention that `CPU_AND_NE` is now an acceptable compute-unit for
  a state-bearing program.

The added read/write state API is a **runtime** convenience — letting
the host peek and poke state buffers between predictions. It does not
imply any change in whether the ANE compiler can schedule
`coreml_update_state`. This is the critical detail that
MIL_PASSES_ADDITIONAL.md §3 overstates ("iOS 26 + coremltools 9.0
landed full ANE stateful support per the 9.0 release notes"). That
statement is not supported by the actual 9.0 notes.

### 3.2 Stateful-models guide

`apple.github.io/coremltools/docs-guides/source/stateful-models.html`
(current revision). The guide's KV-cache example uses
`compute_units=ct.ComputeUnit.CPU_AND_GPU`. There is **no** mention
of ANE compatibility, no alignment advice, no "also supported on
Neural Engine" note. In the stateful-models chapter Apple is, by
omission, still treating ANE as unsupported.

### 3.3 WWDC25 sessions and Apple ML Research

"Meet the Foundation Models framework" (WWDC25 286) introduces a
stateful **session** API in the Foundation Models SDK, but the
on-device model behind it is served by a closed runtime, not a
public CoreML mlpackage. The "stateful session" marketing language
refers to user-visible conversation state, not CoreML MLState on
ANE.

Apple's "On-Device Llama 3.1 with Core ML" writeup (Nov 2024, still
the most detailed public reference) uses MLState and achieves ~33 tok/s
on M1 Max with INT4 KV-stateful — but targets `CPU_AND_GPU` with
an explicit disclaimer: *"we specifically target the GPU, as models
like the Llama-3.1-8B-Instruct are usually constrained by memory
bandwidth, and the GPU offers the best combination of compute FLOPS
and memory bandwidth."* This is Apple stating, in their own flagship
LLM demo, that MLState is a GPU optimization. Nothing in
2025 overrides this.

### 3.4 Other Apple signals

The Apple Foundation Models paper (arXiv 2507.13575) predates iOS 26
and does not discuss the underlying CoreML graph — the on-device
3.18B runs through private APIs and cannot be taken as evidence of
public MLState-on-ANE.

---

## 4. Practical sightings — who is actually shipping stateful KV on ANE?

Exhaustive check of public repos / releases / blog posts that ship
a CoreML LLM targeting ANE:

| Project | Stateful KV? | ANE target? | Notes |
|---|---|---|---|
| Apple On-Device Llama 3.1 (Nov 2024) | Yes (MLState) | **No** (CPU_AND_GPU) | The canonical MLState LLM reference. Explicitly GPU-first. |
| HF Mistral-CoreML (WWDC24 demo) | Yes (MLState) | **No** (CPU_AND_GPU) | "Excellent for GPUs on Mac computers; ANE requires additional adaptations" — Apple's own framing. |
| ANEMLL 0.3.5 (Beta, Apr 2026) | **No** (explicit I/O, IOSurface ping-pong) | Yes | Latest release adds "ANE stability fixes" but does not introduce MLState. Requires `coremltools>=9.0` for tooling reasons, not for state. |
| smpanaro/coreml-llm-cli | **No** (explicit I/O + async KV update chunk) | Yes | Uses the exact "KV-update is a separate mlmodelc" trick we could fall back to. |
| smpanaro/more-ane-transformers | **No** | Yes | Pre-MLState era. |
| Our project (stateless chunks shipping) | No | Yes | Already shipping with explicit KV I/O. |

There is not a single public project in April 2026 that ships MLState
with `CPU_AND_NE` on a real LLM. That is the strongest single signal
in this investigation. If it worked on ANE, someone besides Apple's
private runtime would already be doing it.

---

## 5. Risk/reward recalculation

**Upside if it worked.** FUNDAMENTAL_UNTRIED §2 estimates ×1.3–2.0
decode speedup, median ×1.4, coming from eliminating per-dispatch
IOSurface KV round-trips. MIL_PASSES_ADDITIONAL §2 estimates +15–20%.
On our current 8K decode baseline, median gives ~+40% tok/s on
chunk2, diluted to maybe +15–25% end-to-end (chunk2 is not 100% of
decode). That would be meaningful but not game-changing (we are
chasing ×1.5 to beat LiteRT-LM 56.5 tok/s, and speculative decoding
alone is estimated higher).

**Downside of retry.**

- One day of engineering: re-trace, re-convert with
  `minimum_deployment_target=ct.target.iOS26`, rebuild, copy to
  device, try on Mac ANE + iPhone 17 Pro. Rebuild again if -14.
- Opportunity cost against D1 (chunk consolidation) and D2
  (speculative decoding), both of which have stronger public
  evidence (ANEMLL ships 2-chunk monolithic, MTP/Eagle has its own
  track record).
- **Zero partial credit:** either the ANE compiler accepts
  `coreml_update_state` or it doesn't. There is no "half works"
  state. If -14 repeats we have nothing to salvage.

**Probability-weighted estimate.** P(works on iOS 26 ANE given no
public evidence, no Apple documentation change, no third-party
sighting, and Apple's own guide still using CPU_AND_GPU) ≈ 0.10–0.15.
Expected decode gain = 0.125 × +0.40 = +5%. That is below the
noise floor of our current benchmark methodology (~±3%). Retry is
not justified on decision-theoretic grounds.

**Alternative fixes if we did retry and it failed.** None that have
not already been tried.

- Different state shapes (heads padded to 32): tried, still -14.
- Explicit MIL `read_state`/`write_state` instead of the traced
  mutable-buffer pattern: would emit the same `coreml_update_state`
  op; same outcome expected.
- `CPU_AND_GPU` stateful: possible, but would regress 8K decode
  because our current path is ANE-bound and the GPU is slower on
  the E2B sizes we ship (per baseline speed audit, Track F).

---

## 6. If RETRY (not recommended) — exact steps

Time-boxed to 2 hours. Only execute to collect one clean iOS 26 data
point, not to actually ship.

1. Import `build_stateful.py` from the worktree into `conversion/`
   verbatim. Do not touch the graph.
2. Single-line change: `minimum_deployment_target=ct.target.iOS26`.
3. Keep `CPU_AND_NE`. Keep `nbits=4`.
4. Convert + palettize on host. Copy `chunk2.mlpackage` to
   `Examples/CoreMLLLMChat/CoreMLLLMChat/` alongside the shipping
   stateless one.
5. In `ChunkedEngine.swift`, add a feature-flagged load path that
   allocates an `MLState` via `model.makeState()` and passes it into
   `predictions(from:state:)` for chunk2 only (chunks 1/3/4 stay
   stateless).
6. Run on iPhone 17 Pro under iOS 26.x (current device build). If
   load succeeds and `predict` returns finite outputs, benchmark 10
   × 128-token decode and compare to current baseline. If -14,
   log and move on.
7. If -14 repeats, **do not** attempt pass-pipeline rewrites,
   explicit MIL ops, or further padding. Record in
   `EXPERIMENTS.md` §"MLState stateful KV cache — Rejected
   (iOS 26 retry)" and close.

Do **not** add `canonicalize_inplace_pattern` explicitly — it is
already in `_CLEANUP_PASSES` by default in ct 9.0 and running it
twice is a no-op, not a fix. The earlier speculation that "ordering
fixes -14" in MIL_PASSES_ADDITIONAL §2.4 is not substantiated.

---

## 7. If STAY-REJECTED (current recommendation) — what would change the verdict?

Any one of the following would flip us back to RETRY-NOW:

1. **Apple documentation update:** the stateful-models guide or
   coremltools README explicitly listing ANE as supported for state
   ops. (Currently: silent.)
2. **A public reproduction** — ANEMLL, smpanaro, HF, or an Apple ML
   sample — of an LLM running `CPU_AND_NE` with `states=` and
   measurably faster than the explicit-I/O baseline.
3. **An Apple engineer reply** on forum thread 810987 or coremltools
   issue 2325 describing a fix or constraint that we have not yet
   applied.
4. **A `coremltools >= 9.1` release note** specifically mentioning
   ANE + state-op scheduling.

Until one of those lands, the correct investment is elsewhere:

- D1: chunk consolidation (4→2 chunks) — amortizes dispatch the same
  way, on a path ANE definitely supports (ANEMLL ships it).
- D2: speculative decoding with the already-extracted MTP drafter —
  amortizes dispatch across tokens, orthogonal to KV layout.
- Track A/B: accept-rate tuning on the chat CV residual — already
  in motion and delivering measurable wins.

**Final call: STAY-REJECTED.** Optionally run the 2-hour iOS 26
probe in section 6 next time the device is idle, purely to refresh
the evidence base. Do not rebuild the engine around MLState until
Apple shows us they've wired it to the ANE compiler.

---

## References

- `/Users/majimadaisuke/Downloads/CoreML-LLM/.claude/worktrees/agent-ad21e314/conversion/build_stateful.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/.claude/worktrees/agent-ad21e314/conversion/build_stateful_padded.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/.claude/worktrees/agent-ad21e314/conversion/models/gemma4_stateful_chunks.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/EXPERIMENTS.md` (§MLState rejections, 2026-04-13 and 2026-04-15)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/FUNDAMENTAL_UNTRIED.md` (§2 MLState)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/MIL_PASSES_ADDITIONAL.md` (§2.4, §3, §7.3)
- Apple Developer Forums thread 810987 — state 32-alignment on ANE (unanswered)
- coremltools GitHub issue 2325 — execution plan -14 on stateful GPT-2 (unresolved)
- coremltools 9.0 release notes — iOS26 targets, state read/write API
- Apple ML Research: "On-Device Llama 3.1 with Core ML" (2024-11-01) — MLState + CPU_AND_GPU
- HF blog "WWDC24: Running Mistral 7B with Core ML" — MLState + CPU_AND_GPU
- ANEMLL v0.3.5 Beta — no MLState, explicit IOSurface KV
- smpanaro/coreml-llm-cli — no MLState, async KV-update mlmodelc
