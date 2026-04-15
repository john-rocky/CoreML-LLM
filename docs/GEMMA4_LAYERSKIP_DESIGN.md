# Gemma 4 E2B Structured LayerSkip — Implementation-Ready Design

**Date:** 2026-04-16
**Branch:** `research/conversion-deep-dive`
**Supersedes:** `docs/FUNDAMENTAL_UNTRIED.md` §4 (sketch → concrete plan)
**Companion docs:** `docs/EAGLE3_INTEGRATION_STATE.md` (existing spec-decode
infra), `docs/GEMMA4_FORWARD_ANATOMY.md` (per-layer op inventory),
`docs/ANE_OPTIMIZATION_SURVEY.md` §1 (prefill bypass which shares the same
L15 boundary)

---

## 1. Motivation — why a Gemma 4 specific LayerSkip

The Gemma 4 E2B architecture carries one quirk that no published LayerSkip
paper has exploited: **the post-L14 stack is already structurally a
prediction head, not an autoregressive block**. Concretely:

- L0–L14 (chunk1 + chunk2) own K and V projections. 15 unique KV caches
  are produced here: kv0..kv12 (sliding, W=512, hd=256) plus kv13 (sliding,
  boundary) and kv14 (full-attention, ctx, hd=512).
- L15–L34 (chunk3 + chunk4) are **KV-shared**: they have Q and O
  projections, MLP, QK-norm, RoPE — but **no K/V projection**. All 20
  layers attend to kv13 (sliding) and kv14 (global) that were produced by
  L13 and L14 respectively.

Two corollaries:

1. **Prefill bypass is already on the roadmap** (`ANE_OPTIMIZATION_SURVEY.md`
   §1) and operates on exactly this L15 boundary: for all but the last
   prompt token, chunk3+4 can be skipped because they never write KV. That
   same observation shows the boundary is a *mechanically clean cut* in the
   compiled graph — no crossed data dependencies between chunk2 and
   chunk3 other than kv13/kv14 (already written) and the hidden state.
2. **LayerSkip at L14** gets the same clean cut, but for decode. The
   hidden state emerging from L14 can be turned into a draft prediction
   by a small head, and verification uses the existing chunk3+4 path in
   its `verify_qK` multi-function mode.

Compared to the rejected/deferred alternatives in
`docs/FUNDAMENTAL_UNTRIED.md`:

- **SuffixDecoding** is orthogonal and stronger on chat workloads with
  lookup hits, but collapses to 1.0× on novel generation.
- **MTP drafter** (already converted) uses Google’s tied drafter; its
  acceptance on our custom target is under measurement but the drafter
  itself is an ANE-resident extra dispatch.
- **EAGLE-3** requires a separately shipped 210 MB draft mlpackage with
  its own decode cost and has two blockers that are still open.
- **LayerSkip-L14** ships no extra decoder — just a tiny classifier head
  (≤ 35 MB palettized) bolted onto the tail of chunk2. The draft is the
  same weights already resident on ANE. Nothing else is shipped.

The question is not whether it is structurally clean (it is); the question
is whether acceptance survives the capacity gap between a 14-layer partial
forward and the full 35-layer one. The remainder of this doc is the
implementation-ready version of “measure it.”

---

## 2. Early-exit candidates — where to cut

Four exit depths are plausible. Only L14 and L24 survive cost/benefit:

| Exit depth | Layers run in draft | Draft/Target compute ratio | Natural cut? | Notes |
|---|---|---|---|---|
| **L7** (mid chunk1) | 8/35 = 23% | 0.23× | No | Would leave kv7..kv14 unwritten; post-chunk KV state still needs to be repaired. Reject. |
| **L14 (chunk2 end)** | 15/35 = 43% | 0.43× | **Yes** (KV-share boundary) | All KVs already written; draft has zero side-effect on the verify state. **Primary candidate.** |
| **L24** (mid chunk3) | 25/35 = 71% | 0.71× | No | Cuts through KV-shared stack but also through double-wide MLP layers; marginal speedup even with high acceptance. |
| **Dynamic multi-exit** | variable | variable | — | Complexity too high for first integration; revisit only if single-exit works. |

**Chosen cut: L14.**

Why not L24: with 71% of the forward spent, break-even acceptance is
(1−0.71)/0.29 ≈ 100% — i.e. there is no speedup regime. L24 becomes
interesting only if chunks 3–4 are *themselves* faster than a linear
acceptance model predicts (e.g. because verify at Q=K amortizes per-query
work). That is a second-order optimization we defer.

Why not L7: the speedup ceiling is attractive (1.77× at 100% accept), but
L7 sits inside chunk1 — the draft would have to partially skip chunk1, or
we would have to split chunk1. Neither is free. More importantly, the
kv0..kv6 cache lines still need to be written, because the verify step
depends on them. An early-exit that requires fixing up KV afterward
defeats the “no side effects” property that makes L14 attractive.

**Amortized cost analysis at L14**, assuming K draft tokens per burst and
acceptance rate α:

```
Time_burst        = T_draft × K + T_verify(K)
Tokens accepted   = 1 + α·(K-1)       # first always accepted (target)
Tokens/s          = (1 + α·(K-1)) / Time_burst

Let T_full = 67 ms (current baseline decode per token, summed across
             chunk1-4 at 2K ctx).
T_draft    = 0.43 × 67 = 29 ms (chunks 1+2 only)
T_verify(K)= 31.5 ms measured for Q=3 verify (EAGLE3_INTEGRATION_STATE.md)
```

At K=3, α=0.5: time = 3×29 + 31.5 = 118.5 ms, accepted = 2.0 → 16.9 tok/s
At K=3, α=0.7: time = 118.5 ms, accepted = 2.4 → 20.3 tok/s (vs 14.9 8K baseline → 1.36×)
At K=4, α=0.5: time = 4×29 + 45   = 161 ms, accepted = 2.5 → 15.5 tok/s (no win)

The cliff is steep: to beat baseline the K draft cost has to go down.
**The high-value implementation detail is that *we are running the same
ANE graph as decode* for chunks 1+2**; we do not run them K times
serially. The MTP-style correct formulation is:

```
Burst:
  1. Run chunk1 + chunk2 once (starting from current hidden) → hs_L14
  2. Early-head: hs_L14 → draft_t1 (one token) via cheap classifier
  3. Embed draft_t1, run chunk1+chunk2 → hs_L14'
  4. Early-head → draft_t2
  ...
  (K draft calls to chunks 1+2, each ~29 ms)
  K+1. verify_qK on chunks 3+4 with K drafts
```

K=3 gives 3×29 + 31.5 = 118.5 ms. This is still dominated by the K draft
calls, not the verify. The right lever is **K=2** with a tree (+ sibling)
rather than K=3 linear, matching the Y-tree analysis in
`docs/ANE_OPTIMIZATION_SURVEY.md` §2:

```
K=2 linear, α=0.6: time = 2×29 + 25 = 83 ms, accepted = 1.6 → 19.3 tok/s
K=2 Y-tree (1 draft call, 2 siblings), α=0.6: time = 29 + 25 = 54 ms, accepted = 1.5 → 27.8 tok/s
```

**The Y-tree variant is the only configuration that is credibly
competitive** with the current 28.6 tok/s baseline. This becomes the
reference implementation. Linear K=3 is a fallback for ablation.

---

## 3. Draft head design — three options, one choice

The L14 hidden state is `(B=1, Q=1, 1536)`. A draft prediction must produce
logits over `vocab=262144`. The naive head is 1536×262144 = 403 M params
(~400 MB fp16 / ~100 MB at INT4 palettized). That is too large for a
“tiny” aux head, but exactly the size of the existing lm_head in chunk4.

### Option A — reuse chunk4’s lm_head on L14 hidden directly

Arithmetically: compute `argmax(lm_head(norm(hs_L14)))`. This is applying
the *same* output head that the full stack would use, just 20 layers
earlier. It is equivalent to predicting that L15–L34 are an identity
function (which they are decidedly not).

Accuracy will be poor in the generic case, but there is one regime where
it works: tokens that the network has already decided by L14 and where
L15–L34 mostly refine magnitude rather than identity. Common-word, high
entropy-drop tokens (spaces, frequent bigrams, punctuation, end-of-chunk
tokens) are the expected accept class.

**Ship cost: zero** (reuses existing lm_head weights already in chunk4).

**Inference cost:** the full lm_head matmul is ~200 ms on ANE-unfriendly
layout (measured in chunk4 today); this would be prohibitive if done at
every draft. But we only need **argmax** on the draft path. The ANE-side
top-1 trick (`argmax` + a small gather) is already in chunk4; we can
expose it as a second function of chunk2 that consumes the L14 tail
hidden and emits a token id.

### Option B — small projection head on top of L14

A tiny projection `Linear(1536, V_small=32768)` + argmax. V_small = 32768
(7% of full vocab) covers the 99th percentile of frequency-ranked tokens
on a typical chat/code corpus. On a token predicted out-of-vocab, the
draft emits a sentinel; the verifier sees it, rejects, and we fall back
to a full target decode.

Size: 1536 × 32768 × 2 bytes (fp16) = 100 MB. INT4 palettized: ~25 MB.
This is essentially a shrunk lm_head — conceptually the “vocab pruning”
already explored in `docs/apply_vocab_pruning.py`, applied selectively to
draft-only.

Training: 1-epoch supervised from (L14_hidden, top1_target_token) pairs
collected by running the *full model* on a corpus. Identical data
collection pipeline to EAGLE-3 but:
- Tap at L14 post-FFN-post-norm output (instead of L8/L17/L34)
- Target is the *argmax-over-full-vocab* token, not a hidden state

We already have the pipeline (`conversion/collect_eagle_hidden_states.py`
and the corrected `use_cache=True` reference at
`conversion/debug_l34_parity.py`). One change only: tap address.

### Option C — reuse lm_head as a fixed projection, no training

lm_head is tied to token embeddings. Applying it to L14 hidden is the
same as asking: “which token does L14 project to under the cosine metric
defined by the embedding matrix?” This is **zero-training** and has been
reported in the literature as a respectable probe (see early-exit papers
using LM-head reuse, arXiv 2310.18581).

We can calibrate with a single learned scalar `γ` so that
`logits_draft = γ · lm_head(norm_early(hs_L14))`, where
`norm_early` is a fresh RMSNorm fitted on L14 outputs (rescales variance
to lm_head’s input distribution). ~1500 parameters; trained in minutes.

**Chosen design: B + C hybrid.**

- Start with C (tied lm_head + learned norm) because it is a single-day
  experiment with zero infra changes. If acceptance ≥ 35% at K=2 Y-tree,
  ship it. Measured 2026-04-16 budget: 1 day data collection + 1 day
  calibration fit + 1 day CoreML wiring + 1 day bench = 4 days.
- If C falls short, train B (small vocab projection, 25 MB after
  palettization) and rebench. +1 week.

Reject A because it wastes the per-step argmax latency on ANE without
accuracy justification — B/C win on both axes.

---

## 4. Training the early head

Two tracks run in parallel. Track 1 is necessary for option B; track 2 is
necessary for both.

### Track 1 — corpus collection (reuse EAGLE-3 infra, swap tap)

The EAGLE-3 corpus is already downloaded (`eagle_corpus.jsonl`) and the
collection script reads it. The change:

```python
# conversion/collect_layerskip_corpus.py (new, derived from
# collect_eagle_hidden_states.py):
#   - run our custom Gemma4Model forward with use_cache=True (avoids the
#     HF use_cache=False trap from EAGLE-3 blocker 1)
#   - tap hidden_states[15] (L14 output post-norm post-FFN)
#   - record target argmax token from full 35-layer forward
#   - NO auxiliary taps, NO fusion_layers
```

Corpus size for option B: 30k samples × 512 tokens = 15M (hidden, token)
pairs. Persisted as `layerskip_training_data.pt` (~40 GB fp16 — chunked
to 2 GB shards). Same Colab GPU budget as EAGLE-3 collection (~3 hours).

### Track 2 — head fit

Option C calibration:
```python
# Load lm_head from the converted chunk4 or the HF weights.
# Fit a single RMSNorm(1536) + scalar γ on (hs_L14, token) pairs.
# Objective: maximize P(token | γ·lm_head(norm(hs))) with Adam, 1 epoch.
# Expected output: ~20 minutes on a single GPU, 1537 params.
```

Option B training (if C falls short):
```python
# Architecture: RMSNorm(1536) + Linear(1536, 32768).
# Initialize Linear.weight from lm_head restricted to top-32768 tokens.
# Initialize RMSNorm.weight = 1 (identity).
# Target: top-1 token of 35-layer forward.
# Loss: CE, but only over top-32768 vocab slots (OOV → no-match sentinel).
# 1-2 epochs SGD. ~3 hours on a single A100.
```

In both cases the head is small enough to palettize to INT4 (<10 MB) and
ship bundled with `chunk2.mlpackage` as an `early_head` multi-function
entry.

Evaluation metric for head alone (offline, Python):
- `acc@1 = P(head(hs_L14) == argmax(full_model(x)))` on held-out corpus
- Threshold for proceeding to on-device integration: **acc@1 ≥ 0.45**
- Below that, linear K=2 cannot amortize the extra verify round-trip.

---

## 5. On-device pipeline

The existing infrastructure already solves 80% of the wiring. The Swift
harness `SpeculativeLoop.swift` + `ChunkedEngine.swift` currently runs
EAGLE-3 style burst decoding with these stages:

1. `drawBurst(K)` — runs a draft source K times
2. `verifyCandidates(K)` — runs verify_qK on the target over the K drafts
3. `commitAccepted(N)` — advances state by N accepted tokens

LayerSkip reuses stages 2 and 3 **with no change**. Only stage 1 is
replaced.

### New chunk2 multi-function: `draft_exit_L14`

`build_verify_chunks.py` currently emits `decode_q1` + `verify_qK` as two
functions sharing weights. We add a third function to the chunk2 mlpackage
only:

```
chunk2.mlpackage:
  functions:
    decode_q1:             # existing — full chunk2 decode path
    verify_qK:             # existing — batched verify
    draft_exit_L14:        # new — runs chunk2 with early-head tail
```

`draft_exit_L14` shares the L7..L14 Conv weights with `decode_q1` (no
duplication) and adds the early head: `RMSNorm(1536) → Linear(1536, V) →
argmax`. Output is `int32` token id + `fp16` confidence (softmax max), so
Swift can pass confidence to the Y-tree branching logic.

Cost to chunk2 mlpackage size: +10-25 MB after palettization. Cost to
pipeline compile time: +1 function, negligible.

### Y-tree burst (reference implementation)

```
// Inputs: current state, target-produced token t0.
// Produces K=3 draft candidates in one chunk2 dispatch + one extra.
burst:
  1. ANE: chunk1(t0) → hs_L7 → chunk2(hs_L7) using decode_q1  // existing
     writes kv0..kv14.
     → hs_L14 (already resident, tail tap)
  2. ANE: chunk2.draft_exit_L14(hs_L14)
     → (top1=d1a, top2=d1b, confidence)   // new function, 1 dispatch
  3. ANE: chunk1(d1a) + chunk2(d1a)                               // existing
     → hs_L14_from_d1a
     → chunk2.draft_exit_L14(hs_L14_from_d1a) → d2a
  4. Tree: [t0, d1a, d1b, d2a]  (4 verify positions)
  5. ANE: chunks 3+4 in verify_qK mode with K=4 positions          // existing
     → target argmax per position
  6. Accept walk:
       t0 always accepted
       d1a if target(0)==d1a; else d1b if target(0)==d1b; else reject
       d2a if target(1)==d2a AND d1a was accepted; else reject
  7. commitAccepted(N): rewrite KV only at accepted positions.
```

The **commitAccepted KV rewrite** is Blocker 2 from
`docs/EAGLE3_INTEGRATION_STATE.md`. LayerSkip shares this blocker with
EAGLE-3; the v2 verify chunks (per-T-position K/V outputs) already live
in `conversion/models/gemma4_verify_chunks.py`. We inherit that work
once it lands for EAGLE-3 and reuse directly.

### Why Y-tree is the first integration, not linear K=3

- **1 fewer draft call** (2 instead of 3): saves 29 ms per burst.
- **Siblings (d1a, d1b) are free**: `draft_exit_L14` already outputs top-k;
  reading index [0] and [1] is a ~0 ms cost.
- **Acceptance-wise equivalent or better**: the sibling at depth 1 rescues
  bursts where target argmax was close to draft top-2 but not top-1.
- Matches the spec-decode design principle from `ANE_OPTIMIZATION_SURVEY.md`
  §2: *minimize ANE dispatches per burst, not just per draft*.

---

## 6. Expected speedup — honest intervals

Three scenarios, each conditioned on the acceptance-rate measurement that
the first week of training produces.

### Scenario A — optimistic (α_linear = 0.60 at K=2)

This corresponds to a trained head with acc@1 ≈ 0.55 and a sibling rescue
rate of ~0.15 on the miss cases. Numbers from Sequoia on similarly sized
models put this at the realistic *upper* end for a partial-forward draft.

```
K=2 Y-tree, α=0.60, sibling=0.15:
  time_burst = 29 + 25 = 54 ms  (1 draft + 1 verify, verify is Q=3 because t0 + 2 drafts)
  accepted_per_burst = 1 + 0.60 + 0.60×0.50 = 1.90  (including sibling)
  tok/s = 1.90 / 0.054 = 35.2 tok/s
  vs baseline 28.6 → **1.23× speedup @ 2K**
  vs baseline 14.9 → **2.36× @ 8K**  (verify dominates at 8K; bigger relative win)
```

### Scenario B — median (α_linear = 0.40 at K=2)

Partial-forward LayerSkip on a homogeneous transformer typically lands
here; the KV-share structural prior *should* lift this, but we have no
Gemma-4-specific measurement.

```
K=2 Y-tree, α=0.40, sibling=0.10:
  accepted = 1 + 0.40 + 0.40×0.35 = 1.54
  tok/s    = 1.54 / 0.054 = 28.5 tok/s
  vs baseline 28.6 → **1.00× @ 2K**  (break-even)
  vs baseline 14.9 → **1.91× @ 8K**
```

At 2K, break-even. At 8K, nearly 2× because the per-verify fixed cost is
a smaller fraction of the burst. This is the **minimum-viable regime**
where we keep shipping; below this, turn off.

### Scenario C — conservative (α_linear = 0.25)

```
K=2 Y-tree, α=0.25, sibling=0.08:
  accepted = 1 + 0.25 + 0.25×0.32 = 1.33
  tok/s    = 1.33 / 0.054 = 24.6 tok/s
  vs baseline 28.6 → **0.86× @ 2K**  (regression)
  vs baseline 14.9 → **1.65× @ 8K**
```

At 2K, net loss. Auto-disable at 2K, keep on at ctx ≥ 4K. The live
acceptance-rate counter already implemented in `SpeculativeLoop.swift`
(`rolling_acceptance`) drives the switchover — same mechanism EAGLE-3
uses today.

### Honest caveats

- All three scenarios assume the per-chunk decode time of 29 ms for chunks
  1+2 holds at 8K. At 8K, chunk2 attention against a 2K-slot full-attn
  cache is slower (measured ~37 ms summed). Re-evaluate with 8K numbers
  before shipping.
- `commitAccepted` must be the KV-direct-write variant (blocker 2). With
  the current per-token-replay variant, **none of these numbers hold**.
- No guarantee acceptance doesn’t collapse on novel generation (creative
  writing, code completion outside the training distribution). Track
  per-task α separately, same as SuffixDecoding would.
- The sibling rescue factor (0.10–0.15) is a guess calibrated from
  Sequoia’s published numbers on 7B models. Gemma 4 E2B at 2.7B
  effective may have sharper or flatter top-k distributions. Measure
  during training.

---

## 7. Integration with existing speculative infrastructure

| Method | Status | Composition with LayerSkip |
|---|---|---|
| **EAGLE-3** | `feature/eagle3-speculative`, 2 blockers | Mutually exclusive: both are draft sources. Ship whichever lands stable-first; support runtime switch. |
| **MTP drafter** | Converted, on-device acceptance unmeasured | **Complementary via DrafterUnion.** MTP uses Google’s tied drafter (ANE-resident, shallow); LayerSkip uses partial self-forward. Pick higher-confidence per step. |
| **SuffixDecoding** | Not started (§1 of fundamental-untried) | **Orthogonal.** Suffix tree hit → ship that draft; miss → fall back to LayerSkip. Strictly additive. |
| **Prompt Lookup Decoding (PLD)** | Landed via PR #36 per conversion audit | **Identical position to SuffixDecoding.** Union: PLD on short-context repetition, LayerSkip otherwise. |
| **verify_qK (multi-function)** | Built, unwired (D1 patch plan pending) | **Direct dependency.** LayerSkip needs verify_qK in the chunk3+4 path. D1 wiring patch (`docs/D1_WIRING_PATCH_PLAN.md`) is prereq. |
| **Y-tree verify mask** | Planned (`ANE_OPTIMIZATION_SURVEY.md` §2) | **Direct dependency.** Y-tree burst shape is required; same work feeds MTP Y-tree if MTP lands first. |

The union-drafter pattern (`Sources/CoreMLLLM/DrafterUnion.swift`
already exists per accept-rate-bench harness) is the integration point.
Per-burst pick:

```swift
let candidates = [
    suffixTree.propose(context: recent, k: 3),   // 20 µs CPU
    promptLookup.propose(context: prompt, k: 3), // 5 µs CPU
    layerSkip.propose(context: hs_L14, k: 2),    // 29 ms ANE
    mtp.propose(context: hs_L34, k: 3),          // ~15 ms ANE
]
let draft = candidates.max { $0.confidence < $1.confidence }
```

LayerSkip and MTP should be **mutually exclusive in a single burst** —
they both occupy ANE time. The per-step priority order is:

1. CPU-only drafts (Suffix / PLD) — free, try first
2. If both miss: LayerSkip (cheaper ANE, 29 ms) beats MTP (drafter round
   trip + extra dispatch) if their acc@1 is within 5 points
3. If LayerSkip confidence < threshold: fall back to MTP
4. If MTP confidence < threshold: T=1 decode (no spec)

This aligns with the measured fact that MTP has its own dispatch overhead
and the accept-rate harness comparison can slot LayerSkip in as a new
`DrafterProtocol` conformer without touching verify.

---

## 8. Additional Gemma 4 exploits (beyond the core design)

Three that compose directly with LayerSkip:

### 8.1 Double-wide MLP ratio

L15–L34 have `intermediate_size=12288`, vs 6144 for L0–L14. Per-layer MLP
compute is 2× for the top 20 layers. The draft path avoids *all* of it:
from the MLP-compute perspective, LayerSkip draft is

```
draft_mlp_flops / full_mlp_flops
  = (15 layers × 6144) / (15 × 6144 + 20 × 12288)
  = 92160 / 338MM
  ≈ 0.21
```

Much lower than the naive 43% layer-count ratio. This is why the 29 ms
draft estimate is *pessimistic* — in the MLP-bound regime, partial
forward is closer to 21% of full compute. Current measurements are
dispatch-bound (not MLP-bound), so the layer-count heuristic is the
correct conservative one; but if future MIL graph consolidation lowers
dispatch count, the MLP savings start to count for real.

### 8.2 Sandwich norm clean boundary

Gemma 4 has 4 RMSNorms per layer (input, post_attn, pre_ffn, post_ffn).
The post_feedforward_layernorm of L14 is the natural “clean” exit tap:
residual-stream normalized, FFN-post, no further downstream dependency
in the draft path. No extra norm needed for the early head (Option B’s
RMSNorm is a *distribution matcher*, not a correctness requirement).

### 8.3 layer_scalar distribution probe

Gemma 4 has a learned per-layer scalar multiplier applied after PLE.
`GEMMA4_FORWARD_ANATOMY.md` §3.2 flags that these *may all be 1.0* in
the HF weights. If that holds, there is no extra arithmetic at exit. If
they are not all 1.0, they impose a per-layer residual scaling that must
be faithfully replicated in the early head’s input (the final
`layer_scalar[14]` must be applied to hs_L14 before it enters the head).

**Action item:** dump HF `layer_scalar` values during corpus collection;
propagate to early head if non-trivial.

---

## 9. Alternatives we keep on the bench but don’t ship first

- **Structured layer dropout**: skip random subset, verify. Complex,
  lower accept, no structural prior. Reject for v1.
- **Layer caching on repeat tokens**: cache hidden states across the
  network and short-circuit on repeat decode. Good for punctuation loops
  but covered better by PLD (already shipping). Reject.
- **Multi-exit dynamic routing**: pick exit depth based on hs_L14
  confidence. Adds branching complexity; revisit after single-exit
  baseline.
- **Exit at L24 instead of L14**: already analyzed above; no break-even.
  Reject.

---

## 10. Risks and kill criteria

| Risk | Probability | Impact | Mitigation / kill criterion |
|---|---|---|---|
| acc@1 offline < 0.35 | 40% | Fatal | Kill after corpus collection; do not proceed to device. |
| On-device α decays to < 0.25 in first 100 bursts | 30% | Regression at 2K | Auto-disable via rolling counter; ship at ctx ≥ 4K only. |
| Verify_qK unwired when we integrate | 60% (today) | Blocker | Wait for D1 wiring patch; LayerSkip is strictly downstream. |
| Blocker 2 (KV direct-write) unresolved | 80% (today) | Fatal | Shared with EAGLE-3. Don’t ship LayerSkip until blocker 2 lands. |
| Early-head palettization corrupts acc@1 | 20% | Regression | Eval head at fp16 and INT4 separately during Track 2 fit. |
| Y-tree mask shape breaks verify compile | 15% | Schedule | Fallback to linear K=3 — confirmed ANE-compilable (EAGLE-3 paths). |

**Hard kill criterion:** if on 2026-04-30 (two weeks in) acc@1 offline is
below 0.35 on the holdout set, abandon. The 4-week remainder of the
roadmap below is contingent on passing this gate.

---

## 11. Six-week roadmap

### Week 1 — corpus collection (branches off today)
- Day 1-2: fork `collect_eagle_hidden_states.py` →
  `collect_layerskip_corpus.py`. Tap `hidden_states[15]` (L14 post-norm
  post-FFN). Verify with `debug_l34_parity.py` that the tap is correct
  (rel_diff < 1e-5 vs HF with `use_cache=True`).
- Day 3-4: run collection on Colab. 30k × 512 = 15M pairs.
  `layerskip_training_data.pt` ~40 GB (shard to 2 GB chunks, stream to
  Drive).
- Day 5: split train/val (95/5). Publish to internal storage.

### Week 2 — head fit (parallel track)
- Day 1: Option C calibration (γ + RMSNorm, ~20 min GPU).
  Measure acc@1 on val set.
- Day 2: if C acc@1 ≥ 0.45, stop here. Skip to Week 3.
- Day 2-4: else train Option B (RMSNorm + Linear(1536, 32768)). 2 epochs.
- Day 5: ablation: acc@1 at {top-1, top-2 with sibling rescue, top-4}.

### Week 3 — CoreML conversion
- Day 1-2: add `draft_exit_L14` function to chunk2 mlpackage via
  `build_verify_chunks.py` pattern. Weight dedup with `decode_q1`.
- Day 3: Python smoke test (`/tmp/smoke_layerskip_draft.py`): inputs
  identical to `decode_q1`, output is (token_id int32, confidence fp16).
- Day 4: parity test — chunk2.draft_exit_L14 output vs PyTorch early-head
  forward. Target: max abs diff ≤ 4e-3 in fp16 (same bar as EAGLE-3
  verify chunks passed).
- Day 5: palettize chunk2 including the new head, re-verify parity.

### Week 4 — Swift integration
- Day 1-2: new `LayerSkipDraftSource.swift` conforming to
  `DrafterProtocol`. Internal: holds chunk2 mlmodel, calls
  `draft_exit_L14` function.
- Day 3: wire into `DrafterUnion` with priority order (§7).
- Day 4: Y-tree verify-mask builder in `ChunkedEngine` (depends on D1
  wiring). Shared with MTP Y-tree path if that lands in parallel.
- Day 5: smoke run on Mac + one-off iPhone test, check bookkeeping.

### Week 5 — measurement
- Day 1: thermally stable 10-min bench at 2K (compare 28.6 baseline).
- Day 2: same at 8K (compare 14.9 baseline). Expected: bigger win at 8K.
- Day 3: per-task breakdown (chat / code / RAG). Identify tasks where α
  is below kill threshold; auto-disable list.
- Day 4: tune exit depth if L14 gives marginal results — try L17 via
  `draft_exit_L17` on chunk3 (cut inside chunk3 is non-clean, but if
  acceptance is clearly capacity-bounded not structure-bounded, this may
  help). Defer to Week 6 if no head candidate.
- Day 5: decision gate — ship or kill.

### Week 6 — composition (conditional)
- If shipping: compose with MTP via DrafterUnion. Confidence-based
  picker. Union accept-rate sanity vs either alone.
- If SuffixDecoding lands in parallel: add as CPU-fast first tier.
- Document shipping acceptance distributions in
  `docs/IMPLEMENTATION_LOG_2026_04_15.md` successor.

---

## 12. Summary

**One-line claim:** LayerSkip at Gemma 4’s L14 KV-share boundary is the
only speculative method in our backlog that (a) ships zero extra
decoder, (b) has a mechanically clean cut in the compiled graph, and
(c) gets structurally bigger relative wins at 8K than at 2K, which
matches our deployment target.

**Minimum viable:** Option C (tied lm_head + RMSNorm + γ) in Y-tree K=2
burst, auto-disabled below α=0.25. 4-week path, 0 new weights shipped
beyond ~1.5K calibration parameters.

**Critical prereqs:**
1. D1 multi-function wiring (verify_qK) — separate patch, same branch
2. KV-direct-write `commitAccepted` variant (shared with EAGLE-3)
3. Y-tree verify mask (shared with MTP)

**Dependencies LayerSkip does not have:** no new training infra (reuses
EAGLE-3 collection), no new mlpackage bundle (augments chunk2), no new
draft model to ship (augments lm_head semantics).

**Single highest-leverage measurement to take first:** offline acc@1 of
tied-lm_head-as-draft on 5k held-out (hs_L14, target_token) pairs. One
GPU-hour. Decides whether the remaining 5 weeks are worth doing.

---

## Appendix A — file touch list

| Path | Change |
|---|---|
| `conversion/collect_layerskip_corpus.py` | New. Fork of `collect_eagle_hidden_states.py`. Tap `hidden_states[15]`. |
| `conversion/models/gemma4_swa_chunks.py` | Add `SWAChunk2EarlyExit` class exposing hs_L14 tail + early head. |
| `conversion/build_verify_chunks.py` | Emit 3rd function `draft_exit_L14` on chunk2 only. |
| `conversion/train_layerskip_head.ipynb` | New. Options B and C training. |
| `Sources/CoreMLLLM/LayerSkipDraftSource.swift` | New. `DrafterProtocol` conformer. |
| `Sources/CoreMLLLM/ChunkedEngine.swift` | Y-tree verify mask builder (shared with MTP). |
| `Sources/CoreMLLLM/DrafterUnion.swift` | Add LayerSkip to priority order. |
| `Sources/CoreMLLLM/SpeculativeLoop.swift` | No change (burst shape identical to EAGLE-3). |
| `docs/GEMMA4_LAYERSKIP_DESIGN.md` | This file. |

## Appendix B — references

- LayerSkip: arXiv 2404.16710 (Meta, ACL 2024)
- facebookresearch/LayerSkip (reference impl, HF transformers assisted decoding integration)
- Sequoia tree topology: arXiv 2402.12374
- OPT-Tree: arXiv 2406.17276
- Early-exit probes via tied lm_head: arXiv 2310.18581
- Apple AFM prefill-bypass (KV-share corollary): arXiv 2507.13575
- Maarten Grootendorst — Gemma 4 visual guide (KV-share boundary)
- Internal: `docs/FUNDAMENTAL_UNTRIED.md` §4 (originating sketch);
  `docs/EAGLE3_INTEGRATION_STATE.md` (Blocker 2, verify infra);
  `docs/ANE_OPTIMIZATION_SURVEY.md` §1-2 (prefill bypass + Y-tree);
  `docs/GEMMA4_FORWARD_ANATOMY.md` §3.8 (dual head_dim);
  `docs/D1_WIRING_PATCH_PLAN.md` (verify_qK wiring prereq).
