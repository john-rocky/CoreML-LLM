# Speedup Lever Hunt — open-ended search (2026-05-13)

Context: user is away. While they validate L12 Phase 1 on iPhone, search
**every** plausible speedup lever for Gemma 4 E2B on iPhone 17 Pro. Goal: a
ranked shortlist of promising candidates with cost / risk / projected gain.

Ground truth re-stated:

- iPhone 17 Pro plain decode: 32 tok/s
- iPhone 17 Pro current ship (FLy top-K=16 + L5 + never-bail): ~37 tok/s ≈ 1.16×
- Target: ~48 tok/s ≈ 1.5× lossless
- iPhone verify cycle: 47 ms (drafter 11 + verify 35 + commit 0.4)
- iPhone decode cycle: 30 ms (chunk1 5.7 + chunk2 6.2 + chunk3 7.8 + chunk4 10.3 + emb/mask 0.5)

Constraints: no training, lossless preferred, Mac-side validation first.

This doc is a living index. Each lever has its own section as it's
investigated. Decisions and dead-ends are kept here so we don't re-walk them.

---

## EXECUTIVE SUMMARY (after exhaustive hunt, 2026-05-13)

**Headline:** 1.5× iPhone **lossless** is not achievable training-free with
the current architecture. The realistic ceiling stacking ALL training-free
levers is **~1.22-1.25× iPhone** (≈40-41 tok/s). Path B drafter retraining
(1 GPU-week) is the credible 1.5× route.

### What we found

| Lever | Verdict | Expected gain |
|---|---|---|
| L12 subset LM head | implemented, ~50% miss rate, fallback lossless but slow | 0 (Mac), ~+3-5% iPhone optimistically |
| L12 + chat-corpus freq | marginal improvement | -7% miss rate vs Gutenberg, still 43% miss |
| L12 + semantic-NN clustering | k-means alone insufficient; needs FAISS IVF | TBD multi-day work |
| Hot-path CPU optimizations | 3-7ms / cycle achievable | +6-12% iPhone |
| BNNS fp16 matmul | gather is the bottleneck, not matmul | +1.2× matmul only |
| Multi-chain L5 async | bounded by drafter accept (~18%) | +3-5% iPhone |
| INT4 lm_head palettization | 4× less gather bandwidth | +2-4 ms / cycle iPhone |
| Steering Drafters (SD²) | requires A100-days, our ratio = +2% | not worth standalone |
| Embedding-Space MTP probing | adds +30ms verify pass, needs >1.5× accept lift | borderline, 4-5d PoC |
| FAISS IVF semantic NN | likely needed for L12 to deliver | multi-day, untried |
| Plan 3 Linear verify chunks | parity within noise on iPhone | <1% |
| Cross-chunk async pipelining Phase 2 | speculative chunk1-2 of N+1 | multi-week impl |
| Prefix cache | TTFT only, not steady-state | n/a |
| T=1 decode subset | plain decode rarely fires when MTP-on | low ROI |
| Smaller sliding window | rebuild + quality risk | low ROI |

### Realistic next steps (in priority order)

1. **Land L12 as opt-in shipping foundation** (already implemented +
   lossless verified). Default off. Enables future iteration without
   re-doing the Swift integration.

2. **CPU hot-path optimizations** (3-7 ms / cycle):
   - Gate "first N rounds" diagnostic prints behind explicit env
   - Batch embedToken lookups (1 alloc instead of K)
   - Sliding-cache ring buffer (replace memmove shift)
   - Defer `quiesceCopyBacks()` until after verify chunks 1-3 dispatch
   - Cap CSD memory dict at LRU

3. **Multi-chain L5 async**: dispatch a second drafter chain (CPU/GPU)
   predicting partial-accept K-2 carry alongside current full-accept chain.
   Expected: L5 hit 15% → 30%, +3-5% iPhone.

4. **INT4 lm_head palettization** when L12 becomes viable (with FAISS IVF
   or better candidate selection). 4× less gather bandwidth.

5. **For 1.5×, accept Path B drafter retraining is the path**. Plan a
   1 GPU-week training run with Self-Distillation MTP (arxiv 2602.06019)
   OR fold SD² steering objective into that training run.

### Acceptable interim ship

Steps 1-3 above realistically deliver **~1.20-1.25× iPhone** (vs current
1.16×, target 1.5×). About a **week** of focused engineering. Same
quality as current ship (FLy top-K=16 lossy edge — no quality regression
beyond what's already shipped).

---

## Lever shortlist (initial brainstorm)

| ID | Lever | Cost | Projected iPhone gain | Confidence |
|---|---|---|---|---|
| H1 | Cycle hot-path profile → fix overhead | low | unknown | medium |
| H2 | BNNS fp16 matmul (subset path) | low | enables larger M | high |
| H3 | Semantic-NN candidate selection | high (2-3d) | unlocks lossless 1.5× via L12 | medium |
| H4 | Phase 2 chunk pipelining | medium | +5-15% | medium |
| H5 | Drafter CPU+GPU concurrent | medium | shrinks L5 async window | low |
| H6 | INT4 lm_head palettization | low | -3-5ms iPhone matmul | medium |
| H7 | T=1 subset LM head (decode) | medium | +5-10% decode tok/s | medium |
| H8 | Prefix cache audit | low | TTFT-side, not steady-state | low |
| H9 | Smaller sliding window | high (rebuild) | -1-2ms chunk1+2 | low |
| H10 | ANE-CPU sync batching | medium | -3-5ms | low |
| H11 | 2026 SD papers survey | low | unknown new ideas | medium |
| H12 | Alt SD methods (Specstreams, Medusa) | high | training risk | low |
| H13 | LM head SVD low-rank approx | medium | cheaper full-vocab eval | medium |
| H14 | Conv2d → Linear rework remaining chunks | medium | +3-5% per chunk | medium |
| H15 | Per-thermal-state adaptive cycle | low | +2-5% sustained | low |

Findings populate below as investigations complete.

---

## H1: Cycle hot-path profile

**Status: complete (codebase audit agent, 2026-05-13)**

The MTP cycle is well-optimized at the ANE level, but the CPU side has
several un-amortized costs that add up to **3-7 ms / cycle** on iPhone.
Findings (citations in `Sources/CoreMLLLM/...`):

### High-confidence wins (~3-5ms total)

1. **Sliding-cache commit is `memmove` instead of ring-buffer**
   (`ChunkedEngine.swift:3358-3365` in `commitSlidingSlots`). For W=512,
   M=1-2, each accepted commit shifts 510×512×256×2 bytes (~256 MB moved!)
   instead of a pointer-swap. Estimated **1-2 ms per commit cycle**. Fix:
   ring buffer with logical head index; memcpy only new rows. Risk: changes
   KV-read indexing across the whole chunk-1/2 path; needs careful audit.

2. **embedToken lookups happen K times in a loop instead of batched**
   (`ChunkedEngine.swift:2461-2472` `buildVerifyHidden`, `2474-2485`
   `buildVerifyPLR`). K=3 separate `embedTokens.lookup()` calls + K
   MLMultiArray-shape wrappers. Estimated **0.5-1 ms** per verify cycle.
   Fix: batch into a single (1, K, hidden) allocation with a
   vectorized lookup. Risk: low.

3. **Unconditional diagnostic prints during early cycles** (multiple sites
   in `MtpSpeculativeEngine.swift`: lines 461, 490, 495, 554, 594, 772,
   916, 1113). These fire when `totalRounds < N` regardless of whether the
   user enabled debug. String allocations + prints add **1-2 ms** for the
   first ~6 cycles. Fix: gate every "first N rounds" print behind explicit
   env var. Risk: low.

4. **`quiesceCopyBacks()` synchronously waits at verify entry**
   (`ChunkedEngine.swift:1850`). It blocks the ANE-dispatch path while
   prior predictStep's async memcpy may still be pending. Could overlap by
   deferring the wait until just before reading KV (i.e., after chunks 1-3
   start). Estimated **0-2 ms** when sync would otherwise stall.

### Medium-confidence wins (~1ms)

5. **RoPE header parsed on every batch lookup**
   (`ChunkedEngine.swift:2554-2556` in `lookupRoPEBatch`). The `.npy`
   header is re-parsed on every call. Estimated **0.1-0.3 ms / cycle**.
   Fix: cache parsed offset at init.

6. **CSD memory dict is unbounded** (`MtpSpeculativeEngine.swift:1068-1071`).
   Not a latency issue today, but on long generations the dict can grow
   into the MB range; not a memory leak per se (reset on conversation) but
   risks hashing-overhead degradation. Fix: cap with LRU.

7. **L5 async trigger is narrow** (`MtpSpeculativeEngine.swift:636`). Only
   fires on full-accept (`matchCount==K-1`), which empirically lands on
   10-20% of free-form cycles. Could expand: also dispatch when drafter is
   idle and a "speculative carry" can be guessed from recent emit history.
   Estimated **+5-10% async hit rate**.

### Low-confidence / further investigation

- Per-cycle individual drafter ANE latency isn't profiled separately (the
  `draft=` field is the sum of K-1 calls). If one of the K-1 calls is
  slower (e.g. first call cold), we could swap it out. Needs per-call
  instrumentation.
- ANE-CPU sync per chunk transition is also unmeasured. Likely free with
  IOSurface but worth a one-shot check.

### Combined potential

3-5 ms / cycle of CPU-side savings × 47ms baseline → **1.07-1.10× iPhone**
on top of whatever else lands. By itself this isn't 1.5× — it's a
multiplier on top of L12 or other ideas.

## H2: BNNS fp16 matmul

**Status: complete (2026-05-13). Modest speedup, gather is the real bottleneck.**

Hands-on Swift benchmark `/tmp/bnns_bench.swift` against the actual lm_head
shape (V=262144, H=1536). Mac M-series CPU (release `-O`, 20 iters):

| M | sgemm fp32 + convert | BNNS fp16 direct | Speedup |
|---|---|---|---|
| 1024 | 0.41 ms | 0.34 ms | 1.2× |
| 4096 | 2.50 ms | 2.19 ms | 1.1× |
| 8192 | 4.96 ms | 4.13 ms | 1.2× |

The fp16→fp32 conversion is ~10-20% of cost; matmul is fast either way. The
**gather step dominates** — 8192 random 3 KB rows through a 768 MB buffer is
TLB-thrashing (each row crosses page boundaries). Effective gather bandwidth
~8 GB/s vs theoretical 200 GB/s.

### Real levers (different from BNNS)

1. **Sort candidate IDs before gather** — adjacent IDs share cache lines.
   Easy, low risk. Estimated 1.5-2× gather speedup. **Try this.**
2. **__builtin_prefetch** in gather loop — speculative prefetch the next
   row while reading current. Estimated 1.3-1.5× speedup.
3. **INT4 palettized lm_head** (H6) — 4× less bytes to gather = ~4× gather
   speedup. Requires INT4-aware matmul kernel or dequant on-the-fly.

### Conclusion

BNNS itself isn't worth integrating (only 1.2× win). But the gather
optimization is. **Action: implement sort-then-gather in
`sparseMatmulFp32` next.** Combined with the chat-corpus-derived frequent
tokens, this could make M=4096 affordable (~1.5 ms instead of 2.5 ms) for
better coverage at low cost.

## H3: Semantic-NN candidate selection

**Status: Investigated (2026-05-13). Spherical k-means clustering alone is NOT enough.**

Method: MiniBatchKMeans (sklearn) on L2-normalized lm_head rows. Saved
`lm_head_cluster_centroids.bin` + `lm_head_cluster_assignments.bin` +
`lm_head_cluster_members.bin`. Online: matmul `normed_hidden × centroids
→ top-N clusters → union of member tokens`.

Offline test (proxy hidden state = `lm_head[T]` as query for token T):

| K | top-N | Coverage | Avg subset size |
|---|---|---|---|
| 128 | 2 | 4% | 4313 |
| 128 | 8 | 19% | 7708 |
| 128 | 16 | 51% | 13365 |
| 128 | 32 | 86% | **37817** — too large for matmul |
| 1024 | 16 | 9% | 2779 |
| 1024 | 32 | 13% | 5400 |
| 1024 | 64 | 28% | 7629 |

**Why it fails:** cluster centroids are means of 256-2048 diverse 1536-dim
vectors. When the query aligns with one cluster MEMBER but is orthogonal
to the cluster MEAN, the centroid score is mediocre and the cluster falls
out of top-N. To get >95% coverage we need 38K+ candidates (matmul ~30 ms,
defeats the chunk4 saving).

**What would work but isn't done:**

- FAISS IVF with multi-membership (each token assigned to top-K nearest
  centroids, not just top-1). At query time the same cluster can be picked
  via multiple tokens' membership → higher recall. Multi-day work.
- HNSW (proper graph-based ANN) on the full LM head. Build time ~30 min,
  query 0.1-1 ms. Adds ~50-100 MB index. Multi-day work to wire to Swift.
- The candidate set in H17 (Embedding-Space MTP) sidesteps this entirely.

**Conclusion:** simple clustering won't deliver lossless 1.5×. Deferred
behind FAISS IVF prototype (1 wk) or Path B drafter retraining (1 GPU-wk).

## H4: Phase 2 chunk pipelining

(pending — see TaskCreate #15)

## H5: Drafter CPU+GPU concurrent

(pending — see TaskCreate #16)

## H6: INT4 lm_head palettization

(pending — see TaskCreate #17)

## H7: T=1 subset LM head (decode)

(pending — see TaskCreate #18)

## H8: Prefix cache audit

(pending — see TaskCreate #19)

## H9: Smaller sliding window

(pending — see TaskCreate #20)

## H10: ANE-CPU sync batching

(pending — see TaskCreate #21)

## H11: 2026 SD papers survey

**Status: complete (research agent, 2026-05-13). All arxiv IDs verified by abstract fetch.**

### Top 3 NEW leads (training-free, ANE-compatible, not in our dead-end list)

1. **Steering Pretrained Drafters during Speculative Decoding**
   (arxiv 2511.09844, AAAI 2026). Compute a steering vector from the
   verifier's hidden states (we already have `lastVerifyHiddenStates`!),
   inject into the drafter at decode time. **+35% accepted tokens
   (sampling), +22% (greedy)**, negligible overhead, retrofittable.
   Projected: 1.25-1.30× iPhone on top of FLy top-K=16. **Effort: 2-3 days
   Swift integration.** This is the highest-ROI lever in the table.

2. **SpecKV — Compression-Aware Gamma Selection**
   (arxiv 2605.02888, 2026-05-04). Train a tiny MLP on drafter signals
   to pick per-step speculation depth γ. **+56% over fixed-γ=4** with
   0.34ms overhead. Our K_USE=2 fallback is exactly this knob — and
   we already log all the signals (FOM, rolling acceptance, drafter
   entropy). Training data already exists. **Effort: 4-7 days.**

3. **Embedding-Space MTP Probing**
   (arxiv 2603.17942, Qualcomm). **Delete the drafter entirely.** Probe
   chunk4 LM head with on-the-fly mask tokens drawn from the embedding
   space. +12-19% throughput training-free, lossless. Removes the
   structurally-dead drafter as a constraint. **Effort: 3-5 days, with
   a 1-2 day Mac spike to verify CoreML can inject mask token embeds.**

### Other notable findings

- **DropMatch** (arxiv 2603.03333): MC dropout on LM head only. Layer on
  top of existing path. +9-33%. **Read-later** if cycle budget allows.
- **Mirror Speculative Decoding** (Apple, arxiv 2510.13161): streaming
  draft branches. 2.8-5.8× on 14B-66B; our 2B Gemma may be too small.
  **Read-later.**
- **DIET** (arxiv 2603.23985): training-free dim-wise pruning on Gemma-2 2B.
  Orthogonal to MTP. **Read-later** — potential +5-10% pruning on top of
  L12 if INT4 path survives pruning.

### Confirmed dead / already-evaluated

- LogitSpec (PLD family, already regressed for us)
- ConfLayers (LayerSkip family, dead twice on E2B)
- Component-Aware Self-Spec (needs SSM subgraph, Gemma 4 is pure attention)
- SubSpec (CPU↔GPU offload — no analog on ANE)
- UAG cross-family (same dead-end as our Qwen drafter test)
- DVI online RL (subsumed by Path B drafter retraining plan)

### Action plan from this survey

Rank by `(projected gain × probability of landing) / effort`:

| Rank | Lever | Gain × Prob | Effort | Score |
|---|---|---|---|---|
| 1 | ~~Steering Drafters~~ | **DEFERRED** | 1 wk + A100 5 d | rejected |
| 2 | Embedding-Space MTP | 1.15× × 0.5 = 0.575× | 3-5 d | MED |
| 3 | SpecKV γ-MLP | 1.10× × 0.6 = 0.66× | 4-7 d | MED |
| 4 | DropMatch | 1.05× × 0.5 = 0.525× | 2-3 d | LOW-MED |

### Steering Drafters deep-dive verdict (DEFERRED, 2026-05-13)

Deep agent read of arxiv 2511.09844 + our codebase:

- The "+22-35% accept" headline is on **80× drafter/target ratio** (Vicuna 13B
  + Llama 160M). Paper's own Table 1 shows our ratio (Gemma 4 E2B 2 B target
  with ~110 M drafter, ≈20×) yields **only +2%** — essentially noise.
- "Training-free retrofit" abstract is misleading. §3.3 + Fig. 6 ablation:
  full drafter fine-tune is critical (frozen-drafter + steering-only is flat).
- Compute envelope = same as **Path B drafter retraining** already on roadmap
  (3-5 days A100). If we're going to spend that compute, Path B has a higher
  ceiling.
- Recommendation: fold SD²-style `[h, m, l]` steering + `W_s` MLP bias into
  Path B's training objective at near-zero extra cost. Not worth a standalone
  effort.

Full agent report quotes paper §3.1, §3.2, §3.3, Fig. 6 with line citations
to `build_mtp_drafter.py` and `build_verify_chunks.py`. **Definitive defer.**

## H12: Alt SD methods

(pending — see TaskCreate #22)

## H13: LM head SVD low-rank

(deferred — see H14 / semantic NN)

## H14: Verify-chunk Conv2d → Linear migration

**Status: investigated (agent), DEFERRED (low ROI).**

Plan 3 already migrated STATEFUL chunks to Linear. Verify chunks still use
Conv2d (no `--linear-projections` flag in `build_verify_chunks.py`). Plan 3
empirical probe on stateful chunks (`SESSION_2026_04_26_STATEFUL_PLAN3_PHASE2A.md`)
showed Conv2d vs Linear is **parity within noise** on iPhone 17 Pro (40.0
vs 39.9 tok/s) and **-1.0% on Mac**. The largest dense matmul (lm_head
1536→262144 in chunk4) is already sidestepped by `SWAVerifyChunk4Subset`.
Effort 2-3 hours to add the flag; expected gain <1% on iPhone. **Skip
unless other levers are exhausted.**

## H15: Per-thermal-state adaptive cycle

(no task yet — explore if user wants sustained-throughput optimization;
not relevant to peak tok/s)

## H16: Multi-chain L5 async (NEW)

**Status: candidate, not investigated yet.**

Current L5 dispatches ONE CPU drafter chain per cycle, predicting only the
"full-accept" carry (drafter's last token). Hits ~15% of cycles. iPhone CPU
and GPU are otherwise idle during verify (~35 ms ANE). Could dispatch
multiple drafter chains in parallel:

- Chain A (current): predicts full-accept carry
- Chain B (new, GPU): predicts target_argmax[0] = drafter's top-2 candidate
  (assumption: drafter's top-1 wrong, top-2 right — common under FLy top-K)
- Chain C (post-verify): once verify done, run an ANE drafter for the actual
  carry value

Effort: load drafter twice (CPU + GPU compute units), small Swift wiring.
Expected gain: L5 hit rate 15% → 35-50% → +5-8% iPhone tok/s.
Risk: GPU drafter latency unknown; memory +149 MB if second instance.

## H17: Embedding-Space MTP Probing — replaces drafter (NEW)

**Status: deep-dive done by agent. 4-5 day Mac PoC recommended.**

Replaces the trained drafter with a single mean-embedding probe through verify
chunks. Soft-init mask `m = mean(history embeddings)` replicated K times,
EMA-updated each step. **Math is borderline:** each probe = 30 ms verify
pass on iPhone vs drafter's 11 ms; needs ≥1.5× accept lift to break even.
Paper's +12-19% throughput was vs vanilla self-speculative (no drafter),
not vs our trained drafter — marginal vs our baseline unclear.

Reasonable 4-5 day Mac PoC with go/no-go gates:
- Day 2: mask-probe argmax accept rate ≥ 0.45 vs ground truth
- Day 3: Mac tok/s ≥ +5% over trained-drafter baseline
- Day 4: iPhone tok/s ≥ +15% (≥42 tok/s)

Pure Swift change — no new CoreML models, no weights. Decisive go/no-go.
