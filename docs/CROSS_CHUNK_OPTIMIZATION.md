# Cross-chunk + boundary optimization (R12)

Date: 2026-04-16
Branch: `research/conversion-deep-dive`
Scope: Gemma 4 E2B decode pipeline, 4-chunk default layout at `ctx=2048`.

All byte counts are fp16 (2 B/elem) unless stated. All shapes are copied
verbatim from `conversion/build_verify_chunks.py` and
`conversion/models/gemma4_swa_chunks.py`. Config constants come from
`conversion/models/gemma4.py:39-58`:
`hidden=1536`, `num_heads=8`, `num_kv_heads=1`, `head_dim_sliding=256`,
`global_head_dim=512`, `W=512`, `hidden_size_per_layer_input=256`,
`nlayers=35`, default `ctx=2048`.

---

## 1. What each chunk actually consumes

The four decode chunks share an almost-identical input surface — only
the KV tensors differ. Source: `conversion/build_verify_chunks.py:197-458`.

| Input | Shape | Bytes | Chunk 1 | Chunk 2 | Chunk 3 | Chunk 4 |
|---|---|---:|:-:|:-:|:-:|:-:|
| `hidden_states` | (1,1,1536) | 3 072 | in | in | in | in |
| `causal_mask_full` | (1,1,1,2048) | 4 096 | ✓ | ✓ | ✓ | ✓ |
| `causal_mask_sliding` | (1,1,1,512) | 1 024 | ✓ | ✓ | ✓ | ✓ |
| `update_mask` | (1,1,2048,1) | 4 096 | ✓ | ✓ | ✓ | ✓ |
| `cos_s` | (1,1,1,256) | 512 | ✓ | ✓ | ✓ | ✓ |
| `sin_s` | (1,1,1,256) | 512 | ✓ | ✓ | ✓ | ✓ |
| `cos_f` | (1,1,1,512) | 1 024 | ✓ | ✓ | ✓ | ✓ |
| `sin_f` | (1,1,1,512) | 1 024 | ✓ | ✓ | ✓ | ✓ |
| `per_layer_raw` | (1,1,8 960) | 17 920 | ✓ (raw) | — | — | — |
| `per_layer_combined` | (1,1,8 960) | 17 920 | produced | ✓ | ✓ | ✓ |
| `K/V_sliding_in` | (S, 1, 512, 512) | 512 KB·S·2 | S=7 | S=5 | — | — |
| `K/V_full_in`    | (F, 1, 2048, 512) | 2 MB·F·2 | F=1 | F=2 | — | — |
| `kv13_k`, `kv13_v` | (1, 1, 512, 256) | 256 KB ea. | — | produced | ✓ | ✓ |
| `kv14_k`, `kv14_v` | (1, 1, 2048, 512) | 2 MB ea. | — | produced | ✓ | ✓ |

Outputs (`out1..out4`, same file lines 212/304/397/459):
chunk1 emits the five own-KV updates + `per_layer_combined_out`; chunk2
additionally emits the four `kv13_*/kv14_*` tensors; chunk3 emits only
`hidden_states_out`; chunk4 emits `token_id`, `token_logit`, and
`hidden_states_out` (the last is consumed by MTP/EAGLE drafters, see
`ChunkedEngine.swift:94`).

### Sizes that actually move every decode step

**Per-chunk repeated inputs (1, 2, 3, 4):** `hidden_states` (3 KB),
`causal_mask_full` (4 KB), `causal_mask_sliding` (1 KB), `update_mask`
(4 KB), `cos_s`/`sin_s` (1 KB total), `cos_f`/`sin_f` (2 KB total) =
**15 KB of identical bytes** constructed once on the CPU and referenced
four times in the `MLDictionaryFeatureProvider` dicts built at
`ChunkedEngine.swift:429,453,494,502`. The dicts hold MLMultiArray
*pointers*, so the bytes themselves aren't duplicated in memory — but
they do re-enter the ANE through the IOSurface input path on every
chunk boundary.

**KV in/out traffic on chunks 1 and 2 (own-KV):**

- chunk1 `K/V_sliding_in + out`: 2·(7·1·512·512·2) = **7 MB in + 7 MB out**
- chunk1 `K/V_full_in + out`: 2·(1·1·2048·512·2) = **2 MB in + 2 MB out**
- chunk2 `K/V_sliding_in + out`: 2·(5·1·512·512·2) = **5 MB in + 5 MB out**
- chunk2 `K/V_full_in + out`: 2·(2·1·2048·512·2) = **8 MB in + 8 MB out**

Totals: **chunk1 = 18 MB/step**, **chunk2 = 26 MB/step** of KV across
the Swift⇄ANE boundary *for the own-KV tensors alone*.

**kv13/kv14 cross-chunk handoff (chunk2 → chunks 3, 4):**

- kv13_k + kv13_v: 2·(1·1·512·256·2) = **512 KB**
- kv14_k + kv14_v: 2·(1·1·2048·512·2) = **4 MB**
- Total: **4.5 MB of kv13/kv14** emitted by chunk2, then re-submitted
  to chunks 3 and 4 as inputs. Because the Swift layer re-references
  the same MLMultiArrays (see §4), the bytes are not re-copied — but
  they do leave the ANE at chunk2's output and re-enter at chunks 3/4.

`docs/EXPERIMENTS.md:34` cites "32 MB of fp16 KV I/O per decode step"
for the MergedChunk23 motivation. That number does not match either
ctx=2048 (4.5 MB) or ctx=8192 (16.5 MB) for kv13/kv14 alone; it is
likely the total round-trip volume including the own-KV and double
counting in/out, or was computed for a legacy layout. Recommend the
doc be corrected to **≈4.5 MB at ctx=2048**.

---

## 2. The `copyBack` memcpy is the real per-step I/O cost

`Sources/CoreMLLLM/ChunkedEngine.swift:275-302` allocates
IOSurface-backed MLMultiArrays for every *persistent* KV buffer
(`kSliding1`, `vSliding1`, `kFull1`, `vFull1`, `kSliding2`, `vSliding2`,
`kFull2`, `vFull2`). These are used as **inputs** to chunks 1 and 2
via zero-copy IOSurface aliasing.

But the corresponding *outputs* are not wired to output backings.
`grep outputBacking` and `grep MLPredictionOptions` across
`Sources/CoreMLLLM/` return zero matches. Instead, each chunk's updated
KV tensors are memcpy'd back into the persistent IOSurface after the
prediction returns:

```swift
private func copyBack(_ output: MLFeatureProvider, _ name: String,
                     into buf: MLMultiArray) {
    let src = output.featureValue(for: name)!.multiArrayValue!
    memcpy(buf.dataPointer, src.dataPointer,
           buf.count * MemoryLayout<UInt16>.stride)
}
```
(`ChunkedEngine.swift:1062-1065`)

Called four times per chunk:
`ChunkedEngine.swift:444-447` (chunk1) and `467-470` (chunk2).

**Per-step memcpy volume at ctx=2048:**
- chunk1: 7·1·512·512·2·2 (K+V sliding) + 1·1·2048·512·2·2 (K+V full)
  = 7 340 032 + 2 097 152·2 = **11.0 MB**
- chunk2: 5·1·512·512·2·2 + 2·1·2048·512·2·2
  = 5 242 880 + 4 194 304·2 = **13.0 MB**
- **Total: 24.0 MB of memcpy per decode step** just for KV persistence.

At typical iPhone 17 Pro shared-memory bandwidth of ~70 GB/s for
unmapped CPU copies (the CPU has to read the ANE-produced pages, not
DMA them), 24 MB / 70 GB/s ≈ **0.34 ms of unavoidable CPU stall per
step** — and at the lower 20 GB/s figure for small-burst fp16 copies,
it is **~1.2 ms**. Baseline measured step is 51.7 ms at 19.4 tok/s
(`docs/BASELINE_SPEED_AUDIT.md:60-64`), so `copyBack` costs
**0.7–2.3 % of decode wall-clock** on Mac Studio, almost certainly
more on iPhone (narrower memory bus, shared with LPDDR).

The `copyBack` is only needed because the mlpackage returns fresh
output buffers. Passing `MLPredictionOptions` with
`outputBackings = [name: persistentArray]` would have CoreML write
directly into the same IOSurface — eliminating the memcpy entirely.
This is a strict win with no correctness risk: the persistent buffer
is already the right size and dtype.

**Recommendation R12-A (trivial patch, ≤30 LOC):** set
`MLPredictionOptions.outputBackings` in
`chunk1.prediction(from:options:)` and `chunk2.prediction(from:options:)`
to route `K_sliding_out`, `V_sliding_out`, `K_full_out`, `V_full_out`
directly into the persistent IOSurface KV buffers, and delete the four
`copyBack` calls per chunk. Expected Δ: ~0.5–1.5 ms/step ≈ **+0.2–0.6
tok/s** on-device. Risk: none (API guaranteed since iOS 16).

---

## 3. Constant-foldable inputs

Every chunk currently receives six inputs that are pure functions of
`position` (or entirely static): `causal_mask_full`, `causal_mask_sliding`,
`update_mask`, `cos_s`, `sin_s`, `cos_f`, `sin_f`. They are rebuilt on
every step at `ChunkedEngine.swift:417-423`:

```swift
let maskFull   = try makeCausalMask(position: position, length: ctx)
let maskSliding= try makeSlidingCausalMask(position: position, W: W)
let umask      = try makeUpdateMask(position: position, length: ctx)
let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
let cosF = try lookupRoPE(table: cosFullTable, position: position, dim: 512)
let sinF = try lookupRoPE(table: sinFullTable, position: position, dim: 512)
```

Total per-step work: three fresh mask fills (9 KB of writes) and four
pointer-advance slice allocations out of memory-mapped numpy tables
(3 KB of reads). `docs/BASELINE_SPEED_AUDIT.md:60-64` shows
`mask` ≈ 0.0 ms and `emb` ≈ 0.0 ms — effectively free on Mac Studio.
But every step still materializes **fresh MLMultiArray objects** and
passes them through the ANE input path 4 times each.

### R12-B: bake masks as model constants

The masks are deterministic functions of `position`. All 2 048 possible
`causal_mask_full` vectors fit in 2 048·2 048·2 = **8 MB fp16**. All
2 048 `causal_mask_sliding` vectors fit in 2 048·512·2 = **2 MB**.
All 2 048 `update_mask` vectors fit in 2 048·2 048·2 = **8 MB**.

Baking these as model constants + a single `position` scalar input
would:

1. Remove three input tensors from every chunk → four chunks × three
   inputs × pointer-setup cost, negligible wall-clock but **12 fewer
   `MLFeatureValue` wrappers/step**.
2. Add 18 MB to the mlpackage footprint **once** (post-4-bit palettize
   these tables compress poorly — they're already structured, not
   weight-like — so budget 18 MB actual).
3. Could let the MIL graph constant-fold the mask into the attention
   softmax, which a `gather + select` pattern in the model graph might
   enable.

Trade-off: 18 MB added to each of four mlpackages = 72 MB extra
on-disk, or 18 MB once if the tables become a shared asset loaded by
the runtime. Given the current mlpackage is ~380 MB palettized
(INT4 + fp16 per-channel scales), 72 MB is a **19 % size regression**.

**Verdict on R12-B:** not worth it at ctx=2048 because the mask build
is already ~0 ms. Revisit only if we need to shrink CPU-side prep for
staged pipelining (Phase D1).

### R12-C: bake RoPE cos/sin tables

Same argument, smaller tables:
`cos_f`/`sin_f` at `(2048, 512)` fp16 = 2 MB ea. × 2 = **4 MB**;
`cos_s`/`sin_s` at `(2048, 256)` fp16 = 1 MB ea. × 2 = **2 MB**;
total **6 MB** baked in. The four inputs shrink to one `position`
scalar. Same pattern as masks: negligible wall-clock payoff,
non-trivial size cost.

**Verdict:** skip unless we commit to a "position-only" input
interface for staged pipelining. Not a path to +tok/s on its own.

---

## 4. Per-layer-combined (PLE) routing

`per_layer_combined` is the **only non-KV cross-chunk tensor** that
actually carries information from chunk1 to chunks 2/3/4. Shape
`(1, 1, 8960)` = 17 920 B = **17.5 KB per step**.

It is computed inside chunk1 at `gemma4_swa_chunks.py:224-254` by
`_compute_ple`, then emitted as `per_layer_combined_out`
(`build_verify_chunks.py:213`). Chunks 2-4 consume it unchanged.
Inside each chunk, `_run_layer_swa` slices it per-layer at
`gemma4_swa_chunks.py:168`:

```python
s = layer_idx * config.hidden_size_per_layer_input  # s = layer_idx*256
e = s + config.hidden_size_per_layer_input
per_layer_slice = per_layer_combined[:, :, s:e]
```

The flat 8 960-wide tensor is passed whole to every chunk and indexed
inside. Each chunk only reads 7–10 of the 35 slices (the layers it
owns). Concretely:
- chunk1 reads slices 0-7 = 8·256·2 = **4 KB**
- chunk2 reads slices 8-14 = 7·256·2 = **3.5 KB**
- chunk3 reads slices 15-24 = 10·256·2 = **5 KB**
- chunk4 reads slices 25-34 = 10·256·2 = **5 KB**

### R12-D: pre-slice PLE per chunk

Change chunk2/3/4 input spec to accept `per_layer_combined_part`
shaped `(1, 1, nlayers_in_chunk * 256)` instead of the flat 8 960.
Savings per step:
- chunk2: 17.5 KB → 3.5 KB (-14 KB)
- chunk3: 17.5 KB → 5 KB (-12.5 KB)
- chunk4: 17.5 KB → 5 KB (-12.5 KB)
- Total: **-39 KB/step of cross-chunk PLE traffic**.

That is ~0.1 % of the 48 MB/step KV traffic. Wall-clock impact: well
below the noise floor of the Mac Studio profiler (0.0 ms). **Skip.**

PLE bloats per-function size, not per-step I/O. The 17.5 KB dict-entry
cost is paid through the ANE input path once per chunk; whether it is
17.5 KB or 3.5 KB doesn't change the dispatch latency.

---

## 5. Merged-chunk reality check

`conversion/build_merged_chunks.py:51-52` imports:

```python
from models.gemma4_swa_merged2 import MergedChunk12, MergedChunk34
from models.gemma4_swa_merged1 import MergedChunk1
```

Neither `gemma4_swa_merged1.py` nor `gemma4_swa_merged2.py` exists on
this branch. `ls conversion/models/gemma4_swa_merged*` returns only
`gemma4_swa_merged.py`, which defines `MergedChunk23` (L8-24).

**Finding:** the 2-chunk (MergedChunk12+34) and 1-chunk (MergedChunk1)
variants described in the build script docstring are **unreleased**.
The one merged variant that exists (`MergedChunk23`) merges chunks 2+3
to keep kv14 internal (`gemma4_swa_merged.py:1-8`), but there is no
evidence of it being wired into `ChunkedEngine.swift`'s runtime
dispatch path — grep for `MergedChunk23` / `merged_chunk23` in
`Sources/` returns nothing.

Two planning docs reference the 2-chunk / 1-chunk work as in-progress
(`docs/D1_WIRING_PATCH_PLAN.md`, the branch name
`research/conversion-deep-dive`). The Swift engine has allocation code
that anticipates merged layouts (`ChunkedEngine.swift:479-482` per
subagent's notes), but it isn't reachable.

**R12-E (blocking-prereq):** land the missing Python modules so
`build_merged_chunks.py --mode two` and `--mode one` actually produce
mlpackages. Without those, every downstream optimization that relies
on merged layouts is untestable.

---

## 6. ANE layer-count ceiling

The 4-chunk layout was shaped around an informal "~15 layers per ANE
function" ceiling. Chunk distribution:
- chunk1: **8 layers** (L0-7)
- chunk2: **7 layers** (L8-14)
- chunk3: **10 layers** (L15-24)
- chunk4: **10 layers** (L25-34) + norm + lm_head + argmax

All four fit comfortably under 15. The merged 2-chunk plan would
produce:
- merged_chunk1 (L0-14): **15 layers** — at the boundary
- merged_chunk2 (L15-34 + lm_head): **20 layers** — **past the boundary**

The 1-chunk merged_full (L0-34 + lm_head) is **35 layers** — well
past. No on-device ComputePlanAudit result exists in any doc for
either merged variant. Prior ceiling evidence is indirect:
- chunk3 (10 layers) passes 100 % ANE placement
  (`docs/EXPERIMENTS.md:148-153`).
- The chunking scheme itself implicitly sets the ceiling at 10 —
  chunk3 and chunk4 are the largest shipped chunks, and the split
  points were chosen around ANE stability.

**R12-F:** The 2-chunk merged layout is only guaranteed to work if
merged_chunk2 (20 layers) stays on ANE. If it falls back to GPU or CPU
on iOS 26, the dispatch-halving gain evaporates. ComputePlanAudit must
run **before** any tok/s bench. This is a kill/proceed gate, not an
optimization.

Reasoning about merged_full (35 layers): the *only* upside is
eliminating the kv13/kv14 handoff (4.5 MB/step, ≈0.1 ms at 50 GB/s
bus) and halving dispatch count *again* (2 → 1 dispatch). At 2.3 ms
per ANE dispatch entry fee (`docs/BASELINE_SPEED_AUDIT.md` implied
overhead, consistent with GPU_WHY_FAST.md), 1-chunk saves ~2.3 ms
vs 2-chunk, or ~6.9 ms vs 4-chunk. At current 51.7 ms/step, that is
**+2.9 tok/s** (19.4 → 22.3) if it passes ANE stability. That is the
single biggest lever in this R12 audit — but it requires merged1 to
be written, converted, audited, and benched.

---

## 7. Chunk-boundary choice: is L14/L15 optimal?

The current boundary is natural: L14 is the last own-KV layer (the KV
share boundary is at L15). But from a *wall-clock* balance standpoint,
chunk4 dominates (`docs/BASELINE_SPEED_AUDIT.md:60-64`, 31 % of step
cost vs chunk1's 21 %). The asymmetry is driven by:

1. chunk4 hosts the 256 000-wide `lm_head` + argmax (fp16, baked
   weights), not present in other chunks.
2. chunk4 has no own-KV writes but that saves little — shared-KV
   layers are still 70 % of each attention's work.

**R12-G — 3-chunk asymmetric option (not 4-chunk, not 2-chunk):**
- chunkA: L0-14 (15 layers, own-KV) — boundary at the KV-share line
- chunkB: L15-29 (15 layers, shared-KV)
- chunkC: L30-34 + norm + lm_head (5 layers + head)

Pros: isolates the lm_head into a tiny chunk (chunkC would be mostly
the 250k-wide softmax+argmax), allowing the existing chunk3 (10
layers, 11.8 ms) to be merged into chunkB (15 layers, ~17 ms) while
dropping chunkC to ~10 ms. Total: 3 chunks × ~14 ms = **42 ms/step**
vs 51.7 ms baseline = **+3.9 tok/s**. Requires 15-layer ANE audit to
pass.

Cons: chunkC hosts the lm_head, which is the current chunk4 argmax
CPU fallback. Splitting it further doesn't remove the CPU fallback
unless the argmax is rewritten.

**Verdict:** 3-chunk asymmetric is a dominated strategy — 2-chunk
merged (R12-E + R12-F) delivers the same dispatch-count reduction and
better amortizes the 2.3 ms dispatch entry fee. Only pursue 3-chunk
if 2-chunk's merged_chunk2 (20 layers) fails the ANE audit.

---

## 8. ANE SRAM — not the binding constraint

A28's ANE has 32 MB SRAM and suffers ~30 % throughput degradation when
the working set exceeds it. The Gemma 4 E2B weights, 4-bit palettized,
total ~380 MB — way beyond SRAM at any chunk granularity. The SRAM
therefore operates as a streaming cache, not a resident store.

The *per-layer* working set (input act + Q/K/V temp + attn matrix +
output act) is ~1.5–3 MB, well under SRAM. Chunk sizing is
**dispatch-bound, not SRAM-bound**. This is consistent with
`docs/GPU_WHY_FAST.md`'s observation that ANE latency floor (2.3 ms
per dispatch) dominates decode cost, not compute throughput.

Practical implication: we can merge up to the stability ceiling (15
layers shipped, maybe 20 layers if iOS 26 has loosened the limit)
without hitting any SRAM wall. The only question is whether the ANE
compiler accepts the larger graph.

---

## 9. Ranked recommendations

Gain estimates assume iPhone 17 Pro baseline (32 tok/s from
`docs/MOBILE_2K_COMPETITIVE_PLAN.md`), scaled linearly from Mac Studio's
19.4 tok/s where applicable.

| # | Change | ΔLOC | Risk | Est. Δtok/s (17 Pro) | Notes |
|---|---|---:|:-:|---:|---|
| R12-A | `outputBackings` for chunk1/chunk2 KV outputs, remove `copyBack` | ~40 | low | **+0.5–1.0** | Strict win, iOS 16 API, no correctness risk |
| R12-E + F | Land merged1/merged2 modules, convert + audit merged 2-chunk | ~800 Py + audit | med | **+3–5** | Prereq for R12-F; kill/proceed on merged_chunk2 ANE placement |
| R12-F | Ship merged 2-chunk once R12-E lands & passes audit | Swift wiring | med | **+3–5** | Halves dispatch count; 15-layer ceiling at the limit |
| R12-merged1 | Ship merged 1-chunk (35 layers) if ANE accepts it | Swift | high | **+6–10** | Eliminates kv13/kv14 handoff + 3 dispatches; gated on 35-layer ANE stability |
| R12-G | 3-chunk asymmetric (L0-14 / L15-29 / L30-34+head) | ~400 Py + Swift | med | **+3** | Dominated by R12-F when R12-F works |
| R12-B | Bake masks as model constants | ~200 Py | low | **~0** | 18 MB size regression, no wall-clock payoff at ctx=2048 |
| R12-C | Bake RoPE tables as constants | ~150 Py | low | **~0** | 6 MB size regression, no wall-clock payoff |
| R12-D | Pre-slice PLE per chunk | ~100 Py | low | **~0** | 39 KB/step saved is below noise floor |

### Immediate action (≤1 day each)

1. **R12-A** — wire `outputBackings` in `ChunkedEngine.swift` for the
   eight own-KV outputs. Delete the eight `copyBack` calls. Bench
   before/after with `docs/BASELINE_SPEED_AUDIT.md` methodology.
2. **Correct `docs/EXPERIMENTS.md:34`** — "32 MB of fp16 KV I/O" is
   wrong for ctx=2048. Replace with the measured 4.5 MB (or explain
   what the 32 MB refers to — likely a stale number from a prior
   layout).

### Week-scale work

3. **R12-E** — restore `gemma4_swa_merged1.py` and `gemma4_swa_merged2.py`
   from the worktree (`.claude/worktrees/agent-abf2fc16/…`) into
   `conversion/models/`, make `build_merged_chunks.py --mode two`
   complete. Run ComputePlanAudit on `merged_chunk2.mlpackage`. Kill
   R12-F if merged_chunk2 falls off ANE.
4. **R12-F** — wire the 2-chunk runtime path in `ChunkedEngine.swift`
   (auto-detect merged mlpackages present), bench on iPhone 17 Pro.

### Deferred / conditional

5. **R12-merged1** — only attempt after R12-F ships successfully. The
   35-layer ANE ceiling is unproven on iOS 26; a silent GPU/CPU
   fallback would regress worse than any gain.
6. **R12-G** — only if R12-F fails the ANE audit (i.e. merged_chunk2
   at 20 layers is rejected). The 15/15/5 split stays within the
   confirmed ceiling.

### Rejected

- **R12-B, R12-C, R12-D** — all three optimize tensors whose per-step
  cost is already ≤0 ms on the profiler. Size regressions are not
  justified.

---

## 10. Open questions

- **Exact iOS 26 ANE layer ceiling.** No concrete on-device test of
  merged_chunk2 (20 layers) or merged_full (35 layers) exists.
  ComputePlanAudit on a dummy 20-layer mlpackage would close this,
  independent of the Gemma conversion pipeline.
- **IOSurface on KV *output* path.** Apple's CoreML documentation
  guarantees `outputBackings` works for MLMultiArray but doesn't spell
  out IOSurface behavior. R12-A should validate that the memcpy is
  actually eliminated (not just hidden) by profiling — look for a
  zero-fill pass or an auto-copy in Instruments.
- **Per-layer KV compression.** Out of R12 scope but noted: the full
  KV cache at ctx=2048 is 9 MB (chunk1) + 13 MB (chunk2) = 22 MB
  resident. At ctx=8192 this grows to ~80 MB — still a tiny fraction
  of working memory, but worth tracking if 32K context ships.

---

## Appendix A — verified numbers

| Quantity | Value | Source |
|---|---|---|
| `hidden_size` | 1536 | `gemma4.py:39` |
| `num_attention_heads` | 8 | `gemma4.py:41` |
| `num_key_value_heads` | 1 | `gemma4.py:42` |
| `head_dim` (sliding) | 256 | `gemma4.py:43` |
| `global_head_dim` (full) | 512 | `gemma4.py:44` |
| `sliding_window` | 512 | `gemma4.py:54` |
| `hidden_size_per_layer_input` | 256 | `gemma4.py:58` |
| `nlayers` | 35 | model config |
| Default `CTX` | 2048 | `build_verify_chunks.py:110-111` |
| Baseline tok/s (Mac Studio) | 19.4 | `BASELINE_SPEED_AUDIT.md:64` |
| c1/c2/c3/c4 ms | 11.0 / 12.8 / 11.8 / 16.1 | `BASELINE_SPEED_AUDIT.md:64` |
| chunk1 memcpy volume/step | 11.0 MB | §2 calc |
| chunk2 memcpy volume/step | 13.0 MB | §2 calc |
| kv13_k/v size | 256 KB each | `build_verify_chunks.py:377-378` |
| kv14_k/v size | 2 MB each | `build_verify_chunks.py:379-380` |
| `outputBacking` usages in repo | 0 | `grep -r outputBacking Sources/` |
