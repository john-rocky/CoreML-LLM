# ANE Empirical Guide — consolidated reverse-engineering findings for CoreML-LLM

**Last updated**: 2026-04-15.
**Purpose**: aggregate every concrete ANE measurement from published reverse-engineering work into actionable project items. Where a finding is already in `docs/GPU_WHY_FAST.md` or `docs/FUNDAMENTAL_UNTRIED.md` we only reference it; this document is the **delta** plus the tables/gotchas that are worth having in one place.

Primary sources:
- [Orion (Kumaresan, arXiv 2603.06728, Mar 2026)](https://arxiv.org/abs/2603.06728) — first public hardware characterization of Apple's Neural Engine; 20-constraint catalog; ~27-op IR; GPT-2 124M @ 170 tok/s on M4 Max.
- [mechramc/Orion](https://github.com/mechramc/Orion) — supplementary code + constraint details.
- [maderix "Inside the M4 ANE" Parts 1-2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) — independent M4 microbenchmarks; M4 H16G codename; graph-depth scaling curve.
- [maderix/ANE](https://github.com/maderix/ANE) — training benchmarks; W8A8 1.88x claim (disputed, see §6).
- [Apple ML — "Deploying Transformers on the Apple Neural Engine"](https://machinelearning.apple.com/research/neural-engine-transformers) — Apple's only concrete layout guidance.
- [Apple ML — "Core ML on-device Llama"](https://machinelearning.apple.com/research/core-ml-on-device-llama) — Llama 3.1 KV-cache stateful recipe.
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — community-maintained op/gotcha wiki (pre-Orion).

---

## 1. Executive summary — top 5 empirical findings we can act on

Ranked by how much they change our current plan, not by novelty:

1. **The "(B, C, 1, S) + 64-byte last-axis alignment" constraint is the single most under-exploited Apple-documented rule.** Apple states the last axis of any ANE IOSurface "must be contiguous and aligned to 64 bytes." Mis-alignment costs **32× in FP16, 64× in INT8** memory traffic due to padding. Our KV tensors are `(1, num_kv*n_rep, ctx, head_dim)` where `head_dim=256` already aligns (256 × 2 B = 512 B = 8 × 64 B), but **hidden-state activations** flowing between chunks are `(1, 2560, 1, S)` — for S=1 this is 5,120 B (80 × 64 B) — safe. For `S` that isn't a multiple of 32 (e.g. odd prefill chunks, current test harness uses S=7) we silently pay padding. **Action**: instrument prefillN alignment; round to next multiple of 32 to satisfy the 64-byte = 32 fp16 element rule.
2. **ANE IR has only ~27 ops. Anything outside that list is either rejected or silently offloaded.** The Orion paper's Table 4 lists: `input, const, identity, conv1x1, matmul, add, sub, mul, neg, relu, tanh, sigmoid, exp, pow, sqrt, rsqrt, reduce_{sum,mean,max}, reshape, transpose, split, pad, slice, cast, softmax` (concat is in the list but rejected in practice). **GELU is not a valid MIL op on the ANE path** (constraint #10) — must decompose to tanh approximation. Our `gemma4` exports use `F.gelu(approximate="tanh")`; verify this survives the MIL lowering and does not lose the tanh approximation hint.
3. **Matmul is genuinely 3× slower than Conv1×1, and this is a datapath choice in silicon, not a compiler quirk.** Orion constraint #17 is explicit. Our conversion pipeline already does this for most projections via `nn.Conv2d` on the `(B,C,1,S)` layout, but: **the attention `Q @ K^T` and `attn @ V` are still matmul-shaped in our MIL**. Rewriting them as batched Conv1×1 over head-major layout would reclaim up to 3× on attention compute. This is **the largest single-fix lever still open** on our pipeline.
4. **The 32 MB SRAM cliff applies at the working-set level, not the model-size level.** Orion footnote 3: "throughput drops ~30% when working sets exceed 32 MB SRAM". For us the working set at 8K ctx is dominated not by weights (int4 palettized, ~120 MB total resident but streamed) but by the **per-layer Q·K^T intermediate and the fp16 attention scores matrix**. At `seq_len=512, num_heads=8, head_dim=256, ctx=8192`: scores = `8 × 512 × 8192 × 2 B = 64 MB` — **2× over the cliff**. This suggests our current prefill is already clipped. **Action**: verify with flash-style tiling (which we have in `gemma4_swa_flash.py`) that the rolling tile stays ≤ 24 MB and re-run the chunk-2 prefill benchmark.
5. **Per-dispatch overhead on M4 is ~0.095 ms XPC+IOKit + ~2.3 ms on the first cold IOSurface materialisation, then ~0.5 ms warm.** Our current 4-chunk pipeline pays 4 × 2.3 ms = 9.2 ms on the first decode step and 4 × 0.5 ms = 2.0 ms thereafter. That 2 ms floor is ~30 tok/s just from dispatch, before any compute. **Merging to 2 chunks (already scripted as `build_merged_chunks.py`) brings this to 2 × 0.5 = 1.0 ms → 60 tok/s dispatch ceiling** — which is exactly the regime that decides whether we beat LiteRT 56.5 tok/s.

---

## 2. Orion findings relevant to us (new, not already in GPU_WHY_FAST.md)

### 2.1 Hardware parameters (Table 1, §2.1)

| Parameter | M4 Max (H16G) | Notes |
|---|---|---|
| Cores | 16 | A11 started at 2; scaling is linear in chip area. |
| Peak FP16 | 19 TFLOPS measured | Apple's 38 TOPS spec assumes INT8 + the (incorrect) 2× INT8 multiplier. |
| Eval queue depth | **127** | Public CoreML can't queue past 1 — massive headroom left on floor. |
| Compilations / process | **~119** | Hit this, the process must exec() to reset. Explains Xcode-Instruments "ANE-reset" blips. |
| Min IOSurface | **~49 KB** | Sub-49-KB tensors silently pad. Single-token activations `(1,2560,1,1) = 5 KB` → padded 10×. |
| On-chip SRAM | 32 MB | 30 % throughput cliff when exceeded. |
| Idle power | 0 mW (hard gate) | **Every cold dispatch pays FSM wake.** No user-visible keep-alive signal. |
| Per-core FMAs | Not disclosed by Orion; maderix Part 2 implies 256 FMAs/core based on 19 TF / 16 cores / 3 GHz | Treat as lower-bound. |
| Line size / associativity | Undisclosed | Empirical: 64-byte last-axis alignment matches a 64 B line. |

### 2.2 The 20 documented constraints (Table 3) — map to our conversion scripts

| # | Constraint | Affects us? | Where in our tree |
|---|---|---|---|
| 1 | `concat` rejected by compiler | Yes | `conversion/gemma4_stateless_chunks.py` uses `torch.cat` in KV build — check MIL lowering. |
| 2 | Multi-output buffers require uniform sizes | Yes | Our chunk1 outputs kv13 (sliding) + kv14 (full) + hidden — uneven sizes, likely padded internally. |
| 3 | Outputs ordered **alphabetically by MIL name** | **High risk** | Our Swift runtime reads outputs by **index**, not by name. A rename could silently swap tensors. |
| 4 | Minimum ~49 KB IOSurface | Yes | Single-token decode activations are below 49 KB. |
| 5 | ~119 compilations per process | Low | We compile once at install; user-visible only on adapter hot-swap. |
| 6 | SDPA causal masks **silently ignored** | **High risk** | If we ever adopted `F.scaled_dot_product_attention` with `is_causal=True`, the mask would vanish. We currently materialise the mask by hand, so OK. |
| 7 | Weights baked at compile | Yes | Relevant to LoRA plans. Non-App-Store to work around. |
| 8 | BLOBFILE offset is uint64(64) | Low | Hidden behind coremltools; don't touch. |
| 9 | MIL must be NSData* | Low | Internal. |
| 10 | GELU requires tanh approximation | Yes | Gemma 4 uses `gelu(approximate="tanh")` — confirm lowering. |
| 11 | Empty weight dict must be `@{}` | Low | Internal. |
| 12 | Matmul transpose flags need named constants | Low | coremltools handles. |
| 13 | Convolution doesn't support bias | Yes | Our `nn.Conv2d(bias=True)` MUST lower to conv + separate `add`. coremltools does this by default; confirm no bias parameter survives post-optimization. |
| 14 | Output vars must reference live post-optimization nodes | Low | coremltools handles. |
| 15 | exec() restart overhead ~50 ms | Medium | On process cold-start; hidden behind app launch. |
| 16 | 32K-channel convolutions **rejected** | Yes — relevant to lm_head | Our lm_head projects to 256K vocab. We split it into 8 × 32K along output channels already; verify no single conv exceeds 32K. |
| 17 | Conv1×1 is 3× faster than matmul | **Yes — biggest open lever** | Attention QK^T and attn@V still matmul-shaped. |
| 18 | Multi-input surfaces require uniform alloc sizes | Yes | Our two KV inputs have different sizes; they're already padded. Verify the waste. |
| 19 | Inputs ordered **alphabetically by parameter name** | **High risk** | Same as #3 — our Swift code indexes inputs positionally. |
| 20 | ANE reads flat buffer as packed `[1,C,1,S]` from byte 0 | Yes | If we ever used a non-[1,C,1,S] layout, it would silently misread. |

**Action items**:
- Add a unit test that reads back input/output names, alphabetises them, and confirms the Swift runtime's positional indexing matches.
- Grep our converter for `concat` → audit each one.
- Grep for any `nn.Linear` that hasn't been converted to `nn.Conv2d` along the (B,C,1,S) layout.

### 2.3 Op-level latency (Table 11, §8.4)

Orion reports, on M4 Max:

| Op | CPU | ANE | Speedup |
|---|---|---|---|
| Classifier forward (vocab 32K) | 10.77 ms | 1.06 ms | 10.2× |
| Softmax (vocab 32K) | 81.11 ms | 2.40 ms | 33.8× |
| RMSNorm backward | (near-parity) | ~1× | — |
| ANE decode / token (GPT-2 124M) | — | 4.55 ms | (= 220 tok/s ceiling, before they layer speculation) |
| Prefill cached | — | 5.78 ms | — |

**Implication for us**: classifier/softmax move *to* the ANE, not off. Our existing plan of offloading lm_head to GPU is based on Metal's 256K-vocab dispatch efficiency, **not** because ANE is slow at the op — the 256K output *channels* trip constraint #16 first. If we split vocab across 8 chunks, ANE wins.

### 2.4 Power FSM

Orion explicitly records "0 mW idle, hard power-gated" (Table 1). There is **no published wake-up latency** but the ~2.3 ms cold-IOSurface cost on the first dispatch absorbs it. **Consequence**: any scheme that lets the ANE idle between decode tokens (async prefetch, user typing pause) resets to cold. The CoreML `MLModel.warmup()` hint only primes one prediction; it does **not** hold ANE power.

**Action**: keep a "heartbeat" low-cost prediction running every ~50 ms during active sessions to prevent cold re-entry. Negligible power vs full cold re-wake.

### 2.5 Graph depth utilization curve

From maderix Part 2 + Orion §5.3:

| Dispatch depth (ops) | Measured utilisation |
|---|---|
| 1 | ~30 % |
| 4–8 | ~55 % |
| 16 | ~80 % |
| 32–64 | **~94 %** (plateau) |
| 128+ | plateau, MIL compiler starts hitting undocumented depth caps |

"Ops" here means MIL IR nodes in one submitted program, not transformer layers.

**Our numbers**: Gemma 4 E2B has 35 transformer layers × ~12 MIL ops/layer ≈ **420 ops total**. With a 4-chunk split that's ~105 ops/chunk — deep in the plateau regime. With a 2-chunk split (our `build_merged_chunks.py`) that's ~210 ops/chunk — also safe. **The graph-depth floor is not our current bottleneck**; dispatch count is.

---

## 3. maderix findings relevant to us (M4-specific, 2025-12)

maderix's two-part substack is the most concrete independent measurement of M4 ANE and mostly agrees with Orion. Unique contributions:

- **M4 compilation cost**: first compile 20–40 ms, cache hits effectively instant (matches our observation that an app launch → first decode has a visible "warm-up" step).
- **E5 binary size**: 1024×1024 matmul compiles to 2,688 bytes; 128×128 compiles to 2,680 bytes. **The size is almost independent of problem dimensions**, which strongly implies the compiled binary is a descriptor over a fixed kernel library, not generated code. This matches Orion's "weights baked separately from program" finding and is why **delta compilation** works.
- **IOSurface concrete layout**: a 1024×1024 float tensor has 1024 px width, 1024 px height, 4 bytes/elem, 4096 bytes/row, 4,194,304 bytes total. Our LLM-shaped tensors (1, C, 1, S) degenerate to height=1, which maximises stride efficiency.
- **INT8 W8A8 throughput claim — disputed**: maderix's `maderix/ANE` repo claims "18.6 TOPS FP16 → 35.1 TOPS INT8 W8A8" on a 128×512ch conv, a **1.88× speedup**. Orion directly contradicts: "INT8 and FP16 deliver nearly identical throughput." The contradiction is resolved by noticing that maderix's W8A8 config uses **native INT8 MAC** on a conv path Apple's compiler does NOT target for CoreML (it's the private-API training path). Via CoreML + the public iPhone compiler, INT8 dequantizes to FP16 and there is no speedup — matching both Orion's claim and our own `SPEED_8K.md` §1 A2 null result.
- **No specific byte-alignment numbers** beyond what Apple's transformer blog already states (64-byte last axis).

**Action items**:
- Implement delta compilation if we ever adopt adapter hot-swap. Reference: `docs/ANE_NEW_TECHNIQUES_2026.md §2.6` already triaged this as "research-only" for us.
- Ignore the 1.88× W8A8 claim as an achievable target via CoreML; treat it as a private-API-only result.

---

## 4. Apple-published guidance relevant to us (older but still canonical)

### 4.1 "Deploying Transformers on the Apple Neural Engine" (2022)

Still the only Apple-sanctioned ANE optimisation blog. Key rules:

- **Layout**: (B, C, 1, S) — channels-first 4D. Not (B, S, C).
- **Linear → Conv2d**: mandatory. We already do this.
- **Last-axis 64-byte alignment**. Non-negotiable for fast path.
- **Einsum pattern**: `"bchq,bkhc->bkhq"` maps directly to hardware. Using `transpose + matmul + transpose` costs extra dispatches.
- **Attention**: split Q/K/V by head, run per-head attention as independent conv1×1 sequences, re-concat at the end (but without using the `concat` op — use a multi-output buffer; see Orion constraint #1).
- **distilbert / iPhone 13** baseline: 3.47 ms, 0.454 W, 10× faster than unoptimised.

**Our status**: current `conversion/models/gemma4.py` uses `nn.Conv2d` for projections but **attention uses `torch.matmul`**, which goes through MIL `matmul` rather than the einsum path. Rewriting the attention with einsum following Apple's exact recipe has been on the to-do list since `ANE_CONVERSION_RECIPE_2026.md §5`; Orion constraint #17 re-prioritises it.

### 4.2 "Core ML on-device Llama 3.1" (2024)

Concrete numbers for Llama 3.1 8B on M1 Max:

- **Baseline FP16 (2048 ctx)**: 0.19 tok/s
- **+ KV cache as `MLState`**: 16.26 tok/s (**85×**)
- **+ INT4 block-wise palettization**: 33.67 tok/s

KV structure: `(32, 1, 8, 2048, 128)` = (layers, batch, kv_heads, seq, head_dim). **This matches our `(1, num_kv*n_rep, ctx, head_dim)` layout after squashing layers into separate chunks — we're aligned with Apple's recipe.**

Attention mask: `(B, 1, Q, S)` with `-inf` upper-triangular. **During decode the mask is all-zero** (no future tokens). Our Swift runtime builds a full mask every step even when unnecessary; that's wasted work on warm decode.

### 4.3 WWDC 24/25 sessions

- WWDC 24 (Session 10161 "Deploy machine learning and AI models on-device with Core ML"): introduced `MLState`. Confirms 13× improvement for Llama-class KV.
- WWDC 25 (Session 10222 "Bring your trained ML and AI models to Apple silicon"): quiet year; recommends `coremltools 9` opt-passes but no new ANE-internal changes.
- WWDC 26 (if held June 2026): watch for rumoured "Core AI" framework that could expose the 127-deep eval queue. Currently vapour.

---

## 5. Op-level ANE placement atlas (consolidated)

This is the best current table for deciding "ANE, GPU, or CPU?" per op in our transformer. Combines Orion Table 4, Apple transformer blog, and our measured results.

| Op | ANE fast? | CoreML lowering | Notes |
|---|---|---|---|
| `conv1x1` / `Conv2d(1,1)` | ✅ **fast-path** | `conv` | The native matmul-equivalent. |
| `matmul` | ⚠️ 3× penalty vs conv1x1 | `matmul` | Use einsum → conv1x1 for projections. |
| `linear / nn.Linear` | ❌ via matmul → penalty | `linear → matmul` | Replace with Conv2d. |
| `Conv2d` with `bias=True` | ✅ (compiler splits) | `conv + add` | No runtime cost beyond the add dispatch. |
| `softmax` | ✅ (very fast) | `softmax` | 33× over CPU. Use on-ANE vocab softmax when channels <32K. |
| `layer_norm` | ✅ | `layer_norm` | Dedicated op. |
| `rms_norm` (Gemma) | ✅ | `rms_norm` (iOS 18+) | Newer op; older coremltools decomposes it. |
| `gelu` (exact) | ❌ rejected | — | Constraint #10. |
| `gelu(approximate="tanh")` | ✅ decomposed | `tanh + mul + add` | Our path. |
| `silu / swish` | ✅ decomposed | `sigmoid + mul` | 2-op ANE. |
| `scaled_dot_product_attention(is_causal=True)` | ❌ **mask ignored** | silent compiler bug | Constraint #6. Build mask by hand. |
| `scaled_dot_product_attention(attn_mask=...)` | ✅ (iOS 18+) | `sdpa` fused | Our preferred path. |
| `concat` | ❌ rejected | silent compiler failure | Constraint #1. Use multi-output. |
| `split` | ✅ | `split` | Cheap — slice-view, no copy. |
| `reshape` | ✅ (if alignment preserved) | `reshape` | Free when last-axis alignment holds; costly if it changes. |
| `transpose` | ⚠️ can trigger repack | `transpose` | Apple says "at most one per attention block." |
| `gather` (embedding lookup) | ❌ often CPU fallback | `gather` | We run embed on CPU already. |
| `scatter` / KV-update | ⚠️ supports but unstable | `slice_update` preferred | Prefer `MLState` + `slice_update_along_axis`. |
| `top_k` / `argmax` | ✅ | `topk / argmax` | Apply in-model for 256K vocab; fan out to 8×32K channels to avoid constraint #16. |
| `rope (sin/cos rotation)` | ✅ decomposes to mul/add | — | Decompose by hand; the complex-number torch op does not lower. |
| `cast fp16↔fp32` | ⚠️ implicit penalties | `cast` | Keep everything fp16 end-to-end. |
| Quantize / dequantize (INT4/INT8 weights) | ✅ palettization path | `constexpr_lut_to_dense` | Free on the weight side; activation dequant is NOT free. |
| `quantize_activations` INT8 | ❌ often rejected | `quantize/dequantize` | Our W8A8 repeatedly failed to compile. |
| Dynamic shape | ❌ recompile | — | 4.2 s cold recompile; unacceptable. |

---

## 6. Gotchas (ordered by how badly they bite)

1. **Input/output ordering is alphabetical by MIL variable name, not Python return order.** (Orion #3, #19.) If you rename a tensor from `hidden_out_final` to `final_hidden_out` in the next export, the Swift runtime reading by index will load the wrong tensor with zero warning. Add a regression test that verifies `MLModel.modelDescription.inputDescriptionsByName` ordering matches the runtime assumption.
2. **`F.scaled_dot_product_attention(is_causal=True)` silently emits zero mask on ANE.** (Orion #6.) If someone refactors attention to use the high-level helper, outputs become garbage after the first verification. **Never use `is_causal=True`**; always pass an explicit `attn_mask`.
3. **GELU exact = silent compiler failure.** (Orion #10.) Always `approximate="tanh"`.
4. **32K-channel conv = rejected.** (Orion #16.) 256K-vocab lm_head must be split.
5. **Concat op = rejected.** (Orion #1.) Use multi-output subgraphs instead. Our current `build_merged_chunks.py` uses `torch.cat` on KV tensors; verify the MIL emits multi-output and not `concat`.
6. **Dynamic shapes = full 4.2 s recompile.** Lock all `enumerated_shapes` at convert time. Our conversion already does this; don't regress.
7. **First-token IOSurface allocation is ~2.3 ms cold.** Pre-warm all buffers at model-load time, not on the first token.
8. **`MLState` + ANE + iOS<18 is silently rejected.** Our minimum target is iOS 18 for the MLState path, iOS 17 fallback uses explicit KV I/O. Keep the two paths in sync.
9. **~119 compiles/process.** Irrelevant in production but trips researchers doing sweep runs. Use a subprocess-per-config harness.
10. **Models with >~210 MIL ops per function start hitting undocumented depth caps.** (Orion: 14 new limits beyond the 6 documented.) Our 2-chunk merged plan pushes ~210 ops/chunk; verify compilation succeeds on iOS 26 target before committing.
11. **Last-axis 64-byte alignment violations silently pad.** `(1, 2560, 1, 5) → 5 × 2 B = 10 B last axis → padded to 64 B` is a 6.4× waste per token. Always round `S` to a multiple of 32 for fp16 or 64 for int8.

---

## 7. Sweet-spot parameters for CoreML-LLM on Gemma 4 E2B

Calibrated to iPhone 17 Pro (A19 Pro) with Orion's M4 numbers treated as an upper bound for A-series (A-series ANE is ~half the core count of M4; use 0.7–0.8× for compute, same for dispatch overhead).

| Parameter | Current value | Orion-informed target | Reasoning |
|---|---|---|---|
| Chunks per decode step | 4 | **2** | Dispatch overhead scales linearly; 2 chunks → ~60 tok/s dispatch ceiling. Already scripted in `build_merged_chunks.py`. |
| MIL ops per chunk | ~105 (4-chunk) | 150–250 (2-chunk) | Plateau of 94% util at 32–64+; no upside past ~150 unless we fuse KV update. |
| prefillN (token count per prefill chunk) | 512 | 512 (keep) or 256 if working set > 24 MB | Working set = `num_heads × prefillN × ctx × 2 B`. For Gemma 4: `8 × 512 × 8192 × 2 = 64 MB` — 2× over cliff. Flash tiling brings tile to 8 MB; keep. |
| Last-axis alignment | unchecked | multiple of 32 fp16 elements | 64-byte rule. |
| KV layout | `(1, num_kv*n_rep, ctx, head_dim)` | same | Matches Apple Llama 3.1 recipe. |
| Attention mask on decode | full (B,1,1,S) rebuilt | zero tensor, reuse | No future tokens to mask — wasted host work. |
| Stateful KV (MLState) | not adopted | **adopt** | Apple Llama 3.1 reports 85× speedup when KV moves to state. Priority 1 remaining lever. |
| lm_head vocab split | 8 × 32K | 8 × 32K (keep) | Already under constraint #16. |
| Embedding lookup | CPU | CPU (keep) | Gather is ANE-slow. |
| Sampling | CPU | GPU or ANE top-K | 256K softmax is fine on ANE when split 8× channel-wise. |
| GELU form | `tanh` approx | same | Mandatory. |
| Attention kernel | matmul + softmax + matmul | **einsum-shaped conv1×1** | Reclaim 3× on QK^T and attn@V. Largest open lever. |

---

## 8. Ongoing measurement plan

What we must instrument on device to **validate** each of the claims above.

### 8.1 Per-chunk dispatch latency

Add `os_signpost` around each `chunk.prediction()` call in `Sources/CoreMLLLM/ChunkedEngine.swift`. Target: confirm ~2.3 ms cold / ~0.5 ms warm per chunk. Log to a CSV for post-hoc analysis. Existing infrastructure: `Sources/CoreMLLLM/PerfLogger.swift` already wraps `os.signpost`; extend it.

### 8.2 SRAM working-set probe

In `conversion/`, add a script that walks the MIL graph, computes per-op intermediate tensor sizes, and reports the running max. Flag any op whose live-set exceeds 24 MB (75 % of 32 MB, leaving headroom for weights). This is Orion's "working-set" heuristic, formalised for our graph.

### 8.3 Op-placement audit

`MLComputePlan.forModelStructure` (iOS 18+) returns per-op device target. Script: load each of our 4 chunks, dump per-op target, count `.neuralEngine` vs `.cpu` vs `.gpu`. Any op on CPU/GPU is a candidate for rewrite (or a knowingly-offloaded op like embed/sampling). Reference: `docs/ANE_NEW_TECHNIQUES_2026.md §2.8` already triaged `MLComputePlan`.

### 8.4 Cold vs warm characterisation

Run the same decode step 20 times in sequence (after a 60 s idle) and record per-step wall time. Expected: step 1 ≫ steps 2–20; the delta is the cold dispatch + FSM wake. If step 2 is also elevated, we are getting ANE power-gated mid-sequence (matches §2.4 concern).

### 8.5 Alignment audit

For each input/output of each chunk: print `(shape, dtype, last-axis-bytes, last-axis-bytes mod 64)`. Anything non-zero mod 64 is paying padding. Can run offline with `coremltools.models.MLModel(..).get_spec()` inspection.

### 8.6 Constraint-regression test

A CI test that loads each `.mlpackage`, enumerates all MIL ops, and asserts none is in the "rejected" list (concat, gelu-exact, conv with 32K+ channels, matmul that could be conv1×1, quantize_activations). If the converter regresses, the test fails loudly.

---

## 9. References

Ordered by how often this project cites them.

1. [Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference (arXiv 2603.06728, Kumaresan, Mar 2026)](https://arxiv.org/abs/2603.06728)
2. [Orion HTML rendering on arXiv](https://arxiv.org/html/2603.06728v1)
3. [mechramc/Orion — code release](https://github.com/mechramc/Orion)
4. [Inside the M4 Apple Neural Engine Part 1 — maderix.substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
5. [Inside the M4 Apple Neural Engine Part 2 — maderix.substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
6. [maderix/ANE — training-on-ANE code release](https://github.com/maderix/ANE)
7. [Deploying Transformers on the Apple Neural Engine — Apple ML Research (2022)](https://machinelearning.apple.com/research/neural-engine-transformers)
8. [Core ML on-device Llama 3.1 — Apple ML Research (2024)](https://machinelearning.apple.com/research/core-ml-on-device-llama)
9. [hollance/neural-engine — community wiki](https://github.com/hollance/neural-engine)
10. [philipturner/metal-benchmarks — GPU microarchitecture (cross-reference for hybrid path)](https://github.com/philipturner/metal-benchmarks)
11. [Apple — MLState documentation](https://developer.apple.com/documentation/coreml/mlstate)
12. [coremltools — Stateful Models guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
13. [ANEMLL — Gemma 3 conversion recipe](https://github.com/Anemll/Anemll)
14. [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli)

Internal cross-refs (to avoid duplication when reading this doc):

- `docs/GPU_WHY_FAST.md §5.3` — ANE dispatch anatomy.
- `docs/FUNDAMENTAL_UNTRIED.md §0` — the 0.07 %-of-peak reframing and dispatch-count bottleneck.
- `docs/ANE_NEW_TECHNIQUES_2026.md` — technique-level backlog (multifunction, delta compilation, argmax-in-model, etc).
- `docs/UNEXPLORED_SOURCES.md §A.6` — original SRAM 32-MB cliff note.
- `docs/SPEED_8K.md §1` — W8A8 and INT8 KV null results (confirms Orion's INT8=FP16 finding on our hardware).
- `docs/CHUNK_CONSOLIDATION_BENCH.md` — prior measurements of the 4-chunk dispatch floor.
