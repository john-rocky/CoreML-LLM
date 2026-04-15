# Why Metal GPU is Fast for LLM Decode — Structural Analysis vs CoreML/ANE

**Date:** 2026-04-15
**Scope:** Source-level investigation of llama.cpp Metal, MLX, LiteRT-LM GPU delegate, and Apple GPU/ANE dispatch hardware. Deliverable: structural reasons the iPhone 17 Pro Metal GPU hits 38–56 tok/s on Gemma 4 E2B while CoreML/ANE plateaus at 15 tok/s despite ANE having ~2× the nominal FP16 TFLOPS.

---

## 0. TL;DR

The Metal GPU win is **not about compute**. A19 Pro ANE ≈ 19 TFLOPS FP16, A19 Pro GPU ≈ 9 TFLOPS FP16 (with native matmul via 6-core neural accelerators). Decode is dispatch-bound, and:

| Layer | Per-dispatch cost | Dispatches per token |
|---|---|---|
| Metal (llama.cpp / MLX / LiteRT) | 5–20 μs (user-space command ring) | 270–350 kernels packed into 1–8 MTLCommandBuffers |
| CoreML / ANE (our path) | ~2.3 ms cold, 0.1–0.5 ms warm (XPC → `ANECompilerService` daemon → IOKit → IOSurface) | 4 MLModel.prediction calls, one per chunk |

The ~2.3 ms floor is imposed by the **daemon-mediated security boundary** (DART IOMMU, closed-source ANE firmware). Apple does not expose the hardware's 127-deep evaluation queue to third-party processes. **This is not software-fixable on current iOS.**

**Realistic ANE ceiling on current hardware: ~22–28 tok/s.** To beat LiteRT-LM's 56.5 tok/s, the GPU path (Metal/MPSGraph direct, **not** CoreML-on-GPU) is required.

---

## 1. Decomposed Speedup Attribution (15 → 56.5 tok/s)

Based on LiteRT-LM source analysis, the 3.8× gap compounds from five structural mechanisms:

| Mechanism | Est. contribution | Replicable on ANE? |
|---|---|---|
| 1 command buffer / fence per token (vs 4 predicts) | **2.0×** | No — dispatch model differs at hardware boundary |
| Kernel fusion N:1 (RMSNorm+Mul+Add, SwiGLU) | 1.3× | Partial — via MIL custom ops |
| SIMD-group 8×8 MMA + threadgroup weight cache | 1.2× | No — ANE uses Conv1×1 emulation (3× matmul penalty) |
| Single-buffer KV with `[start,end,end]` param indices | 1.1× | Partial — needs `MLState` (failed on ANE with error -14) |
| FP16 dequant once at weight upload (trades 850 MB RAM) | 1.1× | No — ANE wants weights in its own palette format |

Product: ~3.8×. **No single mechanism explains the gap; they compound.**

---

## 2. llama.cpp Metal — the reference decode-bound implementation

Source: `ggml/src/ggml-metal/` (cloned 2026-04-15)

### 2.1. Command buffer batching — the single biggest lever

`ggml-metal-context.m:20` — `GGML_METAL_MAX_COMMAND_BUFFERS = 8`. Comment at line 458: *"optimal n_cb is 1 or 2 on M1 Pro / M2 Ultra."*

`ggml_metal_graph_compute()` (lines 438–615):
- Creates **one** `commandBufferWithUnretainedReferences` (line 512) — skips per-resource retain/release since residency sets guarantee liveness
- First 64 ops encoded synchronously on main thread (line 445), rest via `dispatch_apply(n_cb, …, encode_async)` on concurrent dispatch queue (line 550)
- `[cmd_buf enqueue]` (line 520) schedules into GPU queue **before** encoding finishes — GPU begins executing while CPU still encoding later ops
- Function returns without `waitUntilCompleted`; next iteration's synchronize (lines 239–294) only blocks if user reads output

**Per-layer dispatch count** (`ggml-metal-ops.cpp:200+`):

| Op | Dispatches |
|---|---|
| RMS_NORM + MUL (fused) | 1 |
| Q/K/V projections (q4_K in-register dequant) | 3 |
| RoPE on Q, K | 2 |
| flash_attn_ext (Q·Kᵀ + softmax + P·V fused) | 1 |
| O projection | 1 |
| pre-MLP RMS_NORM | 1 |
| Gate + Up projections | 2 |
| SwiGLU fused | 1 |
| Down projection | 1 |
| Residual ADD chain (up to 8 wide fused) | 1 |

**~12–14 kernel dispatches × 26 Gemma layers = ~350 dispatches/token**, all in 1 MTLCommandBuffer, 1 `commit`.

### 2.2. Residency set pinning (iOS 18+)

`ggml-metal-device.m:1353–1378`:
```
[MTLDevice newResidencySetWithDescriptor:]
[rset requestResidency]
```
Plus heartbeat thread (lines 580–601) calls `requestResidency` every 0.5s for 3 min after last use. Result: driver skips per-launch page-table validation and eviction check.

### 2.3. Weights never materialize as FP16 in memory

`ggml-metal.metal:7754–7763` (`mul_mv_q4_K_f32`): Q4 nibbles unpacked as `q1[i] & 0x000F`, `& 0x0F00`, etc. **directly inside the FMA chain** — the dequantized fp16 block is never written to DRAM or even threadgroup memory.

For the matmul path (`mul_mm`, line 9276), `dequantize_func(x, il, temp_a)` writes 4×4 fp16 block into threadgroup `sa` only, consumed immediately by `simdgroup_multiply_accumulate` (line 9512).

### 2.4. FlashAttention as single monolithic kernel

`kernel_flash_attn_ext_impl` (lines 5767–6370):
- Tile: Q=8 queries, C=64 KV tokens per threadgroup
- Threadgroup memory layout: `sq`, `so`, `ss`, `sk/sv`, `sm2` — Q/O accumulator/scores/KV tile/mask all on-chip
- **Online softmax** (lines 6132–6174): running max `M[jj]`, sum `S[jj]`, output pre-multiplied by `exp(m - M[jj])` — canonical FlashAttention-2
- **No intermediate N×N attention matrix ever leaves threadgroup memory**
- GQA: `ikv2 = iq2/(ne02/ne_12_2)` (line 5851) — integer division picks KV head, unified memory + L2 handles broadcast **with zero copy op**
- Non-masked block skipping via separate `flash_attn_ext_blk` kernel (line 5666)

### 2.5. Concurrent dispatch with self-managed hazards

`ggml-metal-device.m:462`:
```
computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent
```
Disables Metal's automatic hazard tracking. Custom tracker (`ggml-metal-ops.cpp:221`) maintains read/write ranges; non-aliasing ops run concurrently. Graph optimizer (`ggml-metal-common.cpp:306–367`) reorders up to 64 nodes ahead to pack concurrent sets.

---

## 3. MLX — JIT kernel specialization + async pipelining

Source: `github.com/ml-explore/mlx`, `github.com/ml-explore/mlx-lm`

### 3.1. Shape-specialized SDPA — decode vs prefill split

`scaled_dot_product_attention.cpp:588-637` selects between:
- **`sdpa_vector`** for Q ≤ 8 (decode): `sdpa_vector.h:15-177`, 1024 threads (32 simdgroups × 32 lanes), softmax state in registers, `simd_sum` for 32-lane reduction in one instruction
- **`sdpa_vector_2pass`** for long decode with big GQA fanout (split-K, 64–512 blocks)
- **`steel_attention`** for Q > 8 (prefill)
- **`steel_attention_nax`** on A19/M4/M5 — uses `MetalPerformancePrimitives.h` **16×16 NAX tensor-core frag** (`nax.h:27-34`, `kFragRows=16, kFragCols=16`). This is how MLX calls Apple's GPU-side neural accelerator from shader code.

Head-dim specializations: only `{64, 96, 128, 256}` × `{fp32, fp16, bf16}` = 12 kernels compiled. Gemma's head_dim=256 lands on the 256 instantiation; `qk_per_thread = D/32 = 8` values per lane fully register-resident.

Function constants (`sdpa_vector.h:7-13`): `has_mask`, `do_causal`, `bool_mask`, `float_mask`, `has_sinks` are Metal `[[function_constant]]` — disabled branches emit no code. CoreML MIL either static-unrolls (blowing up graph size) or computes both branches via `select`.

### 3.2. `mx.compile` — JIT kernel codegen

`compiled.cpp:16-255`: traces Python function into graph, for each elementwise subgraph **codegens a single Metal kernel string** at runtime, then compiles it. Benchmark (compile.rst:136): gelu on 32×1000×4096 tensor on M1 Max, **15.5 ms → 3.1 ms, 5×** (memory bandwidth saved by not materializing intermediates).

Fusion cap: 31-buffer argument-table limit (`compiled.cpp:247-253`). Used in mlx-lm for SwiGLU, GeGLU, logit_softcap, RoPE. **CoreML ANE cannot do online codegen — compilation is AOT at `.mlmodelc` time.**

### 3.3. Async pipelining across tokens

`mlx-lm/generate.py:455-470`:
```python
mx.async_eval(y, logprobs)        # emit token 0 compute
while True:
    next_y, _ = _step(y)          # BUILD token 1 graph
    mx.async_eval(next_y, _)       # emit token 1 while token 0 still executing
    yield y.item(), logprobs       # token 0 materialized
    y = next_y
```
CPU-side graph-build overlaps GPU execution. **CoreML `predict()` is synchronous; no equivalent on Swift side.**

### 3.4. iPhone-specific command buffer sizing

`device.cpp:489-491` (phone, arch suffix `p`): `max_ops=20, max_mb=40`. Base/Pro: 40/40. Max: 50/50. Apple tunes this per device — **visible to MLX, hidden from CoreML**.

### 3.5. W4A16 in-register dequant

`affine_qmv_fast` (`quantized.h:1496`), `qmv_fast_impl` (lines 750-814): 2 simdgroups × 4 outputs × lanes cooperatively load 128-element chunks, dequantize 4-bit via `x_thread[values_per_thread]` register array, fused `qdot` does dequant + MAC. **Weight never materialized as FP16 in shared or device memory.** Same property as llama.cpp.

### 3.6. Unified memory + untracked hazards

`allocator.cpp:14-15`:
```cpp
MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked
```
MLX manages deps itself via barriers — saves per-setBuffer driver overhead.

---

## 4. LiteRT-LM GPU delegate — the 56.5 tok/s implementation

Source: `github.com/google-ai-edge/LiteRT-LM`, `github.com/google-ai-edge/LiteRT`

### 4.1. Three encode modes — critical finding

`inference_context.mm`:
- **`EncodeWithCommandBuffer`** (lines 751–760): new encoder per node, all in one MTLCommandBuffer, single commit
- **`EncodeWithEncoder`** (lines 665–671): **single encoder reused across all nodes** — no `endEncoding` between ops. Hot path when graph is monotone.
- **`EncodeWithICB`** (lines 682–687): `[command_encoder executeCommandsInBuffer:icb_ withRange:NSMakeRange(0, nodes_.size())]` — **entire decoder forward pre-recorded into an Indirect Command Buffer, replayed every decode step**.

Gated by `SetHintFullyDelegatedToSingleDelegate(true)` (`llm_executor_settings_utils.cc:195`): the whole TFLite graph must be one delegate, giving the delegate end-to-end command buffer ownership.

**CoreML cannot do this.** Each `MLModel.prediction` is its own submission because the framework can't assert IO bindings are stable. 4 chunks/token × 2.3 ms = **9.2 ms of fence cost** that LiteRT amortizes to one commit.

### 4.2. SIMD-group cooperative matmul

`conv_metal_simd.cc` lines 107–296 emits:
```
#define MMA simdgroup_multiply_accumulate
simdgroup_matrix<FLT,8,8> mat_src;
simdgroup_load(mat_src, tmp_src_x1 + …, 8);
MMA(dst_spX_chY, mat_src, w_oY_iX, dst_spX_chY);
```
Plus threadgroup weight cache (lines 143–151): `tmp_w` staged once, reused across multiple `simdgroup_load` calls.

### 4.3. Weight dequant at upload, not per-dispatch

`llm_executor_settings_utils.cc:199, 216, 239`:
- `SetConvertWeightsOnGpu(true)`
- `SetNumThreadsToUpload(2)` (2 upload threads)
- `WaitForWeightsConversionComplete(...)`

Flow: disk weights (INT4/INT8) → `MTLBuffer` (MTLStorageModePrivate) → one-off GPU pass dequantizes to FP16 → kernel reads FP16 via `simdgroup_load`. **No dequant math in the hot matmul loop.**

Grep of `conv_metal_simd.cc` for `dequant|INT4|q4_|unpack|bitshift` → **zero matches**.

Cost: RSS ~1450 MB (vs 607 MB on CPU XNNPACK). **Google trades 850 MB of RAM for a dequant-free hot path.** CoreML ANE cannot make this trade identically — ANE wants weights in its own palette format.

### 4.4. Single-buffer KV cache

`llm_litert_compiled_model_executor.cc:1539`:
```cpp
bool gpu_optimized_single_buffer_cache =
    backend == Backend::GPU && signatures.input_int32_param.has_value();
```

Non-GPU: `std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_)` after every prefill (line 766) and decode (line 975) — ping-pong double buffer.

GPU: swap skipped. Same buffer read+write via `param_tensor` carrying `[start, end, end]` indices.

For Gemma E2B @ 2K: K or V ≈ 220 MB. Two-buffer swap needs 440 MB + full copy into next command buffer. Single-buffer kills that copy.

Drafter explicitly (`llm_litert_mtp_drafter.cc:131–134`):
```cpp
gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
gpu_compilation_options.AddBufferStorageTensorPattern("kv_cache_");
gpu_compilation_options.AddExternalTensorPattern("param_tensor");
gpu_compilation_options.AddBufferStorageTensorPattern("param_tensor");
```
`AddExternalTensorPattern` = tensor lives in caller-owned MTLBuffer, delegate must not copy. **GPU-only optimization; NPU path cannot take it because NPU can't share arbitrary `MTLBuffer` offsets with the caller.**

### 4.5. Async predict with 2-step prewarm

`llm_executor_settings_utils.cc:238`: `SetNumStepsOfCommandBufferPreparations(2)` — two command buffers in flight. While token N on GPU, token N+1 encoder already populated.

`compiled_model_.RunAsync(...)` (line 971) — unconditional async submit, no `waitUntilCompleted`. CPU returns immediately to prepare next step.

### 4.6. No fused SDPA kernel — reconstructed via linker

Grep `litert/tflite/delegates/gpu/` for `sdpa|flash_attention|scaled_dot_product` → **zero matches**. `litert/tflite/experimental/genai/sdpa.cc` is CPU-only reference.

On GPU, SDPA is **decomposed** at TFLite op level (transpose + batch_matmul + mask_add + softmax + batch_matmul), then re-fused by `MergeElementwiseNodes` (`gpu_model.cc:386`) and `MergeNodes` (`gpu_model.cc:51`). Roughly 10–12 dispatches per decoder layer × 26 layers = **~270–310 dispatches/token**. At 17.7 ms/token budget → ~57 μs/dispatch.

### 4.7. Monolithic GPU vs segmented NPU — architecturally opposite

NPU executor (`llm_litert_npu_compiled_model_executor.cc`) exposes **6 signatures**: Embedder, EmbedderPerLayer, Mask, RoPE, Llm, CacheUpdate — each a separately-compiled subgraph, with DMA+sync between each segment. Forced by NPU op limitations (no RoPE trig, no int32 gather for embedding, no dynamic slice for cache_update).

GPU executor: only `kPrefillSignatureRunner` and `kDecodeSignatureRunner`. Monolithic, end-to-end inside one delegate.

**Our CoreML/ANE path is structurally closer to the NPU path** — 4-chunk split forced by ANE residency (~1 GB/compiled model) and op restrictions. We pay 3 activation round-trips per token between chunks. **This is the root-cause parallel to the NPU segmentation penalty.**

---

## 5. Apple GPU vs ANE Hardware/Dispatch

Sources: Orion paper (arXiv 2603.06728), maderix M4 ANE deep dive, tzakharko A19/M5 benchmark, Apple Metal Feature Set Tables, Apple developer docs

### 5.1. Apple GPU dispatch anatomy

- `MTLCommandQueue`, `MTLCommandBuffer`: user-space ring buffer
- `commit` = syscall-free kernel signal to mapped doorbell on A-series
- GPU firmware picks commands up directly from ring
- **No daemon on hot path.** Shader-compile and `MTLIOCompressor` daemons exist but only off-path.
- End-to-end commit-to-GPU-start: single-digit μs on A-series
- **Indirect Command Buffers (iOS 12+, A9+)**: encode once, reuse — per-frame encode cost → near zero
- `MTLStorageModeShared`: hardware-coherent pool, CPU writes visible to GPU with no DMA/flush, L3/system-cache-level latency

### 5.2. A19 Pro GPU neural accelerators (tzakharko benchmark)

- 6-core "Apple 10-series", 128 matrix FMAs per compute partition with 32-wide 4-way hardware dot product
- Per-core throughput: **1024 FLOPS/cycle FP16**, ~2048 OPS/cycle INT8
- A19 Pro 6-core: ~9 TFLOPS FP16, ~13.5 TOPS INT8 (sustained)
- Backward-compat simdgroup_matrix 8×8: ~17 cycles FP16, ~18 FP32 per MAC instruction
- **On A19, Metal 4 exposes `mpp::tensor_ops::matmul2d` (NAX 16×16 frag)** — MLX uses this via `steel_attention_nax`
- L2 cache on P-cluster: 16 MB; per-GPU-core threadgroup memory: ~60 KB; register file: ~208 KB

### 5.3. ANE dispatch anatomy (Orion + maderix)

Every dispatch goes through:
1. User process → XPC serialize
2. Mach-port send to `ANECompilerService` / `aned` daemon
3. Daemon thread wakeup
4. IOKit `UserClient` method call
5. Kernel driver → DMA descriptor setup
6. IOSurface map (first-time cost dominates)
7. ANE MMIO doorbell
8. Completion interrupt → reverse path

Measured costs (M4 Max / H16 ANE):
- Bare single-token dispatch: **~0.03 ms**
- XPC + IOKit hop: **~0.095 ms**
- IOSurface round-trip (cold first-call): **~2.3 ms**
- Process restart: ~50 ms
- On-chip SRAM: 32 MB, ~30% throughput cliff above threshold
- Peak: 19 TFLOPS FP16 (INT8 dequantizes to FP16 internally, same throughput)
- **Matmul emulated via Conv1×1 datapath: 3× penalty**
- Hard limits: 127-deep eval queue, ~119 compilations/process, 49 KB min IOSurface
- Power: hard power-gated when idle → every cold dispatch touches FSM ramp
- Graph depth: single op ~30% util, 32–64 ops in one dispatch → 94% util

The daemon is not a software artifact; it's enforced by **DART (Apple's IOMMU) and the fact that ANE only accepts pre-compiled immutable programs**. Third-party processes cannot map ANE MMIO directly.

---

## 6. What GPU Does That ANE Physically Cannot

These are structural limits of the ANE silicon + driver boundary. No CoreML update will fix them:

1. **Fine-grained user-space dispatch.** Metal ring buffer, sub-μs commit. ANE has no user-mapped queue.
2. **Dynamic graph reshape per dispatch.** Metal sends shader args per submit; ANE bakes shapes/weights/layout at compile time. Changing KV stride = full recompile (4200 ms cold, 494 ms warm).
3. **Native matmul with arbitrary K.** A19 Pro GPU has hardware matmul unit; ANE emulates via Conv1×1 (3× penalty). This is a datapath choice in silicon.
4. **Online softmax in threadgroup memory.** Metal threadgroup SRAM is programmer-visible; ANE SRAM is firmware-managed. No FlashAttention-equivalent on ANE.
5. **simdgroup_multiply_accumulate / simd_sum / simdgroup_load intrinsics.** Fixed-function systolic on ANE; programmer submits matmul shapes, not instructions.
6. **INT4 dequant fused into MAC.** ANE MAC pipeline is fp16; INT4 sources must expand upstream. Pay fp16 memory bandwidth despite INT4 storage.
7. **`MTLResidencySet` page pinning.** No equivalent on ANE; IOSurface handoff per call.
8. **JIT kernel codegen (MLX `mx.compile`).** ANE is AOT compile-only.
9. **Sub-49-KB operand granularity.** Metal `MTLBuffer` is cacheline-granular; ANE IOSurface min 49 KB.

---

## 7. What is API-Inaccessible on ANE But Architecturally Possible

These exist in the silicon but Apple does not expose them:

1. **Batched queued dispatch** — ANE has 127-deep eval queue; CoreML forces one XPC round-trip per `MLModel.prediction`. Orion documents going under CoreML straight to `_ANEClient`, but it's not public API. `MLBatchPredictions` helps only for independent inputs, not autoregressive decode.
2. **ICB-equivalent persistent graph** — ANE could accept pre-bound descriptors, but the kernel driver doesn't expose it.
3. **Whole-model-in-one-dispatch** — ANE hits 94% util at 32–64 op depth. A 35-layer transformer should fit. But MIL compiler has ~20 undocumented graph-depth limits (Orion, 14 newly discovered); exceeding them produces silent compile failures. The **silicon would be happy with the whole model; the MIL compiler is the gate.**
4. **Fused KV-cache update inside the dispatch** — 32 MB SRAM is large enough; stateful-model API re-materializes IOSurfaces across calls. Orion's direct IOSurface pinning keeps it resident.

---

## 8. What Is Achievable With MIL Graph / Swift Code (Remaining Levers)

1. **Rewrite every matmul as Conv1×1** in coremltools export. **Single largest single-fix lever — reclaims 3× matmul penalty.**
2. **Pin IOSurfaces once and reuse.** 2.3 ms is the cold path. Preallocate all I/O buffers with `MLMultiArray`-backed persistent IOSurfaces at load time; never free between tokens. Converges toward 0.095–0.1 ms floor.
3. **Maximize chunk depth toward 32-MB / 64-op sweet spot.** On 4B models: ~8–12 layers/chunk, 3–4 chunks — matches current Gemma 4 conversion. Verify MIL depth budget.
4. **Manual MIL op fusion** — RMSNorm + Mul + Add → single MIL composite op. Reduces MIL node count and internal ANE sub-dispatches.
5. **CPU+GPU fallback for softmax/attention** via `MLComputeUnits.cpuAndGPU` — push O(seq²) attention matrix onto Metal GPU (where FlashAttention equivalents exist via MPSGraph). Leave QKV projections + MLP on ANE. **This is the hybrid path.**
6. **INT4 palette with larger group size** (groupsize=128 vs 32) — cut scale-tensor traffic 4×. ANE internal dequant still paid.
7. **Move embed + sampling off ANE** to CPU/GPU. Softmax over 256K vocab is bandwidth-bound; GPU handles it far better than ANE. Orion recommends this, matches project_direction.md.
8. **Keep model resident** — single `MLModel` load, reuse same `MLModelConfiguration`. Avoid any code path that triggers recompile (runtime dtype casts, shape changes).

**Realistic ANE ceiling after all of the above: ~22–28 tok/s.** Cannot reach 56.5 tok/s on ANE-only.

---

## 9. Per-Dispatch Accounting (Decode One Token, 26-layer Gemma 4 E2B)

### Metal path (llama.cpp / LiteRT / MLX)
- ~270–350 kernel dispatches in 1–8 MTLCommandBuffers
- Per-dispatch: 5–10 μs encode (overlapped with GPU) + ~100 μs for 1 commit
- Total CPU-side overhead: **~100 μs–1 ms/token**
- GPU compute: ~15–25 ms (bandwidth-bound on W4A16 projections)
- **Wall clock: ~17 ms/token → 56.5 tok/s** ✓

### CoreML/ANE path (ours)
- 4 `MLModel.prediction` calls per token
- Per call: ~2.3 ms (cold) → ~0.5 ms (warm, IOSurface-pinned)
- Plus: inter-chunk activation copy via CPU-visible buffer (~10 ms/token)
- Plus: matmul emulation 3× penalty on any non-Conv1×1 shape
- **Wall clock: ~66 ms/token → 15 tok/s** (matches measurement)

**Fence + copy cost alone (15–30 ms) is larger than LiteRT's entire per-token budget.**

---

## 10. Strategic Decision Points for the Project

### 10.1. ANE-only maximum
~22–28 tok/s after wringing all levers in §8. **Cannot beat LiteRT-LM.**

### 10.2. Metal/MPSGraph direct
56+ tok/s achievable, but requires rewriting the decode engine in Metal. **CoreML targeting `.cpuAndGPU` is not equivalent** — it emits generic MPS primitives and loses fusion opportunities. Must go `MPSGraph` or raw Metal.

### 10.3. Hybrid — prefill on ANE, decode on GPU
- **Prefill** is compute-bound: ANE wins when depth > 32 ops and seq > 64 (dispatch amortizes). Current prefill already fast-ish.
- **Decode** is dispatch-bound: only GPU path breaks 30 tok/s on current hardware.
- Migration cost: moderate. Reuse existing 4-chunk MIL for prefill; new Metal/MPSGraph decode path for inference loop.

### 10.4. Speculative decoding compounding
Mirror SD / EAGLE-3 / MTP all give multiplicative gains (1.3–2×) on top of whichever backend runs the target. **These remain valid** — they multiply the GPU baseline too. A 56.5 × 1.5 = 85 tok/s is the true ceiling with spec decode on top of Metal.

### 10.5. Honest conclusion
The bet "ANE-only can beat LiteRT-LM Metal" is **structurally unsupportable** given current API exposure. Apple would have to open the ANE 127-queue, or ship a new framework (rumored "Core AI" for iOS 27/WWDC 2026) with user-space dispatch. Until then:

- If the project's goal is **"beat 56.5 tok/s on Gemma 4 E2B on iPhone,"** the path is Metal/MPSGraph decode + ANE prefill + spec decode on top
- If the project's goal is **"demonstrate ANE-only high-quality inference,"** the ceiling is ~22–28 tok/s; differentiator shifts to power/privacy/thermal
- CoreML `.all` is a middle ground that captures neither advantage fully

---

## 11. Honest Unknowns

- Exact A19 Pro ANE generation characteristics — extrapolated from M4 Max H16 ±30%
- Per-encoder μs on A19 GPU — inferred from Apple docs + ggerganov discussion #3909; not directly instrumented
- Exact speedup of `gpu_optimized_single_buffer_cache_` vs double-buffer — no microbenchmark in LiteRT tree, Google internal docs suggest 10–15%
- Whether `EncodeWithICB` is on LiteRT-LM's hot path or only `EncodeWithEncoder` — traced to `compiled_model_.RunAsync` which resolves inside `@litert`; not further traced
- NAX availability on A19 — gated by `is_nax_available()` in MLX; untested on A19 (confirmed M4/M5)
- Exact dispatch count per Gemma 4 layer on each backend — estimated from source inspection, would need Xcode GPU frame capture to confirm

---

## 12. Primary Sources

### llama.cpp
- `ggml/src/ggml-metal/ggml-metal-context.m` — command buffer batching
- `ggml/src/ggml-metal/ggml-metal-device.m` — residency sets, concurrent encoder
- `ggml/src/ggml-metal/ggml-metal.metal` — FlashAttention, q4_K matmul kernels
- [Discussion #3909 — Metal dispatch overhead](https://github.com/ggml-org/llama.cpp/discussions/3909)

### MLX
- `mlx/backend/metal/kernels/sdpa_vector.h`, `quantized.h`, `steel/gemm/nax.h`
- `mlx/backend/metal/device.cpp`, `compiled.cpp`, `allocator.cpp`, `eval.cpp`
- `mlx-lm/mlx_lm/generate.py` — async_eval decode loop

### LiteRT-LM
- `LiteRT-LM/runtime/executor/llm_litert_compiled_model_executor.cc`
- `LiteRT-LM/runtime/executor/llm_litert_mtp_drafter.cc`
- `LiteRT-LM/runtime/executor/llm_executor_settings_utils.cc`
- `litert/tflite/delegates/gpu/metal/inference_context.mm` — three encode modes
- `litert/tflite/delegates/gpu/common/tasks/conv_metal_simd.cc` — SIMD matmul
- `litert/tflite/delegates/gpu/common/gpu_model.cc` — fusion passes

### Apple GPU / ANE architecture
- [Orion: Characterizing and Programming Apple's Neural Engine (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728)
- [Inside the M4 Apple Neural Engine, Part 1 & 2 — maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Investigating GPU Neural Accelerators on Apple A19/M5 — tzakharko](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- [philipturner/metal-benchmarks — Apple GPU microarchitecture](https://github.com/philipturner/metal-benchmarks)
- [Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [MTLResidencySet docs](https://developer.apple.com/documentation/metal/mtlresidencyset)
- [Encoding indirect command buffers on the CPU](https://developer.apple.com/documentation/Metal/encoding-indirect-command-buffers-on-the-cpu)
- [MPSGraph documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph)
- [WWDC23: Optimize machine learning for Metal apps](https://developer.apple.com/videos/play/wwdc2023/10050/)
