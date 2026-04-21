# ANE Private API — Hard Constraints Reference (source-verified)

**Date:** 2026-04-22 (updated with source-level verification)
**Primary source:** `maderix/ANE` (`/Users/majimadaisuke/Downloads/workspace/repo-review/maderix-ANE`) — reverse-engineered training on ANE
**Cross-check:** `ANEMLL` (`/Users/majimadaisuke/Downloads/workspace/repo-review/Anemll`) — production ANE LLM inference
**Secondary:** `salescore/776dff7a85f781` Zenn article — author's isolated + E2E benchmarks on M4 Max

**Corrections from earlier README-only draft:**
- Private API names were wrong. Real: `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`. `_ANEClient` / `_ANECompiler` not used in maderix.
- `attn_mask` constraint framing was wrong. Not "silently ignored" — it **forces CPU-side decomposition** (Q@Kᵀ on ANE, softmax+mask on CPU, scores@V on ANE). ANEMLL **contradicts the premise**: they pass an explicit `fullCausalMask` MLMultiArray and decode works on ANE.
- Compile limit: 100 hardcoded (`MAX_COMPILES` in `stories_config.h:26`), not 119. 119 was the author's observed figure; code pins it to 100 with exec()-restart.
- SwiGLU fusion is **adjacent concat**, not true MIL-op fusion. RMSNorm backward is separate kernel in training path.
- Dynamic weight packing is a **training-time** IOSurface-update trick; inference weight swaps still require recompile.

---

## TL;DR — does this change our decisions?

### Unchanged
- **Drafter-on-E2B death stands.** Bottleneck is oracle-live acc gap (3-9×), a distribution problem neither source addresses.
- **W4A8 headroom stands.** INT8 1.88× is peak-ratio from weaker baseline; not additive.
- **L2 SRAM int8-activation trick stands.** Already captured in our W4A8.

### Changed by source verification
- **`attn_mask` is not a hard blocker.** ANEMLL passes explicit causal mask MLMultiArrays and attention runs on ANE. Maderix decomposes; ANEMLL does not. Our architectural decision should be informed by ANEMLL's approach, not maderix's.
- **Compile-budget math tightens.** 100/process, not 119. Shape-bucket ceiling tightens accordingly.
- **Private API surface is narrower than I claimed.** Only 4 classes, all `_ANEInMemoryModel*` / `_ANERequest` / `_ANEIOSurfaceObject` — not a `_ANEClient`/`_ANECompiler` pair.

---

## 1. Hardware / driver constraints (verified)

### 1.1 `attn_mask` in SDPA — decompose or pass explicit mask

- **Maderix path:** `training/test_ane_causal_attn.m:82-125` — non-causal SDPA baseline, then causal path is split into Q@Kᵀ on ANE, CPU-side mask+softmax, scores@V on ANE.
- **ANEMLL path:** `anemll/models/gemma3_model.py:416-712` — builds explicit `fullCausalMask` MLMultiArray input, passes to ANE SDPA, works. Sliding-window alternation: separate `infer_rotate` / `prefill_rotate` compiled models per chunk (for position > window).
- **Implication for us:** Attention *can* stay on ANE on our stack by the ANEMLL approach. Memory cost: full-seq causal mask. For sliding-512 half-layers, two model variants per chunk. This contradicts my earlier claim that attention on ANE is physically gated.

### 1.2 Single-input constraint (confirmed)

- `maderix training/training_dynamic/mil_dynamic.h:37-46` — dynamic matmul input is `[1, IC, 1, SEQ+OC]`, weights and activations packed into the spatial dim.
- `mil_dynamic.h:56-62` — SDPA forward packs `xnorm | Wq | Wk | Wv` into a single `[1, DIM, 1, SEQ+Q_DIM+KV_DIM+KV_DIM]` input.
- Violation fails silently (off-by-one slicing, wrong results, no error).

### 1.3 Compile budget — 100/process (corrected)

- `maderix training/stories_config.h:26` — `#define MAX_COMPILES 100`.
- `maderix training/train_large_ane.m:350-361` — `if (g_compile_count + kernels_needed > MAX_COMPILES)` triggers `execl(argv[0], "--resume", "--ckpt", ...)`.
- Stories110M uses ~60 kernels/batch (5/layer × 12 layers), so budget = 1 batch + overhead.
- **Not software-configurable** per author; hardcoded ANE runtime limit.
- Zenn's 119 was observed; maderix uses 100 conservatively.
- ANEMLL does not hit this in the inspected code path (`anemll-swift-cli` uses precompiled `.mlmodelc`, not runtime in-memory compile).

### 1.4 Multi-output IOSurface binding is alphabetical (source-unconfirmed)

- Only Zenn article, not maderix or ANEMLL. Treat as **unverified** until reproduced.
- Action: when any multi-output ANE kernel is wired, test slot ordering experimentally before trusting.

### 1.5 IOSurface layout `[1, C, 1, S]`, I/O fp32, internal fp16

- `maderix bridge/ane_bridge.m:61-67` — IOSurface shape assembly confirms `[1, C, 1, S]`.
- Internal fp16 is stated in both sources.

### 1.6 Private API surface (corrected)

- `maderix bridge/ane_bridge.m:33-56` — resolution chain:
  - `dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)`
  - `NSClassFromString(@"_ANEInMemoryModelDescriptor")`
  - `NSClassFromString(@"_ANEInMemoryModel")`
  - `NSClassFromString(@"_ANERequest")`
  - `NSClassFromString(@"_ANEIOSurfaceObject")`
- Call sequence (`bridge/ane_bridge.m:94-156`): `modelWithMILText:weights:optionsPlist:` → `inMemoryModelWithDescriptor:` → `compileWithQoS:options:error:` → `loadWithQoS:options:error:` (with 100ms retry on slot reclaim) → IOSurface + `_ANERequest` → `evaluateWithQoS:options:request:error:`.
- No `_ANEClient` or `_ANECompiler` symbol. My earlier reference claim was wrong.

---

## 2. MIL op compatibility (Zenn-only, unverified against maderix code)

From salescore Zenn testing (macOS 15, M4 Max):

| MIL op | Behavior | Workaround |
|---|---|---|
| `matmul` | Error 0x1d | `conv` kernel=1×1 |
| `rsqrt` | SIGSEGV | `1 / sqrt(x)` |
| `gelu` | Compile fails | `x * sigmoid(1.702 * x)` |
| `log` | No fix | — |
| `slice_by_index` | No fix | — |

**Source status:** maderix uses MIL text emitted as hand-written NSString in `training/ane_mil_gen.h` and `mil_dynamic.h` (no ops from the failing list in the paths I checked). Zenn's list **not contradicted** but **not independently reproduced** in these repos. Treat as hypothesis until we hit them.

---

## 3. Shape / layout gotchas

### 3.1 D=2^n SRAM bank conflict (Zenn-only, unverified)

- Zenn: D=4096, F=14336 → isolated fused FFN 0.55× Metal; pad to D=4160 (64×65) → 2.06×.
- Not reproduced in maderix (no 4096-dim model in their code — Qwen3 0.6B at D=1024, Stories110M at D=768).
- For Gemma 4 E2B (D=1536 = 2^9 × 3), **rule does not trigger**.

### 3.2 Isolated vs E2E ANE-win collapse at D≥2560 (Zenn-only)

- Zenn Phase 3 table (D=896 → 2.47×, D=2560 → 0.82×) is author's measurement, no source in maderix.
- MLX source inspection (`/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/mlx`) **refutes** the mechanism claimed ("MLX lazy eval keeps intermediates in GPU L2"): MLX fusion is **compile-time** op-inlining (`mlx/backend/metal/compiled.cpp`), with no L2-pinning logic and no ANE-aware scheduler.
- **Net:** The *measured* isolated-vs-E2E gap may be real (Zenn's bench), but the *explanation* is not. Trust the numbers cautiously; don't trust the causal story.

---

## 4. Peak / realized throughput (verified)

- `maderix ane_int8_bench.m:220-264` — conv 512×512, 64×64 spatial, 128-deep: FP16 14.8ms → 18.6 TOPS; INT8 7.8ms → 35.1 TOPS. **1.88× ratio confirmed.**
- `maderix training/train_large_ane.m:734` — utilization print: `100*ane_flops/(total_train_ms*1e9)/15.8` — using 15.8 TFLOPS peak. Training-side 5-9% utilization confirmed, limited by CPU ops (RMSNorm, softmax, element-wise).
- Realized throughput on ANE transformer workloads: **1-3 effective TOPS**, not 35.

---

## 5. Kernel fusion — scope clarified

- `maderix training/stories_config.h:87-91` — **inference forward** fused kernels: `sdpaFwd` (QKV + SDPA + output), `ffnFused` (W1, W3, SiLU, W2).
- **Training backward NOT fused** (same file, lines 91-95): `ffnBwdW2t`, `ffnBwdW13t` split for memory.
- **SwiGLU is not a single MIL op**: `mil_dynamic.h:285` — `concat(x_next, h1, h3, gate)`, gate computed separately. "Fusion" here = single ANE kernel dispatch containing multiple MIL ops, not true op fusion.
- **Implication:** CoreML converter outputs for our existing chunks may already achieve equivalent dispatch density. The Private API fusion advantage is less than the Zenn article implies.

---

## 6. ANEMLL counter-evidence / cross-check

- Chunking: balanced-by-param layer split, `anemll/utils/calc_chunk_split.py:245-248`. No mid-layer splits.
- KV cache: ring buffer depth 16 (monolithic), ping-pong depth 2 (chunked) — `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:55-60`, 99-100.
- RMSNorm-as-LayerNorm: Python-side `concat([x, -x])` → LayerNorm → slice, fold gain — `anemll/models/gemma3_model.py:264, 308`.
- Output buffers: pre-allocated modulo-indexed pool (not NSCache) — `InferenceManager.swift:1738-1759`.
- Serial `DispatchQueue` for ANE prediction — `InferenceManager.swift:99-100, 1417-1449`.
- Attention on ANE with explicit causal mask input: `anemll/models/gemma3_model.py:416-712`. Contradicts "attn_mask ignored" framing.
- Speculative decoding: **absent**. No EAGLE / MTP / n-gram in codebase.
- Quantization: LUT 4/6/8-bit only; no W4A8, no int8 activations. `anemll/ane_converter/gemma3_converter.py:418-516`.

---

## 7. Dynamic weight packing — scope (verified)

- `maderix training/test_weight_patch.m:323` — "Change weight dynamically — NO recompile!" Implementation is IOSurface write + reload; works for training gradient updates (weights in spatial dim, sliced at runtime).
- **Inference weight swap still requires recompile** — no evidence of load-time swap for non-training scenarios.

---

## 8. What this means for the current roadmap (updated)

1. **Drafter reversal: still NO.** Nothing in maderix or ANEMLL addresses oracle-live acceptance gap.
2. **ANE-side decoder speedup for Gemma 4 E2B via Private API:** Limited. W4A8 captures the INT8 bandwidth gain; Private API fusion may not exceed what CoreML converter already produces.
3. **Attention-on-ANE is not physically gated.** ANEMLL proves it works with explicit causal mask. Reject the earlier "attention can't go to ANE" claim.
4. **Non-drafter speedup angles worth chasing from these sources:**
   - Ring buffer / ping-pong for our multi-chunk KV transfer (ANEMLL-verified pattern).
   - IOSurface-backed CVPixelBuffer for prediction I/O on our current CoreML stack.
   - Separate `_rotate` model variants for sliding-window crossover (ANEMLL Gemma 3 pattern; applies to our Gemma 4 E2B sliding-512).
5. **Private API direct path remains speculative.** Upside bounded; engineering cost high.

---

## 9. Pre-flight checklist for ANE optimization proposals

- "Put attention on ANE" → OK with explicit mask MLMultiArray input (§1.1, ANEMLL pattern). Sliding window needs `_rotate` variant per chunk.
- "Add N shape buckets" → ≤100/process, max 2 reuse → practically ~80 unique shapes over full session (§1.3).
- "Multi-output fused ANE kernel" → alphabetical binding unverified (§1.4). Test slot ordering.
- "Use `rsqrt` / `matmul` / `gelu` in Private API path" → Zenn reports failures (§2). Verify with target macOS before investing.
- "D=4096 model candidate" → pad to break 2^n factor if Zenn's claim reproduces (§3.1).
- "Isolated microbench shows N×" → E2E gap real per Zenn numbers, but mechanism explanation is wrong; don't over-trust.
- "INT8 W8A8 1.88× on ANE" → our W4A8 already captures this; not additive.

---

## 10. Sources

- maderix/ANE: `/Users/majimadaisuke/Downloads/workspace/repo-review/maderix-ANE/`
- ANEMLL: `/Users/majimadaisuke/Downloads/workspace/repo-review/Anemll/`
- MLX: `/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/mlx/`
- Zenn article: https://zenn.dev/salescore/articles/776dff7a85f781
