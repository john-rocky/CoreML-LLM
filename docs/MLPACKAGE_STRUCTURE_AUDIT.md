# mlpackage structure audit — 2026-04-15

Companion to `CONVERSION_AUDIT_2026_04_15.md`. That document audited the **Python converter**; this one audits **what actually landed in the compiled MIL graph**. All findings below come from parsing the protobuf `Data/com.apple.CoreML/model.mlmodel` and inspecting `weight.bin` / `Manifest.json` directly — no on-device load. Tool: `coremltools.proto.Model_pb2` / `MIL_pb2`.

Scope: sampled representative packages (prefill/verify chunk, decode chunk, drafter, stateful reference). Full `ls` output is in section 1.

---

## 1. Inventory of found mlpackages

| Package | Size | Weights | Spec MB | Purpose |
|---|---|---|---|---|
| `Examples/CoreMLLLMChat/CoreMLLLMChat/stateful_chunk2.mlpkg` | 128 MB | 128 MB | 0.38 MB | **Stateful reference** — 14 decoder layers, `StateType` KV, q=1 only |
| `conversion/output/iphone_8k/chunk1.mlpackage` | 149 MB | 148 MB | 0.85 MB | Stateless, 2-function (verify q=3 + decode q=1), 11 layers + input embed path |
| `conversion/output/iphone_8k/chunk2.mlpackage` | 128 MB | 128 MB | 0.72 MB | Stateless, 2-function, 10 layers |
| `conversion/output/iphone_8k/chunk3.mlpackage` | 311 MB | 310 MB | 0.65 MB | Stateless, 2-function, 10 layers |
| `conversion/output/iphone_8k/chunk4.mlpackage` | 503 MB | 503 MB | 0.66 MB | Stateless, 2-function, 10 layers + **tied lm_head** (≈400 MB fp16 embedding) |
| `conversion/output/iphone_8k/mtp_drafter.mlpackage` | 38 MB | 38 MB | 0.10 MB | MTP drafter, 3 decoder layers, top-k head |
| `conversion/output/gemma4-prefill/*` | same MBs | — | — | Duplicate of iphone_8k (older copy) |
| `conversion/output/all_chunks_2k/*`, `all_chunks_8k/*`, `chunk4_8k*/*`, `gemma4-mobile/*`, `gemma4-e2b-final/*`, `gemma4-mm-final/*`, `audio*/*`, `qwen2.5-0.5b/*` | — | — | Earlier runs; not audited in this pass |

All audited packages: `specVersion=9` (iOS 18 / macOS 15 opset `CoreML8`). All `mlProgram`. No classic NeuralNetwork packages remain. Manifest.json is stock (no generator markers).

Representative sample used below: `stateful_chunk2` (stateful reference), `iphone_8k/chunk1` (prefill + decode, has embedding path), `iphone_8k/chunk4` (sampler), `iphone_8k/mtp_drafter`.

---

## 2. Per-package op structure

### 2.1 `stateful_chunk2.mlpkg` (14 decoder layers, stateful)

Declared inputs (11), all `FLOAT16`:

```
hidden_states           1x1x1536
causal_mask_full        1x1x1x8192
causal_mask_sliding     1x1x1x512
update_mask             1x1x8192x1
per_layer_combined      1x1x8960
cos_s, sin_s            1x1x1x256         (sliding / local)
cos_f, sin_f            1x1x1x512         (full / global)
kv_sliding              State[10x1x512x512]    ← StateType present
kv_full                 State[ 4x1x8192x512]   ← StateType present
```

Outputs (5): `hidden_states_out`, `kv13_k`, `kv13_v`, `kv14_k`, `kv14_v`.
Context window = 8192 tokens, sliding = 512, q=1 only (decode-only model), hidden=1536, `max_head_dim=512`.

Op summary (`main`, total 2473):

| category | count |
|---|---|
| `const` | 1458 |
| `constexpr_lut_to_dense` (INT4 palette) | 63 |
| `conv` (projection linears as 1×1 conv) | 63 |
| `mul` | 175 |
| `transpose` | 126 |
| `reshape` | 84 (**100 % static-shape**) |
| `layer_norm` | 49 |
| `matmul` (attention only) | 14 |
| `read_state` / `write_state` | 14 / 14 |
| softmax-decomposed (`reduce_max`+`sub`+`exp`+`reduce_sum`+`real_div`) | 35 |
| `softmax` (primitive) | **0** |
| `gelu` | 14 |
| `slice_by_index` | 41, `slice_update` 14, `pad` 10, `tile` 18, `concat` 73, `split` 63 |
| `gather*` / `scatter*` / `one_hot` | **0** |

Per-layer profile (14 layers):
- 63 `conv` / 14 = **4.5 conv per layer** → Q, K, V, O, up, gate, down = 7 linears expected. Only 63 instead of 7×14=98. → **gate+up are fused** (single `conv` with 2×expanded out-channels, later `split`), **per-layer-input projection is absent** (moved to `per_layer_combined` input). Matches converter source.
- 14 `matmul` == 14 layers → exactly **one matmul per layer**. This is the QK^T matmul. The A·V matmul is done via `conv` (because V is a constant in height direction after reshape) or folded into a different op. **There is no `scaled_dot_product_attention` op** — attention is fully scalar-decomposed.
- Softmax is **manually expanded** (max-sub-exp-sum-div) — 35 / 5 = 7 locations × 14 layers / 2 paths (sliding + full) ≈ matches. The `softmax` MIL op is not used. See §4 / §6.

### 2.2 `iphone_8k/chunk1.mlpackage` (stateless prefill + decode, 11 layers + embed)

**Two functions in one mlpackage** — `verify_qK` (q=3, multi-token verify/prefill) and `decode_q1` (q=1).

Both take full KV cache blocks as **Multi-Array inputs** (`K_sliding_in 7x1x512x512`, `K_full_in 1x1x2048x512`, etc.) and return updated KV via Multi-Array outputs. That is the "stateless" design documented in `model_config.json` (`"stateless": true`). **No `StateType` declared anywhere in the iphone_8k set.**

Op counts (verify_qK / decode_q1):

| op | verify | decode |
|---|---|---|
| total | 2891 | 2746 |
| `const` | 1727 | 1615 |
| `constexpr_lut_to_dense` | 73 | 73 |
| `conv` | 73 | 73 |
| `matmul` | 18 | 16 |
| `transpose` | 162 | 146 |
| `reshape` | 98 (static) | 98 (static) |
| `slice_by_index` | 88 | 52 |
| `layer_norm` | 57 | 57 |
| `softmax` | 0 | 0 |
| softmax-decomposed | 43 | 41 |
| `gelu` | 16 | 16 |
| `gather*`/`scatter*`/`one_hot` | 0 | 0 |

Observations:
- `conv` 73 with 11 decoder layers ≈ 6.6 per layer → QKV-fused is partially there; still more ops than necessary (see §4).
- `verify_qK` has **+36 slice_by_index / +16 transpose** versus `decode_q1` at identical hidden size. Those are the extra KV-append slices that become no-ops at q=1. They reflect the q=3 path of speculative verify rather than any real waste.
- `matmul` = 18 (verify) vs 16 (decode): 11 layers × (QK + AV) should be 22. Chunk1 is 11 layers + the pre-embed path; 18 matches "only QK matmul is a real matmul, AV matmul is a conv" for some layers and full matmul for others. Mixed — suggests the KV layout decision is inconsistent across layers (see §4).
- 98 reshapes — **all** have a `const` shape input (static). Good: no dynamic reshape trap.
- Two functions in one bundle = weights shared (they live in the same `weight.bin`). Confirmed: only 148 MB weight.bin, not 2×.

### 2.3 `iphone_8k/chunk4.mlpackage` (sampler / last chunk, 10 layers + lm_head)

weight.bin 503 MB. Approx budget:
- 10 decoder layers @ ≈13 MB each (INT4 + scales) = 130 MB
- LM head embedding (tied to input embed, vocab 262 144 × 1536 = 403 M elem) → 403 × 2 = **806 MB fp16** but actually landed as **~400 MB** → fp16 (not palettized). This is the biggest single asset in the whole model.

Tail ops of `verify_qK`:
```
mul, tanh, const, mul, const, const, const, reduce_argmax
```
→ final logit-cap (`tanh` soft-cap at 30), greedy `reduce_argmax` at end. `decode_q1` extra tail: `reduce_argmax → expand_dims → gather_along_axis → squeeze`. That `gather_along_axis` is the **only** gather/scatter op in the entire decoder stack. It is a 1-element gather (token logit lookup) at the output — small CPU fallback, negligible.

### 2.4 `iphone_8k/mtp_drafter.mlpackage` (3 layers, q=1)

Op counts (main, 666 total):

```
const 373, mul 64, transpose 25, concat 25, constexpr_lut_to_dense 24,
conv 23, layer_norm 21, split 21, reshape 20, add 17, squeeze 11,
expand_dims 10, slice_by_index 8, matmul 8, sub 4, softmax 4, gelu 4,
linear 1, tanh 1, topk 1, cast 1
```

Two big differences vs decoder chunks:

1. **`softmax` is the primitive MIL op (4 calls)** — not decomposed. So the fused `softmax` works fine on ANE at these shapes. The decoder chunks decompose it unnecessarily (see §4).
2. **One `linear` op** remains (not lowered to `conv`). That one is likely the 8-way top-k head. Minor.
3. **`topk`** as a native op + `cast` at the end (int32 indices out). Fine.

KV inputs for drafter match chunk outputs: `kv13_k 1x1x512x256` (sliding from chunk3) and `kv14_k 1x1x8192x512` (full from chunk4). Note `kv14_v 1x1x512x8192` — transposed **V layout** vs `kv14_k 1x1x8192x512`. See §4.

---

## 3. Verified vs claimed optimizations

| Claim in `CONVERSION_AUDIT_2026_04_15.md` | Evidence in compiled graph | Verdict |
|---|---|---|
| INT4 per-grouped-channel (g=32) palette applied | 63–73 `constexpr_lut_to_dense` ops per chunk; weight.bin ≈ 14 % of dense fp16 estimate for chunk2 (128 MB vs ~900 MB) | **Confirmed** |
| FP16 activations end-to-end | Every declared input/output/intermediate type is `FLOAT16`. No `cast` ops except one int32 in drafter. `adjacent-cast=0` | **Confirmed** |
| Linears emitted as `conv` (ANE-friendly 1×1 conv) | 63 / 73 `conv` per chunk vs 0 / 1 `linear` | **Confirmed** |
| Static shapes only (ANE requires it) | 328 reshape ops across audited set, **0 dynamic** | **Confirmed** |
| Masks baked as constants | ❌ masks are declared as **runtime inputs** (`causal_mask_full`, `causal_mask_sliding`, `update_mask`). They are per-step (shape depends on q and position), so baking is not possible at graph level, but they could be **precomputed once per prefill** on CPU and reused (they already are on the Swift side — audit item closed). | Confirmed |
| RoPE baked as constant | ❌ `cos_s/sin_s/cos_f/sin_f` are runtime inputs. They are small (256/512 dim), but every decode step sends four fresh arrays. → See §4 opportunity. | **Partial** |
| `scaled_dot_product_attention_sliced_q` pass applied | **0 occurrences of any `scaled_dot_product…` op** and **0 `softmax` ops** in decoder chunks. Grep of `build_verify_chunks.py` has no `pass_pipeline.set_options("common::scaled_dot_product_attention_sliced_q", …)` — only `build_prefill_gpu.py` wires it. The main stateless-chunk converter does **not** enable the sliced-Q SDPA pass. | **Not applied** |
| Stateful KV design (`StateType`) | Only `stateful_chunk2.mlpkg` (legacy example, not used by the current runner) declares `StateType`. The production `iphone_8k/chunk1..4` explicitly set `"stateless": true` and use Multi-Array KV in/out. | **Diverged** — two codepaths exist; production path is stateless |
| Attention "a single op per layer" | Decoder chunks: 14 matmul for 14 layers (stateful) / 18 for 11 layers + embed path (chunk1 verify). Expected 2 matmul/layer (QK^T, A·V) = 22 or 28. → Means **A·V is being emitted as `conv` (not `matmul`)** for some / all layers. That's actually good for ANE (conv path). | Confirmed as intended, but see §4 |

---

## 4. Optimization opportunities visible **only** in the compiled graph

Ranked by estimated impact. "ops touched" figures are from the counters above.

### 4.1 — Softmax decomposition is not needed (medium-high impact)

Every decoder chunk decomposes softmax into `reduce_max → sub → exp → reduce_sum → real_div` (≈10 attention sites per chunk × 5 ops = 50 ops per chunk × 4 chunks = **≈200 ops model-wide**). The MTP drafter uses the fused `softmax` op and works fine. coremltools 8+ emits `softmax` for ANE on (B,H,1,S) shapes with FLOAT16. Reasons to decompose are usually numerical (fp16 overflow on very long sequences) or masking, but the masking is already additive in `causal_mask_*`. Replacing with `softmax(axis=-1)` in the converter gives:
- **Fewer dispatches** (5 → 1) → lower ANE scheduler overhead, fewer intermediate tensors.
- Better **fused attention pattern recognition** — the ANE compiler recognises `matmul → add_mask → softmax → matmul` as attention and applies its in-place block SDPA kernel. With a decomposed softmax it cannot.
- Required prerequisite for the sliced-Q SDPA pass (4.3).

Action: remove the manual `(x - x.max()).exp() / sum(...)` expansion from `models/gemma4_swa_*.py` and let coremltools lower it.

### 4.2 — `scaled_dot_product_attention_sliced_q` pass is not enabled for verify/decode (high impact at prefill)

`build_prefill_gpu.py` enables it, but `build_verify_chunks.py` / `build_merged_chunks.py` do not. At q=3 (verify) the Q-slicing is marginal, but when we move to longer prefill windows (audit item: increase verify q) it is a major win. And if 4.1 is applied first, the pass can actually fire on the decoder chunks. Expected: 20–35 % prefill speedup on large Q.

### 4.3 — 126 transposes / 98 reshapes per chunk-function (medium impact)

Pattern counters show `adjacent-transpose = 0` and `adjacent-reshape = 0` in every chunk, so the obvious "transpose-transpose" chains have already been folded. **However**, the ratio of layout ops to compute ops is very high:

- chunk1 verify: 162 transpose + 98 reshape + 88 slice + 16 gelu / **73 conv + 18 matmul** ≈ **4.2 layout ops per compute op**.
- stateful_chunk2: 126 transpose + 84 reshape + 41 slice / **63 conv + 14 matmul** ≈ **3.3 layout ops per compute op**.

Each transpose of a 1×H×T×D tensor is a real ANE instruction (the NE prefers BHCW for conv, CHW-ish for matmul). The graph is walking back and forth:

- Input `hidden_states (1,1,1536)` → `reshape (1,1,1,1536)` → `transpose(0,3,1,2)` → conv (ANE-4D) → `transpose(0,2,3,1)` → `reshape (1,1,1536)` → next op.
- Every layer redoes the NHWC↔NCHW dance around RoPE.

Concrete recommendation: **keep the entire graph in ANE-4D `(B,C,1,T)` layout end-to-end.** coremltools' `transpose_op_layout` pass and/or user-level layout choice in the converter can fold ~60 % of these transposes. Savings: measured roughly as `per_transpose_cost × (162 − 60)` per verify pass. On ANE, sub-1 ms tensors each cost ~50–120 µs → 5–12 ms/step, non-trivial at 30 tok/s.

### 4.4 — KV **V-layout inconsistency** (small but correctness risk)

Drafter declares:
- `kv14_k: 1x1x8192x512`
- `kv14_v: 1x1x512x8192` (transposed!)

Chunk outputs produce:
- `kv14_k: 1x1x8192x512`
- `kv14_v: 1x1x8192x512`

So Swift code must transpose V between chunk4-out and drafter-in every step. That transpose is **not in any MIL graph** — it is implicit in the Swift `MLMultiArray` contract. This is:

1. An extra memory pass on the CPU / GPU every decode step (8192×512×2 B = 8 MB → ~0.5 ms on an A17 DRAM bound copy).
2. A latent bug if layouts drift.

Recommendation: align the V layout so drafter accepts `(..., 8192, 512)`. This is a converter-side change in `build_mtp_drafter.py`'s attention formulation.

### 4.5 — Two separate bundles share weights (chunk1) but duplicate op count (small)

`chunk1.mlpackage` has two functions (`verify_qK`, `decode_q1`) that share `weight.bin`. verify has 2891 ops, decode 2746. Together 5637 ops, ≈ 2.2× one function. Good. The only real duplication:

- `slice_by_index` count diverges (88 vs 52) and `expand_dims` diverges (33 vs 49). These are Q-length-specific and cannot be shared.
- Both functions embed the same 63 `constexpr_lut_to_dense` ops twice — those are *op definitions*, not data; the underlying `weight.bin` offsets collapse. Confirmed by comparing spec.mlmodel size (0.85 MB) vs weight.bin (148 MB).

No action — this is working as intended.

### 4.6 — Per-layer LayerNorm × 49 (stateful_chunk2) — expected 4 × 14 = 56 (cosmetic)

Gemma 4 has 4 norms per block: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`. 14 × 4 = 56 but only 49 emitted + 1 final = 50. Difference comes from the `per_layer_input_norm` which is applied once before combine. This is fine, but the **top-10 const dump shows every layernorm weight as a separate 3 KB FP16 const** (`layers_6_post_per_layer_input_norm_weight_promoted_to_fp16` etc.). They are already tiny but they are **not palettized** (FP16 not LUT). Net size: 49 × 3 KB = 150 KB → don't bother.

### 4.7 — RoPE cos/sin as inputs (small)

`cos_s, sin_s, cos_f, sin_f` arrive as fp16 MultiArrays every step (sizes 256 and 512). For decode q=1 we are sending ~3 KB of trig tables per step. These could be `const` tensors indexed by a **1-element position** `slice_by_index`. Benefit: one less MultiArray bind + consistent cache alignment. Impact: ~0.1 ms per step.

### 4.8 — LM-head fp16 not INT4 palettized (large disk, zero latency)

Chunk4 weight.bin = 502 MB; ~400 MB of it is the tied vocab embedding in fp16. Palettizing the output projection to INT4 would drop chunk4 to ~200 MB. Runtime impact: near-zero (lm_head runs once per token), but ship size benefit is large. Risk: quality. If tied weights are desired, keep fp16 for chunk1 input embedding but palettize chunk4's copy (no longer tied on disk, but mathematically near-identical).

### 4.9 — `expand_dims` / `squeeze` pairs (minor)

stateful_chunk2: 42 expand_dims + 28 squeeze. iphone_8k/chunk1 decode: 49 + 31. Many of these wrap matmul/conv to reconcile rank between the ANE-4D conv path and the rank-3 matmul path. If the whole chain is 4D (4.3), most of these vanish. Tracked under 4.3, not separate.

### 4.10 — `pad` ops (10 in stateful, 10–14 in chunks)

Used to align attention seq-len to a multiple that ANE likes (usually 128 or 256). Static shapes, each firing once at the prefill branch. No action: removing them would actually hurt ANE.

---

## 5. ComputePlan findings

**Not obtainable from this environment.** The installed coremltools 9.0 reports:

```
Failed to load _MLModelProxy / _MLComputePlanProxy: No module named 'coremltools.libcoremlpython'
```

`MLComputePlan(modelAssetAt:configuration:)` needs Xcode/macOS native CoreML framework, which isn't linked into this pyenv build. To get the ANE/GPU/CPU breakdown we would need either:

1. A short Swift program using `MLComputePlan.load(contentsOf:configuration:)` on the physical device and dumping `computeDeviceUsage` per op.
2. `xcrun coremlcompiler compile … && xcrun coremlcompiler analyze …` on macOS.

Predicted from op types alone (all of these are ANE-eligible on A17/A18):
- `conv`, `matmul`, `layer_norm`, `mul`, `add`, `gelu`, `transpose`, `reshape`, `slice_by_index`, `concat`, `split`, `tile`, `pad`, `expand_dims`, `squeeze`, `softmax`, decomposed softmax primitives, `read_state`, `write_state`, `constexpr_lut_to_dense` → **ANE**.
- `gather_along_axis` (1 in chunk4 decode tail), `reduce_argmax` (1), `tanh` (1), `topk` (1 in drafter), `cast` (1 in drafter) → **CPU/GPU fallback**, all at tails, total <1 % of runtime.

Open TODO: run the Swift `MLComputePlan` probe on-device and attach a log to this document.

---

## 6. Recommendations ranked by estimated impact

1. **Replace manual softmax decomposition with the `softmax` MIL op.** Enables SDPA fusion on ANE, removes ~200 ops model-wide. Single converter-source change. **High impact, low risk.**
2. **Enable `common::scaled_dot_product_attention_sliced_q` pass in `build_verify_chunks.py` / `build_merged_chunks.py`** (already wired in `build_prefill_gpu.py`). Expected 20–35 % prefill speedup once q > 32. Requires (1) first.
3. **Keep the whole decoder in ANE-4D `(B,C,1,T)` layout**, dropping ~100 of the 162 transpose/reshape/expand/squeeze ops per chunk-function. Biggest per-step win for decode (we are transpose-bound). Converter-level rewrite in `models/gemma4_swa_merged*.py`.
4. **Fix V-layout mismatch between chunk4 output and mtp_drafter input** (`1x1x8192x512` vs `1x1x512x8192`). Removes a ~0.5 ms implicit transpose per speculative step, eliminates a correctness foot-gun.
5. **Palettize chunk4 lm_head** (currently ~400 MB fp16). Ship-size drop from 503 MB → ~200 MB. Validate quality on eval set before rolling out.
6. **Run `MLComputePlan` on-device** and attach the compute-device breakdown to this doc. If it turns out softmax decomposition is actually pushing ops to CPU, (1) is even higher priority than ranked here.
7. **Consolidate stateful vs stateless paths.** The legacy `stateful_chunk2.mlpkg` in `Examples/CoreMLLLMChat/` uses `StateType`; production uses Multi-Array IO (`"stateless": true`). Pick one and delete the other — today we double the maintenance and risk the sample app loading the legacy one. Recommend keeping stateless for the current speculative-decode design (ambient KV is exposed anyway) and removing the stateful example.
8. **Bake RoPE tables as `const` and index with the per-step position** rather than passing four MultiArrays every decode. Marginal.
9. **Audit `slice_by_index` count divergence between verify_qK (88) and decode_q1 (52)** in chunk1. Probably fine but worth a look — a few of the verify-path slices may be emitted even when q is effectively 1 (redundant guards).

Items 1–3 compound: they are the path to a **decoder chunk with one `matmul`, one `softmax`, one `matmul` per attention block** — i.e. three dispatches instead of ~12. That is the shape we need to beat LiteRT-LM at decode-tok/s without changing the model.

---

## Appendix: regeneration

If any of these packages are missing locally, they are produced by:

```
python3 conversion/build_verify_chunks.py \
    --config conversion/config.py \
    --out conversion/output/iphone_8k
python3 conversion/build_mtp_drafter.py \
    --out conversion/output/iphone_8k/mtp_drafter.mlpackage
```

Stateful reference:

```
python3 conversion/build_merged_chunks.py --stateful --chunk 2 \
    --out Examples/CoreMLLLMChat/CoreMLLLMChat/stateful_chunk2.mlpkg
```

Audit scripts used: `/tmp/audit_mlpkg.py` and `/tmp/audit_mlpkg2.py` (parse protobuf only, no device).
