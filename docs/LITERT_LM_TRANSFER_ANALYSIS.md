# LiteRT-LM transfer analysis — what's transferable to our CoreML/ANE W4A16 stack

**Date:** 2026-04-26
**Branch:** `research/litert-lm-transfer`
**Scope:** Mac-side analysis only. iPhone validation deliberately deferred per
"Mac Studio first" rule. No Sources/ changes made by this analysis pass —
findings hand off to per-stage feature branches if/when adopted.

This is a focused follow-up to `docs/LITERT_LM_ARCH_VERIFIED.md` (LiteRT-LM
runtime source-read, 2026-04-22) and `docs/LITERT_CONTAINER_ANALYSIS.md`
(LiteRT-LM `.litertlm` container layout, 2026-04-14). Read those first if
you haven't.

---

## 0. TL;DR

> The headline LiteRT-LM techniques the original task asked us to port —
> external mmap embedding (Phase 2) and per-layer-embedding externalization
> (Phase 3) — **are already shipping in our W4A16 stateful path.**
> What's NOT yet ported: (a) a clean inventory of how much of the bundle
> each piece accounts for, (b) `outputBackings` to suppress unused / state-
> aliased outputs on the prediction call, and (c) the unnecessary 6 MB/step
> KV alias copy from chunk2's MLState back through Swift into chunk3/chunk4.

Concrete results:

1. **Bundle size truth check.** The "148 MB W4A16" in `STAGE1_W4A8_FINAL.md`
   is *only chunk_1*. The full Gemma 4 E2B stateful bundle is **3.71 GB**:
   1.09 GB of mlmodelc decoder chunks + 2.20 GB external PLE + 384 MB
   external token embedding + ~30 MB sidecars (RoPE, projection, scales,
   norm). LiteRT-LM `.litertlm` equivalent is ~2.21 GB — **we're 1.68×
   bigger**, almost entirely due to the PLE table (2.20 GB INT8 vs 1.28 GB
   in LiteRT) and the token embedding (384 MB INT8 vs 104 MB in LiteRT).

2. **External embedding is already done.** `Sources/CoreMLLLM/EmbeddingLookup.swift:25`
   uses `Data(contentsOf: dataURL, options: .mappedIfSafe)` for the 384 MB
   `embed_tokens_q8.bin` and the 2.2 GB `embed_tokens_per_layer_q8.bin`.
   INT8 → FP16 dequantization is vectorized via vDSP + vImage. The
   embedding table never enters the CoreML decoder graph; the runtime
   feeds `hidden_states` and `per_layer_raw` as inputs.

3. **PLE is already external** — and is the elephant in the room. The
   `embed_tokens_per_layer_q8.bin` file alone is **2.2 GB**, dwarfing the
   1.09 GB decoder chunks. Phase 3 of the task ("search for PLE in graph,
   externalize if present") is moot — the architectural decision was made
   in v1.4.x.

4. **outputBackings is unused everywhere.** `grep` across `Sources/` and
   `Examples/` shows zero call sites set `MLPredictionOptions.outputBackings`.
   Every prediction allocates fresh `MLMultiArray` for every declared output.
   The biggest concrete waste: chunk2 emits 4 KV alias outputs (kv13_k/v,
   kv14_k/v ≈ 6 MB at ctx=2048) that get materialized to Swift and re-fed
   into chunk3/chunk4 — those slices already exist in chunk2's MLState and
   the materialize-then-copy round trip is structurally redundant. At
   ~40 tok/s decode that's ~240 MB/s of avoidable bandwidth.

5. **What LiteRT-LM does that we cannot replicate cheaply:** keeps KV state
   on GPU across MTP draft / verify / decode in one Metal command queue.
   That advantage is GPU-Metal-structural (per `LITERT_LM_ARCH_VERIFIED.md`
   §1) and does not transfer to ANE. Already documented at line 191 of that
   doc — "Do not pursue 'beat LiteRT on ANE.'"

The remaining transferable wins are small (Phase 5 outputBackings is single-
digit % at best). The largest realistic optimization on this stack is
*shrinking the PLE table* — but that's an architectural change to Gemma 4,
not a runtime port.

---

## 1. Phase 1 — Size and graph audit

### 1.1 Method

`scripts/print_coreml_size_breakdown.py` walks the MIL `mlProgram` proto:

- For `const` ops with `blobFileValue`, reads the 32-byte
  `0xDEADBEEF`-prefixed header at the recorded offset to get the true byte
  size.
- For `constexpr_*` ops (palettization), enumerates each input
  `Argument.value.blobFileValue` (lut, indices, scale, zero_point) — which
  is where the W4 LUT data actually lives in MIL.
- Deduplicates by physical `(file, offset)` so weights shared between
  function specializations (`decode_q1` and `verify_qK` reference the same
  blobs) count once.

The script's per-blob byte total within a few hundred bytes of the actual
`weight.bin` file size for every chunk inspected (delta = alignment padding
between aligned blocks).

### 1.2 Decoder mlmodelc breakdown (chunks-k8 path, W4 LUT, FP16 activations)

Source: `/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/output/gemma4-e2b/chunks-k8/chunk{1,2,3,4}.mlpackage`

| chunk | total | weights | const tensors | dominant role | layer range (E2B) |
|---|---:|---:|---:|---|---|
| chunk_1 | 149.05 MB | 148.24 MB | 200 | early decoder + PLE projection | L0-L7 |
| chunk_2 | 128.45 MB | 127.76 MB | 174 | mid decoder (smaller MLP) | L8-L14 |
| chunk_3 | 310.84 MB | 310.21 MB | 192 | mid decoder (wider MLP) | L15-L24 |
| chunk_4 | 503.10 MB | 502.47 MB | 195 | late decoder + **lm_head (192 MB)** | L25-L34 |
| **decoder total** | **1,091.44 MB** | | 761 unique blobs | | |

(The ~2× MLP-row growth between chunk_2 and chunk_3 is Gemma 4's variable
MLP intermediate-size schedule — early layers are narrower, late ones wider.
Confirmed via the per-tensor sizes inside each chunk.)

In every chunk, ~99.9% of bytes live in `constexpr_lut_to_dense/indices`
(the W4 packed indices), with `lut` tables (FP16 LUT centroids) + plain
RMSNorm consts making up the remaining 0.1%. So the iOS18 ML Program is
almost entirely INT4 indices with tiny FP16 LUT lookup tables and FP16
RMSNorm scales.

### 1.3 What's IN the decoder graph

- All 35 transformer layers (L0-L34): `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj` (W4 LUT each), plus FP16
  RMSNorm weights for `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm`,
  `post_per_layer_input_norm`, `q_norm`, `k_norm`.
- The PLE *projection* (`per_layer_model_projection_weight_palettized`,
  6.56 MB) — i.e. the post-lookup matrix that mixes per-layer embedding
  vectors into the residual stream.
- The lm_head matrix in chunk_4 (`squeeze_10_palettized`, **192 MB**) —
  named `squeeze_*` because MIL's optimizer rewrote the unsqueeze-then-
  matmul into a squeeze; the size matches `vocab_size × hidden ÷ 2 nibbles`
  = 262144 × 1536 ÷ 2 = 192 MB exactly.

### 1.4 What's NOT in the decoder graph (verified)

- The **token embedding table**. Lives entirely in
  `embed_tokens_q8.bin` (384 MB INT8 + 512 KB FP16 scales) on disk.
  Never appears as a graph const.
- The **per-layer embedding (PLE) lookup table.** Lives entirely in
  `embed_tokens_per_layer_q8.bin` (**2.20 GB INT8** + 512 KB FP16 scales).
  Never appears as a graph const.
- The **RoPE cos/sin tables** (full + sliding). Live as
  `cos_full.npy`, `cos_sliding.npy`, `sin_full.npy`, `sin_sliding.npy` —
  feeding into the decoder graph as `cos_s/sin_s/cos_f/sin_f` inputs each
  step.
- KV cache buffers (when stateful). Materialized as `MLState` instances by
  the iOS18 runtime, not as graph consts. (For `chunks-k8` non-stateful
  build, KV is round-tripped via input/output tensors on every step.)

### 1.5 Full bundle size

Source: `/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/build/gemma4_stateful_ab/linear/gemma4_e2b_stateful_chunks/`

| component | size | format | mmap'd by Swift |
|---|---:|---|---|
| `chunk_1..4.mlmodelc` (decoder) | 1,091.40 MB | iOS18 ML Program | yes (CoreML-managed) |
| `embed_tokens_q8.bin` | 384.00 MB | INT8 + FP16 scales | **yes** (`.mappedIfSafe`) |
| `embed_tokens_scales.bin` | 0.50 MB | FP16 | **yes** |
| `embed_tokens_per_layer_q8.bin` | **2,240.00 MB** | INT8 + FP16 scales | **yes** (`.mappedIfSafe`) |
| `embed_tokens_per_layer_scales.bin` | 0.50 MB | FP16 | **yes** |
| `per_layer_projection.bin` | 26.25 MB | FP16 | yes |
| `per_layer_norm_weight.bin` | 0.00 MB | FP16 | yes |
| `cos_full.npy` / `sin_full.npy` | 8.00 MB each | FP32 | yes (`.npy` mmap) |
| `cos_sliding.npy` / `sin_sliding.npy` | 4.00 MB each | FP32 | yes |
| `model_config.json` | <1 KB | JSON | n/a |
| `hf_model/` (tokenizer + config) | 31 MB | tokenizer.json, etc | n/a |
| **Total** | **~3,798 MB (3.71 GB)** | | |

### 1.6 Comparison with LiteRT-LM `.litertlm` (per `LITERT_CONTAINER_ANALYSIS.md`)

| component | ours (CoreML W4A16) | LiteRT-LM | delta |
|---|---:|---:|---|
| Decoder graph | 1,091 MB W4 LUT | 818 MB W4/W8 mix | +273 MB (33% bigger) |
| Token embedding | 384 MB INT8 | 104 MB | +280 MB (3.7× bigger) |
| PLE | 2,240 MB INT8 | 1,284 MB | +956 MB (75% bigger) |
| RoPE + projection + scales | ~57 MB | (in TFLite) | n/a |
| Vision/audio + MTP drafter | (separate / not shipped) | 372 MB | n/a |
| **Comparable subtotal** | **~3,715 MB** | **~2,206 MB** | **+1,509 MB (1.68×)** |

Why we're bigger:
- **PLE storage:** LiteRT likely uses fewer bits per element (INT4-grouped?)
  or a smaller PLE table altogether. Our `embed_tokens_per_layer_q8.bin`
  layout is `vocab × num_layers × per_layer_dim` INT8, no group quant.
- **Token embedding:** Same shape (`vocab × hidden`), same INT8 dtype on
  paper, but we ship 384 MB vs their 104 MB. The cleanest hypothesis is
  that LiteRT either (a) does a vocab subset lookup with on-the-fly hashing,
  (b) shares the embedding table with lm_head (tied weights, no separate
  storage), or (c) uses INT4 grouped. The `.litertlm` "Embedder" section
  is a TFLite subgraph with internal layout we don't have parity-source for
  — would need to dump and inspect.
- **Decoder:** 33% delta is the smallest. Probably a function of LiteRT
  using a tighter quant scheme (some W8, some W4) than our uniform W4 LUT.

The PLE delta is the single-largest possible bundle-size win on this stack.
**Follow-up probe ran on 2026-04-26 — see `docs/PLE_INT4_PROBE.md`.**
TL;DR: per-row cos numbers were optimistic (INT4 g=32 mean cos 0.995 vs
BF16); the e2e prefill test then **flipped ~35% of token argmaxes**
even at g=8. PTQ INT4 PLE is **not viable** as a drop-in. Saving the
~980 MB requires QAT or vocab pruning, both larger investments. The
realistic quick win on this stack is tied-weight embedding↔lm_head
dedup (-192 MB), not PLE compression.

---

## 2. Phase 2 — External mmap embedding: ALREADY SHIPPED

The original task wanted us to:
1. Export embedding to a separate binary file. ✅ Already at
   `embed_tokens_q8.bin` (INT8) + `embed_tokens_scales.bin` (FP16 per-row
   scales).
2. mmap it. ✅ `Sources/CoreMLLLM/EmbeddingLookup.swift:25`:
   `try Data(contentsOf: dataURL, options: .mappedIfSafe)`.
3. Add a path: `token_ids → external lookup → CoreML decoder input_embeddings`.
   ✅ `Gemma4StatefulEngine.swift:257` calls `embedTokens.lookup(token, ...)`
   then feeds the returned MLMultiArray as `hidden_states` to chunk_1.
4. Validate cos = 1.0. **N/A — we built the path, not ported it.** The
   reference is the on-disk INT8 table itself; the Swift dequant +
   feed-through path is the production ground truth.

The dequant is non-trivial and worth noting:
- Storage: `int8[vocab][dim]` + `float16[vocab]` per-row scales.
- Lookup: `output[i] = int8[token][i] × (scale[token] / 127) × global_embed_scale`
- vDSP path: `vDSP.convertElements(int8 → f32)` →
  `vDSP.multiply(scale, …)` → `vImageConvert_PlanarFtoPlanar16F` to FP16.
  ~4-6× scalar speedup per the file's own comment.

**No code changes recommended.** Phase 2 of the task is already in production.

---

## 3. Phase 3 — Per-layer embedding (PLE): ALREADY SHIPPED, AND IS THE BIGGEST FILE

The PLE *table* is `embed_tokens_per_layer_q8.bin` — **2.20 GB INT8**.

- Same Swift codepath as the token embedding: a second `EmbeddingLookup`
  instance with the per-layer-flattened shape.
  `Gemma4StatefulEngine.swift:142-146` instantiates it.
- Per-token, per-layer dequant happens via the same vDSP+vImage path. The
  resulting `[1, 1, num_layers × per_layer_dim]` FP16 array is fed as
  `per_layer_raw` to chunk_1.
- chunk_1 then computes `per_layer_combined_out` (mixing PLE through the
  projection matrix, which IS in the graph at 6.56 MB). chunks 2/3/4 take
  `per_layer_combined` as input.

So:
- The PLE *lookup* is external and mmap'd.
- The PLE *projection* (the only PLE-related weight matrix inside the
  decoder) is W4 LUT'd in chunk_1's `weight.bin`.
- This is structurally identical to LiteRT-LM's split (Section 1
  "Per-layer embedder" external, projection inside the main TFLite).

**No code changes recommended.** Phase 3 of the task is already in production.

The one open question — whether the 2.20 GB on-disk PLE table can be made
smaller — is an architecture / quantization-strategy question, not a
runtime port. Documented in §1.6 as the largest potential bundle-size win.

---

## 4. Phase 5 — outputBackings / KV copyBack audit: ACTUAL GAPS HERE

### 4.1 outputBackings is unused

```
$ grep -rn "outputBackings" Sources/ Examples/
(no matches)
```

Every prediction call in `Gemma4StatefulEngine.step` (chunks 1/2/3/4) and
in `Qwen3VL2BStatefulGenerator` uses default `MLPredictionOptions()`. CoreML
allocates a fresh MLMultiArray for every declared output every step.

### 4.2 KV state stays in MLState — good

`Gemma4StatefulEngine.step:289-309`:
- chunk_1: `chunk1.prediction(from: p1, using: states.s1, options: opts)` —
  the `using: state1` argument means CoreML's slice_update updates the
  K_sliding / V_sliding / K_full / V_full state buffers in place. Swift
  never sees those tensors as outputs.
- chunk_2: same pattern with `states.s2`.

This is the right architecture. The state is held by `MLState`, not copied.
✅

### 4.3 KV alias outputs are the real waste

chunk_2 has 4 declared *outputs* — `kv13_k`, `kv13_v`, `kv14_k`, `kv14_v`
— that exist purely to feed chunks 3 and 4 (which are stateless and need
the KV slices as regular tensor inputs). Per `Gemma4StatefulEngine.swift:295`:

```swift
guard let h2 = out2.featureValue(for: "hidden_states_out"),
      let kv13k = out2.featureValue(for: "kv13_k"),
      let kv13v = out2.featureValue(for: "kv13_v"),
      let kv14k = out2.featureValue(for: "kv14_k"),
      let kv14v = out2.featureValue(for: "kv14_v")
```

These tensors are slices of the MLState that already lives inside chunk_2.
Materializing them as outputs forces:
- CoreML to allocate fresh MLMultiArray buffers each step.
- Data to be copied from ANE-side state out to those buffers.
- Swift to immediately re-feed those same buffers into chunks 3/4 as
  inputs, where ANE has to bring the data back in.

Rough sizes (ctx = 2048, fp16):
- kv13_k: 1×1×2048×256 = 1.00 MB
- kv13_v: 1×1×256×2048 = 1.00 MB
- kv14_k: 1×1×2048×512 = 2.00 MB
- kv14_v: 1×1×512×2048 = 2.00 MB
- **Total: 6 MB / decode step.** At 40 tok/s = **240 MB/s** of ANE↔Swift↔ANE
  bandwidth that pays no compute and is structurally a no-op.

### 4.4 chunk_4 unused outputs

chunk_4 declares 3 outputs (`token_id`, `token_logit`, `hidden_states_out`).
`step()` reads only `token_id`. The other two are allocated and discarded.

- `token_logit`: 1×1×262144 fp16 = **524 KB / step** = ~21 MB/s at 40 tok/s.
- `hidden_states_out`: 1×1×1536 fp16 = 3 KB / step.

(`hidden_states_out` exists for hooking a drafter / hidden-state probe;
none of the production paths use it. It's in the graph as a holdover.)

### 4.5 What outputBackings would do

`MLPredictionOptions.outputBackings = [name: MLBuffer]` lets us:
1. Provide a pre-allocated buffer for outputs we *do* need (e.g. `token_id`),
   avoiding per-step allocation. Negligible on its own.
2. Bind unused outputs to a single shared scratch buffer. The runtime still
   has to write to it, but the allocator pressure goes away. Saves a few
   MB/s of malloc / autorelease churn.
3. **Most importantly,** with `MLBuffer` initialized over an `IOSurface`,
   the receiving prediction (chunks 3/4) can take the SAME `IOSurface` as
   an input — so the only memory bandwidth is the original write inside
   chunk_2's state. This is what eliminates the 240 MB/s round-trip.

### 4.6 The bigger win: fold KV-aliased layers into chunk_2's MLState

The chunk2 → chunk3/4 alias roundtrip is structural to the current chunk
split. The most aggressive fix is to share MLState across chunks — i.e.
make chunks 3 and 4 also `using:` chunk_2's `MLState`. That requires:
- Either a single multi-function CoreML model (chunks 2/3/4 all reference
  the same kv13/kv14 state buffers in their MIL graphs).
- Or iOS18's cross-model state sharing (MLState handles tied to a
  shared name) — this is undocumented as of cml9 and may not exist.

Both are larger surgery than a 1-day task. Recording as a follow-up.

### 4.7 Concrete recommendation for a follow-up branch

Order of operations, smallest engineering first:

1. **One-line outputBackings on chunk_4 to bind `token_logit` and
   `hidden_states_out` to scratch.** Saves ~21 MB/s + allocator pressure.
   ~1 hour. Branch: `feat/chunk4-outputbackings`.
2. **Same on chunk_2** for the KV alias outputs, with chunks 3/4 reading
   from the same backings. Saves ~240 MB/s. **iPhone validation required**
   — outputBackings + IOSurface interaction with ANE is hardware-dependent,
   not Mac-only verifiable. ~1-2 days. Branch: `feat/chunk2-kv-iosurface`.
3. **Multifunction merge of chunks 2-4 + shared state** — eliminates the
   alias problem at the graph level. Significant, ~1-2 weeks.
   Out of scope for v1.7.0; queue for v1.8.0.

None of (1), (2), (3) requires changing the W4A16 weight format or the
external embedding pipeline.

---

## 5. What we deliberately did NOT do (and why)

- **Phase 4 (KV layout audit):** `LITERT_CONTAINER_ANALYSIS.md:127` already
  established that LiteRT-LM uses K=(1,1,ctx,head_dim), V=(1,1,head_dim,ctx),
  and that this matches our ChunkedEngine. No layout transpose is needed.
- **Phase 6 (prefill_bN):** Already in flight on `stage3-prefill-bn` branch
  per `docs/ROADMAP_2026_04_26.md:331` and the project memory record. Mac
  shipped, iPhone validation pending. Re-doing it on this branch would be
  duplicate work.
- **W4A8 / W8A8 activation quantization:** explicitly excluded by the task
  framing. `STAGE1_W4A8_FINAL.md` is the canonical close.
- **iPhone benchmarks:** scoped out by user choice. The `outputBackings`
  recommendations in §4.7 are explicitly gated on a future iPhone session.

---

## 6. Appendix — chunks-k8 size breakdown raw output

Generated by `scripts/print_coreml_size_breakdown.py` on 2026-04-26.
Saved as `/tmp/chunks_k8_breakdown.txt` during this analysis pass.

Top tensors per chunk (rounded):

```
chunk_1 (149 MB):
   6.56 MB  per_layer_model_projection_weight_palettized  (PLE projection)
   4.50 MB × 24  layers_{0..7}_mlp_{gate,up,down}_proj_weight_palettized
   ... + RMSNorm weights, q/k_norm, etc.

chunk_2 (128 MB):
   4.50 MB × 21  layers_{8..14}_mlp_{gate,up,down}_proj_weight_palettized
   1.69 MB × 7   layers_{8..14}_self_attn_{k,v}_proj
  13.52 MB      attention Q (W4) total

chunk_3 (311 MB):
   9.00 MB × 30  layers_{15..24}_mlp_{gate,up,down}_proj_weight_palettized
                 (~2× larger MLP than chunks 1/2 — Gemma 4 schedule)

chunk_4 (503 MB):
 192.00 MB     squeeze_10_palettized  (lm_head matrix, vocab 262144 × 1536 W4)
   9.00 MB × 30  layers_{25..34}_mlp_{gate,up,down}_proj_weight_palettized
```

To regenerate:

```bash
python3.12 scripts/print_coreml_size_breakdown.py \
  /Users/majimadaisuke/Downloads/workspace/CoreML-LLM/output/gemma4-e2b/chunks-k8/chunk{1,2,3,4}.mlpackage
```

---

## 7. Cross-references

- `docs/LITERT_LM_ARCH_VERIFIED.md` — LiteRT-LM runtime source-read.
- `docs/LITERT_CONTAINER_ANALYSIS.md` — `.litertlm` container layout.
- `docs/STAGE1_W4A8_FINAL.md` — Stage 1 W4A8 close-out (W4A16 is the
  operating point).
- `docs/ROADMAP_2026_04_26.md` — current stage roadmap; Stage 3
  prefill_bN claims Phase 6 of the original task spec.
- `Sources/CoreMLLLM/EmbeddingLookup.swift` — the production
  external-mmap-INT8 embedding loader (Phase 2 + 3 already implemented).
- `Sources/CoreMLLLM/Gemma4StatefulEngine.swift:289-348` — the prediction
  call site where outputBackings would slot in (Phase 5 work).
- `scripts/print_coreml_size_breakdown.py` — the size-audit tool produced
  by this branch.
