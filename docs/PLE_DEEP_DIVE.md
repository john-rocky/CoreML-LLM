# Per-Layer Embeddings (PLE) Deep Dive — Gemma 4 E2B on ANE

**Date:** 2026-04-15
**Scope:** `Gemma4Model.embed_tokens_per_layer` and associated per-layer input
pipeline. Sizes measured from the shipping artifact
(`Sources/CoreMLLLM/ModelDownloader.swift:532-533` — `embed_tokens_per_layer_q8.bin`
is **2,348,810,240 bytes** = 2.19 GiB INT8 weight + **524,288 bytes** scales).
Runtime instrumentation from `docs/EMBEDDING_BYPASS_FINDINGS.md`. LiteRT layout
from `docs/LITERT_CONTAINER_ANALYSIS.md`.

---

## 1. Executive summary

**The biggest realistic win is vocabulary pruning combined with low-rank
factorization of the PLE table**, targeting a ~90% reduction of
`embed_tokens_per_layer` from 2.24 GB (INT8) down to ~200 MB — or ~50 MB if we
accept a retraining step for low-rank. Everything else (PLE compute on ANE,
compute-skipping, INT4) is already largely won or yields <2 tok/s. Storage, not
compute, is the PLE pain point: the gather+combine is already ~0.4 ms of our
~30 ms/step budget, and moving it further is a rounding error. The on-disk and
in-RAM footprint of the raw table is what forces us to mmap, what blocks a
single-mlpackage ship, and what puts us behind LiteRT-LM's <1.5 GB GGUF. Vocab
pruning to ~60K tokens (drop non-Latin, non-CJK scripts) is **already
implemented as a dry-run** (`conversion/prune_vocab.py`) and saves **~1.6 GB**
immediately — it is the single highest impact × feasibility item on the list.
Recommended action plan is in §6.

---

## 2. What PLE does (from the Gemma 3n / Gemma 4 architecture)

### 2.1 Architectural role

Per-Layer Embeddings are a **Gemma 3n-family innovation** (carried into Gemma 4
E2B/E4B) that trade parameter count for effective memory on accelerators. The
short version from Google's own framing (ai.google.dev/gemma/docs/gemma-3n):

> "PLE data can be generated separately, outside the operating memory of the
> model, cached to fast storage, and then added to the model inference process
> as each layer runs. This approach allows PLE parameters to be kept out of
> the model memory space, reducing resource consumption while still improving
> model response quality."

Concretely, **PLE lets Google report E2B as "2B effective parameters" while
shipping ~5B total params** — the PLE table is the extra 3B, it is
CPU-resident / disk-mmapped, and a 17.5 KB slice per step is streamed into
the accelerator. The accelerator side (GPU/ANE/XNNPACK) never needs to hold
the 2.3B PLE tensor in its tight working set.

### 2.2 Forward math (reference: HuggingFace `modeling_gemma3n.py`)

From the model-level forward (`Gemma3nTextModel`):

```python
# Per-token PLE vector — 35×256 = 8960 dims for E2B
per_layer_inputs = embed_tokens_per_layer(input_ids) * sqrt(hidden_size_per_layer_input)

# Project hidden state into the same 8960-dim space
per_layer_projection = per_layer_model_projection(inputs_embeds)  # 1536 → 8960
per_layer_projection *= hidden_size ** -0.5
per_layer_projection = per_layer_projection_norm(per_layer_projection)

# Combine and rescale
per_layer_combined = (per_layer_projection + per_layer_inputs) / sqrt(2)
```

Inside each decoder layer (`Gemma3nTextDecoderLayer`), the 256-dim slice for
that specific layer index is used as a **gating signal**:

```python
gated = per_layer_input_gate(hidden)     # 1536 → 256 Conv1x1
gated = gelu_tanh(gated)
gated = gated * per_layer_slice          # elementwise (256-d)
gated = per_layer_projection(gated)      # 256 → 1536 Conv1x1
out   = post_per_layer_input_norm(gated)
hidden = hidden + out                    # residual
hidden = hidden * layer_scalar
```

This repo's implementation is in `conversion/models/gemma4.py:116-127`
(embeddings), `conversion/models/gemma4_swa_chunks.py:164-178` (per-layer
input application inside `_run_layer_swa`), and
`conversion/models/gemma4_swa_chunks.py:224-254` (PLE combination inside
`SWAChunk1._compute_ple`, which was moved from Swift → ANE for an 8 ms/step
win — see commit log note on line 260: "Compute PLE internally (8ms savings
vs Swift BLAS)").

### 2.3 Training objective — why PLE works

There is **no separate training objective** for PLE. It is trained end-to-end
with the rest of the model under the standard next-token cross-entropy loss.
The reason it works as a memory-saving trick is:

1. **PLE weights are gradient-sparse.** Only the row corresponding to the
   current input token sees a gradient step. After training, most rows are
   lookup tables, not compute.
2. **The gating architecture makes the contribution additive.** The PLE vector
   modulates a GELU-gated projection (which starts small) and is added into
   the residual stream. This means the transformer can function reasonably
   well even if PLE contributions are degraded — unlike the main embedding
   table, which is on the critical path.
3. **Per-token-per-layer specialization.** Because each of the 262,144 tokens
   has its own 8960-dim vector sliced across 35 layers, the network can store
   "if token is X, at layer L, modulate channel C this way" — a form of
   token-conditioned layer-local bias. Empirically this lets Google use a
   smaller transformer trunk (1536 hidden, 35 layers ≈ 1.9B params) without
   losing quality, because the PLE table carries ~2.3B params of
   "token-specific adjustment".

No community paper ablates PLE directly; the closest work is
[antimatter15/reverse-engineering-gemma-3n](https://github.com/antimatter15/reverse-engineering-gemma-3n)
which confirms the gating structure but does not measure a PLE-off variant.

---

## 3. Current implementation cost (storage + compute)

### 3.1 Storage

| Tensor | Shape | Dtype | Bytes |
|---|---|---|---|
| `embed_tokens_per_layer.weight` | (262144, 8960) | INT8 + scales | **2,348,810,240 + 524,288 = 2.19 GiB** |
| `per_layer_model_projection.weight` | (8960, 1536, 1, 1) | fp16 / palettized | 27.5 MB fp16, ~6.9 MB INT4 |
| `per_layer_projection_norm.weight` | (256,) | fp16 | 0.5 KB |
| 35 × `per_layer_input_gate.weight` | (256, 1536, 1, 1) each | fp16 / palettized | 27.5 MB fp16 total |
| 35 × `per_layer_projection.weight` | (1536, 256, 1, 1) each | fp16 / palettized | 27.5 MB fp16 total |
| 35 × `post_per_layer_input_norm.weight` | (1536,) each | fp16 | 0.1 MB total |

**Total PLE-family storage:**
- Un-palettized fp16: 4.73 GB (2×2.35 for the main table + 83 MB for projections)
- Current shipping (INT8 PLE + palettized projections): **~2.24 GB**
- This is **>50% of the entire model package** — the transformer trunk is
  ~0.8 GB palettized.

In the shipping artifact (per `ModelDownloader.swift:532`), the PLE table is a
**standalone quantized blob** (`embed_tokens_per_layer_q8.bin` +
`_scales.bin`), separate from the mlpackage. Swift does the gather on CPU via
mmap (confirmed in `Sources/CoreMLLLM/ChunkedEngine.swift:375-378,
1565-1568`). It does **not** ride inside any ANE chunk — that would blow past
the ANE working-set budget.

### 3.2 Runtime cost per decode step

From the Swift path (`ChunkedEngine.swift:1565`, `EmbeddingLookup.swift`):

1. **Raw PLE gather:** INT8 memcpy of 8960 bytes from the mmapped blob →
   fp16 in the scratch pool. Measured 0.30-0.45 ms in warm runs with vDSP
   (`docs/EMBEDDING_BYPASS_FINDINGS.md:89-90`). This is before any matmul.
2. **PLE combine on ANE (inside `SWAChunk1._compute_ple`):**
   - `per_layer_model_projection`: Conv1x1 1536→8960, FLOPs = 2×1536×8960 =
     27.5 MFLOPs (single token).
   - Reshape + concat-trick RMSNorm: trivial (<1 MFLOP).
   - Add + scale: 8960 fmadd ≈ 18 KFLOPs.
3. **Per-layer gating × 35 layers:**
   - `per_layer_input_gate`: Conv1x1 1536→256 = 0.79 MFLOPs × 35 = 27.5 MFLOPs.
   - `per_layer_projection`: Conv1x1 256→1536 = 0.79 MFLOPs × 35 = 27.5 MFLOPs.
   - Total per-layer-input compute = **55 MFLOPs/step**.

**Aggregate PLE-related compute per decode step ≈ 83 MFLOPs**
(27.5 combine + 55 gating). The transformer MLP alone is 35 × 2 × 1536 ×
6144 × 3 = 1.98 GFLOPs. So **PLE compute is <5% of the decoder FLOPs.**

### 3.3 Conclusion of §3

The PLE problem is overwhelmingly a **storage / memory-bandwidth** problem,
not a compute problem. The 2.24 GB on-disk blob is the dominant cost. Every
decode step reads 17.5 KB from it — at 50 tok/s that is only 875 KB/s, well
under mmap page-cache bandwidth. The model load-time cost of warming that
mmap to RSS is the real hit (hundreds of ms on iPhone 17 Pro, observable as
TTFT).

---

## 4. Optimization techniques ranked by feasibility × impact

Feasibility: ★ = paper only, ★★ = requires retraining, ★★★ = retrain-free.
Impact ranked as (storage MB saved) / (tok/s gained).

### 4.1 Vocab pruning — ★★★ feasibility, HIGHEST impact

**Status: tool already written (`conversion/prune_vocab.py`), dry-run only.**

Gemma 4's 262,144-token vocab is dominated by non-Latin, non-CJK scripts
(Cyrillic, Arabic, Devanagari, etc.). The dry-run analyzer already classifies
tokens by Unicode block. For an English + Japanese target:

- Keep ~60,000 tokens (ASCII/Latin + Hiragana/Katakana/CJK + symbols + special)
- Prune ~200,000 tokens (~77%)

**Savings:**
- `embed_tokens_per_layer`: 2.19 GB → **~513 MB** (1.68 GB saved)
- `embed_tokens`: 385 MB → **~88 MB** (297 MB saved)
- `lm_head` (tied): same as `embed_tokens`

**Total package reduction: ~2.0 GB**. This also reduces the LM head softmax
over the vocab dimension, but in the current export `lm_head.weight` is part
of chunk4 and is palettized — pruning rows shrinks it proportionally.

**Quality cost:** zero for in-distribution text. For pruned scripts the model
cannot even generate the character (tokenizer will emit UNK or split into
bytes). This is fine if the product is English/Japanese-only.

**Required work:** The existing dry-run just prints stats. `apply_vocab_pruning.py`
exists (per Grep hit at `conversion/apply_vocab_pruning.py:145-152`) and handles
the PLE reslicing (`embed_tokens_per_layer` row-slice with `new_ple` tensor
assembly, `conversion/apply_vocab_pruning.py:145-148`). What's missing: a
re-export step that regenerates the int8+scales blobs and updates the
tokenizer.

**Impact on tok/s:** Small but non-zero. The `lm_head` matmul is currently
262144 output channels; reducing to 60K cuts chunk4 compute proportionally
on the LM head arm. Expect +1-2 tok/s. TTFT (first-token) drops meaningfully
because the mmapped PLE is smaller — fewer page faults.

### 4.2 Per-chunk PLE slicing — ★★★ feasibility, MEDIUM impact

PLE layout is (262144, 35×256) = (vocab, layer_idx × per_layer_dim). Chunk *i*
only needs layers in its range:

- Chunk1: layers 0-7 → slice `[:, 0:8*256]` = (262144, 2048)
- Chunk2: layers 8-14 → (262144, 7×256) = (262144, 1792)
- Chunk3: layers 15-24 → (262144, 10×256) = (262144, 2560)
- Chunk4: layers 25-34 → (262144, 10×256) = (262144, 2560)

If we abandon the "all chunks share one PLE blob" design and split into 4
chunk-specific blobs, each chunk only mmaps its own slice. This **does not
reduce total storage** but makes each chunk's RSS smaller and enables
per-chunk loading / unloading. For iOS memory-pressure resilience this is
worthwhile — iOS kills apps exceeding the 2-3 GB foreground footprint on
iPhone 17 Pro. The current PLE sitting mmapped at 2.2 GB RSS leaves very
little headroom for the system, the keyboard, the chat UI, and WKWebView
instances.

**Effort:** ~1 day. Small Swift change in `ChunkedEngine.swift` to maintain 4
`EmbeddingLookup` instances and dispatch per-chunk-local slices.

**Impact:** No tok/s change, but eliminates OOM termination above 4K context
on devices with 6 GB RAM (iPhone 16 / 15 Pro). No impact on iPhone 17 Pro
(8 GB). Skip if the product ships 17 Pro+ only.

### 4.3 INT4 PLE — ★★★ feasibility, MEDIUM impact

LiteRT-LM ships the PLE inside their single-container model at ~1.28 GB total
(from `docs/LITERT_CONTAINER_ANALYSIS.md:15` — "Section 1 Per-layer embedder
1284.5 MB"). That is close to INT4 for the (262144, 8960) tensor:
262144×8960×0.5 B = 1.12 GB + scales.

Our current INT8 PLE is 2.19 GB. Going INT4 with per-channel scales
(say 64-column groupsize for a palettized lookup) gets us to **~1.17 GB**
— matches LiteRT. Embeddings are empirically the most quantization-tolerant
tensors in a transformer (they're used as lookups, not matmul operands, so
quantization noise doesn't cascade). LiteRT ships INT4, so we know it works
for quality.

**Effort:** ~2 days. We already palettize via coremltools; PLE is a
standalone `.bin`, not inside the mlpackage, so we need a custom INT4
quantizer + Swift dequant path. vImage doesn't have INT4→fp16 directly, so
add a small vDSP unpack-and-scale.

**Impact:** Saves **~1.0 GB**. Compound with vocab pruning (4.1) for
**60K × 8960 × 0.5 B ≈ 256 MB PLE total** — a 9× reduction from current.

### 4.4 Low-rank factorization — ★★ feasibility, HIGHEST ceiling

`embed_tokens_per_layer: (262144, 8960)` ≈ U @ V with U: (262144, r), V: (r,
8960). For **r=256**:

- U size: 262144×256 = 67M params (67 MB INT8 → 33 MB INT4)
- V size: 256×8960 = 2.3M params (negligible)
- **Total: ~34 MB INT4** — a **64× reduction** vs current INT8 PLE.

Quality cost: unknown, but the analogous trick works well for the main
embedding table (Matryoshka, tied LM head with low-rank projection). The PLE
is structurally more redundant because:
- Many tokens share morphological features (gating signals likely factorize
  well).
- Adjacent layer slices are often correlated (layer-wise gating is smooth).

**Requires SVD on the pre-trained PLE table + a short LoRA-style finetune
(~1000 steps on a small corpus) to recover any quality drop.** This is the
only item on the list that is not retrain-free, but the amount of retraining
is small (we're only tuning U and V; the transformer trunk is frozen).

**Effort:** 3-5 days for a proof-of-concept (SVD init, short finetune,
quality eval on a held-out perplexity set). Piggyback on the existing
EAGLE-3 training infra (MEMORY.md: eagle3_retrain with a custom Gemma4Model
teacher already exists; add a PLE-factorized student).

**Impact:** If rank=256 works, **PLE table drops from 1.2 GB (INT4) to
34 MB** — PLE is effectively free. Combined with vocab pruning this takes
the whole model under 1 GB.

### 4.5 Skip PLE projection / make it identity — ★★ feasibility, LOW impact

Question: could `per_layer_model_projection` (1536 → 8960, 13.8 M params) be
omitted? It is the only place where the hidden state influences PLE (the raw
lookup `embed_tokens_per_layer[token_id]` is token-only).

Without it, the PLE contribution becomes purely token-conditional (like a
static per-layer bias lookup), losing the "hidden-state-modulated" character.
In principle the per-layer-input gate inside each decoder layer still mixes
the hidden state with the PLE slice, so this might be survivable. But:

- It would change the training objective — there's no pretrained checkpoint
  with this ablation.
- Savings are tiny: 13.8 M params = 27 MB fp16. We already palettize this.

**Verdict:** Not worth it. Mentioned only because the prompt asked.

### 4.6 Frequent-token PLE cache — ★★★ feasibility, NEGATIVE impact

Prompt suggested precomputing PLE for the top 100 tokens in RAM and streaming
the rest from disk. This is backwards: the current mmap path already caches
frequently-accessed pages (the OS does this for free via the unified buffer
cache). Adding an explicit LRU on top would duplicate work and add latency.
**Skip.**

### 4.7 Off-ANE PLE (keep on CPU, stream to ANE) — ★★★ feasibility, ALREADY DONE

Per `docs/EMBEDDING_BYPASS_FINDINGS.md:7-9` and ChunkedEngine.swift:719, the
PLE gather is **already on CPU**. The `per_layer_raw` input to
`SWAChunk1.forward` is produced by `embedPerLayer.lookupRaw(tokenID)` on the
main thread. ANE only sees a (1, 1, 8960) fp16 tensor per step. The CPU→ANE
feed is 17.5 KB = negligible on the unified memory architecture (A19 has
~100 GB/s bandwidth; that's 200 ns at the theoretical limit).

Nothing to do here.

---

## 5. Vocab pruning interaction with PLE

The existing `apply_vocab_pruning.py` tool handles PLE as a first-class
concern (line 6: "…by slicing embed_tokens, embed_per_layer (PLE — the
largest tensor), and …", line 145 pulls `embed_tokens_per_layer` off the
inner model and row-slices it). Combined with the classifier in
`prune_vocab.py`, pruning to ~60K tokens produces this matrix of options:

| Config | PLE size (INT8) | PLE size (INT4) | PLE + low-rank (r=256) |
|---|---|---|---|
| Full 262K vocab | 2.19 GB | 1.17 GB | 67 MB |
| 60K vocab (EN+JP) | **513 MB** | **275 MB** | **16 MB** |
| 30K vocab (EN only) | 257 MB | 138 MB | 8 MB |

Each column composes with any row. We can ship "full vocab + INT4 PLE" to
match LiteRT, or "60K vocab + INT4 PLE" to beat them (275 MB vs their 1.28 GB).

**Caveat:** vocab pruning breaks round-trip with HF's tokenizer. The
tokenizer needs the same subset and the `lm_head` output index space must
be remapped. `apply_vocab_pruning.py:204` already reports new PLE size;
confirm it also patches `tokenizer.json` and the output decoding in Swift.

---

## 6. Recommended action plan

Ordered by ROI. Each step is independent; ship in sequence.

1. **Activate vocab pruning (2-3 days).** `prune_vocab.py` is a dry-run;
   `apply_vocab_pruning.py` exists but unverified. Run the end-to-end flow:
   prune → re-export safetensors → re-quantize PLE int8 blob → re-convert
   mlpackage → update tokenizer.json → update
   `Sources/CoreMLLLM/ModelDownloader.swift:532-533` size constants and
   `vocab_size` in ModelConfig. Expected: **~2.0 GB package reduction**,
   +1-2 tok/s, TTFT improvement from smaller mmap warming.
2. **Convert PLE INT8 → INT4 (2 days).** Use blockwise (64-col groups)
   quantization. Add Swift dequant path (vDSP unpack). Expected: **additional
   ~230 MB saved** (on top of vocab pruning), zero tok/s change.
3. **Low-rank factorization POC (5 days).** SVD-init U, V from the current
   PLE table. Run 1000 steps of LoRA-style tuning on C4+OpenAssistant, eval
   perplexity on held-out set. If perplexity regression <2%, ship it.
   Expected: **additional ~250 MB saved** (PLE total: ~16 MB); quality
   validation required.
4. **Per-chunk PLE slicing (1 day, optional).** Only if we target iPhone 15
   Pro / 16 (6 GB RAM). Skip for 17 Pro-only products.
5. **Do NOT pursue:** frequent-token PLE cache (4.6), PLE projection removal
   (4.5), any compute-side optimizations — PLE is <5% of decoder FLOPs and
   <1 ms of wall time.

**Aggregate end state:** Vocab 60K + INT4 PLE + low-rank factor →
**~16 MB PLE table**, down from 2,200 MB. Whole mlpackage under 900 MB.
Beats LiteRT-LM's 2.58 GB container by **2.8×**, puts us in the same size
class as on-device Qwen2.5-0.5B while keeping the Gemma 4 quality.

---

## 7. References

- Google AI for Developers, Gemma 3n model overview —
  [ai.google.dev/gemma/docs/gemma-3n](https://ai.google.dev/gemma/docs/gemma-3n)
  (PLE "kept out of model memory space" framing; 1.91B effective params for E2B).
- HuggingFace `transformers/models/gemma3n/modeling_gemma3n.py` — reference
  forward for `Gemma3nTextModel.get_per_layer_inputs`,
  `project_per_layer_inputs`, and the decoder layer's
  `per_layer_input_gate` / `per_layer_projection`.
- antimatter15, *Reverse Engineering Gemma 3n* —
  [github.com/antimatter15/reverse-engineering-gemma-3n](https://github.com/antimatter15/reverse-engineering-gemma-3n)
  (confirms PLE gating structure from the released weights).
- HuggingFace `rishiraj/matformer-in-gemma-3n` —
  [huggingface.co/blog/rishiraj/matformer-in-gemma-3n](https://huggingface.co/blog/rishiraj/matformer-in-gemma-3n)
  (MatFormer framing; PLE as one of three memory-saving tricks alongside
  selective parameter activation and KV cache sharing).
- In-repo references:
  - `conversion/models/gemma4.py:116-127` — PLE module definitions.
  - `conversion/models/gemma4_swa_chunks.py:164-178, 205-254` — PLE forward
    math inside chunk1 (ANE).
  - `Sources/CoreMLLLM/ChunkedEngine.swift:375-378, 1565-1568` — CPU-side
    mmap gather.
  - `Sources/CoreMLLLM/ModelDownloader.swift:532-533` — ground-truth PLE
    blob sizes (2.35 GB + 0.5 MB scales).
  - `conversion/prune_vocab.py` — vocab classifier (dry-run).
  - `conversion/apply_vocab_pruning.py:145-204` — actual PLE/embed row-slice
    applier.
  - `docs/EMBEDDING_BYPASS_FINDINGS.md` — confirms PLE is already CPU-resident,
    ANE never holds it; runtime timing (0.3-0.4 ms/step gather).
  - `docs/LITERT_CONTAINER_ANALYSIS.md:15` — Google's shipping PLE is
    1284.5 MB (INT4, confirms the INT4 approach works for quality).
