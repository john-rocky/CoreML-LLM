# EmbeddingGemma-300M on CoreML

Gemma 3 bidirectional encoder (`google/embeddinggemma-300m`, arXiv:2509.20354)
that produces a 768-d unit-norm sentence embedding, Matryoshka-truncatable to
512 / 256 / 128.

## Architectural fit

This is **not** a causal LM. The existing `exporter.py` assumes a stateful
decoder with `input_ids (1,1)` + KV cache; that doesn't fit here. EmbeddingGemma
gets a dedicated stateless build script.

| Feature | Gemma 3 decoder | EmbeddingGemma encoder |
|---|---|---|
| Attention | Causal | **Bidirectional** |
| KV cache | Yes | — |
| Input | `(1, 1)` per step | `(1, L)` full sequence |
| Output | next-token logits | (L, 768) hidden, pooled to (1, 768) |
| `sliding_window_pattern` | every 6th full | every 6th full (same pattern, bidirectional band) |
| Post-backbone | `lm_head` → argmax | mean-pool → dense(768→3072) → dense(3072→768) → L2 |

Dims per arXiv:2509.20354 §3: `n=24`, `dₘ=768`, `dᵤ=3072`, `d=768`.

Source files:

- `conversion/models/gemma3_encoder.py` — bidirectional Gemma-3 layer, per-layer
  RoPE + sliding-window band mask, ANE-friendly layouts (Conv2d(1x1) for
  projections, `repeat_kv_ane` for GQA, fp16 softmax on dim=−1, `−1e4` mask).
- `conversion/models/embeddinggemma.py` — encoder + mean-pool + 2 dense + L2.
- `conversion/build_embeddinggemma_bundle.py` — standalone converter.

## Conversion

```bash
# fp16 bundle at 512-token max input (default)
python conversion/build_embeddinggemma_bundle.py --max-seq-len 512

# Longer inputs (2048 = native EmbeddingGemma context)
python conversion/build_embeddinggemma_bundle.py --max-seq-len 2048

# INT4-palettized shippable bundle
python conversion/build_embeddinggemma_bundle.py --max-seq-len 512 --quantize int4
```

Or via the unified CLI (same dispatch as Gemma 4):

```bash
python conversion/convert.py --model embeddinggemma-300m --context-length 512
```

### Why fixed-length input, not RangeDim

`ct.RangeDim` forces the model off the ANE (`docs/ANE_OPTIMIZATION_SURVEY.md`,
`docs/SPEED_8K.md`). The bundle builder emits a single fixed-length mlpackage;
to serve multiple length buckets, run the builder multiple times and ship all
variants:

```bash
for L in 128 256 512 1024 2048; do
  python conversion/build_embeddinggemma_bundle.py --max-seq-len $L \
      --output output/embeddinggemma-300m/bundle_L$L
done
```

Swift picks the smallest bucket that fits the tokenized input, pads with
`attention_mask=0.0` on the unused positions.

## Output layout

```
output/embeddinggemma-300m/bundle/
    encoder.mlpackage       # stateless; fp16 or INT4
    model_config.json
    hf_model/
        tokenizer.json  tokenizer_config.json  ...
```

## I/O contract

| Name | Shape | dtype | Role |
|---|---|---|---|
| `input_ids` | (1, L) | int32 | token ids; pad with 0 |
| `attention_mask` | (1, L) | fp16 | 1.0 for valid tokens, 0.0 for pad |
| → `embedding` | (1, 768) | fp16 | L2-normalized |

**Matryoshka truncation** (768 → 512 / 256 / 128): slice the leading dim of the
output and renormalize in Swift:

```swift
var vec = embedding  // (1, 768)
var truncated = Array(vec.prefix(targetDim))
let norm = sqrt(truncated.reduce(0) { $0 + $1 * $1 })
for i in 0..<truncated.count { truncated[i] /= Float(norm) }
```

## Task prefixes

EmbeddingGemma is trained with task prefixes (HF model card). The bundle's
`model_config.json` includes:

```json
"task_prefixes": {
    "retrieval_query": "task: search result | query: ",
    "retrieval_document": "title: none | text: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "similarity": "task: sentence similarity | query: ",
    "code_retrieval": "task: code retrieval | query: ",
    "question_answering": "task: question answering | query: ",
    "fact_verification": "task: fact checking | query: "
}
```

Prepend the relevant prefix to the input text **before** tokenization.

## Parity check

```bash
pip install sentence-transformers   # only needed for this test
python conversion/test_embeddinggemma_parity.py --max-seq-len 512
```

Encodes 16 multilingual sentences with HF SentenceTransformer and with our
ANE-style PyTorch model side-by-side. Passes when:
- mean cosine(hf, ours) at d=768 ≥ 0.995
- mean cosine after Matryoshka truncate-to-128 + renormalize ≥ 0.98

## Using it together with Gemma 4 / FunctionGemma

Three independent mlpackages; a Swift integrator can compose them into:

- **Model picker** — each bundle stands alone; the chat app can offer
  EmbeddingGemma / FunctionGemma / Gemma 4 E2B as separate entries.
- **RAG hybrid** — EmbeddingGemma encodes documents + query, cosine-ranks
  local chunks, feeds the top-k into Gemma 4 as retrieved context.
- **Tool-calling hybrid** — FunctionGemma as a small, fast tool-selector head
  that emits `<start_function_call>...<end_function_call>`, with Gemma 4
  handling free-form answer generation.

Swift glue for these compositions is out of scope for this PR; the three
bundles publish self-describing `model_config.json` files so the orchestration
lands cleanly upstream.
