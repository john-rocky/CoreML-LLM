# FunctionGemma-270M on CoreML

Gemma 3 270M decoder fine-tuned for function calling (`google/functiongemma-270m-it`).
Same Transformer shape as vanilla Gemma 3 270M; the only architectural
difference is the **chat template** — `developer` role for function
declarations, and `<start_function_call>` / `<end_function_call>` markers in
the output.

## Architectural fit

Strict subset of Gemma 4 E2B (see `docs/experiments/GEMMA4_ARCHITECTURE_VERIFIED.md`):

| Feature | Gemma 4 E2B | Gemma 3 270M (FunctionGemma) |
|---|---|---|
| Sandwich norms (4 per layer) | ✓ | ✓ |
| QK norm | ✓ | ✓ |
| GeGLU with tanh-approx GELU | ✓ | ✓ |
| GQA | ✓ (8/1) | ✓ (4/1) |
| Tied embeddings | ✓ | ✓ |
| Dual RoPE (local θ / global θ) | ✓ (10K / 1M) | ✓ (10K / 1M) |
| `sliding_window_pattern` alternation | every 5th full | every 6th full |
| Per-layer embeddings (PLE) | ✓ | — |
| `layer_scalar` | ✓ | — |
| `v_norm` (scaleless) | ✓ | — |
| KV sharing (layers 15–34) | ✓ | — |
| Dual head_dim (256 / 512) | ✓ | single head_dim (256) |
| Logit softcap | ✓ (30.0) | — (270M does not use it) |

Result: the Gemma 3 decoder reuses the same `exporter.py` path as Gemma 4 with
a thinner wrapper. See:

- `conversion/models/gemma3.py` — Gemma 3 Conv2d/ANE decoder (model + config + weight loader).
- `conversion/models/gemma3_wrapper.py` — monolithic tracing wrapper for `exporter.py`.
- `conversion/build_functiongemma_bundle.py` — bundle builder.

## Conversion

Prerequisite: `huggingface-cli login` (FunctionGemma is gated).

```bash
# fp16 bundle for parity testing
python conversion/build_functiongemma_bundle.py --ctx 2048 --quantize none

# Shippable INT4-palettized bundle (~135 MB encoder weights on disk)
python conversion/build_functiongemma_bundle.py --ctx 2048 --quantize int4
```

Output at `output/functiongemma-270m/bundle/`:

```
model.mlpackage               # monolithic stateful decoder
cos_sliding.npy  sin_sliding.npy
cos_full.npy     sin_full.npy    # RoPE tables (max_pos = 2·ctx)
model_config.json
hf_model/
    tokenizer.json  tokenizer_config.json  chat_template.jinja  ...
```

## I/O contract

Same as Gemma 4 E2B's monolithic export (see `conversion/exporter.py:236–254`):

| Name | Shape | dtype | Role |
|---|---|---|---|
| `input_ids` | (1, 1) | int32 | current token |
| `position_ids` | (1,) | int32 | current position (for RoPE index_select) |
| `causal_mask` | (1, 1, 1, ctx) | fp16 | additive; −1e4 outside the window, 0 inside |
| `update_mask` | (1, 1, ctx, 1) | fp16 | 1.0 at current position, 0.0 elsewhere |
| → `token_id` | (1,) | int32 | argmax |
| → `token_logit` | (1,) | fp16 | value at argmax |
| state `kv_cache_0` | (2·L, kv_heads, ctx, head_dim) | fp16 | unified K/V buffer |

The Swift runtime is expected to pre-bake the mask per layer (sliding-window
rows have −1e4 outside the window). Same convention Gemma 4 uses today.

## Using the function-calling chat template

The prompt format (from `chat_template.jinja`) looks roughly like:

```
<start_of_turn>developer
{"name": "get_weather", "parameters": {"location": "string"}}
<end_of_turn>
<start_of_turn>user
What's the weather in Tokyo?
<end_of_turn>
<start_of_turn>model
<start_function_call>get_weather{"location": "Tokyo"}<end_function_call>
```

Tokenize the full template with `AutoTokenizer.from_pretrained(hf_model/)` and
let the model decode. Parse the text between `<start_function_call>` and
`<end_function_call>` as JSON-like arguments.

The `model_config.json` emitted by the bundle builder embeds:

```json
"chat_format": "functiongemma",
"function_call_markers": {
    "start": "<start_function_call>",
    "end": "<end_function_call>"
}
```

so a downstream Swift integrator can look up the markers without parsing the
Jinja template itself.

## Parity check

```bash
python conversion/test_gemma3_parity.py --ctx 512
```

Streams 32 tokens through HF `AutoModelForCausalLM` (fp32) and through our
`Gemma3MonolithicWrapper` (fp16) side-by-side, asserts ≥ 90% top-1 agreement.
This is a PyTorch-level check; the `.mlpackage` itself should be validated on
device once the Swift runtime is wired up.

## Swift integration (not in this PR)

Out of scope for the initial conversion work. The bundle layout matches what
the existing Gemma 4 chunked engine can consume *except* the monolithic
`model.mlpackage` lives in a single file rather than four chunks. A minimal
adapter in `Sources/CoreMLLLM/` will call the mlpackage directly (load state,
build mask per step, argmax). See `docs/ADDING_MODELS.md` for the general
checklist.
