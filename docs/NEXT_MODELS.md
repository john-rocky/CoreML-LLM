# Next ANE-Friendly Models to Port

**Last updated:** 2026-04-25

Shortlist of small (1–4B) decoder-only LLMs that map cleanly onto our existing
Core ML / ANE conversion infrastructure. Ranked by ease-of-port × ecosystem value.
Excluded: models we already support (Qwen2.5, Gemma 3 / 4, FunctionGemma,
EmbeddingGemma) and models that fundamentally don't fit ANE (per-block
quantized BitNet/Bonsai-class — see `TERNARY_BONSAI.md`).

## Top picks

### 1. Qwen3-1.7B-Instruct / Qwen3-4B-Instruct  (top pick)

- HF: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B-Instruct-2507`
- 1.7B (28 layers, hidden=2048, GQA 16/8, head_dim=128, tied embed, 32K ctx)
- 4B (36 layers, GQA 32/8, similar tied / RoPE / QK-norm story)
- Apache 2.0; bf16 official; community AWQ/GGUF abundant
- Why ANE-friendly: dense decoder, GQA, RoPE, **QK-norm** — exactly what
  `conversion/models/qwen3.py` already supports. Drop-in via existing
  `convert.py --model qwen3-1.7b` once registered.
- Speed estimate: 1.7B ≥ 25 tok/s INT4 chunked, 4B ~12-14 tok/s (Gemma 4 E4B class).
- ANEMLL has a working CoreML port that validates feasibility.

**Effort to port:** add 1-2 entries to `MODEL_REGISTRY` in `config.py`. Done.

### 2. Gemma 3 1B / 4B  (best architectural match)

- HF: `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-270m`
- 1B is text-only (32K ctx); 4B is multimodal (128K ctx)
- Same **5-local-SWA : 1-global** pattern as Gemma 4 — our `gemma4_swa_*.py`
  chunking infra reuses ~80%
- Google ships **QAT INT4** weights officially:
  `google/gemma-3-4b-it-qat-q4_0-unquantized` — palette-friendly, no per-block
  scales (unlike Bonsai), so they actually run on ANE
- Gemma terms; commercial OK with usage policy; released Mar 25, 2025

**Effort to port:** Gemma 3 has FunctionGemma support already
(`conversion/models/gemma3.py`); 1B / 4B differ mostly in size + dual SWA/full
window. Reuse Gemma 4 SWA chunking; ~1-2 days.

### 3. Llama-3.2-1B-Instruct / 3B-Instruct  (lowest risk)

- HF: `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`
- 1B: 16 layers, hidden=2048, GQA 32/8; 3B: 28 layers, hidden=3072, GQA 24/8
- 128K ctx via Llama-3 RoPE scaling
- Vanilla decoder-only — no QK-norm, no SWA — simplest possible ANE port
- Llama 3.2 Community License (commercial OK with MAU caps; not EU)
- Reported: ~47–62 tok/s on iPhone 17 Pro for Llama-3.2-1B via ANEMLL → our
  build should hit similar
- Existing reference: `smpanaro/Llama-3.2-1B-Instruct-CoreML` on HF
- Released Sep 2024 — older but mature & widely requested

**Effort to port:** new `conversion/models/llama.py` (mirror `qwen2.py`),
verify Llama-3 RoPE scaling. Half a day.

### 4. SmolLM3-3B  (best quality/size in 3B class)

- HF: `HuggingFaceTB/SmolLM3-3B`
- 3B, decoder-only, GQA, **NoPE 3:1** (no positional embed every 4th layer)
- 64K native ctx (128K via YaRN)
- Apache 2.0
- Outperforms Llama-3.2-3B and Qwen2.5-3B at 3B scale (HF benchmarks Jul 2025)
- Risk: 3:1 NoPE pattern needs a small wrapper change (skip RoPE on every 4th layer)

**Effort to port:** ~1 day, mostly the NoPE wrapper.

## Honorable mentions

| model | HF | why interesting | why not first |
|---|---|---|---|
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | 3.8B, MIT, popular; INT4 ONNX exists | fractional RoPE (25% NoPE per head) → custom RoPE op. Qwen3-4B fills same niche |
| Ministral-3-3B-Instruct-2512 | `mistralai/Ministral-3-3B-Instruct-2512` | 3B, Apache 2.0, multimodal, 256K ctx | released Dec 2025, too new to validate |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | reasoning-tuned | comes free via existing `qwen2.py` path; just register |
| Apple OpenELM-3B | `apple/OpenELM-3B-Instruct` | Apple-built | layer-wise variable-width FFN doesn't fit our chunking; weak instruct quality |

## Skip / deprioritize

- **Phi-3.5-mini** — superseded by Phi-4-mini
- **TinyLlama** — too old, dwarfed by SmolLM3
- **Liquid LFM2 / IBM Granite 4.0 / Hymba / Falcon-Mamba** — Mamba/SSM hybrid;
  needs new SSM kernel beyond our Gated-DeltaNet code (different op set)
- **Apple Foundation Models 3B (Apple Intelligence)** — weights not released
- **Cohere R7B / Mistral 7B / Qwen3-8B** — exceed iPhone ANE compile budget;
  Mac-only would still need 4-chunk split

## Recommended porting order

1. **Qwen3-1.7B-Instruct** — almost zero code; validates the new `qwen3.py`
   path on a real instruct model; ships fastest.
2. **Gemma 3 4B (QAT)** — reuses Gemma 4 SWA chunking; native INT4 weights;
   Google brand recognition.
3. **Llama-3.2-3B-Instruct** — most-requested by community; ANEMLL parity check.
4. **SmolLM3-3B** — best quality/size at 3B class; Apache 2.0; differentiator.
5. (later) Qwen3-4B-Instruct-2507, Phi-4-mini, Ministral-3-3B.

## What infrastructure we have ready

- `conversion/models/qwen3.py` — Qwen3 architecture (QK-norm, tied embed)
  ready for Qwen3-1.7B / 4B / 8B
- `conversion/models/qwen2.py` — Qwen2.5 architecture, also fits
  DeepSeek-R1-Distill-Qwen-* finetunes
- `conversion/models/gemma3.py` + Gemma 4 SWA chunks — Gemma family backbone
- `docs/DECODE_STATE_LAYOUTS.md` — decode-time state pattern checklist for
  any new model
- `docs/ADDING_MODELS.md` — end-to-end walkthrough for adding a new arch

## Sources for the shortlist

- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen3 tech report](https://arxiv.org/pdf/2505.09388)
- [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it), [4B](https://huggingface.co/google/gemma-3-4b-it), [tech report](https://arxiv.org/html/2503.19786v1)
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Ministral-3-3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
- [smpanaro/Llama-3.2-1B-Instruct-CoreML](https://huggingface.co/smpanaro/Llama-3.2-1B-Instruct-CoreML), [ANEMLL](https://www.anemll.com/)
