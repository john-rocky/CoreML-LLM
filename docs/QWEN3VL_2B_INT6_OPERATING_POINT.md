# Qwen3-VL 2B stateful — INT4 dead, **INT6 g=32 is the working operating point**

**Date:** 2026-05-03
**Branch:** `research/qwen3vl-2b-int4-g32`
**Model:** `Qwen/Qwen3-VL-2B-Instruct` text decoder, stateful 4×7-layer chunks
**Conclusion:** Every INT4 weight-only variant tried (k-means palettize at g=32 / g=16 / g=8, INT4 linear quantize, INT4 + per-channel-scale, mixed precision INT8-attn / INT8-MLP) is unusable on Qwen3-VL 2B. **INT6 k-means palettize at `granularity="per_grouped_channel", group_size=32` is shippable** — coherent EN + JP, 25.78 % avg top-1 match against fp16, **−15 % bundle vs INT8 (1917 MB vs 2266 MB) at +13 % Mac decode tok/s (31.2 vs 27.7)**.

---

## Headline numbers

Mac Studio M4, coremltools 9.0, ANE-only at runtime (`compute_units=CPU_AND_NE`), greedy argmax decode, `max_new=64`, chat template applied. Match rate = greedy top-1 token agreement vs fp16 reference, averaged over 6 prompts × 64 generated tokens.

| Variant | Bundle (body+head, no embed) | Mac decode tok/s | Top-1 match vs fp16 | Quality |
|---|---:|---:|---:|---|
| fp16 (reference) | 3,442 MB | 11.7 | — | coherent |
| **INT8 per_tensor (current production)** | **1,645 MB** | **27.5** | **62.5 %** | coherent + factually correct |
| **INT6 g=32 per_grouped_channel** ⭐ | **1,168 MB** | **31.1** | **25.78 %** | **coherent + factually correct + valid JP** |
| INT4 g=32 per_grouped_channel (Gemma 4 recipe) | 825 MB | 31.4 | 1.30 % | repetition collapse, JP blank |
| INT4 g=16 per_grouped_channel | ~830 MB | 31.0 | 1.82 % | repetition collapse, JP blank |
| INT4 g=8 per_grouped_channel | ~830 MB | 31.2 | 4.95 % | repetition collapse (Paris answered once before loop) |
| INT4 linear quantize per_channel symmetric | 825 MB | 19.1 | 1.82 % | digit/whitespace spam (also poorly ANE-routed) |
| INT4 + `enable_per_channel_scale=True` (AWQ-style) | — | — | — | **MPS runtime rejects op shape on macOS 26** (`mps.dequantize` operand error) |
| Mixed: INT8 attn (q/k/v/o) + INT4 MLP | 891 MB | 30.4 | 5.99 % | Paris answered, JP digits-only |
| Mixed: INT8 MLP (gate/up/down) + INT4 attn | 1,329 MB | 28.3 | 7.81 % | starts valid JP recipe ("もちろんです！…ステップ1") then drifts EN |

Bundle artifacts (kept on Mac for follow-up):
- `/tmp/qwen3vl-int4-g32-build/_fp16_intermediate/` — fp16 reference chunks (3.2 GB)
- `/tmp/qwen3vl-int8-pertensor/` — INT8 baseline (2.2 GB)
- `/tmp/qwen3vl-int6-g32/` — **the winner** (1.8 GB total inc. embed)

JSONs: `/tmp/qwen3vl-int4-g32-build/bench_int6.json` (the 4-bundle final), plus `bench_g16.json`, `bench_g8.json`, `bench_lin.json`, `bench_mixed.json` for the failed paths.

---

## What "broken" looks like (vs what works)

**INT4 g=32 (broken)** — pure collapse:

```
[en_factual_paris] " One sentence. One sentence. One sentence. ..."
[en_factual_japan] " The Norwegian Wood (the novel) was written by Norwegian Wood (the novel). ..."
[jp_greeting]      "                                                                "   (blank)
[jp_recipe]        "                                                                "   (blank)
[reason_math]      " How do they move?  The first train leaves at 9:00, and the second at 9:30, so they meet at 9:30. ..." (wrong fact, semi-coherent)
[code_python]      " The function should be named ```````````````````````` ..."
```

**INT6 g=32 (works)** — coherent across all 6 prompts:

```
[en_factual_paris] " The capital of France is Paris.  a. Paris  b. Rome  c. London  d. Berlin  a. Paris ..."
[en_factual_japan] " The novel Norwegian Wood was written by Haruki Murakami. It is a novel that was published in 1987. It is a fictional novel that is set in Japan. It is a novel that is known for its surreal and dreaml…"
[jp_greeting]      "   はい、元気です。   ありがとうございます。   どうぞ、お気軽にお知らせください。   ありがとうございます。 ..."
[jp_recipe]        "   また、その中で「お好みの具材」は、どのようなものを使えばよいでしょうか？  ---  ### ステップ1：材料の準備  まず、餃子の基本的な材料を準備します。  - **餃子皮**：小麦粉（または玄米粉）と水"
[reason_math]      " Also, what is the formula for the distance traveled by an object moving at a constant speed?  The formula for the distance traveled by an object moving at a constant speed is ( d = v cdot t ), whe…"
[code_python]      " The function should take an integer `n` as input and return the nth Fibonacci number. The function should be efficient and handle edge cases such as `n = 0` and `n = 1`.  Here is the code: ```python"
```

**INT8 per_tensor (production)** for comparison:

```
[en_factual_paris] " The capital of France is Paris.  user: What is the capital of France?  assistant: The capital of France is Paris. ..."
[en_factual_japan] " The novel Norwegian Wood was written by Haruki Murakami. It is a novel that was published in 1987. ..."
[jp_greeting]      "   私は元気です。   私は元気です。 ..." (greedy loop in fp16 too; not a quant artefact)
[jp_recipe]        "   また、どの食材を用いるべきか、そしてその理由を説明してください。  ---  **ステップ1：材料の準備** ..."
[reason_math]      " Also, what is the formula for the distance traveled by an object moving at a constant speed? ..."
[code_python]      " The function should take an integer `n` as input and return the Fibonacci number at position `n`. ..."
```

INT6's answers are *different paths* from fp16/INT8 (e.g. multiple-choice Paris, longer Norwegian Wood, formal greeting style) but all factually correct + grammatically valid in their target language. The 25.78 % top-1 match understates the quality because greedy-decode A/B punishes any divergence even if both completions are correct; INT8 gets 62.5 % because it stays bit-near fp16, not because INT6 outputs are unusable.

Greedy repetition on JP greeting is a property of the model + decode strategy (no temperature, no rep_penalty, short context) — fp16 does the exact same loop. Production runtime applies sampling + rep_penalty so this doesn't surface in real chat.

---

## Why INT4 fails and INT6 works

Per-grouped-channel k-means at INT4 only has 16 LUT entries per group. Qwen3-VL 2B weight rows have wider dynamic range than Gemma 4's (extreme outliers per group), so 16 quantization levels can't cover the range without dropping critical signal in the bulk weights. Smaller group sizes (g=16, g=8) only marginally help because the LUT entry count is fixed.

INT6 has 64 LUT entries per group — 4× the headroom. Empirically that's enough to capture Qwen3-VL 2B's per-group distribution.

This was not predictable from Gemma 4's success at INT4 g=32. The recipe is model-specific.

---

## Untried levers (priority order if revisited)

1. **Calibration-aware palettization via `coremltools.optimize.torch`** (GPTQ/AWQ at the PyTorch level, then re-export). Gemma 4's W4A8 attempt died on activation quant, not weight quant — so weight-only AWQ for Qwen3-VL is still open. Would need PyTorch-level fine-tune pipeline.
2. **INT4 + `enable_per_channel_scale=True` patched for macOS 26**. The `mps.dequantize` op shape error happens at ANE compile/load — coremltools 9.0 emits an unsupported tensor layout. Either upgrade to a fixed coremltools / iOS, or hand-patch the MIL output. Could unlock real INT4.
3. **QAT (Quantization-Aware Training)** — last resort, requires multi-day GPU job.

---

## What's reproducible

Branch `research/qwen3vl-2b-int4-g32` adds:

- `conversion/build_qwen3_vl_2b_stateful_chunks.py` — `--granularity per_grouped_channel --group-size N` flags (default unchanged: `per_tensor`).
- `conversion/repalettize_qwen3vl_stateful.py` — fp16 chunks → any quant variant in ~1–3 min/chunk:
  - `--mode palettize --nbits {2,3,4,6,8} --granularity {per_tensor,per_grouped_channel} --group-size N [--per-channel-scale]`
  - `--mode linear --nbits {4,8} --linear-granularity {per_tensor,per_channel} --linear-mode {LINEAR,LINEAR_SYMMETRIC}`
  - `--mode mixed_attn8_mlp4` / `mixed_mlp8_attn4` (auto-enumerates const op names per chunk for op_name_configs since coremltools uses exact-match dict, not glob).
- `conversion/bench_qwen3vl_stateful_quant.py` — Mac stateful greedy-decode A/B/C harness. Per-step top-1 match vs reference, decode tok/s warm, repetition / language-switch heuristic flags.

Reproduction of the winning path:

```bash
PY=/Users/majimadaisuke/.venv-coreml-llm-py312/bin/python

# 1. Build fp16 once (~15 min on Mac Studio M4) — same as the existing builder.
$PY conversion/build_qwen3_vl_2b_stateful_chunks.py \
    --out-dir /tmp/qwen3vl-build --nbits 0 --keep-fp16

# 2. Palettize fp16 → INT6 g=32 (~5 min for 5 chunks).
$PY conversion/repalettize_qwen3vl_stateful.py \
    --fp16-dir /tmp/qwen3vl-build/_fp16_intermediate \
    --out-dir /tmp/qwen3vl-int6-g32 \
    --mode palettize --nbits 6 \
    --granularity per_grouped_channel --group-size 32 \
    --copy-embed-from /tmp/qwen3vl-build/qwen3_vl_2b_stateful_chunks

# 3. Bench against fp16 + the existing INT8 production baseline.
$PY conversion/bench_qwen3vl_stateful_quant.py \
    --bundle fp16=/tmp/qwen3vl-build/_fp16_intermediate \
    --bundle int8_pt=/path/to/existing/int8/bundle \
    --bundle int6_g32=/tmp/qwen3vl-int6-g32 \
    --max-new 64 \
    --out-json /tmp/bench_int6.json
```

---

## Decision

**Adopt INT6 g=32 per_grouped_channel as the new Qwen3-VL 2B stateful operating point**, replacing INT8 per_tensor. Need:

1. Generalize the `--granularity / --nbits` flags into the production `build_qwen3_vl_2b_stateful_chunks.py` defaults.
2. Iphone 17 Pro validation: confirm INT6 chunks load on iPhone ANE (palette-LUT path is ANE-supported on iOS 18, but need to verify decode tok/s and visual quality on a real photo).
3. New HF artefact: `mlboydaisuke/qwen3-vl-2b-stateful-int6-coreml` (or replace the existing INT8 repo).
4. README + picker bundle URL update.

These are out of scope for this Mac probe — see `docs/INFLIGHT.md` if you pick this up.
