# LM-head split iPhone A/B/C deploy procedure (2026-04-25)

**Goal.** Compare chunk3 latency and end-to-end tok/s on iPhone 17 Pro for
three variants of the lm_head Conv2d:

| variant   | lm_head structure                       | folder name                         |
|-----------|------------------------------------------|-------------------------------------|
| baseline  | 1 × Conv2d(2304 → 262144)                | `gemma4-e2b-lmsplit-baseline`       |
| lmsplit8  | 8 × Conv2d(2304 → 32768) + concat        | `gemma4-e2b-lmsplit8`               |
| lmsplit16 | 16 × Conv2d(2304 → 16384) + concat       | `gemma4-e2b-lmsplit16`              |

ANEMLL's `qwen_model.py:1006-1124` claims 16-way is the right choice for
vocab=128k Qwen3. Vocab=262k Gemma 4 E2B is 2× larger so we're testing
16-way (anemll-style), 8-way (intermediate), and 1-way (current production)
to find the knee.

## Prerequisites — what's already done (Mac)

- `conversion/models/gemma4_swa_chunks.py` — `SWAChunk4_LMSplit(n_splits)`
  class added. Smoke-tested: bitwise-identical token_id vs SWAChunk4 across
  n=2/8/16 (`conversion/smoke_lmsplit.py`).
- `conversion/build_gemma4_3way.py` — `--lm-splits {1,2,4,8,16}` flag.
- mlpackages built for chunk3 lmsplit8 + lmsplit16:
  - `output/gemma4-e2b/chunks_3way_lmsplit8/chunk3_3way.mlpackage` (503 MB)
  - `output/gemma4-e2b/chunks_3way_lmsplit16/chunk3_3way.mlpackage` (503 MB)
- mlmodelc compiled (same dirs).
- `conversion/sanity_chunk3_lmsplit.py` — Mac CPU_AND_NE smoke run passes
  for both variants. (Mac timing irrelevant — first-load compile cache
  dominates; iPhone is the decision point.)

## Step 1 — Build baseline chunks (running)

```bash
python3.12 conversion/build_gemma4_3way.py --model gemma4-e2b
xcrun coremlcompiler compile output/gemma4-e2b/chunks_3way/chunk1_3way.mlpackage \
    output/gemma4-e2b/chunks_3way/
xcrun coremlcompiler compile output/gemma4-e2b/chunks_3way/chunk2_3way.mlpackage \
    output/gemma4-e2b/chunks_3way/
xcrun coremlcompiler compile output/gemma4-e2b/chunks_3way/chunk3_3way.mlpackage \
    output/gemma4-e2b/chunks_3way/
```

Produces `output/gemma4-e2b/chunks_3way/chunk{1,2,3}_3way.mlmodelc`.

## Step 2 — Assemble three bundles

```bash
./scripts/assemble_lmsplit_bundles.sh
# → build/lmsplit_bundles/{baseline,lmsplit8,lmsplit16}/
```

Each bundle is **fully self-contained for 3-chunk decode** (~4.8 GB):

| File | Source | Notes |
|------|--------|-------|
| `chunk1.mlmodelc` | renamed from our `chunk1_3way.mlmodelc` | L0-7, identical to 4-chunk's chunk1 |
| `chunk2_3way.mlmodelc` | baseline build | L8-24 merged |
| `chunk3_3way.mlmodelc` | variant build (1/8/16-way) | L25-34 + LM head |
| `prefill_chunk{1-4}.mlmodelc` | staging-2k-fast-prefill | shared 4-chunk-format prefill |
| `embed_tokens_q8.bin` + scales | staging | INT8 embedding lookup |
| `embed_tokens_per_layer_q8.bin` + scales | staging | per-layer embed (2.2 GB) |
| `per_layer_projection.bin`, `per_layer_norm_weight.bin` | staging | |
| `cos_sliding.npy`, `sin_sliding.npy`, `cos_full.npy`, `sin_full.npy` | staging | RoPE tables |
| `hf_model/tokenizer*.json`, `config.json` | staging | tokenizer assets |
| `model_config.json` | staging | runtime config |

Vision/audio/MTP drafter omitted — text-only A/B. Total push to device:
~14.4 GB (3 bundles × 4.8 GB).

## Step 3 — Add ModelInfo entries (Swift)

In `Sources/CoreMLLLM/ModelDownloader.swift`, mirror the
`gemma4e2bLookaheadProbe` pattern. Add three entries after it:

```swift
public static let gemma4e2bLMSplitBaseline = ModelInfo(
    id: "gemma4-e2b-lmsplit-baseline",
    name: "Gemma 4 E2B (lm_split=1, baseline)", size: "1.6 GB",
    downloadURL: "",
    folderName: "gemma4-e2b-lmsplit-baseline")

public static let gemma4e2bLMSplit8 = ModelInfo(
    id: "gemma4-e2b-lmsplit8",
    name: "Gemma 4 E2B (lm_split=8)", size: "1.6 GB",
    downloadURL: "",
    folderName: "gemma4-e2b-lmsplit8")

public static let gemma4e2bLMSplit16 = ModelInfo(
    id: "gemma4-e2b-lmsplit16",
    name: "Gemma 4 E2B (lm_split=16, anemll-style)", size: "1.6 GB",
    downloadURL: "",
    folderName: "gemma4-e2b-lmsplit16")
```

…then in `defaults`:

```swift
if experimental {
    list.insert(gemma4e2bEagle3, at: 2)
    list.insert(gemma4e2bLookaheadProbe, at: 3)
    list.insert(gemma4e2bLMSplitBaseline, at: 4)
    list.insert(gemma4e2bLMSplit8, at: 5)
    list.insert(gemma4e2bLMSplit16, at: 6)
}
```

The runtime contract is identical to the production gemma4-e2b chunks
(same input/output names per `SWAChunk4_LMSplit`), so `ChunkedEngine`
needs no changes. The picker-side change is purely metadata.

## Step 4 — Sideload to iPhone

```bash
DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)

for variant in baseline lmsplit8 lmsplit16; do
    xcrun devicectl device copy to \
        --device "$DEVICE" \
        --domain-type appDataContainer \
        --domain-identifier com.example.CoreMLLLMChat \
        --source "build/lmsplit_bundles/$variant" \
        --destination "Documents/Models/gemma4-e2b-$variant"
done

# Verify
xcrun devicectl device info files \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --subdirectory Documents/Models
```

Production `gemma4-e2b/` bundle is NOT touched.

## Step 5 — Measure on iPhone 17 Pro

In Xcode:
1. Scheme → Run → Arguments → Environment, add **all three**:
   - `LLM_SHOW_EXPERIMENTAL=1` — exposes the 3 lm-split bundles in the picker
   - `LLM_3CHUNK=1` — **required** so the runtime loads
     `chunk1 + chunk2_3way + chunk3_3way` instead of the (absent) 4-chunk
     decode set
   - `LLM_PROFILE_EVERY_STEP=1` — emits per-step `[Profile]` lines
2. Build + run.
3. Switch model picker through the three lm-split variants in turn.
4. For each, type the same prompt (e.g. `"Write a short paragraph about
   sushi."`), generate ~150 tokens, copy the steady-state `[Profile]`
   log lines (cycles 30+).

Record:

| variant   | c1 ms | c2 ms | c3 ms | sum | tok/s | notes |
|-----------|-------|-------|-------|-----|-------|-------|
| baseline  |       |       |       |     |       |       |
| lmsplit8  |       |       |       |     |       |       |
| lmsplit16 |       |       |       |     |       |       |

Run each variant with the **same prompt and same generation length** for
fair comparison. Take median over 3 generations to filter thermal /
warm-up noise.

## Decision rule

| Result | Action |
|--------|--------|
| Best variant ≥ +5 % E2E tok/s vs baseline | Ship it as production default; deprecate other splits |
| Best variant +2 — 5 % | Keep as opt-in `LLM_SHOW_EXPERIMENTAL=1`, document in `project_chunk_pipeline_phase1.md`-style note |
| All within ±2 % | Drop variant code, mark "anemll lm_split is workload-specific, not universal — vocab=262k baseline is fine" in REJECTED_APPROACHES.md |
| Any variant has token_id divergence (different output text) | Bug — investigate before shipping; check argmax / softcap chain |

## Expected outcome (priors, not predictions)

Theoretical chunk3 weight-byte share of lm_head: ~54 % (302/562 MB).
ANEMLL's claim is that 16-way reduces lm_head latency 20-30 % via better
SRAM tile fit. If true:

- chunk3 latency change: −10 to −16 %
- E2E tok/s change: chunk3 is ~30 % of step → +3 to +5 %

A null result (no measurable change) is also informative — it means the
single Conv2d(2304→262144) is already ANE-optimal for this vocab size,
and we can stop worrying about lm_head splits forever.

## Open question for follow-up (only if a winner emerges)

ANEMLL also has `ENABLE_LOGITS2=True` mode where the 16 logits return
**separately** (no in-model concat) and the runtime does argmax over 16
arrays. The motivation is that the concat itself runs on CPU. We did NOT
test that variant yet; if 16-way concat-in-model wins, the concat-out
variant might be faster still. Tracked as a follow-up — don't implement
unless step-5 results justify it.
