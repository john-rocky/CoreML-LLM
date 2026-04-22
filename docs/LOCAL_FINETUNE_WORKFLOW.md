# Local Fine-Tune → iPhone Workflow (Gemma 4)

How to take a locally fine-tuned Gemma 4 E2B / E4B checkpoint (LoRA merge,
quantization experiment, patched weights — anything) and produce a sideload-ready
CoreML bundle that the CoreMLLLMChat app will recognize as "downloaded".

Companion reference to [`CONVERSION.md`](CONVERSION.md). This document focuses
on the gotchas specific to *custom weights*; for base-model conversion, the
quick-start in `CONVERSION.md` still applies.

## TL;DR

```sh
# 1. Produce HF-format safetensors with model.language_model.* naming (see below).
# 2. Build the on-device bundle:
python conversion/build_gemma4_bundle.py \
    --model gemma4-e2b \
    --hf-dir /path/to/your_merged_model \
    --ctx 2048 \
    --output ./output/your_model/

# 3. USB sideload to Documents/Models/<folderName>/ on device (see USB_MODEL_SIDELOAD.md).
# 4. Register a ModelInfo entry in Sources/CoreMLLLM/ModelDownloader.swift
#    with downloadURL: "" so the app treats the sideloaded folder as
#    already-downloaded.
```

## The `--hf-dir` contract (important)

`build_gemma4_bundle.py` reads weights from `--hf-dir` in two places:

1. **`build_gemma4_bundle.py` itself** — extracts `embed_tokens`, PLE,
   `per_layer_model_projection`, tokenizer.
2. **`build_verify_chunks.py`** (invoked as a subprocess) — loads the full
   text decoder and builds the four transformer chunks.

Both read from the same `--hf-dir`. If you see the build log print something
like

```
Loading gemma4-e2b from .../output/gemma4-e2b/hf_model...
```

when you passed `--hf-dir /something/else`, the subprocess call is broken.
As of PR #123 this is fixed — if it regresses, grep for `_run_chunks_build`
and make sure `hf_dir` is forwarded to the subprocess command.

Symptom of the old bug: embeddings are fine-tuned but the transformer layers
are base. Output looks coherent, is *in the right language*, but knows
nothing your LoRA taught it.

## Weight naming schemes you may encounter

`build_gemma4_bundle.py` expects HF multimodal naming:

```
model.language_model.embed_tokens.weight
model.language_model.layers.{i}.self_attn.q_proj.weight
model.language_model.norm.weight
model.vision_tower.*         (optional; passthrough only for text bundle)
model.embed_vision.*         (optional)
model.audio_tower.*          (optional)
model.embed_audio.*          (optional)
```

Training frameworks save in different conventions. The three we've seen:

| Scheme | Example key | Source |
|--------|-------------|--------|
| `hf_multimodal` (target) | `model.language_model.embed_tokens.weight` | `google/gemma-4-E2B-it` as-is |
| `mlx_nested` | `language_model.model.embed_tokens.weight` | `mlx_lm.fuse` output |
| `text_only_flat` | `model.embed_tokens.weight` | Hypothetical text-only `Gemma4ForCausalLM` |

If your merged checkpoint is not in `hf_multimodal` layout, rename weights
before calling `build_gemma4_bundle.py`. A reference remapper lives at
`finetune/scripts/02b_post_fuse.py` — it detects the scheme, remaps keys,
and backfills non-text weights (vision/audio towers) from the original HF
repo so the output directory mirrors the full multimodal checkpoint.

## Why backfill vision/audio weights?

`build_gemma4_bundle.py` today only consumes text weights + `embed_tokens_per_layer`
and friends, so strictly speaking a text-only merged dir works. But mirroring
the full multimodal layout has two practical benefits:

1. `convert_gemma4_multimodal.py` (the multimodal variant) will work on the
   same directory without re-assembling anything — useful when Step 2 (vision
   fine-tune) comes along.
2. `config.json` in a multimodal checkpoint has `text_config` / `vision_config`
   sections. `build_gemma4_bundle.py` reads
   `cfg.get("text_config", cfg)`, so either layout works — but a flat
   text-only config won't carry the fields that `convert_gemma4_multimodal.py`
   later needs.

The overhead is one `huggingface_hub.snapshot_download` of the base repo
(~9 GB) plus ~10 GB disk for the recombined safetensors file. The bundle
builder still extracts only the text pieces it needs.

## Step-by-step: merging a LoRA adapter

Using [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon:

```sh
# Assumes you trained with mlx_lm.lora and saved adapters.safetensors.
mlx_lm.fuse \
    --model google/gemma-4-E2B-it \
    --adapter-path ./runs/my_lora \
    --save-path ./merged/my_model-raw

# Normalize naming + backfill non-text weights:
python /path/to/finetune/scripts/02b_post_fuse.py \
    --merged ./merged/my_model-raw \
    --output ./merged/my_model \
    --base-repo google/gemma-4-E2B-it

# Build bundle (fix for #123 must be present):
python conversion/build_gemma4_bundle.py \
    --model gemma4-e2b \
    --hf-dir ./merged/my_model \
    --ctx 2048 \
    --output ./output/my_model
```

Expected log lines that confirm the FT weights are actually used:

```
Model:   gemma4-e2b
HF dir:  /path/to/merged/my_model          ← must match your --hf-dir
...
Building 4 chunks via build_verify_chunks.py --model gemma4-e2b --ctx 2048
Loading gemma4-e2b from /path/to/merged/my_model...   ← NOT output/.../hf_model
```

## Registering the new model in the picker

`Sources/CoreMLLLM/ModelDownloader.swift`:

```swift
public static let myModel = ModelInfo(
    id: "my-model-ft",
    name: "Gemma 4 E2B (My FT)",
    size: "3.7 GB",
    downloadURL: "",            // sideload-only → isDownloaded checks folder
    folderName: "my-model-ft")  // matches sideload destination

public static var defaults: [ModelInfo] {
    var list: [ModelInfo] = [..., myModel, ...]
    ...
}
```

With `downloadURL: ""` the app treats the folder as "downloaded" whenever
`Documents/Models/<folderName>/model_config.json` exists. No HF hosting
needed for personal / internal FT variants.

## Sideload

See [`USB_MODEL_SIDELOAD.md`](USB_MODEL_SIDELOAD.md) for the `xcrun devicectl`
command. Skip the intermediate `chunks/` subdirectory (it holds uncompiled
`.mlpackage` files used only at build time; compiled `.mlmodelc` folders in
the bundle root are what the runtime loads).

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| On-device output is fluent but ignores your FT content | `--hf-dir` not honored by chunk builder | Make sure you're on PR #123 or later |
| `KeyError: model.language_model.X` during bundle build | Merged checkpoint uses non-HF naming | Run the remapper (`02b_post_fuse.py`) |
| Picker doesn't show your sideloaded model | Folder name mismatch, or `downloadURL` is non-empty | Match `folderName` exactly; `downloadURL: ""` |
| ANE compile failure on device | Chunks stuck in CPU/GPU cache from prior variant | Delete `Documents/Models/.../chunks/` (not needed at runtime); force re-compile by removing the folder and re-sideloading |
| `aten::Int` in trace during conversion | Your LoRA rank / fusion introduced shape-dependent ops | See `docs/ADDING_MODELS.md` → "Common fixes for aten::Int" |

## See also

- `conversion/build_gemma4_bundle.py` — orchestrator
- `conversion/build_verify_chunks.py` — chunk builder (subprocess)
- `finetune/` (sibling repo) — reference LoRA pipeline with safe defaults
  (prompt caching for data synth, `--hf-dir`-aware scripts, checkpoint swap
  for overfit avoidance)
