# USB model sideload — skip the HF round-trip

For iterating on decode-side optimizations you don't want to pay HF
download cost every time the model bundle changes. Xcode and the app
are already set up for direct USB-C transfer:

- `Info.plist` has `UIFileSharingEnabled=YES` and
  `LSSupportsOpeningDocumentsInPlace=YES`, so `Documents/Models/` is
  reachable from Finder and `xcrun devicectl`.
- Bundle ID is `com.example.CoreMLLLMChat`.
- Team ID matches the workspace and sibling repos, so installing from
  one clobbers the other while preserving `Documents/`.

## Pull the current app data

```bash
DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
xcrun devicectl device copy from \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source Documents/Models/gemma4-e2b \
    --destination ~/Downloads/coreml-llm-artifacts/backup-iphone-2k
```

This gives a faithful copy of whatever the app currently has, including
`chunk{1-4}.mlmodelc`, `prefill_chunk{1-4}.mlmodelc`, embeddings, RoPE
tables, `model_config.json`, `hf_model/tokenizer.json`, etc.

## Push a staged model

`push-model.sh` lives in `~/Downloads/coreml-llm-artifacts/` (not in the
repo — it's session-scoped and depends on a hardcoded device UUID).
The basic `devicectl` command is:

```bash
xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source /path/to/staging/gemma4-e2b \
    --destination Documents/Models/gemma4-e2b \
    --remove-existing-content true
```

- `--remove-existing-content true` clears the target folder first so
  leftover files from an earlier variant don't linger.
- `--destination` must be the full final path. Passing a bare
  `Documents/` as destination silently mis-places the file even though
  the command reports success (seen with a smoke-test `.txt`).
- USB-C gigabit transfer runs at roughly 500 MB/s, so a 5.4 GB model
  pushes in ~10 s.

## Staging directory layout

The app's `ModelDownloader.localModelURL` probes, in order:

1. `chunk1.mlmodelc/weights/weight.bin` — compiled
2. `model.mlmodelc/weights/weight.bin` — legacy monolithic
3. `model.mlpackage/Data/com.apple.CoreML/weights/weight.bin` — package

For chunked models, the app also looks for:

```
gemma4-e2b/
├── chunk{1-4}.mlmodelc/       (or .mlpackage)
├── prefill_chunk{1-4}.mlmodelc   optional — falls back to per-token
├── model_config.json          context_length, sliding_window, etc.
├── embed_tokens_q8.bin, embed_tokens_scales.bin
├── embed_tokens_per_layer_q8.bin, embed_tokens_per_layer_scales.bin
├── per_layer_projection.bin, per_layer_norm_weight.bin
├── cos_sliding.npy, sin_sliding.npy   W=512 rows
├── cos_full.npy, sin_full.npy         ctx rows (2K → 2048, 8K → 8192)
├── hf_model/{tokenizer.json,config.json,tokenizer_config.json}
├── vision.mlmodelc, audio.mlmodelc    optional — multimodal
├── audio_config.json, mel_filterbank.bin   optional — audio input only
└── output_proj_*.npy, embed_proj_weight.npy   optional — audio projection
```

The RoPE tables' row count is authoritative for `context_length`, and
chunk `model.mil`'s `causal_mask_full` tensor shape is the ground truth
for what the compiled chunk expects. Mismatch (e.g. 2K chunks with 8K
config.json) will trip `isChunkCtxMismatched` and auto-invalidate on
load.

## prefillN is declared by the prefill chunks, not the config

`ChunkedEngine.load` reads `prefill_chunk1`'s `hidden_states` input
shape to set `prefillN`. A 2K model built with `prefill_N=64` will
per-token-decode every prompt token past 64, dramatically hurting TTFT.
The faster 2K variant built with `prefill_N=512` batches up to 512
prompt tokens in a single dispatch.

## Caveats

- `devicectl copy to --destination <dir>/` (trailing slash) is broken.
  The file goes somewhere unretrievable and the command reports success.
  Always specify the full final path.
- Background `URLSession` can hold partial-download file handles past
  what a trash-button delete expects; the app now routes deletions
  through a rename-to-`.graveyard/` step so removal succeeds even when
  iOS returns "no permission" on direct `removeItem`.
- The workspace and sibling repos share a bundle ID. Whoever builds
  last wins the install on device. Reinstalling the sibling restores
  its copy without touching `Documents/`.
