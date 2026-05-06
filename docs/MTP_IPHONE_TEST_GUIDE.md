# iPhone test guide — centroid MTP drafter

## Prerequisites

* iPhone 17 Pro connected via USB, unlocked, paired (`xcrun devicectl
  list devices` should show it as `connected`).
* CoreMLLLMChat app installed (build from Examples/CoreMLLLMChat with
  Xcode).
* Gemma 4 model already downloaded into the app.
  - Open CoreMLLLMChat
  - Tap model picker → "Gemma 4 E2B (4-chunk legacy)" (or "Gemma 4 E4B
    (text-only)")
  - Wait for the ~3 GB download to complete
  - Confirm with: `xcrun devicectl device info files --domain-type
    appDataContainer --domain-identifier com.example.CoreMLLLMChat
    --subdirectory Documents/Models` and look for `gemma4-e2b/` (or
    `gemma4-e4b/`).

## Quick path: drafter-only swap

The shipping chunks tolerate the new centroid drafter (older build,
position-agnostic). Just swap the drafter:

```bash
# E2B drafter swap (fastest test):
bash scripts/push_centroid_drafter.sh e2b drafter

# E4B drafter swap:
bash scripts/push_centroid_drafter.sh e4b drafter
```

The script:

1. Verifies iPhone 17 Pro is connected.
2. Pushes `mtp_drafter.mlmodelc` into `Documents/Models/<variant>/`.
3. Lists the bundle's `mtp_drafter*` files for verification.

**Pre-built artifacts expected at:**

* E2B: `/tmp/mtp_drafter_centroid_out/mtp_drafter_centroid.mlmodelc`
* E4B: `/tmp/mtp_drafter_centroid_e4b_out/mtp_drafter_centroid_e4b.mlmodelc`

Build (if missing — 30 s + 30 s compile each):

```bash
# E2B
~/.pyenv/versions/lama-cml/bin/python conversion/build_mtp_drafter.py \
  --hf-repo google/gemma-4-E2B-it-assistant \
  --output /tmp/mtp_drafter_centroid.mlpackage \
  --sliding-window 512 --context-length 2048 --centroid-lm-head
xcrun coremlcompiler compile /tmp/mtp_drafter_centroid.mlpackage \
  /tmp/mtp_drafter_centroid_out

# E4B
~/.pyenv/versions/lama-cml/bin/python conversion/build_mtp_drafter.py \
  --hf-repo google/gemma-4-E4B-it-assistant \
  --output /tmp/mtp_drafter_centroid_e4b.mlpackage \
  --sliding-window 512 --context-length 2048 --centroid-lm-head --target e4b
xcrun coremlcompiler compile /tmp/mtp_drafter_centroid_e4b.mlpackage \
  /tmp/mtp_drafter_centroid_e4b_out
```

## Full path: drafter + fresh chunks

If you also want the freshly-rebuilt INT4 chunks (`constpm1`-tuned,
slightly higher accept on Fibonacci):

```bash
bash scripts/push_centroid_drafter.sh e2b all   # ~3 GB push, ~5-10 min
bash scripts/push_centroid_drafter.sh e4b all   # ~5 GB push, ~10-20 min
```

Pre-built chunks expected at:

* E2B: `/tmp/gemma4_chunks_K3_fresh/chunk{1..4}.mlpackage`
* E4B: `/tmp/gemma4_e4b_chunks_K3/chunk{1..4}.mlpackage`

## Run + observe

In Xcode (CoreMLLLMChat scheme → Edit Scheme → Run → Arguments → Environment Variables):

* `SPECULATIVE_PROFILE` = `1` (turn on `[SpecProfile mtp ...]` log lines)
* Optional: `MTP_DRAFT_POS_MODE` = `constpm1` (default in current binary)
* Optional: `MTP_DRAFTER_DEVICE` = `cpu` / `gpu` / `ane` (default `ane`)

Launch the app, type a prompt, watch the Xcode console for:

```
[MTP] Drafter loaded (K=3)
[SpecProfile mtp #0001] draft=Xms verify=Yms accepted=A/B emitted=N rolling=Z
```

### Expected vs prior full-vocab drafter (Mac numbers)

| variant | content | base tok/s | centroid tok/s | speedup |
|---|---|---|---|---|
| E2B | Repetitive | 33 | **73** | **2.20×** |
| E2B | Fibonacci | 33 | 51 | 1.52× |
| E2B | chat code | 33 | 49 | 1.47× |
| E4B | Repetitive | 15 | **36** | **2.30×** |
| E4B | Fibonacci | 15 | 14 | 0.92× ↓ |

iPhone 17 Pro numbers will differ from Mac (different ANE generation).
**E2B is the safer ship candidate**: positive on all content types.
**E4B regresses on chat code** because base is slow; only ship if your
workload is repetitive/structured.

## Troubleshooting

* **`accept = 0.00`**: iPhone ANE 18 may demote ints differently than
  Mac. Set `MTP_DRAFTER_DEVICE=cpu` in scheme env to bypass ANE for the
  drafter (drafter then runs on CPU, ~5 ms vs ~3 ms on ANE — small
  hit, big correctness gain).
* **App crashes on first MTP burst**: layout swaps via `devicectl` can
  leave orphan files that override the new push silently. Delete the
  app from the iPhone (long-press → Remove App), reinstall via Xcode,
  re-download the model, then re-run the push script.
* **Drafter loads but verify shape error**: only happens if you push an
  E2B drafter into an E4B bundle (or vice versa). The drafter's
  `kv13_k` input shape encodes `num_kv_heads` (E2B=1, E4B=2).
* **Garbage output (multiples-of-32 token IDs)**: pre-fix-build
  drafter, can't happen with the current `--centroid-lm-head` build
  (the `add_int16_cast` pass is dropped from the pipeline). If it does
  happen, rebuild from `main`.
