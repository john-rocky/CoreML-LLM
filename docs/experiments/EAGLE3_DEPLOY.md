# EAGLE-3 Speculative Decoding — Deployment Guide

Trained draft checkpoint → iPhone ANE pipeline. This doc is the shared contract
between training (Colab), conversion (Mac), target-chunk surgery
(build_speculative.py owner), and Swift integration.

**Status:** draft is training on Colab · conversion scaffold landed
(`build_eagle3.py`) · target-chunk surgery + Swift verify loop pending · fixed
K=3 for v1, dynamic tree for v2.

---

## 1. Pipeline overview

```
  Colab training                  Mac conversion                    iPhone runtime
  ──────────────                  ──────────────                    ──────────────
  eagle_corpus.jsonl              eagle3_draft_best.pt
    │                               │
    ▼                               ▼
  train_eagle3_draft.ipynb        build_eagle3.py                  fusion.mlpackage
    │                               │                               │
    ▼                               ▼                               ▼
  eagle3_draft_best.pt            eagle3_draft.mlpackage  ───────▶ draft.mlpackage
  eagle3_config.json              eagle3_fusion.mlpackage            │
                                                                     ▼
  build_speculative.py            decode chunks + hidden_mid  ─────▶ decode_chunks/
  (modified)                      verify_chunks_K3 (EnumShapes)      verify_chunks/
                                                                     │
                                                                     ▼
                                                               Swift ChunkedEngine
                                                               + SpeculativeLoop
```

---

## 2. Prerequisites

- `eagle3_draft_best.pt` from `conversion/train_eagle3_draft.ipynb` (training on Colab).
- `eagle3_config.json` from same training run.
- Mac with coremltools 8.x+, Python 3.10+, ≥ 16 GB RAM.
- Existing target decode + prefill chunk mlpackages (built by `build_speculative.py`).
- iPhone 17 Pro (A19 Pro) for on-device benchmarks.

---

## 3. Step-by-step

### 3.1 Train draft (Colab)
- Open `conversion/train_eagle3_draft.ipynb` in Colab, A100 preferred (T4 = 10–14 h).
- Run all cells. Checkpoints land in `/content/drive/MyDrive/eagle3_draft/`.
- Validation target: **acc[0] ≥ 55%**, **expL ≥ 2.0**. Below that, train longer
  or enlarge corpus.

### 3.2 Download artifacts to Mac
```
/content/drive/MyDrive/eagle3_draft/
  ├─ eagle3_draft_best.pt     ← required
  ├─ eagle3_config.json       ← required
  └─ eagle3_eval.json         ← nice to have (perf projection)
```

### 3.3 Sanity-check the draft (Colab, before CoreML)
```
python conversion/test_eagle3_infer.py \
    --ckpt /content/drive/MyDrive/eagle3_draft/eagle3_draft_best.pt \
    --prompt "The capital of Japan is" \
    --max-new 64 --K 3
```
Expected: `outputs match: True`. Accept rate ≥ 50%. If False, stop — it is a bug
in the speculative loop or the checkpoint, not worth converting.

### 3.4 Convert draft to Core ML (Mac)
```
python conversion/build_eagle3.py \
    --ckpt ./eagle3_draft/eagle3_draft_best.pt \
    --output ./eagle3_draft.mlpackage \
    --fusion-output ./eagle3_fusion.mlpackage \
    --palettize-int4
```
Produces two mlpackages (see §4 for I/O contract).

### 3.5 Target-chunk surgery — **OWNERSHIP: `build_speculative.py` editor**
Add extra outputs to decode chunks so the draft's fusion can read multi-layer
hidden states. Fusion layer indices are in `eagle3_config.json`:

```
fusion_layers = [8, 17, 34]   (from config; change only if retrained)
```

Per-chunk output additions (layer → chunk mapping must be computed from the
existing chunk boundaries):

| chunk    | layers           | add output (fp16, shape (1, 1, H)) |
|----------|------------------|------------------------------------|
| chunk1   | L0–L7            | `hidden_at_L8_pre` *(if L8 output lives in chunk2, alternatively emit hidden at chunk1 exit)* |
| chunk2   | L8–L14           | `hidden_at_L17_preview` – skip, L17 is in chunk3 |
| chunk3   | L15–L24          | `hidden_at_L17`, `hidden_at_L22` candidate |
| chunk4   | L25–L34          | `hidden_at_L34` (already available as pre-argmax hidden) |

If `fusion_layers = [8, 17, 34]`, the chunks that must add extra outputs are
the ones that *contain* layer 8, 17, 34. Exact indexing must be resolved against
the current chunk split in `build_speculative.py`. Output tensors should be
`(1, 1, H)` fp16 matching the target's per-layer hidden dim.

### 3.6 Verify chunks with EnumeratedShapes — **OWNERSHIP: same as 3.5**
Build `verify_chunk{1..4}_K3.mlpackage` with seq-dim enumerated over {1, 3}:
- seq_dim=1: normal decode step
- seq_dim=3: speculative verify step (batch of K=3 candidate tokens)

Per coremltools docs, EnumeratedShapes (not RangeDim) is required to keep all
enumerated shapes on ANE. Weights are identical to decode chunks, only the
shape envelope differs.

### 3.7 Swift integration — **OWNERSHIP: `ChunkedEngine.swift` editor**
New file (proposed): `Sources/CoreMLLLM/SpeculativeLoop.swift`.

Responsibilities:
1. Load `eagle3_fusion.mlpackage` + `eagle3_draft.mlpackage` + verify chunks.
2. After each target-verified commit, call `fusion(h_low, h_mid, h_high)` → `h_fused`.
3. Seed draft: call `draft(h_fused, embed(t_tok_next))` → `(h_out, tok_pred_1, logit_1)`.
4. Loop K−1 more times: `draft(h_out, embed(tok_pred_k))` → `(h_out, tok_pred_{k+1}, logit_{k+1})`.
5. Feed candidate tokens into verify chunks (seq_dim=3): produces target's own
   argmax at each position. Compare to draft proposals. Accept prefix up to
   first disagreement.
6. On reject: take target's correction token at the disagreement position,
   continue from there.
7. On low acceptance rate (< 30% rolling average): disable speculation for a
   few steps (fallback), saves wall-clock on hard distributions.

See §5 for detailed Swift call pseudocode.

### 3.8 Benchmark
- Target-only (K=1 fallback path): baseline tok/s (should match current).
- Speculative (K=3): measure on same prompts, iPhone 17 Pro, thermal-stable
  (10-min sustained run).
- Expected: **31 → 55–70 tok/s @ ctx=2048, 15 → 30–35 tok/s @ ctx=8192** at
  expL ≈ 2.2.

---

## 4. mlpackage I/O contracts

### 4.1 `eagle3_fusion.mlpackage`
| Name     | I/O    | Shape      | Dtype | Notes |
|----------|--------|------------|-------|-------|
| h_low    | input  | (1, 1, H)  | fp16  | target hidden at fusion_layers[0] (e.g. L8)  |
| h_mid    | input  | (1, 1, H)  | fp16  | target hidden at fusion_layers[1] (e.g. L17) |
| h_high   | input  | (1, 1, H)  | fp16  | target hidden at fusion_layers[2] (e.g. L34) |
| h_fused  | output | (1, 1, H)  | fp16  | fused hidden to seed draft                   |

Compute unit: `.cpuAndNeuralEngine`. ~27 MB (3× Linear(H, H) palettized INT4).

### 4.2 `eagle3_draft.mlpackage`
| Name     | I/O    | Shape      | Dtype  | Notes |
|----------|--------|------------|--------|-------|
| h_prev   | input  | (1, 1, H)  | fp16   | previous step's hidden (or h_fused on step 0) |
| e_next   | input  | (1, 1, H)  | fp16   | embed(previous token) × embed_scale           |
| h_out    | output | (1, 1, H)  | fp16   | hidden at this step, feed back as next h_prev |
| token    | output | scalar int | int32  | argmax of draft logits                         |
| logit    | output | scalar     | fp16   | value at argmax (for acceptance bookkeeping)   |

Compute unit: `.cpuAndNeuralEngine`. ~50 MB (INT4 palettized). T=1, no KV cache.

### 4.3 Decode chunk output additions (spec for bench session)
Each `chunk{i}_decode.mlpackage` containing a layer index `idx in fusion_layers`
must emit the POST-layer hidden state at that layer as an extra output:

```
  output name: hidden_at_L{idx}
  shape:       (1, 1, H)
  dtype:       fp16
```

### 4.4 Verify chunks (spec for bench session)
`verify_chunk{i}.mlpackage` is structurally the same as `chunk{i}_decode.mlpackage`
but with the Q seq_dim enumerated over {1, 3}. Weights identical (share via
symlink or build-time dedup). Must produce the same logit output shape, scaled
by seq_dim.

---

## 5. Swift call pseudocode

```swift
// Per decoding burst:
let (hLow, hMid, hHigh) = try chunksDecodeStep(tokens: [prevToken])  // extra outputs
let hFused = try fusion.predict(h_low: hLow, h_mid: hMid, h_high: hHigh).h_fused
let tTokNext = previousTargetArgmax  // from last chunk's argmax output

// Draft K=3 tokens autoregressively
var hPrev = hFused
var eNext = embed(tTokNext) * embedScale
var proposals: [Int32] = []
for _ in 0..<3 {
    let r = try draft.predict(h_prev: hPrev, e_next: eNext)
    proposals.append(r.token)
    hPrev = r.h_out
    eNext = embed(r.token) * embedScale
}

// Verify: run target chunks with seq_dim=3 on [tTokNext, proposals[0..1]]
let verifyTokens = [tTokNext, proposals[0], proposals[1]]
let targetArgmax = try verifyChunksStep(tokens: verifyTokens)   // returns [Int32] length 3

// Accept greedily
var accepted: [Int32] = [tTokNext]
for k in 0..<3 {
    if proposals[k] == targetArgmax[k] {
        accepted.append(proposals[k])
    } else {
        accepted.append(targetArgmax[k])  // target's correction
        break
    }
}
// Append `accepted` to running output; continue next burst.
```

---

## 6. Quality gates before ship

| Check                                  | Threshold                        |
|----------------------------------------|----------------------------------|
| `test_eagle3_infer.py` outputs match   | `outputs match: True`            |
| Acceptance rate on held-out prompts    | ≥ 50 % (targets 60–70 %)         |
| Draft mlpackage ANE residency          | ≥ 99 % via MLComputePlan         |
| Fusion mlpackage ANE residency         | ≥ 99 % via MLComputePlan         |
| Verify chunks ANE residency (K=3)      | ≥ 99 % via MLComputePlan         |
| Thermal stability (10-min sustained)   | < 5 % tok/s drift                |

If acceptance < 30% in production, fall back to K=1 (target-only) for the
remainder of the session, report as warning.

---

## 7. Known risks & mitigations

| Risk | Cause | Mitigation |
|---|---|---|
| Target chunks fail to compile after adding hidden_mid output | New graph output near chunk boundary may trigger ANE compile errors | Start by emitting hidden at the *layer that owns* the output (not cross-chunk); test chunk by chunk |
| Verify chunks with seq_dim=3 fall back to CPU/GPU | `RangeDim` vs `EnumeratedShapes` mixup | Use `EnumeratedShapes([(1, 1, 1536), (1, 3, 1536)])`; do not use RangeDim |
| Draft acceptance much lower on iPhone than Mac | fp16 numerical drift between CPU/GPU and ANE | Calibrate in fp16 from the start; verify logit agreement on a sample set before deploy |
| Extra I/O (h_low, h_mid, h_high to fusion) blows memory bandwidth budget | 3 × (1, 1, 1536) fp16 = 9 KB/step — trivial | — |
| Draft hidden divergence across K steps | Draft was trained with T=valid causal attention; inference is T=1 degenerate | Confirmed exact simplification (§build_eagle3.py); no divergence expected |

---

## 8. Next actions

| Owner                 | Task                                                                     |
|-----------------------|--------------------------------------------------------------------------|
| training / Colab      | finish epoch 2, collect `eagle3_draft_best.pt` + `eagle3_eval.json`      |
| conversion / this doc | run `build_eagle3.py`, produce two mlpackages                            |
| **build_speculative.py editor** | add `hidden_at_L{i}` outputs for i ∈ fusion_layers      |
| **build_speculative.py editor** | emit `verify_chunk{1..4}_K3.mlpackage` via EnumeratedShapes |
| **ChunkedEngine editor**        | new `SpeculativeLoop.swift`; wire fusion + draft + verify  |
| everyone              | benchmark on iPhone 17 Pro: ctx=2048 + ctx=8192, K=1 vs K=3              |

---

## 9. References
- Training notebook: `conversion/train_eagle3_draft.ipynb`
- Sanity test:      `conversion/test_eagle3_infer.py`
- Conversion:       `conversion/build_eagle3.py`
- EAGLE-3 paper:    <https://arxiv.org/abs/2503.01840>
- EAGLE-3 repo:     <https://github.com/SafeAILab/EAGLE>
- Full speed roadmap: `docs/SPEED_8K.md`
