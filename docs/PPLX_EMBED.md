# pplx-embed — Perplexity embedding models on the ANE

Adds a **bidirectional Qwen3 encoder** path that converts Perplexity's pplx-embed models to CoreML
and runs them on the Apple Neural Engine (macOS Tahoe / `macOS26`):

- `perplexity-ai/pplx-embed-v1-0.6b` — **plain** sentence embeddings (mean-pool → 1024-d int8).
- `perplexity-ai/pplx-embed-context-v1-0.6b` — **late chunking** (per-chunk embeddings via a
  `pool_matrix` matmul; one encoder pass over the whole window).

The encoder is a 28-layer bidirectional Qwen3-0.6B (GQA 16/8, head_dim 128, SwiGLU, QK-norm,
RoPE θ=1e6) built on the existing ANE primitives (Conv2d-1×1 projections, native RMSNorm,
`repeat_kv_ane`, `stable_attention`). Output matches the model's own `st_quantize.py` exactly:
int8 = `clamp(round(tanh(x)·127), −128, 127)` (`torch.round`, half-to-even), plus `binary`
(sign) and `ubinary` (packbits).

## Files

| what | where |
|---|---|
| Model registry | `conversion/config.py` → `pplx-embed`, `pplx-embed-context` |
| Encoder | `conversion/models/qwen3_encoder.py` |
| Bundle builder | `conversion/build_pplx_embed_bundle.py` |
| Golden fp32 reference (oracle) | `conversion/pplx_embed_reference.py` |
| Parity test | `conversion/test_pplx_embed_parity.py` |
| ANE RMSNorm A/B | `conversion/experiment_ane_rmsnorm.py` |
| HF uploader | `conversion/upload_pplx_embed.py` (single repo, per-bucket subfolders) |
| Swift runtime | `Sources/CoreMLLLM/PplxEmbed.swift` (+ `pplx-embed-demo`, `pplx-embed-bench`) |

## Build

```bash
# A fixed-shape ANE bucket (the fast path), plain int8 output:
python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 512
# Context (late chunking) variant:
python conversion/build_pplx_embed_bundle.py --model pplx-embed-context --max-seq-len 512
# The flexible GPU catch-all for inputs larger than the biggest bucket (up to 8192):
python conversion/build_pplx_embed_bundle.py --model pplx-embed --dynamic-upper 8192
```

Verify fidelity against the fp32 reference (CPU, fast):

```bash
python conversion/test_pplx_embed_parity.py        # pooled ≥0.999, int8 ≥0.997
```

## Use (Swift)

```swift
let embedder = try await PplxEmbed.load(bundleDir: URL(fileURLWithPath: "output/pplx-embed"))
let vectors = try embedder.embed(["hello world", "bonjour le monde"])   // [[Int8]] (1024-d)
// also: embedBinary / embedUBinary; embedContext([[String]]) for late chunking
```

`embed()` tokenizes, selects the **smallest fixed bucket** that fits, pads/masks, and runs on the
ANE. Inputs larger than the biggest bucket are routed to the flexible RangeDim model on the GPU
(non-padded). Run the CLI demo with `swift run -c release pplx-embed-demo --bundle-dir output/pplx-embed --text "…"`.

### Download prebuilt models from Hugging Face

End users can **download** the prebuilt CoreML buckets instead of regenerating them (conversion
needs the toolkit + minutes per bucket). The buckets live in one repo with per-bucket subfolders
(`<account>/pplx-embed-coreml`; final id confirmed at publish time) plus a `manifest.json`
inventory. `PplxEmbed.load(repo:)` reads the manifest and **selectively** downloads only the
requested buckets (+ the dynamic catch-all) — never the whole repo:

```swift
let embedder = try await PplxEmbed.load(
    repo: "<account>/pplx-embed-coreml",
    buckets: [512, 1024, 2048],      // only these subfolders + tokenizer are fetched
    into: appSupportDir)             // preferCompiled: true → pulls .mlmodelc (no on-device compile)
let vectors = try embedder.embed(["hello world"])
```

Each bucket is published in **both** formats — precompiled `.mlmodelc` (default; no on-device
compile) and the portable `.mlpackage` (`preferCompiled: false`). The repo hosts both, but the
client downloads only one format's ~1.1 GB weights per bucket. The demo takes `--repo`:
`swift run -c release pplx-embed-demo --repo <account>/pplx-embed-coreml --buckets 512 --text "…"`.

Publish with `conversion/upload_pplx_embed.py` (single repo, per-bucket subfolders, `--compile` to
ship both `.mlmodelc`+`.mlpackage`): it compiles, **stages a clean repo tree** (hardlinks +
`manifest.json` + README card), ensures the repo exists, and prints a resumable
`hf upload-large-folder <repo> <stage> --repo-type=model` command (parallel, xet-accelerated,
realtime progress; re-run to resume). Every bucket's `weight.bin` is now **byte-identical** (the
RoPE cos/sin tables are built once to a fixed length — `max_position_embeddings`, 32768 — and
gathered to `S` at runtime via `position_ids` derived from `attention_mask`, so they no longer scale
with `max_seq_len`; verified L512≡L1024 by sha256). So HF LFS stores the ~1.2 GB blob **once** across
all buckets (was ~7 GB for 6 buckets), and `.mlmodelc`↔`.mlpackage` within a bucket still dedup too.
The runtime gather is fold-proof — a plain static `[:S]` slice gets const-folded back to a per-bucket
constant — and needs no new model input / Swift change. Fidelity unchanged (CoreML L512 cosine vs the
fp32 oracle 0.99996; ANE residency 99.3%).

## Design notes

- **Fixed-shape buckets, one `.mlpackage` per bucket.** Flexible shapes (EnumeratedShapes/RangeDim)
  force CPU fallback on the ANE and are ~10× slower; fixed buckets stay 99.8% on the ANE. Pad each
  input to the smallest fitting bucket. Latency is O(L²) with a sharp knee at L=1024→2048.
- **Flexible GPU model is the >max-bucket catch-all only.** Built with `--dynamic-upper N`
  (RangeDim 1..N), it runs on the GPU non-padded for unbounded length — correct (cos 0.999) but
  ~10× slower than a fixed bucket, so it's used only when no bucket fits.
- **L=8192 is NOT an ANE bucket — the largest fixed ANE bucket is 4096.** A fixed L=8192 bucket
  *statically* plans to 99.81% ANE, but the ANE **runtime fails to execute it**
  (`ANEProgramProcessRequestDirect status=0x15: Program Inference error`,
  `conversion/measure_l8192_bucket.py`): at 8192 the full bidirectional-attention intermediates
  (16 heads × 8192² fp16 ≈ 2 GB per score tensor) exceed ANE buffer limits, and the ANE graph
  compile itself takes ~25 min. So inputs of 4097–8192 tokens stay on the **dynamic GPU
  catch-all** (which already covers them). `chunk-and-pool` to stay within a bucket would change
  plain-embedding semantics, so it's a separate design question, not a drop-in.
- **Native RMSNorm (`norm_impl="native"`, the default).** The 5 encoder norm sites use native
  `x·rsqrt(mean(x²)+eps)·w` rather than the shared `ane_ops.ANERMSNorm` cat([x,−x])→LayerNorm
  trick. The trick predates a fast native ANE rsqrt; on M4 Max / macOS 26 / coremltools 9 native
  is **12–21% faster** on the ANE at identical 99.81% residency and cosine 0.99998
  (`conversion/experiment_ane_rmsnorm.py`; see [`PPLX_EMBED_GPU_RESIDENCY.md`](PPLX_EMBED_GPU_RESIDENCY.md)
  and [`ANE_RMSNORM_FOLLOWUP.md`](ANE_RMSNORM_FOLLOWUP.md)). Build with `--norm-impl ane_cat` to
  fall back. This is local to the pplx-embed encoder — the shared decoder `ANERMSNorm` is untouched.
- **fp16 residual rescale (K=8).** The 28-layer `down_proj` accumulation overflows fp16; scaling
  `embed_tokens`/`o_proj`/`down_proj` by 1/K is exact for a pre-norm net (scale-invariant norms)
  and keeps activations in range. K=8 is the fidelity/overflow sweet spot.
- **macOS26 native int8 output** is not readable from the Python CoreML bridge; read it in Swift
  (the `pplx-embed-bench` harness does). Fidelity is otherwise measured via a `pooled_fp16`-output
  variant in Python.
- **Throughput:** ANE batch-1 at the smallest bucket is both the lowest-latency and
  highest-throughput path; batching is not a useful lever on CoreML (see below).

See [`PPLX_EMBED_W8A8.md`](PPLX_EMBED_W8A8.md) (weight/activation quantization is not viable for
this model), [`PPLX_EMBED_BATCHING.md`](PPLX_EMBED_BATCHING.md) (batching is not a useful
throughput lever — the ANE is batch-1 by design), and
[`PPLX_EMBED_GPU_RESIDENCY.md`](PPLX_EMBED_GPU_RESIDENCY.md) (why the GPU `CPU_AND_GPU` path has
low GPU residency — an inherent CoreML partitioner behavior at B=1, not a fixable issue).
