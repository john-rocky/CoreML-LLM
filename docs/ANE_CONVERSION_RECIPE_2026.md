# ANE LLM Conversion Recipe (2026 edition)

Definitive best-practices recipe for converting HuggingFace-format LLMs (specifically Gemma 4 E2B) into chunked CoreML packages targeting the Apple Neural Engine on iPhone 17 Pro / iOS 26. Sources are at the bottom; everything here is implementable, with code excerpts that map directly onto the existing `conversion/` scripts.

---

## 1. Executive summary — the canonical 2026 ANE LLM recipe in 10 bullets

1. **Tensor layout `[B, C, 1, S]`** end-to-end. Replace every `nn.Linear` with `nn.Conv2d(in, out, 1)`; keep activations as 4-D channels-first the whole way through. The sequence axis must be the unpacked last dimension (64-byte aligned).
2. **Custom LayerNorm/RMSNorm** that operates on the `C` axis with `keepdims=True` and uses `rsqrt` (no division). Apple's `LayerNormANE` is the reference; an RMSNorm variant is a 5-line edit.
3. **Per-head attention split via `tensor.split` along dim=1`**, no `permute`/`reshape`. Compute scores with `einsum('bchq,bkhc->bkhq', q_i, k_i)` per head — this maps onto a single ANE op; the standard MHA reshape pattern triggers fallbacks.
4. **Stateful KV cache** via `register_buffer` + `ct.StateType` (iOS 18+). Stateless KV→I/O is the #1 perf bug; stateful gave Apple 13× over I/O in the Llama 3.1 blog.
5. **`scaled_dot_product_attention_sliced_q` MIL pass** — explicitly enable in `ct.PassPipeline` for prefill chunks with seq ≥ 256 (34 % faster, 45 % less memory on ANE per Apple).
6. **Block-wise INT4 weight quantization**: `OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int4', granularity='per_block', block_size=32)`. This is the Apple-blessed default since coremltools 8.0; in WWDC24 it gave 2× over per-grouped-channel palettization on the same model.
7. **Per-grouped-channel 4-bit palettization** only for the LM head (`group_size=16`, K-means LUT). Matches Apple Foundation Model 2025 recipe (`every 16 columns/rows share the same constants`).
8. **Chunk model into ≤ 1 GB sub-packages** (embed / N transformer chunks / lm-head). Stay under the ANE's mmap budget; load chunks asynchronously.
9. **Set runtime hints** at load time: `computeUnits = .cpuAndNeuralEngine`, `optimizationHints.reshapeFrequency = .infrequent` (prefill model), `.frequent` (decode model with dynamic seq), `specializationStrategy = .fastPrediction` for the decode chunk.
10. **iOS 26 deployment target** (`ct.target.iOS26`) — unlocks Int8 I/O for embeddings/lm-head, and the read/write state API needed for KV warm-starts and speculative reverts.

---

## 2. Apple `ml-ane-transformers` pattern — what to replicate

### 2.1 LayerNorm (drop-in for Gemma's RMSNorm with one tweak)

Apple's reference (`ane_transformers/reference/layer_norm.py`) — note the BSC→BC1S migration in `forward`:

```python
class LayerNormANE(nn.Module):
    """LayerNorm over the C axis of a [B, C, 1, S] tensor."""
    def __init__(self, num_channels, clip_mag=None, eps=1e-5,
                 elementwise_affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # Migrate BSC -> BC1S if needed
        if x.dim() == 3 and x.size(2) == self.num_channels:
            x = x.transpose(1, 2).unsqueeze(2)
        if self.clip_mag is not None:
            x.clamp_(-self.clip_mag, self.clip_mag)
        mean   = x.mean(dim=1, keepdims=True)
        zm     = x - mean
        denom  = (zm.pow(2).mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out    = zm * denom
        if self.elementwise_affine:
            out = (out + self.bias.view(1, -1, 1, 1)) * self.weight.view(1, -1, 1, 1)
        return out
```

**RMSNorm variant for Gemma 4** — same skeleton, drop the mean-subtract:

```python
class RMSNormANE(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps
    def forward(self, x):  # x: [B, C, 1, S]
        denom = (x.pow(2).mean(dim=1, keepdims=True) + self.eps).rsqrt()
        return x * denom * self.weight.view(1, -1, 1, 1)
```

Why this matters: Apple's MIL converter recognises this exact pattern and emits a single fused `layer_norm` op on the ANE. PyTorch's stock `LayerNorm` traces to a sequence (`reduce_mean`, `sub`, `pow`, `reduce_mean`, `add`, `rsqrt`, `mul`, `mul`, `add`) and fuses unreliably under tracing.

### 2.2 Multi-head attention (Conv2d + per-head einsum)

Apple's reference (`multihead_attention.py`):

```python
self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
self.v_proj = nn.Conv2d(embed_dim, self.d_v,  1)
self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)

# q,k,v all shape [B, d, 1, S]
mh_q = q.split(self.d_qk // self.n_head, dim=1)
mh_k = k.transpose(1, 3).split(self.d_qk // self.n_head, dim=3)  # [B,1,S,d_h]
mh_v = v.split(self.d_v  // self.n_head, dim=1)

attn_weights = [
    torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * self.q_normalize_fact
    for qi, ki in zip(mh_q, mh_k)
]
```

For Gemma 4 (GQA, 8 KV heads / 16 Q heads), share each `k_i`/`v_i` across two `q_i`s — repeat the list, do not call `.repeat_interleave` (it breaks the layout):

```python
kv_factor = self.n_q_head // self.n_kv_head  # 2 for Gemma 4 E2B
mh_k = list(itertools.chain.from_iterable([(ki,)*kv_factor for ki in mh_k]))
mh_v = list(itertools.chain.from_iterable([(vi,)*kv_factor for vi in mh_v]))
```

### 2.3 FFN (Conv2d sandwich)

```python
class FFN(nn.Module):
    def __init__(self, d, ffn_d):
        super().__init__()
        self.up   = nn.Conv2d(d, ffn_d, 1)
        self.gate = nn.Conv2d(d, ffn_d, 1)   # Gemma's SwiGLU
        self.down = nn.Conv2d(ffn_d, d, 1)
    def forward(self, x):                    # x: [B, d, 1, S]
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

### 2.4 What is deprecated from the 2022 reference

| Reference (2022)                              | 2026 replacement                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| Custom `q@k.T` einsum kernel                  | Keep for prefill ≥ 256; for decode use `F.scaled_dot_product_attention`  |
| ReLU FFN                                      | SwiGLU — Conv2d gates work fine                                          |
| Float16 KV-cache I/O                          | `ct.StateType` stateful KV (iOS 18+)                                     |
| Manual chunking via Python loop               | `coremltools.utils.bisect_model` (ct 8.0+)                               |
| `load_state_dict_pre_hook` for Linear→Conv2d  | Still valid and recommended                                              |

---

## 3. coremltools version history (what applies to us)

| Version | Date         | Feature                                                                                        | Use it?  |
| ------- | ------------ | ---------------------------------------------------------------------------------------------- | -------- |
| 7.0     | 2023-09      | flexible shapes (`RangeDim`), iOS 17 target                                                    | yes      |
| 7.2     | 2024-02      | ExecuTorch CoreML partitioner                                                                  | n/a      |
| 8.0     | 2024-06      | **stateful models (`ct.StateType`)**, **per_block INT4**, **`bisect_model`**, SDPA op support  | **yes**  |
| 8.1     | 2024-09      | torch.export 68 % parity, enumerated shapes, MLComputePlan bindings                            | maybe    |
| 8.2     | 2025-01      | `scaled_dot_product_attention_sliced_q` MIL pass introduced                                    | **yes**  |
| 8.3     | 2025-04-29   | MLModelValidator/Comparator/Inspector/Benchmarker; remote-device benchmarking                  | yes (debug) |
| 9.0     | 2025-11-10   | **iOS 26 target**, **Int8 I/O**, **state read/write API**, Python 3.13, PyTorch 2.7            | **yes**  |

For Gemma 4 E2B targeting iPhone 17 Pro you want **coremltools ≥ 9.0** and `minimum_deployment_target=ct.target.iOS26`.

### Stateful KV cache — the canonical pattern (ct 8.0+)

```python
# In the PyTorch wrapper:
self.register_buffer("k_cache", torch.zeros(kv_shape))
self.register_buffer("v_cache", torch.zeros(kv_shape))

# At conversion time:
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name="k_cache"),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name="v_cache"),
]
mlmodel = ct.convert(
    traced, inputs=inputs, outputs=outputs, states=states,
    minimum_deployment_target=ct.target.iOS26,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    skip_model_load=True,
)
```

Apple's measured progression on Llama 3.1 8B / M1 Max:

| Configuration                         | tok/s         |
| ------------------------------------- | ------------- |
| Baseline FP16, KV recomputed          | 0.19          |
| KV cache as I/O                       | 1.25          |
| Stateful KV cache                     | 16.26         |
| + INT4 per_block(32) quantization     | **33.67**     |

That last row is the recipe we are replicating.

---

## 4. The 2026 optimal quantization config for Gemma 4 E2B

Three blocks; apply each only to the components it suits.

### 4.1 Transformer blocks — INT4 per_block(32), linear-symmetric

```python
import coremltools.optimize.coreml as cto

cfg = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,          # WWDC24 sweet spot for ANE
        weight_threshold=512,    # leave tiny tensors in fp16
    ),
)
quant_blocks = cto.linear_quantize_weights(blocks_mlmodel, config=cfg)
```

This is what Apple uses for Mistral 7B (HF blog) and Llama 3.1 (Apple blog); it gives ~4× over fp16 with negligible quality loss (Apple reports < 0.5 PPL drift). On Gemma 4 E2B (~5 GB fp16 weights) this should land at ~1.3 GB.

### 4.2 LM head — 4-bit per-grouped-channel palettization (group_size = 16)

```python
cfg_head = cto.OptimizationConfig(
    global_config=cto.OpPalettizerConfig(
        nbits=4,
        granularity="per_grouped_channel",
        group_size=16,           # matches Apple Foundation Model 2025
        mode="kmeans",
        weight_threshold=2048,
    )
)
quant_head = cto.palettize_weights(lm_head_mlmodel, config=cfg_head)
```

Apple's tech report (arXiv 2507.13575): *"Every 16 columns/rows share the same quantization constants and are quantized using K-means."* Gemma 4 E2B vocab head (262 144 × 2304 = 600 M params) benefits the most from palettization because the LUT-cost amortises over a huge tensor.

### 4.3 Embedding table — Int8 I/O passthrough (iOS 26)

```python
embed_inputs  = [ct.TensorType(shape=(1, seq), dtype=np.int32, name="ids")]
embed_outputs = [ct.TensorType(dtype=np.int8,  name="embeds")]   # ct 9.0
```

Then dequantize on the GPU side or on the first conv of the next chunk. Saves 50 % of inter-chunk bandwidth.

### 4.4 What we tested and rejected (matches MEMORY.md)

| Scheme                                           | Outcome                                       |
| ------------------------------------------------ | --------------------------------------------- |
| W8A8 via `linear_quantize_activations`           | falls back to CPU on ANE; **avoid**           |
| Per-tensor INT8                                  | 1.5 PPL drift, no speed win vs INT4 per-block |
| INT2 weights (Apple's QAT recipe)                | requires QAT pipeline we do not have          |
| LUT4 globally                                    | anemll observed quality loss on FFN           |
| INT8 KV cache                                    | needs ct 9.0 + custom QAT; defer              |

---

## 5. Optimization passes and flags

### 5.1 Custom `PassPipeline` for transformer chunks

```python
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline

pipeline = PassPipeline.DEFAULT
# Critical for prefill (seq >= 256)
pipeline.insert_pass(
    index=-1,
    pass_name="common::scaled_dot_product_attention_sliced_q",
)
pipeline.set_options(
    "common::scaled_dot_product_attention_sliced_q",
    {"slice_size": 128},   # tune 64/128/256
)

mlmodel = ct.convert(
    traced,
    inputs=inputs, outputs=outputs, states=states,
    pass_pipeline=pipeline,
    minimum_deployment_target=ct.target.iOS26,
)
```

Apple measured **34 % faster, 45 % less memory** on Depth-Anything (seq 1814) on the ANE with this pass.

### 5.2 Passes that are automatic (don't fight them)

These are in `PassPipeline.DEFAULT`; you only need to verify they fired with `MLModelInspector` (ct 8.3+):

* `common::fuse_linear_bias`
* `common::fuse_layernorm_or_instancenorm`  *(the canonical fused-LN pass; it also fires for the RMSNorm pattern in §2.1 if you use `keepdims=True` and `rsqrt`)*
* `common::fuse_transpose_matmul`
* `common::fuse_activation_into_conv`
* `common::fuse_pad_into_conv`
* `common::const_elimination`
* `common::dead_code_elimination`

### 5.3 Passes you may want to skip

```python
pipeline.remove_passes([
    "common::add_fp16_cast",          # if you're already FP16 throughout
    "common::const_deduplication",    # break per-chunk weight pruning
])
```

### 5.4 `ct.optimize.torch` vs `ct.optimize`

* `ct.optimize.coreml.*` — **use this**, operates on the converted `.mlpackage`. Idempotent, scriptable, no retraining.
* `ct.optimize.torch.*` — only if you intend to do QAT (Apple's 2-bit recipe). For PTQ post-conversion, the `coreml` namespace is strictly better because it sees the actual MIL graph and can per-op-skip.

---

## 6. iOS 18 / iOS 26 runtime hints (Swift side)

Drop-in for `CoreMLLLMChat`'s model loading code. `MLOptimizationHints` was added in iOS 18.0, expanded in 18.2.

```swift
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndNeuralEngine          // never .all for our chunks

if #available(iOS 18.0, *) {
    let hints = MLOptimizationHints()
    // Decode chunk: shape changes every step (KV grows). Use .frequent.
    // Prefill chunk: fixed shape per call. Use .infrequent.
    hints.reshapeFrequency = isDecode ? .frequent : .infrequent
    hints.specializationStrategy = .fastPrediction
    cfg.optimizationHints = hints
}

if #available(iOS 18.2, *) {
    cfg.allowLowPrecisionAccumulationOnGPU = true   // GPU fallback path
}
```

Per-chunk recommendations:

| Chunk          | computeUnits        | reshapeFrequency | specializationStrategy |
| -------------- | ------------------- | ---------------- | ---------------------- |
| `embed`        | `.cpuAndNeuralEngine` | `.infrequent`  | `.fastPrediction`      |
| `prefill_block_*` (seq=256) | `.cpuAndNeuralEngine` | `.infrequent` | `.fastPrediction`  |
| `decode_block_*` (seq=1)    | `.cpuAndNeuralEngine` | `.frequent`   | `.fastPrediction`  |
| `lm_head`      | `.cpuAndNeuralEngine` | `.infrequent` | `.fastPrediction`     |

Loading: always use the **async** loader (`MLModel.load(contentsOf:configuration:)`) and warm each chunk with one zero-input prediction before measuring tok/s — the first prediction triggers the ANE specialization compile (~200–800 ms) and is what makes naive benchmarks under-report.

---

## 7. Comparison to other published ANE LLM implementations

### 7.1 anemll (github.com/anemll/anemll)

* Chunk via `--chunk N` flag; for 8B Llama uses 8 chunks, 4 transformer blocks per chunk.
* Quant: LUT4 for FFN, LUT6 for LM head, FP16 elsewhere. Notes *"LUT4 quality is fairly low due to lack of Block Quantization on Apple Neural Engine"* — an artefact of pre-ct-8.0; we should use `per_block(32)` instead.
* RoPE: pre-computed cos/sin baked as constants per chunk; fixed context length per package.
* Context length: 512–1024 sweet spot; 4 K verified but slower.
* Uses `coremltools >= 9.0` — confirms we are aligned.

### 7.2 smpanaro/coreml-llm-cli (Llama 2 7B)

* Achieves 13.92 tok/s on M3 Max — well below our 22–28 target because:
  * Uses **non-stateful** I/O KV cache (no `ct.StateType`).
  * Uses `(B,C,8,8)` reshape trick around MLP — interesting but only ANE-microarchitecture-specific (A14/A15); not reproduced on A19 (iPhone 17 Pro).
  * IOSurface-backed `CVPixelBuffer` for I/O — **steal this**, ~2 ms/chunk saved.
  * Async KV combine outside the sync path — **steal this** too.

### 7.3 HuggingFace `apple/mistral-coreml` (WWDC24)

* Exact recipe in §3 above. Our convert.py already matches the structure; we are missing (a) the explicit `pass_pipeline` insert and (b) the `iOS26` deployment target bump.

### 7.4 Apple `corenet/llama-coreml` blog

* Same conversion as Mistral, plus they expose the *kv-prep* model as a separate `mlpackage` so the prompt can be processed in chunks of 64 tokens. Our `build_prefill_gpu.py` already does this; verify chunk sizes are 64-multiples for ANE 64-byte alignment.

---

## 8. What Apple Foundation Models 2025 reveals about their internal recipe

From arXiv **2507.13575** + the "Apple Foundation Models Tech Report 2025" blog:

1. **3 B-parameter on-device model**, deployed via CoreML on ANE.
2. **2-bit QAT** with palette `{-1.5, -0.5, +0.5, +1.5}` — *"a balanced 2-bit set yields smoother training with fewer training loss spikes."* This is novel; not available in coremltools out of the box, requires their training pipeline.
3. **Embedding table at 4-bit**, jointly trained with base weights during QAT.
4. **KV cache at 8-bit per weight** — confirms 8-bit KV is on Apple's roadmap; ct 9.0 has the API but we'd need calibration.
5. **Per-group palettization, group_size = 16** along columns or rows. K-means LUT.
6. **KV-cache *sharing* across layer pairs** — two consecutive transformer blocks share one KV buffer. We could test this on Gemma 4 by tying layers (i, i+1) to the same `register_buffer`. Estimated 1.4× over independent KV at the cost of ~0.3 PPL.
7. **Average ~3.5 bits per weight** for the on-device model after mixing 2-bit base, 4-bit embed, 8-bit KV.

Translated to our shipping recipe (no QAT budget):
* INT4 `per_block(32)` everywhere blocks live — we're on the right side of the curve.
* Palettize the LM head per-grouped-channel with group_size=16 — direct port of their pattern, no QAT needed for the head.
* Trial a **layer-pair KV share** for blocks 6–7 and 14–15 (Gemma 4 E2B has 26 blocks); the perf upside is large and the change is local to `base_model.py`.

---

## 9. References

* Apple ML Research, *Deploying Transformers on the Apple Neural Engine* — https://machinelearning.apple.com/research/neural-engine-transformers
* Apple ML Research, *On-Device Llama 3.1 with Core ML* — https://machinelearning.apple.com/research/core-ml-on-device-llama
* Apple ML Research, *Apple Intelligence Foundation Models, Tech Report 2025* — https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025
* arXiv 2507.13575, *Apple Intelligence Foundation Language Models: Tech Report 2025* — https://arxiv.org/abs/2507.13575
* Apple, `ml-ane-transformers` repo — https://github.com/apple/ml-ane-transformers
  * `ane_transformers/reference/layer_norm.py`
  * `ane_transformers/reference/multihead_attention.py`
  * `ane_transformers/reference/ffn.py`
* coremltools releases — https://github.com/apple/coremltools/releases
* coremltools quantization API — https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html
* coremltools palettization API — https://apple.github.io/coremltools/docs-guides/source/opt-palettization-api.html
* coremltools stateful models guide — https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
* coremltools MIL graph passes — https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html
* Apple Developer, `MLOptimizationHints` — https://developer.apple.com/documentation/coreml/mloptimizationhints
* HuggingFace blog, *WWDC 24: Running Mistral 7B with Core ML* — https://huggingface.co/blog/mistral-coreml
* anemll/anemll — https://github.com/anemll/anemll
* smpanaro/coreml-llm-cli — https://github.com/smpanaro/coreml-llm-cli
