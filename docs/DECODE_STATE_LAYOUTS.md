# Decode-Time KV State Layouts — Lessons

**Last updated:** 2026-04-24 (from Ternary-Bonsai-1.7B port)

Actionable knowledge about how Core ML / ANE compiles KV cache update patterns.
Read this before designing the decode path of a new model or reworking an
existing one.

## TL;DR

1. On ANE, per-step cost is dominated by `O(state_length)` in attention (not
   weight bandwidth). Shrinking state length is the biggest single lever.
2. For Stateful models, **mask-based circular rotating buffer** is a strictly
   safer default than shift-based `cat`. Same semantics, strictly wider op
   compatibility with ANEC. Use it unless you have a measured reason not to.
3. Palettize with `mode="kmeans"` by default. Linear quantization is a
   div-by-zero hazard on sparse tensors.
4. If you're tracing a module that mutates a registered buffer (KV cache),
   pass `strict=False` to `torch.jit.trace`. The warning is benign; the graph
   is correct.

## 1. `O(state_length)` is the decode-time bottleneck on ANE

Measured on Ternary-Bonsai-1.7B INT4 kmeans, 2-chunk, Mac ANE (M-series),
"The capital of France is" decode:

| ctx / state | decode tok/s |
|---|---|
| 2048 | 9.4 |
| 1024 | 24.1 (2.56× vs 2048) |
| 4096 | 4.9 |
| 4096 + SWA W=1024 | 25.6 |

Halving state gave **2.56×**, which is much larger than the weight-bandwidth
explanation (INT8 → INT4 only gave +12% for the same ctx). Attention
softmax-over-state and KV-cache state read/write dominate per-step latency
for 1–2B class models on ANE.

**Corollary**: for any new model, **start with ctx=1024** and measure before
pushing higher. If you need larger effective context, switch to SWA — see §3.

Qwen3.5-2B handoff (`docs/QWEN35_2B_CHUNKED_HANDOFF.md` §2) made the same
observation for the 2B class; Bonsai-1.7B confirms it for the 1.7B class.

## 2. Monolithic bundles > ~1.4 GB silently fall back to GPU

Confirmed a third time (after Gemma 4 E4B and Qwen3.5-2B): Ternary-Bonsai-1.7B
monolithic INT8 at 1.94 GB runs at 8.3 tok/s on Mac ANE, but audit shows 0%
ANE placement — Core ML routes to GPU without error because
`MLComputeUnits.cpuAndNeuralEngine` is a preference, not a requirement.

**Rule of thumb**: if `du -sh model.mlpackage` > ~1.0–1.4 GB,
you will hit silent GPU fallback on iPhone ANE and likely jetsam-kill on
load. Split into chunks. See `docs/QWEN35_2B_CHUNKED_HANDOFF.md` §3 for the
chunking pattern.

## 3. Mask-based circular rotating buffer vs shift-based `cat`

**The problem** (discovered on Bonsai / Qwen3 port):

```python
# gemma4_swa_chunks.py-style shift-based SWA update:
K_new = torch.cat([K_cache[:, :, 1:, :], k], dim=2)
```

For Qwen3ForCausalLM + `ct.StateType` + `tie_word_embeddings=True`, Apple's
ANEC compiler **rejects this pattern** with `error code: -14`. The produced
mlpackage loads but:
- `ct.models.MLModel(path).get_compiled_model_path()` throws.
- `MLModel.make_state()` throws with "This model was not loaded with the Core
  ML Framework."
- Opening with `CPU_ONLY` compute units succeeds, so the graph itself is
  valid; the failure is ANE-specific op-lowering.

The exact conditions that trigger -14 are architecture-sensitive. Gemma 4's
sliding layers use the same `cat`-shift pattern and **do** ship on ANE. Our
best current guess: interaction between the shift op, the tied-embedding
lm_head, and Stateful buffer aliasing.

**The fix**: mask-based circular rotating buffer.

```python
# SWA, state buffer sized = W (not ctx):
# Host passes update_mask with 1.0 at `pos % W`, 0 elsewhere.
k_broadcast = k.expand_as(K_cache)          # (1, kv, W, d)
K_new = K_cache * (1 - update_mask) + k_broadcast * update_mask
```

This is the exact op pattern used for non-SWA mask-based absolute-position
writes — so it inherits the ANE-proven lowering path. With this pattern the
Bonsai-1.7B build hits 92% ANE placement at INT4, 25.6 tok/s at ctx=4096
effective / W=1024.

**Correctness**:
- RoPE is applied to K *before* the blend, so cached K holds position-encoded
  values. The slot index in the buffer is independent of position — just a
  physical location.
- After wraparound, slot order is scrambled (e.g. slot 0 holds pos W, slot 1
  holds pos 1, …). Attention softmax is permutation-invariant over the key
  axis, so this is fine.
- During warm-up (pos < W-1), unfilled slots are masked to `-1e4` in
  `causal_mask`. Mask value -1e4 (not `-inf`) is the ANE FP16 convention.

**Host-side per step**:
```python
W = sliding_window
write_slot = pos % W
update_mask = np.zeros((1, 1, W, 1), dtype=np.float16)
update_mask[0, 0, write_slot, 0] = 1.0
causal_mask = np.full((1, 1, 1, W), -1e4, dtype=np.float16)
valid_count = min(pos + 1, W)
for i in range(valid_count):
    causal_mask[0, 0, 0, (pos - i) % W] = 0.0   # most-recent valid slots
```

Reference build: `conversion/build_bonsai_17b_decode_chunks.py`
Reference host-side: `conversion/test_bonsai_chunks_inference.py`

### When to prefer shift-based `cat` anyway

- Your model is Gemma 4 (already ships with shift, production-proven).
- You measured a speedup from shift over mask-based on your specific arch.
- You are not using `ct.StateType` (non-stateful, KV passed as I/O tensors).

Otherwise, default to mask-based rotating.

## 4. Palettization: kmeans INT4 first, not linear INT8

Bonsai-1.7B INT8 linear quantization logged multiple
`RuntimeWarning: invalid value encountered in divide / cast` during
`linear_quantize_weights`, caused by zero-valued tensors in some layers
(scale = max_abs/127 → 0 → div-by-zero NaN → cast to int8 produces garbage).
The model still loaded but first-token logits were less stable.

k-means palettization (`mode="kmeans"`) is centroid-based, doesn't hit
div-by-zero on zero tensors, and compresses to the same disk size. INT4
kmeans is usually the sweet spot for ANE — further quantization (INT3, INT2)
has dramatically less compiler / kernel support.

Reference: `ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_tensor")`.

Observed first-token logit stability for "Paris" prompt (INT8 linear →
INT4 kmeans): 17.938 → 18.234 (~1.6% delta, top-1 preserved, top-3 preserved).

## 5. Tracing gotchas for Stateful models

**Problem**: `torch.jit.trace` re-runs the traced function to validate outputs.
If the module mutates a registered buffer (kv_cache), the second run sees
different state and the tracer logs:

```
TracerWarning: Output nr 1. of the traced function does not match...
```

The graph is still correct — the mutation is captured, the validation just
runs a different branch.

**Fix**: pass `strict=False`.

```python
traced = torch.jit.trace(module, sample_inputs, strict=False)
```

Match the pattern in `conversion/build_qwen35_2b_decode_chunks.py`.

## 6. `audit_ane` can throw on -14 models; wrap it

If conversion logs `error code: -14` (ANE compiler rejection), the saved
mlpackage is a CPU-stub. `get_compiled_model_path()` will throw with "This
model was not loaded or compiled with the Core ML Framework." Don't let that
kill your build pipeline — audit is diagnostic only.

```python
def audit_ane(pkg_path: Path) -> float:
    try:
        m = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        compiled = m.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"    ANE audit skipped: {e}")
        return -1.0
    # ... placement counting
```

The real signal (the model is bad, usually from shift-cat or unsupported
op pattern) is already in the saved-time warning; the audit crash is just
noise.

## 7. Checklist for a new model's decode path

- [ ] Reference HF arch. If it has `tie_word_embeddings=True` and you plan
      to use `ct.StateType`, **default to mask-based rotating, not shift**.
- [ ] Measure monolithic INT4 bundle size. If > ~1.4 GB, chunk (see
      `docs/QWEN35_2B_CHUNKED_HANDOFF.md` §3).
- [ ] Start with `context_length=1024`. Decide on larger ctx only after
      measuring per-step latency at 1024.
- [ ] For ctx > 1024: use SWA (mask-based rotating) with W=1024 by default.
      State buffer size = W, not ctx.
- [ ] Palettize with `mode="kmeans"` first. Only try linear quant if kmeans
      produces measured quality regression.
- [ ] `torch.jit.trace(..., strict=False)` for stateful graphs.
- [ ] Wrap `audit_ane` in try/except; -14 warnings kill the check.
- [ ] Parity test (PyTorch vs our-model) before CoreML conversion.
      See `conversion/bonsai_reference_oracle.py` for the pattern.

## 8. Per-(row, block) palette is rejected by ANEC

For weight quantization schemes where each (row, block) of a matrix needs its
own scale — e.g., 1.58-bit ternary BitNet-style models like Bonsai — Core ML
has the right MIL ops (`constexpr_lut_to_dense` with multi-axis LUT, or
`constexpr_lut_to_dense` + `constexpr_blockwise_shift_scale` two-op chain),
and the resulting mlpackage saves and validates fine.

**But Apple's ANE compiler rejects both forms** with `error code: -14`. The
saved model loads as a CPU-stub: `MLModel(...)` returns successfully, but
`get_compiled_model_path()` and `make_state()` both throw "This model was not
loaded with the Core ML Framework." Tested with iOS 18 / coremltools 9.0 on
Qwen3 + Stateful + tied embed.

ANE in iOS 18 supports `per_tensor` and `per_grouped_channel` palette
granularities (one LUT shared across the tensor, or one LUT per N
output-channels). It does **not** support a separate LUT per `(row, block)`
pair, which is what BitNet/Bonsai need to preserve their per-block
independent scales.

The available approximation — `nbits=2 per_grouped_channel + enable_per_channel_scale`
— compiles for ANE but factorizes the scale matrix as `s_{r,b} ≈ c_b · d_r`
(rank-1 outer product), losing the per-(row, block) independence. For models
whose value is precisely that independence, this defeats the purpose.

**Practical guidance**:
- Don't try to bit-exact ternary / 1.58-bit on ANE today. Either ship an
  approximation (per-tensor / per-channel kmeans) and accept that you've
  effectively quantized a Qwen3/Llama equivalent — or ship via MLX (Apple
  Silicon GPU), which natively executes packed ternary via `mx.quantized_matmul`.
- This applies to: Bonsai, BitNet b1.58, Era of 1-bit LLMs and any
  derivative with per-block scales.

Investigation details: [`TERNARY_BONSAI.md`](TERNARY_BONSAI.md).

## 9. Related docs

- `docs/TERNARY_BONSAI.md` — the Bonsai-specific landing page.
- `docs/QWEN35_2B_CHUNKED_HANDOFF.md` — original chunking + jetsam findings.
- `docs/ANE_OPTIMIZATION_SURVEY.md` — broader ANE tricks (prefill bypass,
  ping-pong buffers, etc.).
- `docs/ADDING_MODELS.md` — the end-to-end "I want to add a new model"
  walkthrough; this doc is its decode-path companion.
- `docs/GEMMA4_ROTATING_BUFFER_PORT.md` — applies this decode-path knowledge
  to Gemma 4.
