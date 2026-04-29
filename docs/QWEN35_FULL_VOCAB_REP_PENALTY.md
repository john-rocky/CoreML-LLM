# Full-vocab rep_penalty workaround for iPhone ANE fp16 bias

**Discovery: 2026-04-29**
**Result: Qwen3.5 0.8B on iPhone 17 Pro = 45-50 tok/s clean output**
(beats prior 33 tok/s ceiling, matches Mac M4 50.4 tok/s)

## The problem

iPhone A18 ANE's fp16 reduction has hardware-level non-deterministic
tie-breaks on large-vocab (248320) matmul. On prompts where the
top-1 token's fp32 logit is within fp16 epsilon (~0.015 at logit
magnitude 25) of another candidate, A18 picks the "wrong" tied token
where Mac M4 picks the right one. This is NOT software-fixable —
Apple's silicon-level reduction order differs across chip generations.

Symptoms (Qwen3.5 0.8B + chunk_d body+head fused on ANE fp16):
- `こんにちは` → `おはる、おはる、おはる、…` tight loop
- `Hello.` → `🌟🌟🌟🌟…` star wall (some runs)
- `What is the capital of France?` → `Paris-mark-mark-mark-…`
- Mac M4 same model: clean output for all of the above at 50.4 tok/s

Verified on Apple coremltools issues:
- [#2359 — Mish fp16 ANE precision](https://github.com/apple/coremltools/issues/2359)
- [#2625 — MobileNetV3 fp16 large errors](https://github.com/apple/coremltools/issues/2625)
- [hollance/neural-engine — ANE is fp16-only](https://github.com/hollance/neural-engine/blob/master/docs/16-bit.md)

Apple's documented workaround: `FP16ComputePrecision(op_selector=...)`
to mark problematic ops as fp32. This forces those ops off ANE onto
CPU/GPU, costing dispatch overhead. For the lm_head matmul (the heavy
op), splitting it off ANE drops 49 tok/s → 26-33 tok/s on iPhone.

## What we tried (everything that DIDN'T work)

| Approach | Speed | Output |
|---|---|---|
| OLD MLKV unified, body+head fp16 ANE, in-graph topk[k=40] | 49 tok/s | Wrong on Japanese |
| Body fp16 ANE + chunk_head fp32 GPU (CoreML) | 14 tok/s | Clean |
| Body fp16 ANE + chunk_head fp32 CPU (CoreML) | 26-33 tok/s | Clean |
| Body fp16 ANE + chunk_head INT8 fp32-graph CPU | 31 tok/s | Clean |
| Body fp16 ANE + Swift cblas_sgemv on fp32 buffer | 16 tok/s | Clean |
| Body fp16 ANE + chunk_d emits normed_hidden + Swift K=40 fp32 rerank | 36 tok/s | **WRONG** (biased normed_hidden) |
| Body fp16 ANE + chunk_d emits normed_hidden + Swift K=2048 fp32 rerank | 4.7 tok/s | **WRONG** (still biased) |
| Mixed precision via op_selector (conv+topk fp32) | n/a | **WRONG** (other ops drift) |
| INT8 weight + fp16 graph | n/a | **WRONG** (rolls back to fp16 reduction) |

## What worked: full fp16 logits + Swift full-vocab rep_penalty

```python
# build_qwen35_decode_chunks_mlkv.py — MLKVTailChunk
class MLKVTailChunk(MLKVBodyChunk):
    """Body + final_norm + Conv2d lm_head, FULL fp16 logits (no topk)."""
    def forward(self, hidden_in, cos, sin, causal_mask, current_pos, *ssm_states):
        body_out = super().forward(...)
        h = body_out[0]
        ssm_outs = body_out[1:]
        h = self.final_norm(h)
        logits = self.lm_head(h)            # (1, 1, V) fp16
        return (logits, *ssm_outs)
```

Output spec: `ct.TensorType(name="logits", dtype=np.float16)`. Whole
chunk stays ANE-resident — same compute as `OLD unified + topk[k=N]`,
just outputs the raw 248K logits to host instead of pre-sorted top-K.

```swift
// Qwen35MLKVGenerator.swift
private func sampleFromFullLogits(logits: MLMultiArray) -> Int32 {
    let V = logits.count
    let p = logits.dataPointer.assumingMemoryBound(to: UInt16.self)

    // 1. fp16 → fp32 batch convert via vImage (~1 ms for 248K).
    var srcBuf = vImage_Buffer(data: ..., rowBytes: V * 2)
    var dstBuf = vImage_Buffer(data: ..., rowBytes: V * 4)
    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)

    // 2. rep_penalty over FULL VOCAB (not just top-K).
    for tok in recentTokens {
        let i = Int(tok)
        if logitsBuf[i] > 0 { logitsBuf[i] /= samplingRepPenalty }
        else                { logitsBuf[i] *= samplingRepPenalty }
    }

    // 3. fp32 argmax via vDSP_maxvi.
    var maxV: Float = 0
    var maxIdx: vDSP_Length = 0
    vDSP_maxvi(bp, 1, &maxV, &maxIdx, vDSP_Length(V))
    return Int32(maxIdx)
}
```

## Why it works

Apple ANE A18 still picks the WRONG top-1 ("おはる" instead of
"こんにちは") on the first generated token. But by the second/third
step, ALL of "おはる"'s top-K continuations ("おはる、", "おはる！",
"おはる～") have been seen recently. With **top-K-only rep_penalty**
the demotion only reaches the candidates ANE bothered to put in
top-40, so the loop stays in the おはる cluster.

**Full-vocab rep_penalty** demotes EVERY recently-seen token across
the entire 248K. After 2-3 おはる-flavored picks, all variants are
demoted enough that **fresh tokens** (whose fp16 logits we previously
ignored) win argmax. Model self-corrects, output reads correctly.

The fp16 ANE bias isn't fixed — just MASKED by aggressive rep_penalty
reach. The first 1-2 generated tokens may still be "off" from Mac
output, but the trajectory recovers within 2-3 tokens and continues
sensibly.

## Performance

| Device | Speed | Output |
|---|---|---|
| Mac M4 ANE | 50.4 tok/s | Clean ✓ |
| iPhone 17 Pro ANE | 45-50 tok/s peak / sustained | Clean ✓ |

Costs:
- Output transfer: 248K × 2B = 485 KB/step (~2 ms iPhone bus)
- Swift fp16→fp32: ~1 ms via vImage
- rep_penalty: ~64 ops, negligible
- vDSP_maxvi argmax: ~1 ms

Net Swift head cost ≈ 4-5 ms vs the alternative chunk_head fp32 model
at 8-10 ms. Plus ANE chunk_d retains the OLD unified speed (no extra
ANE dispatch).

## Generalization

**Any Apple-Silicon LLM port** where lm_head matmul on ANE produces
non-deterministic top-1 due to fp16 reduction ties:
1. Output the full fp16 logits — drop in-graph topk if you have one.
2. Sampling in Swift fp32 with rep_penalty applied over the FULL
   vocab (not just top-K candidates).
3. fp32 argmax / softmax / top-p as needed.

Body still runs ANE-native fp16 at full speed. Only sampling moves
to Swift, which is cheap when batched via Accelerate.

## What this doesn't fix

- The first 1-2 tokens after a "biased" prompt may still be wrong
  (e.g., こんにちは → おは…). The recovery starts at step 2-3.
- Pure greedy with rep_penalty=0 wouldn't help — needs the demotion
  to reach fresh tokens.
- For very short generations (< 5 tokens) on biased prompts, the
  output may not have recovered yet.
