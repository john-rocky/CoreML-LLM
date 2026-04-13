# Unexplored Approaches — Round 2 (runtime mechanics)

Five additional candidates surfaced during the 2026-04-13 investigation that
asked: "what radical, untried direction could move the iPhone 8K @ 14.5 tok/s
needle, given everything in `SPEED_8K.md` / `ALTERNATIVE_APPROACHES.md` /
`UNEXPLORED_APPROACHES.md` is already on the board?"

These are **research notes only** — none implemented, none measured on device.
They are kept here so the next iteration starts with the option set complete.

The honest framing up front:

- None is a Copernican turn. They are **runtime-mechanics tweaks** layered on
  the existing chunked SWA pipeline.
- All numbers below are **expectations from literature + code reading**, not
  measurements. Treat as upper bounds; ANE wall-clock often comes in lower.
- The biggest leverage item is the **diagnostic** (#G2, MLComputePlan): it
  doesn't itself add a feature — it tells us *where the silent loss is*, and
  whether items in the existing roadmap are over- or under-priced. Cheapest
  step that could change the priority order of every other item.

---

## G1. Multi-function mlpackage with Q=1 decoder + Q=K verifier (shared weights)

### Idea
`coremltools` `MultiFunctionDescriptor` lets a single `.mlpackage` expose
multiple entry points that **share deduplicated weight blobs**. Build chunk3
and chunk4 with two functions:
- `decode_q1` — current Q=1 path used for autoregressive steps.
- `verify_qK` — Q=K (e.g. K=4) batched-token forward, used by EAGLE-3 to
  verify K draft tokens in a single ANE dispatch.

KV cache layout, weight tensors, RoPE tables stay identical. Only the Q
sequence dimension differs. Switch at runtime via
`MLModelConfiguration.functionName`.

### Why it could win
- ANE dispatch overhead is fixed-per-call. Going from 4 single-token verifies
  to 1 four-token verify replaces 4 dispatches with 1.
- The Q@K^T matmul reads K once and broadcasts across K queries, instead of
  4 separate reads. K is the bandwidth-bound side of attention at long ctx.
- Composes multiplicatively with EAGLE-3's draft acceptance.

### Relationship to `SPEED_8K.md §3 P2.2` (KV-share Q-batching)
Different axis:
- Doc's Q-batching: stack Q from L19/24/29/34 (different layers, same step)
  against shared kv13/kv14. **Layer axis.**
- This proposal: stack Q from K consecutive tokens (same layer, different
  steps) against the layer's own K. **Time axis.**

The two are **orthogonal and composable**. End state could be `4 layers × 4
tokens = 16 Q rows / matmul`.

### Expected
- 1.6–2.2× decode throughput on the speculative path, contingent on EAGLE-3
  acceptance ≥ 60%.
- 14.5 → ~22–32 tok/s @ 8K once EAGLE-3 lands. Pre-EAGLE-3, no win.

### Cost
- Re-export chunk3 / chunk4 with a second function (`MultiFunctionDescriptor`).
- Swift: hold two `MLModel` handles or one model with two function names; route
  by mode in the speculative loop.
- ~2 days engineering, 0 training (uses the same EAGLE-3 weights).

### Novelty caveat
- Multi-function packages, batched verify, and Q>1 prefill (Stephen Panaro's
  CLI uses Q=64 for prefill) all exist independently in the public ANE-LLM
  space. The specific combination — **batched verifier sharing weights with
  the Q=1 decoder via multi-function** — is not visible in published OSS
  (ANEMLL, smpanaro/coreml-llm-cli, Apple's ml-ane-transformers, EAGLE
  reference impls). Could be elsewhere; we did not exhaustively search.
- Conceptually a natural extension of EAGLE-3 once it ships, not a
  fundamentally new technique.

### Sources
- [coremltools — Multifunction Models](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)
- [Apple — Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli)

---

## G2. MLComputePlan-driven silent-fallback audit

### Idea
iOS 17+ exposes `MLComputePlan.deviceUsage(for:)` and
`MLComputePlan.estimatedCost(of:)` per MIL operation. Run it once per chunk
on device, dump every op whose `preferred` device is **not** Neural Engine
along with its estimated cost. Fix the costly ones; ignore the rest.

### Why it could win
The `compute_units = .cpuAndNeuralEngine` setting is *advisory*: the CoreML
compiler decides per-op which device runs. Gemma-family graphs are known to
trip this:
- coremltools issue [#2560 — Gemma-3 `__ior__` falls back](https://github.com/apple/coremltools/issues/2560)
- Mask-based KV updates (`K * (1 - umask) + new_k * umask`) sometimes route
  through CPU when broadcast shapes don't fit ANE templates.
- `gather` over RoPE tables with non-constant indices can drop to CPU.

We have no current visibility into which ops, if any, are on CPU/GPU per
chunk. **Every CPU-bound op on the per-step path costs ≥ a CPU↔ANE round
trip, ~0.05–0.5 ms each.** A single one repeated 35× (per-layer) is a
multiple-tok/s loss.

### Why this is the highest-leverage *next* step
This doesn't add a feature — it **prices** every other item in the roadmap.
If MLComputePlan reveals 5 ops on CPU at 1.5% of total cost each, that's a
~7% direct win and reorders the priority list. If it reveals nothing, we
narrow the search to genuine compute or bandwidth.

### Expected
- 1.05–1.25× depending on what surfaces. Could be zero if the compiler is
  already perfect — also a valuable result.
- 14.5 → 15–18 tok/s @ 8K floor; could be more if a hot loop op is found.

### Cost
- ~50 lines of Swift to walk `MLModelStructure.Program.Block.operations` and
  print non-ANE ops with cost. Run once at first launch behind a debug flag.
- 0 training, 0 reconversion, 0 model surgery.
- Half-day to write the tool, additional day(s) per offender to fix.

### Novelty caveat
- `MLComputePlan` is documented Apple API since iOS 17. `freedomtan/coreml_modelc_profling`
  uses it for general Core ML profiling. Not novel; just **not yet applied
  to this codebase**, and the docs in this repo do not mention it.

### Sources
- [MLComputePlan — Apple Developer](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)
- [WWDC23 — Improve Core ML integration with async prediction](https://developer.apple.com/videos/play/wwdc2023/10049/) (introduces compute plans)
- [freedomtan/coreml_modelc_profling](https://github.com/freedomtan/coreml_modelc_profling)
- [coremltools issue #2560 — Gemma-3 __ior__ failure](https://github.com/apple/coremltools/issues/2560)

---

## G3. GQA via broadcast matmul (drop `repeat_interleave` materialization)

### Observation
`conversion/models/gemma4_swa_chunks.py:132`:
```python
K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)
```
materializes K and V across the GQA replication axis before the attention
matmul. For full-attention layers at ctx=8192 with `n_rep=4`:
- K_for_attn: `(1, num_kv, 8192, 256)` ≈ 8 MB
- K_expanded: `(1, num_kv*n_rep, 8192, 256)` ≈ 32 MB

That's ~24 MB of redundant writes per K-cache **per full-attn layer** per
decode step. ×7 full-attn layers = ~168 MB/step of speculative bandwidth
waste at ctx=8192 (V doubles it).

### Note
`conversion/ane_ops.py` already defines `repeat_kv_ane` (an `unsqueeze +
repeat + view` variant) and `repeat_kv` (the HF-standard `expand + reshape`
pattern, where `expand` is zero-copy). Neither is wired up in
`gemma4_swa_chunks.py`. The model uses `repeat_interleave` directly.

### Idea
Replace materialization with broadcast matmul:
```python
q_grouped = q.view(1, num_kv, n_rep, 1, hd)            # (1, kv, rep, 1, hd)
K_b = K_for_attn.unsqueeze(2)                           # (1, kv, 1, S, hd)
attn_weights = torch.matmul(q_grouped, K_b.transpose(-1, -2))
# attn_weights: (1, kv, rep, 1, S) by broadcast
attn_weights = ane_softmax(attn_weights + mask, dim=-1)
V_b = V_for_attn.unsqueeze(2)                           # (1, kv, 1, S, hd)
attn_output = torch.matmul(attn_weights, V_b)           # (1, kv, rep, 1, hd)
attn_output = attn_output.view(1, num_kv * n_rep, 1, hd)
```
CoreML MIL `matmul` supports broadcast on outer dims; the ANE compiler
*should* emit this as a tiled matmul that reads K once.

### Risk
- "Should" is not "will." If the ANE compiler still materializes internally,
  we get the same wall-clock with a different graph shape.
- fp16 reduction order differs between the two formulations — last-bit
  numerical drift possible. Needs A/B parity check on a held-out prompt
  set before shipping.
- Requires reconverting and re-uploading the chunks.

### Expected
- 1.05–1.15× if ANE honors the broadcast (saves the 24 MB×7-layer write
  bandwidth on the K side, plus the same on V).
- 0 if the compiler materializes anyway.
- 14.5 → 15–17 tok/s @ 8K best case.

### Cost
- ~20 LoC change in `_run_layer_swa`.
- Reconvert all 4 decode chunks.
- A/B numerical parity test (~1 hour) + LongBench gate.

### Novelty caveat
- Standard pattern in HF transformers (`repeat_kv` uses `expand`). Not novel;
  just **not yet applied here**. Listed because it's a concrete code-level
  inefficiency in the current shipping converter.

### Sources
- [coremltools MIL `matmul` op](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.linear.matmul) — broadcast semantics on outer dims
- HF `transformers.models.llama.modeling_llama.repeat_kv` — the `expand`-based reference

---

## G4. Async / pipelined chunk dispatch across tokens

### Idea
Today's decode loop in `ChunkedEngine.predictStep` calls
`chunk1.prediction → chunk2 → chunk3 → chunk4` strictly serially for token
*t*, and only then starts chunk1 for token *t+1*. With CoreML's async
prediction API (WWDC23) we could overlap:

```
t=0:  c1 → c2 → c3 → c4
t=1:        c1 → c2 → c3 → c4
t=2:              c1 → c2 → c3 → c4
              ↑ pipelined, ANE busy ~always
```

### Catch
The chunks are **data-dependent**:
- `c2(t)` needs `h1(t)` from `c1(t)`.
- `c3(t)` and `c4(t)` need `kv13(t)` and `kv14(t)` from `c2(t)`.
- And critically: `c1(t+1)` needs the **token id** from `c4(t)`.

So within autoregressive greedy decoding, true pipelining across tokens is
**not possible** — token *t+1* can't start until *t* finishes. This idea
*only* unlocks once we have:
- A draft model (EAGLE-3) producing tokens *t+1, t+2, …* without waiting for
  the target's `c4(t)`. Then `c1(t+1)` runs with the draft token while
  `c4(t)` is still verifying token *t*.
- Or N-gram / retrieval drafting (training-free) for repetitive text.

### Expected
- Standalone (no draft): ~0 (no pipelinable opportunity in greedy decode).
- With EAGLE-3 + draft: 1.15–1.30× *additional* on top of speculative
  speedup, by hiding draft latency under target verify latency.
- Mirror Speculative Decoding (`UNEXPLORED_APPROACHES.md §B`) is a related
  but more ambitious version (NPU+GPU split). This is the single-accelerator
  variant.

### Cost
- Rewrite decode loop with `TaskGroup` / async predictions.
- Double-buffer the KV state inputs to avoid hazards (ping-pong IOSurfaces).
- ~1 day engineering. Useless without a draft source.

### Novelty caveat
- `MLModel.predictions(from:)` async batch API is documented since iOS 17.
- Not new technique; **not yet wired in our engine**, and not mentioned in
  existing docs as a stack item.

### Sources
- [WWDC23 — Improve Core ML integration with async prediction](https://developer.apple.com/videos/play/wwdc2023/10049/)

---

## G5. In-model top-K (not just argmax) on the LM head

### Observation
`SWAChunk4.forward` already does in-model `argmax` (`InModelArgmax` in
`conversion/ane_ops.py`) and returns only `(token_id, token_logit)` — so the
~512 KB fp16 logit vector never crosses ANE↔host. Good.

But every speculative-decoding scheme needs **top-K** (or top-p) candidates
per step, not just the argmax. Today, enabling speculative would force us to
either:
- Return the full logit vector → re-introduce the 512 KB/step transfer that
  `InModelArgmax` was added to eliminate, or
- Sample on CPU with the full vector → same problem.

### Idea
Add an `InModelTopK` op: returns `(top_indices[K], top_values[K])`,
K=8 or 16. Total: 8 × (int32 + fp16) = 48 bytes/token vs 512 KB/token for
the full vector — five orders of magnitude less.

For the 16-way split LM head (if applicable), each shard returns its local
top-K, then a final on-device reduction picks the global top-K from
16 × K candidates.

### Expected
- Standalone in greedy decoding: 0 (we already use argmax; there's no extra
  data to drop).
- As **prerequisite** for speculative decoding (G1, EAGLE-3 verify, Mirror
  SD): keeps ANE↔host transfer flat as we go from greedy to speculative,
  preventing a 5–12% regression.
- Worth it only if we ship speculative.

### Cost
- ~10 LoC change in `SWAChunk4.forward` (replace `argmax` with `topk`).
- Reconvert chunk4.
- Swift sampler accepts top-K instead of single index.

### Novelty caveat
- Direct extrapolation of ANEMLL's `--argmax` flag pattern. Not novel.
- Listed for completeness because speculative decoding work in this repo
  needs to land it before any speculative scheme can be measured fairly.

### Sources
- [ANEMLL — Artificial Neural Engine Machine Learning Library](https://github.com/Anemll/Anemll)
- `conversion/ane_ops.py :: InModelArgmax` (existing reference)

---

## Summary table

Numbers below are **expected**, never measured. Read with skepticism.

| # | Approach | Standalone gain @ 8K | With EAGLE-3 | Effort | Risk | New code or model? |
|---|---|---|---|---|---|---|
| G1 | Multi-function Q=K verifier | 0 (needs draft) | **×1.6–2.2** → 22–32 tok/s | 2 d | medium | both |
| G2 | MLComputePlan audit | ×1.05–1.25 → 15–18 | same multiplier | 0.5 d + fixes | low | Swift only |
| G3 | GQA broadcast matmul | ×1.05–1.15 → 15–17 | same | 1 d + reconvert | medium (numerical) | model only |
| G4 | Async chunk pipeline | 0 (greedy serial) | ×1.15–1.30 | 1 d | low-medium | Swift only |
| G5 | In-model top-K | 0 (argmax already) | prerequisite for G1 | 0.5 d + reconvert | low | both |

### Recommended sequencing

1. **G2 first.** Half-day diagnostic, no model change, tells us whether to
   reprioritize anything else. Run it before any other work in this list.
2. **G3 second** if EAGLE-3 is still in training: model-side win that ships
   before speculative lands.
3. **G5 + G1 + G4 as a bundle** once EAGLE-3 weights land. They are mutually
   reinforcing: G5 keeps host bandwidth flat, G1 batches verify, G4 hides
   draft latency.

### What this list is *not*

- Not a fundamental architectural breakthrough. Items in
  `ALTERNATIVE_APPROACHES.md` (#1 distill, #5 from-scratch ANE-native) and
  `SPEED_8K.md` Tier C (StreamingLLM+QLoRA, MHA→MLA retrofit) remain the
  only paths to >2× standalone wins.
- Not measured. Every number is an expectation. Real wall-clock requires the
  on-device benchmark already running on `claude/diag-per-chunk-timing`.
- Not exhaustive. Deliberately omitted: hand-rolled MPS shaders (violates
  ANE-only decode), private ANE APIs (App Store risk), full graph rewrite
  to Conv2d-only attention (multi-week effort).

---

## References

- [coremltools — Multifunction Models](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)
- [Apple — Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli)
- [Stephen Panaro — In Pursuit of Fast KV-Cached Attention for ANE](https://stephenpanaro.com/blog/kv-cache-for-neural-engine)
- [ANEMLL repo](https://github.com/Anemll/Anemll)
- [MLComputePlan — Apple Developer](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)
- [WWDC23 — Improve Core ML integration with async prediction](https://developer.apple.com/videos/play/wwdc2023/10049/)
- [WWDC24 — Deploy ML and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/)
- [freedomtan/coreml_modelc_profling](https://github.com/freedomtan/coreml_modelc_profling)
- [coremltools #2560 — Gemma-3 __ior__ failure](https://github.com/apple/coremltools/issues/2560)
- [SqueezeBits — Disaggregated Inference on Apple Silicon (NPU prefill + GPU decode)](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [HeteroLLM (arXiv 2501.14794)](https://arxiv.org/abs/2501.14794)
