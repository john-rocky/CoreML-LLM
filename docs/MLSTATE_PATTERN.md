# Core ML `MLState` + `slice_update` — what actually works on iPhone ANE

Distillation of three independent ports onto iPhone 17 Pro ANE
(VL Phase 1, Gemma 4 stateful, Qwen3.5 MLKV). Reusable recipe for any
new model that wants per-step KV cache without round-tripping state
through `MLFeatureProvider`.

## Verified-good envelopes

| Model | StateTypes per chunk | Names | Decode tok/s on iPhone 17 Pro |
|-------|----------------------|-------|-------------------------------|
| Qwen3-VL 2B (Phase 1) | 1 | `kv_cache_0` | 24.4 |
| Gemma 4 E2B stateful  | 2 | `kv_cache_sliding`, `kv_cache_full` | 31 (default ship) |
| Qwen3.5 0.8B MLKV     | 1 | `kv_cache` (KV only; SSM via I/O) | ~34 (Mac M4 51 / ratio 1.5; iPhone bench pending) |
| Qwen3.5 2B   MLKV     | 1 | `kv_cache` (KV only; SSM via I/O) | ~21 (Mac M4 32 / 1.5) |

## Verified-bad: 3+ StateTypes per chunk (2026-04-28, Qwen3.5 attempt)

Three `ct.StateType` per chunk (`kv_cache` + `conv_state` + `rec_state`):
- coremltools converts and saves the mlpackage cleanly.
- ANE compile reports 100% / 99.8% placement.
- **Predict-time runtime fails**:
  `ANEProgramProcessRequestDirect Failed with status=0x1d (Error 11)`.
- CPU runtime miscompiles `slice_update` on multi-state and outputs
  garbage tokens (Chinese spam where the prompt asked about Paris).
- GPU runtime is the only working path on the multi-state model.

A 4D-aligned conv_state shape (rank-4 instead of rank-3) does not
help. Cause is structural in the iOS 18 ANE runtime — multi-state
+ slice_update interaction.

## The recipe

When a model has multiple state buffers per chunk:

1. **Pick the biggest state** (typically full-attention KV — 12-24 MB
   per step for 24-layer models at max_seq=2048). Put just it in
   `ct.StateType`.
2. **Push the rest through input/output tensors**, keyed by absolute
   layer index (`conv_state_<i>` / `new_conv_state_<i>` etc.) so
   each chunk pulls only its own layers' state from a Swift-side dict.
3. **One `ct.StateType` per chunk** — even if you have logically two
   types of attention (sliding + full), pack them into a single big
   buffer with an offset convention (Gemma 4 packs `(2*L_sliding,
   nkv, W, hd)` and `(2*L_full, nkv, ctx, hd)` — 2 StateTypes total,
   still ANE-runnable).
4. **slice_update via Python slice-assign**:
   ```python
   kv_cache[k_idx:k_idx + 1, :, current_pos:current_pos + 1, :] = \
       k_write.unsqueeze(0)
   ```
   `current_pos` is `int32` shape `(1,)`. coremltools lowers this to
   `ios18.slice_update`.
5. **Re-slice the full layer K/V after the write**:
   ```python
   k_full = kv_cache[k_idx:k_idx + 1, :, :, :]
   ```
   Required because subsequent attention reads need to see the just-
   written position alongside the historical positions.

## Concrete example: Qwen3.5 hybrid SSM + full-attention

`conversion/qwen3_5_decode_layer_mlkv.py` defines two layer types:
- `MLKVLinearAttnDecodeStep`: SSM (Gated DeltaNet). State (`conv_state`,
  `rec_state`) flows through input/output tensors. Per-step Python /
  Swift dictionary update.
- `MLKVFullAttnDecodeStep`: GQA. KV cache lives in chunk-owned
  MLState; per-step writes via slice_update at `current_pos`.

`conversion/build_qwen35_decode_chunks_mlkv.py` declares:
```python
ct_states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name="kv_cache",
    ),
]
```
**Single** StateType per chunk. SSM-state inputs/outputs are declared
on `ct_inputs` / `ct_outputs` per the linear-attention layer indices
that fall in this chunk.

## Swift glue

```swift
// One handle per chunk; persists across decode calls.
states = bodyChunks.map { $0.makeState() }

// Per step:
for (chunk, state) in zip(bodyChunks, states) {
    let provider = try MLDictionaryFeatureProvider(dictionary: [
        "hidden_in":   hiddenFV,
        "cos":         fvCos,
        "sin":         fvSin,
        "causal_mask": fvMask,
        "current_pos": fvCurPos,
        // SSM state for this chunk's lin layers:
        "conv_state_<i>": ssmConv[i].mlfv,
        "rec_state_<i>":  ssmRec[i].mlfv,
        // ...
    ])
    let out = try await chunk.prediction(from: provider, using: state, options: opts)
    // Read SSM updates for the next step.
}
```

A reference implementation lives in
`Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen35MLKVGenerator.swift`
(SSM + full-attn) and `Qwen3VL2BStatefulGenerator.swift` (full-attn
only).

## Causal mask + position handling

- `causal_mask` shape `(1, 1, 1, max_seq)` fp16, with `-1e4` for
  positions strictly greater than `current_pos` and `0` otherwise.
  Build once per step on the Swift side.
- `current_pos` shape `(1,)` int32. Used both as the slice_update
  index AND inside the attention scores' causal addition.
- RoPE `cos` / `sin` shape `(1, 1, rotary_dim)` fp16. Precompute the
  full `(max_seq, rotary_dim)` table at load time, copy one row per
  step.

## Diagnostics — how to tell when MLState is broken

| Symptom | Likely cause |
|---------|--------------|
| `ANEProgramProcessRequestDirect Error=(11)` at predict time | multi-StateType ANE incompat (use MLKV) |
| CPU output is "coherent garbage" (Chinese on English prompt) | CPU runtime miscompiled slice_update on multi-state |
| ANE compile passes, predict succeeds, but state appears unchanged | rank/shape mismatch between buffer and StateType — rebuild and re-trace |
| Error code -14 at compile time | older coremltools; upgrade to 9.0+ |
| Compile crashes with E5RT MILCompilerForANE Error=(11) | per-chunk graph too big; chunk further or reduce ops |

## When NOT to use MLState

- The state is small (say < 1 MB / step). Marshaling cost is
  negligible; stay with stateless I/O for simpler debugging.
- The state semantics need cross-turn replay (e.g., regenerating from
  a saved prompt). Stateless makes this trivial; MLState requires
  saving + restoring `MLState` blobs explicitly.
- The model has 3+ logically distinct state types per layer that
  don't naturally pack. See "Verified-bad" above.
