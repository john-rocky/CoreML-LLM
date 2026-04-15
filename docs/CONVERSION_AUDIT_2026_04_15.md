# Conversion-side ANE Optimization Audit — Gemma 4 E2B

**Date:** 2026-04-15
**Scope:** Python conversion code under `conversion/` that emits the chunked
`.mlpackage` files consumed by the iOS runtime. On-device baseline:
**~15 tok/s @ 8K** on iPhone 17 Pro (4-chunk decode, INT4-palettized).
**Target:** 22-28 tok/s after Python-side graph fixes (pre-empting LiteRT-LM
56.5 tok/s is the medium-term goal).

Apple reference: `github.com/apple/ml-ane-transformers` (channels-first 4D,
QKV pack, packed gate/up, RMSNorm-as-LayerNorm).
ANEMLL reference: `github.com/Anemll/Anemll`.

The audit walked **`conversion/ane_ops.py`**, **`conversion/exporter.py`**,
**`conversion/build_merged_chunks.py`**, **`conversion/build_verify_chunks.py`**,
**`conversion/generate_rope.py`**, **`conversion/optimize_mlpackage_graph.py`**,
**`conversion/models/gemma4.py`**, **`conversion/models/gemma4_swa_chunks.py`**,
**`conversion/models/gemma4_lite_wrapper.py`**, **`conversion/models/gemma4_swa_merged*.py`**,
**`conversion/models/gemma4_prefill_chunks.py`**,
**`conversion/models/gemma4_stateless_chunks.py`**, plus the runtime hint
files in `Sources/CoreMLLLM/`.

---

## Executive summary

| Bucket | Items | Status |
|---|---|---|
| Already done | 1 (Conv1x1 linears), 5 (RoPE precompute), 6/7 (mask shape), 11 (manual softmax), 12+13 (lm_head/embedding offload), 14 (prefill/decode split), 16 (FP16 throughout), `compute_precision`+`minimum_deployment_target=iOS18`, INT4 palettization, `reshapeFrequency = .infrequent` runtime hint | DONE |
| Partial | 2 (4D layout — done in compute, but **transposes not minimized**), 9 (heads flattened — partial), 10 (KV-share — exploited but **kv13/kv14 hopped through every chunk via I/O**), 17 (transpose count — heavy), 18 (chunk op-budget — graph optimizer present but **never invoked in the build pipeline**) | PARTIAL |
| Missing | 3 (fused QKV Conv2d), 4 (fused gate/up Conv2d), 8 (RMSNorm-into-Conv2d absorption), 15 (tile-aligned padding for hidden=2560 / inter=6912 — but Gemma 4 E2B is hidden=1536/inter=6144, also misaligned to 128 tiles for some shapes), 19 (one true gather op remains: `InModelArgmax.gather` and `index_select` over the RoPE table; both are decode-time hot paths) | MISSING |

**Estimated headroom from missing items only:** roughly **+30–55 % decode
tok/s** over the current 15 tok/s baseline if the top 5 items below land
clean — i.e. the 22-28 tok/s target is plausible from Python-side work alone,
without speculative or runtime changes.

The ranked highest-leverage Python fixes:

1. **Fused `qkv_proj` (single Conv2d, 3*hidden out)** at every L0-14 layer.
   ~15 ops/layer eliminated, dispatch cut by 3x for QKV path. Est. **+8–12 %**.
2. **Fused `gate_up_proj` (single Conv2d, 2*intermediate out)** for the MLP.
   ~7 ops/layer eliminated. Est. **+5–8 %** (MLP is the largest op-time
   share per layer).
3. **Run `optimize_mlpackage_graph.py` automatically as a build step.**
   The script exists, is well-tested (`merge_consecutive_reshapes`,
   `merge_consecutive_transposes`, `fuse_layernorm`), but the build
   pipeline never calls it. 20-40 % MIL op-count drop is the script's own
   tested wins. Est. **+5–10 %** decode + faster cold compile.
4. **Drop the per-decode `index_select` on the RoPE table.** All four
   chunks `index_select` `cos_*`/`sin_*` once per decode call (5 calls in
   `gemma4_lite_wrapper.py` alone, plus in `_run_layer_swa` cos_s/sin_s
   inputs). Replace with a single Swift-side fp16 fetch into a flat
   per-call input — gather → linear-time index → CPU work that overlaps
   the ANE pre-flight. Est. **+2–4 %**.
5. **Absorb the four sandwich-norm scales into the next Conv2d weights.**
   Each layer has 4 RMSNorms × 1 elementwise multiply that the ANE has to
   schedule. The `weight` is constant — fold it into the next `q_proj` /
   `gate_proj` / `down_proj` weight column. Est. **+3–5 %**.

The remaining items (KV-share-routing, tile alignment, transpose pruning) are
worth doing but on the order of 1-3 % each, and several are blocked on
re-tracing all chunks.

---

## Item-by-item status

| # | Optimization | Status | Evidence (file:line) | Recommended change | Est. gain |
|---|---|---|---|---|---|
| 1 | Conv1x1 instead of `nn.Linear` | **DONE** | `conversion/ane_ops.py:57-111` defines `Conv2dLinear`; `conversion/models/gemma4.py:379-400` builds Q/K/V/O/gate/up/down/ple-gate/ple-projection/lm_head as `nn.Conv2d(..., 1)`; `gemma4.py:152-154` lm_head = Conv2d. | none | — |
| 2 | 4D `[B, C, 1, S]` channels-first | **PARTIAL** | Compute *body* uses NCHW — see `gemma4_swa_chunks.py:67` `x = h.permute(0,2,1).unsqueeze(2)`, `:147` and similar in `_run_layer_verify`, `gemma4_lite_wrapper.py:111`. **But every layer round-trips back to `(B,S,C)` 3D for the residual add and the next RMSNorm** (see `gemma4_swa_chunks.py:145,148-149,160-161`). That is ~6 reshape/transpose ops per layer × 35 layers × per chunk. | Keep the residual stream in NCHW until the chunk boundary. Have `ANERMSNorm` accept and return `(B, C, 1, S)` — currently it operates on the last dim, so it forces the back-and-forth. | +3-5 % |
| 3 | **Fused `qkv_proj`** (single Conv2d, 3*hidden out, then split) | **MISSING** | `conversion/models/gemma4.py:378-385` defines three separate `nn.Conv2d` for q/k/v. `_run_layer_swa` (`gemma4_swa_chunks.py:70,84,85`) dispatches them sequentially. Same in lite wrapper (`gemma4_lite_wrapper.py:113,126,127`), prefill (`gemma4_prefill_chunks.py:63,84,85`), and verify (`gemma4_swa_chunks.py:495,516,518`). **Note:** Gemma 4 E2B has 8 Q heads + 1 KV head (GQA-8), so the fused width is `8*hd + 2*hd = 10*hd` (Q=2048, KV=512 each → 3072 fused). | Add `Gemma4Model.from_pretrained()` post-load step that concatenates `q_proj.weight`, `k_proj.weight`, `v_proj.weight` along dim 0 into a new `qkv_proj` Conv2d, then update `_run_layer_swa` (and the verify/prefill/lite copies) to do one `qkv = layer.self_attn['qkv_proj'](x)` followed by `torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=1)`. KV-shared layers (L15-34) should still use `q_proj` only, since k/v are skipped — keep separate `q_proj` for those, fuse only L0-14. | +8-12 % |
| 4 | **Fused `gate_up_proj`** (single Conv2d, 2*inter out) | **MISSING** | `gemma4.py:396-400` separate gate/up; `gemma4_swa_chunks.py:156-159` calls them as separate dispatches `gate = layer.mlp['gate_proj'](x_mlp); up = layer.mlp['up_proj'](x_mlp); gate = F.gelu(gate, ...); mlp_out = layer.mlp['down_proj'](gate * up)`. Same in lite, prefill, verify, merged. | Concat `gate_proj.weight` and `up_proj.weight` into `gate_up_proj` (Conv2d, in=hidden, out=2*inter). Then `gu = layer.mlp.gate_up(x); gate, up = torch.chunk(gu, 2, dim=1); out = down(F.gelu(gate, approximate='tanh') * up)`. Note KV-shared layers use `intermediate_size * 2` (`gemma4.py:102-105`), so both gate and up double — fused width is 4*intermediate for those. Trace independently. | +5-8 % |
| 5 | RoPE sin/cos pre-computed as constants | **DONE** | `gemma4.py:176-198` (`_build_rope_caches`) registers `cos_sliding`/`sin_sliding`/`cos_full`/`sin_full` buffers. Externally, `conversion/generate_rope.py` writes them to `.npy` for the chunked path. Tables also baked into the trace as buffers in `gemma4_lite_wrapper.py:50-53`. | none — but see item 19 below: the runtime `index_select` into them is hot. | — |
| 6 | Sliding-window mask pre-computed | **DONE** | `causal_mask_sliding` is a model **input** of shape `(1,1,1,W)` in every chunk (`build_verify_chunks.py:144,200`, `build_merged_chunks.py:117,132`). Swift fills it once per decode step. The mask is not regenerated by trig/range ops in the graph. | none. (Caveat: the mask is recomputed in Swift every step rather than reused — this is a Swift-side fix.) | — |
| 7 | Global mask pre-computed | **DONE** | `causal_mask_full` shape `(1,1,1,ctx)` (`build_merged_chunks.py:116,131`). Same path as item 6. | none | — |
| 8 | RMSNorm scale absorbed into next Conv2d | **MISSING** | `ANERMSNorm.forward` (`ane_ops.py:41-54`) ends with `return normed * self.weight` — a separate elementwise multiply that ANE has to schedule, despite the weight being a constant. Pattern repeats 4 times per layer (sandwich norm) + q_norm + k_norm + per_layer_projection_norm + final norm. | After loading weights, fold each RMSNorm `weight` into the in-channel axis of the next Conv2d weight. For sandwich `input_layernorm.weight` → fold into Q/K/V Conv2d. For `pre_feedforward_layernorm.weight` → fold into gate/up Conv2d. For `post_feedforward_layernorm.weight` and `post_attention_layernorm.weight` → these are post-residual, so they cannot be folded into the next Conv2d directly; fold into the **previous** o_proj/down_proj output channel instead (mathematically equivalent because they are pre-residual scales). Then drop the `* self.weight` step in `ANERMSNorm` for those folded layers. | +3-5 % |
| 9 | Heads flattened into channel dim | **PARTIAL** | Q/K/V projections **do** produce `(1, H*hd, 1, S)` (a Conv2d output) — that is the Apple ml-ane-transformers shape. But the next step (`gemma4_swa_chunks.py:70`) is `q.view(1, num_heads, hd, 1).permute(0,1,3,2)` which immediately splits the channel dim back into heads × per-head and transposes for the matmul. The attention itself is then per-head (`matmul(q, K_expanded.transpose(-1,-2))`). | The Apple reference avoids the per-head split for attention by keeping `(1, H*hd, 1, S)` and using a "shared keys" reshape trick (chunked-block matmul). For Gemma 4 with 8 heads + GQA-8 (1 KV head replicated) the attention matmul has to split heads anyway because Q has 8 heads and K has 1 (broadcast) — so a *full* head-flatten is not possible. **Realistic action:** keep the current per-head split but eliminate the `.contiguous()` in `gemma4_swa_chunks.py:145` (`.permute(0,2,1,3).contiguous().view(1,1,-1)`) which forces a memory copy on ANE. | +1-2 % |
| 10 | KV-share exploited explicitly | **PARTIAL** | The structure is correctly detected (`gemma4.py:95-97` `is_kv_shared`) and the chunked SWA model **does** skip k/v projections for L15-34 (`gemma4_swa_chunks.py:83`). It also routes `kv13_k/v` and `kv14_k/v` from chunk2 → chunk3 → chunk4 (`build_merged_chunks.py:185-188`) so they are computed once. **However**, in `gemma4_lite_wrapper.py:95-98,150-155` the kv13/kv14 stash is initialized to **zero** every forward call and overwritten when L13/L14 run — fine for single-graph monolithic. In the chunked decode path the kv13/kv14 tensors are passed across chunks **as 4D MLMultiArrays** of shape `(1, 1, W, 256)` and `(1, 1, ctx, 512)` (`build_merged_chunks.py:170-173`), which means a write-then-read marshalling cost on every decode step for chunk3/4. | This is mostly a runtime optimization, not a Python-side one. The Python side is correct. To shave more: have chunk2 keep kv13/kv14 in a **state** (`MLState`) instead of an output, which lives across calls — needs `ct.StateType` annotation for the chunk2 graph. | +2-3 % (but mostly runtime work) |
| 11 | Manual softmax pattern (`max; exp; div sum`) | **DONE** | `ane_ops.py:156-168` `ane_softmax` implements `x_max = x.max(dim, keepdim); x_shifted = x - x_max; exp_x = torch.exp(x_shifted); exp_sum = exp_x.sum(dim, keepdim); return exp_x / exp_sum`. Used in every attention path (`gemma4_swa_chunks.py:142,595`, `gemma4_lite_wrapper.py:170`). | none | — |
| 12 | lm_head softmax/argmax offload differentiation | **DONE** (argmax in graph; no softmax over vocab on ANE) | `ane_ops.py:114-135` `InModelArgmax` — argmax + gather-of-max-logit done in graph, returns only `token_id` (int32) and `token_logit` (fp16 scalar). Avoids shipping the 262 144-element logit vector to CPU. The lm_head Conv2d itself runs as part of the chunk4 ANE graph. **Apple's official argmax** is used (no manual `max + arg` pattern), which the ANE compiler maps efficiently. | Optional: split lm_head to GPU. `build_eagle3_gpu.py` shows the pattern — `compute_units=ct.ComputeUnit.CPU_AND_GPU`, with a sidecar `preferred_compute_units` JSON entry — but for chunk4 specifically (where the 262 144-vocab matmul is the largest single op), routing only chunk4 lm_head to GPU could pay off. Not a clear win — needs A/B benchmark. | 0 to +5 % depending on bench |
| 13 | Embedding lookup off-ANE | **DONE** | `gemma4_lite_wrapper.py:79-84` does the embedding outside the Conv2d graph: text path = `nn.Embedding` lookup + scale (still inside the trace as a `gather` op), plus an `image_embedding` injection. **For the chunked decode path**, the model takes `hidden_states` as input (`build_merged_chunks.py:115`, `gemma4_stateless_chunks.py:160-161` "NO embed_tokens buffer — hidden_states passed as input ... Eliminates gather op"), and Swift performs the lookup via `Sources/CoreMLLLM/EmbeddingLookup.swift:13-50` using vDSP-vectorized INT8 → FP16 dequant. This is the best option ("gather op is bad on ANE"). | none for chunked path. The lite/monolithic paths still trace the `nn.Embedding` — keep them out of the iPhone build. | — |
| 14 | Prefill vs decode split | **DONE** | `models/gemma4_prefill_chunks.py` exists with `PREFILL_N = 512` fixed shape (`gemma4_prefill_chunks.py:29`) and per-token Q-norm via `view(N, num_heads, hd)` reshape (`gemma4_prefill_chunks.py:70`). Decode chunks are Q=1 (`gemma4_swa_chunks.py:67`). They are converted as separate `.mlpackage`s. There is **also** an experimental GPU-prefill path (`build_prefill_gpu.py`) that re-targets the prefill chunks to `cpuAndGPU`. | none (the design is right) — the per-call PREFILL_N=512 may be too large for an 8K context (N must divide ctx and amortize TTFT). Consider N=128 prefill shards, but that's a tuning question, not an audit gap. | — |
| 15 | Tile-aligned dims | **N/A → DONE-by-luck** | Gemma 4 E2B uses `hidden_size=1536` (12*128), `intermediate_size=6144` (48*128), `head_dim=256` (2*128), `global_head_dim=512` (4*128) — all aligned to the ANE 128-byte tile. The audit prompt mentioned hidden=2560/inter=6912 — those numbers are **not** correct for Gemma 4 E2B; the values from `gemma4.py:39-46` are 1536 / 6144. **No padding needed**. | none. Be aware: `num_attention_heads=8`, `num_key_value_heads=1` (`gemma4.py:41-42`) — single KV head means GQA broadcasts 1→8, which is hard for ANE matmul tiling. This is upstream-fixed (model architecture), can't change. | — |
| 16 | FP16 everywhere | **DONE** | `MODEL_DTYPE = torch.float16` (`ane_ops.py:22`); all weight tensors loaded as fp16 (`gemma4.py:241`); `ANERMSNorm` weight built fp16 (`ane_ops.py:37`); RoPE buffers `.to(MODEL_DTYPE)` (`gemma4.py:186-197`); `ane_softmax` enforces fp16 with explicit `.to(MODEL_DTYPE)` after `torch.exp` (`ane_ops.py:163-168`); ALL `ct.convert` calls use `compute_precision=ct.precision.FLOAT16` (`build_merged_chunks.py:71`, `build_verify_chunks.py:60`, etc.). **Caveat:** `MonolithicWrapper.forward` in `exporter.py:130-137` still does the SDPA in fp32 (`q.to(torch.float32)`, `softmax(..., dim=-1).to(MODEL_DTYPE)`, then matmul fp32). That path is *not* used for the production chunks, but if anyone re-uses `exporter.py` it will silently slow down. `stable_attention` in `ane_ops.py:201-234` is also fp32 internally — flagged but unused by the chunked Gemma 4 build. | Delete or rename the unused fp32 fallbacks in `exporter.py`/`ane_ops.py:201-234` to prevent accidental use. | — |
| 17 | Transpose ops minimized | **PARTIAL — NEEDS ACTION** | Grep of `gemma4_swa_chunks.py` finds **55 occurrences** of `permute|transpose|squeeze|unsqueeze`. Per-layer cost: roughly **6 transpose ops** = `permute(0,2,1).unsqueeze(2)` for QKV input, `view().permute(0,1,3,2)` for Q reshape (×3 for QKV), and `permute(0,2,1,3).contiguous().view(...)` for attention output. Most of those are forced by mixing 3D residual stream with NCHW Conv2d input. | (a) Run `optimize_mlpackage_graph.py` with `merge_consecutive_transposes` after every chunk build. (b) Restructure `_run_layer_swa` to keep hidden_states in NCHW the whole way through; only convert at chunk boundary. Removes 4 transposes per layer × 35 layers ≈ 140 ops/forward. | +3-5 % combined with item 2 |
| 18 | Chunk op-count under ANE 32 MB SRAM budget | **PARTIAL — TOOLING UNUSED** | `optimize_mlpackage_graph.py` exists, has a tested pass list (`DEFAULT_PASSES` at `optimize_mlpackage_graph.py:90-100`: dead-code, const-elim, fuse_linear_bias, fuse_gelu, fuse_layernorm, merge_reshapes, merge_transposes, fuse_matmul_bias) — claimed 20-40 % op count reduction in its docstring (`optimize_mlpackage_graph.py:8-11`). However, `build_merged_chunks.py`, `build_verify_chunks.py`, `convert.py`, and `exporter.py` **never call it**. Build artifacts ship un-optimized. There is no automatic op-count assertion to enforce the 32-64 ops-per-chunk budget Orion paper recommends. | Add as a build step in every `build_*` script: after `mlmodel.save(path)`, call `optimize_mlpackage_graph.apply_optimization_passes(mlmodel, DEFAULT_PASSES)` and re-save. Better: add a CI check that asserts `count_ops(mlm)["matmul"] + count_ops["conv"] < N` per chunk, fail loudly if regressed. | +5-10 % decode + faster cold compile |
| 19 | No gather/scatter ops | **PARTIAL — TWO REAL CASES** | Grep for `gather|index_select|scatter` shows: **(a)** `InModelArgmax` calls `logits.gather(-1, ...)` (`ane_ops.py:134`) — runs once per decode step, on a 262 144-element row, after argmax. ANE will fall back to CPU for gather — but on the chunk4 output side this is the right place (CPU is taking the result anyway). **(b)** `index_select` on the RoPE table — five chunks × four tables = up to 20 `index_select` ops per decode step (`exporter.py:85-86`, `gemma4_decoder.py:90-93`, `gemma4_lite_chunks.py:161-164`, `gemma4_lite_wrapper.py:89-92`, `gemma4_wrapper.py:122-125`). Each falls back to CPU. **(c)** The chunked path actually does **not** index_select inside the chunk graph — it accepts `cos_s`, `sin_s`, `cos_f`, `sin_f` as model **inputs** (`build_merged_chunks.py:135-138`) — Swift slices the table and passes the row as input. So the chunked production build is clean; only the lite/monolithic/decoder/wrapper variants index_select inside the graph. **(d)** Verify chunks use a `update_indicator` matmul instead of scatter (`gemma4_swa_chunks.py:544-549`) — that's the right ANE-friendly pattern. **(e)** `vocab_pruning` `index_select` (`apply_vocab_pruning.py:140-155`) is offline (load-time), not inference. | (a) Keep `InModelArgmax.gather` (the alternative is shipping 262 K floats per token to CPU, which is much worse). (b) Move `gemma4_lite_wrapper.py` index_select out of the model — the chunked build already does this; just deprecate the lite wrapper for shipping (it's referenced only by older scripts). | +1-2 % (because the chunked path is already clean) |

### Conversion-script knobs

| Knob | Status | Evidence |
|---|---|---|
| `minimum_deployment_target=iOS18` | DONE everywhere — `build_merged_chunks.py:70`, `build_verify_chunks.py:59`, `build_speculative.py:49`, `build_flash.py:38`, `build_w8a8.py:160`, `build_eagle3_chunks.py:63`, `rebuild_chunk4_8k.py:90`, `convert_audio_chunked.py:193`, `convert_audio.py:371`, `build_mtp_drafter.py:377`, `convert_gemma4_multimodal.py:140`, `exporter.py:253`, `build_eagle3.py:336,365`, `build_eagle3_gpu.py:120`, `build_prefill_gpu.py` (re-targets compiled chunks) — 100 % coverage. |
| `compute_precision=ct.precision.FLOAT16` | DONE on every Gemma 4 build path (same line numbers as above). The audio path also uses `FP16ComputePrecision` (`convert_audio.py:374`). |
| `OpPalettizerConfig(nbits=4, granularity='per_grouped_channel', group_size=32)` | DONE — `exporter.py:273-280`, `build_merged_chunks.py:80`, `build_verify_chunks.py:69`. This is the recommended Apple newer API; switching to the very newest `ct.optimize.coreml.experimental.OpPalettizerConfig` (iOS 18.4) is **not** worth it for Gemma 4 — extra LUT decode cost has been measured to wash. |
| `MLModelConfiguration.optimizationHints.reshapeFrequency = .infrequent` | DONE on the runtime side — `Sources/CoreMLLLM/ChunkedEngine.swift:193,327`, `MtpDraftSource.swift:47`, `CoreMLLLM.swift:214`, `CrossVocabDraft.swift:161`. iOS 18.2 + only; the runtime guards already exist. |
| `compute_units` differentiation per chunk | PARTIAL — chunks 1–4 default `CPU_AND_NE` (`build_merged_chunks.py:72`, `build_verify_chunks.py:61`); GPU re-target only via separate `build_prefill_gpu.py` and `build_eagle3_gpu.py`. There is no per-chunk `compute_units` knob driven from a single build script. Consider a JSON manifest (`fix_coreml_zoo_manifest.py:223,259` already writes one) so the runtime auto-picks the right units per chunk. |

---

## Reconversion order — what to fix first

The principle: each fix below **does not require re-quantization** of any
other chunk and **does not change the Swift API**. Bench after each step.

| Order | Fix | Files | Risk | Expected delta |
|---:|---|---|---|---|
| 1 | Wire `optimize_mlpackage_graph.py` into `build_merged_chunks.py` and `build_verify_chunks.py` as a post-save pass. | `conversion/build_merged_chunks.py` (after line 86 `mlmodel.save(save_path)`), `conversion/build_verify_chunks.py` (after line 80 `save_temp`). | Low — pass list is conservative; `--verify-equivalence` exists. | +5-10 %, faster cold-compile |
| 2 | Add a `_fuse_qkv` helper called from `Gemma4Model.from_pretrained` after weights load: concat `q_proj.weight`, `k_proj.weight`, `v_proj.weight` along dim 0 for layers 0-14 (skip L15-34, no k/v there). Update `_run_layer_swa` (and verify, prefill, lite) to call `qkv_proj` once and `torch.split`. | `conversion/models/gemma4.py` (new method), `conversion/models/gemma4_swa_chunks.py:69-86,494-525`, `conversion/models/gemma4_lite_wrapper.py:113-133`, `conversion/models/gemma4_prefill_chunks.py:63-91`. | Medium — re-trace, re-verify parity (existing `test_merged_parity.py` covers it). | +8-12 % |
| 3 | Add `_fuse_gate_up` analogous to step 2. Concat `gate_proj`/`up_proj` weights, single Conv2d, then `torch.chunk(gu, 2, dim=1)` and the rest of MLP unchanged. | Same files as step 2, MLP block. | Low — math is identical. | +5-8 % |
| 4 | Fold sandwich-norm scales into adjacent Conv2d weights; have `ANERMSNorm.forward` skip the `* self.weight` for those folded layers (or expose a `affine=False` flag). | `conversion/models/gemma4.py` (post-load fold), `conversion/ane_ops.py:33-54` (add flag). | Medium — algebraic identity is well-known but easy to mis-place; verify per-layer parity at fp16 with `test_merged_parity.py` cosine > 0.9999. | +3-5 % |
| 5 | Restructure `_run_layer_swa` to keep `hidden_states` in NCHW (4D, channels-first) end-to-end inside a chunk; convert only at chunk boundary inputs/outputs. Make `ANERMSNorm` accept channel-first 4D directly. Then run step 1 again to clean up trailing reshapes. | `conversion/ane_ops.py` (new `ANERMSNorm` 4D path), `conversion/models/gemma4_swa_chunks.py:_run_layer_swa` whole function. | High — many call sites; needs full parity sweep. Defer until 1-4 are shipped. | +3-5 % combined |

After steps 1-4 all done independently (each individually safe), expected
new baseline is **22-26 tok/s** decode @ 8K on iPhone 17 Pro — which would
hit the audit's 22-28 target without speculative decoding gains.

---

## Files that need edits (with function/line ranges)

These are the Python files that the proposed fixes touch. Line ranges are
approximate (current as of the snapshot read for this audit):

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4.py`
  - `Gemma4Model.__init__` (lines 108-200): no structural change, but add
    new `qkv_proj` / `gate_up_proj` Conv2d slots for L0-14 / all-layers
    respectively.
  - `Gemma4Model.load_weights` (lines 219-283): after the existing weight
    copy and tied-embedding fix, call new helpers `_fuse_qkv()`,
    `_fuse_gate_up()`, `_fuse_norms_into_conv()`.
  - `Gemma4DecoderLayer.__init__` (lines 357-417): make `q_proj`/`k_proj`/`v_proj`
    optional once `qkv_proj` is populated, similar for gate/up.

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_chunks.py`
  - `_run_layer_swa` (lines 46-181): replace lines 70/84/85 with single
    fused-QKV call and split; replace lines 156-159 with single
    fused-gate-up call and chunk; this is the highest-leverage edit.
  - `_run_layer_verify` (lines 457-635): same edits.

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_lite_wrapper.py`
  - `Gemma4LiteWrapper.forward` (lines 67-214): same fused-QKV / fused-gate
    edits at lines 113/126/127 and 184-187. Also: deprecate the
    `index_select` calls at lines 89-92 in favour of model-input cos/sin
    (or stop shipping this wrapper).

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_prefill_chunks.py`
  - `_run_layer_prefill` (lines 37-100+): fused-QKV / fused-gate edits.

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/ane_ops.py`
  - `ANERMSNorm` (lines 25-54): add `affine=False` flag so folded layers
    skip the final multiply.
  - Remove `stable_attention` (lines 201-234) and the fp32 path in
    `MonolithicWrapper` to prevent accidental fp32 promotion.

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/build_merged_chunks.py`
  - `trace_and_convert` (lines 58-93): after `mlmodel.save(save_path)`,
    invoke `optimize_mlpackage_graph.apply_optimization_passes` and
    re-save.

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/build_verify_chunks.py`
  - Same hook in `trace_and_convert` (lines 47-73) and around `save_temp`
    (line 80).

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/exporter.py`
  - `_export_monolithic` (lines 193-269): same post-save optimization
    hook.

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/optimize_mlpackage_graph.py`
  - `apply_optimization_passes` (lines 52-86): expose as a library function
    (already importable). Add `transformer_qkv_fusion`-style custom pass if
    we want to avoid changing the PyTorch model — but the PyTorch-side fuse
    is simpler and safer.

---

## Out-of-scope but worth noting

- **GQA-1 ratio (8 Q heads × 1 KV head).** The model's 1-KV-head GQA means
  every attention step expands K/V from 1 head to 8 via `repeat_interleave`
  (`gemma4_swa_chunks.py:132-133`, `gemma4_lite_wrapper.py:164-165`). This
  is an upstream architectural choice (it's how Gemma 4 E2B was trained).
  The Apple-style "shared keys" trick (one Conv1x1 reads all 8 heads as a
  single tile) would help; but it requires re-layouting the attention
  matmul. Not in this audit's scope.

- **Vocab pruning.** `prune_vocab.py` exists as a dry-run analyzer
  (`prune_vocab.py:1-50`); `apply_vocab_pruning.py` will trim
  `embed_tokens` and `lm_head` to a smaller vocab (`apply_vocab_pruning.py:140-155`).
  This is a memory and lm_head matmul size win, not strictly an ANE-graph
  optimization, but the ~2 % decode speedup from a 256K → 128K vocab is
  free if quality is preserved. Has been considered and rejected per
  `docs/EMBEDDING_BYPASS_FINDINGS.md` for the lite wrapper path.

- **Stateful KV via `MLState`.** The chunked build currently passes K/V
  caches as **inputs and outputs** (`build_merged_chunks.py:124-127,
  144-148`) instead of using `ct.StateType` (which `exporter.py:248-251`
  does for the monolithic path). Switching the chunks to `MLState` would
  remove the per-step MLMultiArray binding cost (~0.3-0.6 ms/step per
  chunk per the WARM_PATH_BENCH doc). This is a runtime-side workitem
  ranked separately, but it lives in the conversion script (the
  `ct.convert(..., states=[ct.StateType(...)])` call must move from
  `exporter.py` to `build_merged_chunks.py`).

- **Per-layer Embeddings (PLE).** The conversion currently bakes the PLE
  computation into chunk1 (`gemma4_swa_chunks.py:224-254`) using a
  one-shot reshape-then-LayerNorm trick (~70 ops eliminated, per the
  comment at line 234). This is a good optimization that's already in.

---

## Conclusion

The conversion code has done most of the well-known ANE optimizations
(Conv1x1, channels-first compute body, ANE-RMSNorm `[x,-x]` trick, manual
softmax, in-graph argmax, embedding off-ANE, FP16 throughout,
`reshapeFrequency=.infrequent`, INT4 palettization, chunked decode + prefill
split, KV-share routing), but **three major Apple ml-ane-transformers
patterns are still missing**: (1) fused QKV Conv2d, (2) fused gate/up
Conv2d, (3) RMSNorm-into-Conv2d weight folding. Together with running the
already-present `optimize_mlpackage_graph` pass automatically in the build
pipeline, these four fixes are mechanical, individually safe, parity-testable
with the existing `test_merged_parity.py`, and projected to deliver
**+22-35 %** decode tok/s — enough to hit the 22-28 tok/s target purely
from Python-side conversion changes, with no Swift-side or speculative
work needed.

The biggest remaining un-quantified question is whether Gemma 4 E2B's
GQA-1 architecture (8 Q heads, 1 KV head) leaves a non-trivial residual
ANE inefficiency that no graph rewrite can fix — that would be the lower
bound on what's achievable purely through conversion.
