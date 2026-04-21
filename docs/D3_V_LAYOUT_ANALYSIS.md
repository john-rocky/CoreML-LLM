# D3 — kv14_v Layout Mismatch: Analysis and Decision

Status: 2026-04-15. Producer (Gemma 4 target chunks) emits `kv14_v` with layout `(1, 1, 8192, 512)` ("seq, D" — standard attention). Consumer (Google-ported MTP drafter mlpackage) declares `kv14_v` as `(1, 1, 512, 8192)` ("D, seq" — TFLite pre-transposed). The two disagree on a pure layout axis swap; the data is identical bit-for-bit under a permute. This document is the D3 decision memo: where the fix goes, why, and how to validate.

The same mismatch exists on the sliding cache (`kv13_v`: producer `(1,1,512,256)`, drafter `(1,1,256,512)`). It is treated identically to `kv14_v` throughout.

---

## 1. Root cause reconstruction

The drafter is a byte-accurate re-materialization of Google's Gemma 4 E2B MTP drafter from their LiteRT-LM (`.litertlm`) release. TFLite stores `V` pre-transposed so that `attn @ V` can execute as a plain `matmul` without an on-graph transpose — the weights in `section_10.tflite` are organized assuming `V` arrives as `(B, 1, head_dim, seq_len)`. Our extraction (`conversion/extract_mtp_drafter.py`) and PyTorch re-build (`conversion/mtp_drafter_model.py`) preserve that calling convention verbatim. The PyTorch reference forward documents it explicitly at `conversion/mtp_drafter_model.py:113` (`kv_v: (B, 1, head_dim, ctx) int8 → dequant to fp`) and `:243` (`kv14_v: (B, 1, 512, ctx) target's full V`), then `untransposes` inside the kernel at `:145-146`:

```python
v_t = v.transpose(-2, -1)                      # (B, 1, ctx, D)
out = torch.matmul(attn.float(), v_t.float())  # (B, H, 1, D)
```

`build_mtp_drafter.py:211` ports the same pattern into the ANE graph:

```python
v_t = kv_v.transpose(-2, -1).to(MODEL_DTYPE)   # (1, 1, ctx, hd)
out = torch.matmul(attn.float(), v_t.float())  # (1, nh, 1, hd)
```

So the drafter always produces one extra internal `transpose` op on V. That transpose landed as a real MIL op in the compiled package — confirmed by the audit (`docs/MLPACKAGE_STRUCTURE_AUDIT.md:§2.4` lists 25 `transpose` ops in the drafter main block).

The Gemma 4 target, on the other hand, never pre-transposes V. `conversion/models/gemma4_swa_chunks.py:_run_layer_swa` is the sole SWA kernel reused by every chunk build (stateless, stateful, merged, WFA, flash, prefill, decoder, lite). V is produced as `(1, num_kv_heads, 1, hd)` from the `v_proj` Conv2d and stored in the slot as `(1, num_kv_heads, seq, hd)` by `torch.cat([V_sliding_slot[:, :, 1:, :], v_padded], dim=2)` (`models/gemma4_swa_chunks.py:109`) for sliding or `V_full_slot * (1 - update_mask) + v_padded.expand_as(...) * update_mask` (`:102`) for full. The output exports are taken directly from that slot:

```python
# models/gemma4_swa_chunks.py:117-121
if layer_idx == 13:
    kv_store_13_v = V_sliding_out[..., :256]  # (1, 1, W, 256)
elif layer_idx == 14:
    kv_store_14_v = V_full_out[..., :512]     # (1, 1, ctx, 512)
```

No transpose ever happens on the producer side. The chunk spec's declared output shape is `kv14_v (1, 1, CTX, 512)` at `conversion/build_verify_chunks.py:175` / `:191`. This is the canonical HuggingFace attention layout and it is what chunk3 and chunk4 expect back on their input port (same file, `:417`, `:479`) — so all downstream target consumers agree with the producer.

The drafter therefore sits on its own layout island, wholly inherited from TFLite.

The archaeology confirms the mismatch was hit and partially patched during the original MTP Path A bring-up:

- Commit `e72cbeb` ("fix: transpose target V cache for MTP drafter", 2026-04-14) added `transposeLastTwoDims(_:)` in `MtpSpeculativeEngine.swift` and applied it to `lastKV13V` / `lastKV14V` right before passing them to `MtpDraftSource`. Error message in the commit: `MultiArray shape (1 x 1 x 512 x 256) does not match the shape (1 x 1 x 256 x 512) specified in the model description` — proving the runtime mismatch was real.
- Commit `aef01ee` ("docs: MTP Path A integration results — acc0=0%, recommend fallback", same day) parked MTP Path A after acceptance measured 0 %. The transpose helper was removed from the current tree in a later cleanup (grep `transposeLastTwoDims` in `/Sources` returns no matches today). `MtpSpeculativeEngine.swift:82-86` now passes `lastKV{13,14}V` straight through, which would raise the same shape error if the MTP path were re-enabled with the current drafter package.

So: **the current production build has the mismatch latent** (MTP path is dormant behind `mtpEnabled` + DrafterUnion gating, but the fix commit is no longer in the tree). The audit's "~0.5 ms Swift-side transpose per decode step" refers to the cost that existed between `e72cbeb` and the removal — a cost we would pay again the moment MTP is re-enabled, unless we pick a real fix now.

---

## 2. Evidence table

| Role | File | Line | Shape / op |
|---|---|---|---|
| Producer: where V is stored | `conversion/models/gemma4_swa_chunks.py` | 109 | `V_sliding_out = torch.cat([V_sliding_slot[:, :, 1:, :], v_padded], dim=2)` → `(1, nkv, W, hd)` |
| Producer: where V is stored | `conversion/models/gemma4_swa_chunks.py` | 102 | `V_full_out = V_full_slot * (1-u) + v_padded.expand_as(...) * u` → `(1, nkv, ctx, hd)` |
| Producer: sliding export | `conversion/models/gemma4_swa_chunks.py` | 117 | `kv_store_13_v = V_sliding_out[..., :256]` → `(1, 1, 512, 256)` |
| Producer: full export | `conversion/models/gemma4_swa_chunks.py` | 121 | `kv_store_14_v = V_full_out[..., :512]` → `(1, 1, 8192, 512)` |
| Producer: declared output spec | `conversion/build_verify_chunks.py` | 175, 191 | `ct.TensorType(name="kv14_v", shape=(1, 1, CTX, 512), dtype=fp16)` |
| Consumer: declared input spec | `conversion/build_mtp_drafter.py` | 361 | `ct.TensorType(name="kv14_v", shape=(1, 1, 512, C), dtype=fp16_type)` |
| Consumer: actual use | `conversion/build_mtp_drafter.py` | 211 | `v_t = kv_v.transpose(-2, -1)` → `(1, 1, ctx, hd)`, then `matmul(attn, v_t)` |
| Consumer (PyTorch ref): same op | `conversion/mtp_drafter_model.py` | 145-146 | `v_t = v.transpose(-2, -1)`; `out = torch.matmul(attn, v_t)` |
| Swift transpose (removed) | `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` | 82-95 | Currently **no** transpose. In `e72cbeb` was `kv13V = try transposeLastTwoDims(kv13VRaw)` / `kv14V = try transposeLastTwoDims(kv14VRaw)` with a hand-rolled scalar loop inside `transposeLastTwoDims`. |
| Historical transpose impl | `git show e72cbeb` | 200-218 | `for i in 0..<N { for j in 0..<M { dstPtr[j*N + i] = srcPtr[i*M + j] } }` — scalar UInt16 loop over the full `(N, M)` grid. No vDSP / MPS. |
| Drafter MIL ops count | `docs/MLPACKAGE_STRUCTURE_AUDIT.md` | §2.4 | 25 transposes, 8 matmuls in the drafter main; two of those transposes are exactly the per-layer `kv_v.transpose(-2,-1)` (drafter has 4 layers but only 2 groups use V in transposed form — 1 SWA shared + 1 full — note the SWA stack reuses the same cached V across 3 layers). |

---

## 3. Cost analysis

The historical transpose (`e72cbeb`, now removed) ran two scalar loops per speculative cycle:

- `kv13_v`: N=256, M=512 → 131 072 fp16 loads + stores = 512 KB traffic.
- `kv14_v`: N=512, M=8192 → 4 194 304 fp16 loads + stores = 16 MB traffic.

A17/A18 DRAM-bound fp16 copy is ~40-50 GB/s best case on a 16 MB payload (saturating the SoC memory controller), so `kv14_v` alone is ~0.3-0.4 ms memory-bound. The observed ~0.5 ms includes the scalar loop overhead (8 M iterations × ~0.1 ns/iter in the CPU E-core for a non-vectorized access pattern) plus `kv13_v`. The audit's "~0.5 ms/step" is therefore consistent with the scalar implementation; a vectorized `vDSP_mtrans` implementation would cut this to ~0.1-0.15 ms (bandwidth bound).

Integration cost per session, given MTP runs the drafter once per draft step with K=3 per burst:

| Scenario | Bursts/token | Drafter calls/burst | Transposes/burst | ms/burst lost | Tokens/s impact @ baseline 32 tok/s |
|---|---|---|---|---|---|
| Current (transpose would run if path re-enabled) | 1 | 3 (K=3) | 1 (transpose is once per burst in `e72cbeb`, not per call — the inputs are read-only across the K-loop) | 0.5 ms | ~1.6 % @ 32 tok/s; ~0.8 % @ 16 tok/s measured |
| Worst case (transpose per call) | 1 | 3 | 3 | 1.5 ms | ~4.5 % |
| Long chat, 1000 tok output | 1000 bursts | — | 1000 transposes | 500 ms total | Measurable but sub-second over full generation |

`e72cbeb` did it once per burst (outside the K-loop), so the realistic cost is the 1.6 % / 0.5 ms number. Note that the drafter is called K times per burst and the **drafter internally** does its own `transpose(-2,-1)` on the same V tensor every call — so the "savings" from fixing this ALSO applies to eliminating those K internal ANE transposes per V input per burst, which is the second-order win. The internal ANE transposes are small per-call (8 MB on ANE ≈ 0.1-0.2 ms) and folded into the drafter's overall ~1.5-2 ms latency; eliminating them would yield a measurable but smaller drop in the drafter's own wall-clock.

Projected forward: at our beat-LiteRT target of 56.5 tok/s, a 0.5 ms per-token overhead is 2.8 % — not negligible but also not decisive. The real reason to fix is correctness, not speed (see §5).

---

## 4. Why the drafter came with a different layout

The drafter's V layout is not a design choice we made — it is an ABI constant inherited from TFLite. Google's LiteRT-LM stores all KV caches in the "pre-transposed-V" convention. When we extracted the drafter TFLite and traced it in `mtp_drafter_model.py`, we preserved the caller contract verbatim so the extracted fp weights would map 1:1 onto our PyTorch modules. Changing the convention at extraction time would have required relabeling every axis in the TFLite → PyTorch weight dictionary, which was deemed risky during the initial bring-up when parity was already fragile (TFLite → PyTorch cosine was 0.82, `docs/MTP_INTEGRATION_RESULTS.md`).

Conclusion: the layout mismatch is a historical artifact of cross-framework porting, not an architectural invariant. Fixing it is purely a software engineering question.

---

## 5. Fix option comparison

### Option A — Producer-side fix (transpose V in the chunks)

Scope: change `kv_store_14_v = V_full_out[..., :512]` to `kv_store_14_v = V_full_out[..., :512].transpose(-2, -1)` (and identical for kv_store_13_v) in **every** chunk variant that produces those outputs. Then flip the declared output shape in `build_verify_chunks.py` and the input shape on every **consumer** chunk (chunks 3, 4, merged chunks, stateless chunks) that reads `kv14_v`.

- Files touched: `gemma4_swa_chunks.py` (×4 call sites in `_run_layer_swa` / `_run_layer_verify`), `gemma4_swa_merged.py`, `gemma4_swa_merged1.py`, `gemma4_swa_merged2.py`, `gemma4_swa_wfa.py`, `gemma4_swa_flash.py`, `gemma4_stateless_chunks.py`, `gemma4_lite_chunks.py`, `gemma4_lite_wrapper.py`, `gemma4_prefill_chunks.py`, `gemma4_decoder.py`, `gemma4_wrapper.py`, `build_verify_chunks.py`, `build_speculative.py`, `build_w8a8.py`, `build_flash.py`, `build_eagle3_chunks.py`, `build_merged_chunks.py`, `build_prefill_gpu.py`, plus the Swift side: every `lastKV14V` consumer in `ChunkedEngine.swift` that forwards it to chunk3/4 (`:807`, `:931`, `:1100`, `:1234`, `:1364`).
- LOC: ~40 producer lines, ~25 consumer-spec lines, ~15 Swift kv14_v/kv13_v pass-through sites; plus rebuild of **all** chunks (~1.2 GB of mlpackages). Estimate: 80-120 LOC, 1-2 hours to write, 3-4 hours to reconvert and revalidate every chunk variant.
- Correctness risk: **high**. The target chunks 3-4 reuse kv14_v internally for `attn @ V` (`_run_layer_swa` line 143: `attn_output = torch.matmul(attn_weights, V_expanded)`). `V_expanded` is derived from `V_for_attn = kv_store_14_v`, and the matmul expects it as `(B, H, ctx, hd)`. If we transpose at the export point, we must also re-transpose before the matmul on the consumer side — which turns a 1-site fix into a two-site fix and loses the original motivation (eliminate a transpose, not move it). A transpose that gets re-transposed one hop later is net zero on op count.
- Rebuild impact: all 4 chunks must be rebuilt and re-shipped. ~5 GB re-download per user.
- Breaks: EAGLE-3 retrain pipeline (Track C) assumes the current producer layout — would need sync.

Verdict: large blast radius, negligible net win.

### Option B — Consumer-side fix (rewrite drafter to accept producer's layout)

Scope: (a) drop `kv_v.transpose(-2, -1)` from `build_mtp_drafter.py` and the PyTorch reference; (b) rewrite the `attn @ V` matmul to consume V as `(1, 1, ctx, D)` directly; (c) flip declared input specs for `kv13_v` / `kv14_v`; (d) re-convert only the drafter mlpackage (38 MB rebuild).

- Files touched: `build_mtp_drafter.py` (2 lines in `MtpLayerANE.forward`, 2 shape specs in the converter), `mtp_drafter_model.py` (2 lines in `DrafterAttention.forward`, for the parity reference), optionally `test_mtp_local.py` (kill `to_drafter_v` and the transpose helper).
- LOC: 8-12 LOC. Single source change. Single 38 MB reconversion.
- Weights: **layout-agnostic** — the drafter's V input is a raw cache tensor, not a trainable weight. There is no `kv_v.weight` anywhere; only `q_proj.weight`, `o_proj.weight`, `q_norm.weight`, and the MLP/norm weights. All weights are permuted off the `head_dim` axis which is NOT the axis we're transposing (we're swapping the `ctx` axis with `head_dim` only at the V-cache level; `q/k_proj` still output `(1, H, 1, D)`). Confirmed by reading `MtpLayerANE.__init__` (build_mtp_drafter.py:169-176) — no V-specific weight exists because the drafter has Q-only attention.
- Correctness risk: **low**. The change is a single op: swap `torch.matmul(attn, kv_v.transpose(-2,-1))` for `torch.matmul(attn, kv_v)`. Semantically this is the same operation; we are just removing the need to pre-transpose at the producer.
- Rebuild impact: only `mtp_drafter.mlpackage` (38 MB) is rebuilt; `build_mtp_drafter.py --palettize-int4` runs in ~2 min on CPU.
- Parity risk: TFLite → our-drafter parity will shift slightly (by fp rounding order) but the numerical operation is identical. We already have the TFLite parity harness in `test_mtp_parity.py`; the test still compares against the TFLite runtime using `kv_cache_v_*` arrays fed in whatever convention TFLite itself expects, independent of our mlpackage's input convention. One harness update: feed V as (1,1,ctx,D) instead of (1,1,D,ctx) when calling our model, leaving TFLite calls untouched. ~5 LOC.

Verdict: local change, no knock-on to target, cheap to validate.

### Option C — Swift-side band-aid (keep transposing at the seam)

Scope: restore `transposeLastTwoDims(_:)` in `MtpSpeculativeEngine.swift` (the same implementation as `e72cbeb`, or a vDSP-accelerated rewrite). Keep both mlpackages on their current layouts.

- LOC: ~25 (vDSP version ~15 using `vDSP_mtransD` — wait, no: vDSP has no fp16 mtrans. Would need BNNS (`BNNSCopy` with transpose flag) or a manual fp16 → fp32 → mtrans → fp16 roundtrip, which is slower. Alternative: `vImageConvert_Planar16FtoPlanarF` + `vDSP_mtrans` + `vImageConvert_PlanarFtoPlanar16F` — too expensive. Realistic best: `MPSMatrixCopy` on GPU with `transpose=true` but that requires `MPSMatrixDescriptor` bookkeeping. Or keep the scalar loop and measure whether it's actually 0.5 ms.).
- Latency cost: 0.5 ms scalar → 0.1-0.15 ms with an Accelerate-backed impl. Measurable overhead at 56 tok/s target (~1.6 % → ~0.6 %).
- Future-proofing: drafter is called K times per burst and the **internal** drafter transpose still runs on each call. Option C does nothing about that second-order cost — only the Swift-seam copy.
- Correctness risk: zero (simple data repacking).
- Maintenance: the Swift code has to know the drafter's layout convention forever, which is exactly the foot-gun Option B removes.

Verdict: ships quickly but locks in tech debt and costs 0.1-0.5 ms forever.

### Option D — Do nothing, accept 0.5 ms/step

Documented cost on the critical path. 1.6 % at 32 tok/s, 2.8 % at 56 tok/s target. Correctness-blocking as long as MTP path is re-enabled without Option C's helper — which was the state of the tree as of 2026-04-14 (`aef01ee`). So "do nothing" is only tenable while MTP stays parked, i.e. while the drafter's root acceptance-rate problem (0 %) is unresolved. The moment we move back to an MTP path (either EAGLE-3 retrain that lands in the same mlpackage or self-trained MTP heads), this layout mismatch becomes a blocker again.

Verdict: only viable short-term; not a real fix.

---

## 6. Recommendation

**Option B. Rewrite the drafter to consume V in the producer's native layout.**

Reasoning:

1. The consumer side is strictly smaller in blast radius: 8-12 LOC vs Option A's 80-120 LOC, one mlpackage rebuild vs 4-5, and zero impact on target-side code paths (including the EAGLE-3 retrain pipeline and the stateful/stateless chunks in `Examples/CoreMLLLMChat/`).
2. The drafter weights are layout-agnostic — there is no `v_proj` trained weight tied to a specific V axis order. The only change is the attention kernel's matmul line. This was verified by reading `MtpLayerANE.__init__` (build_mtp_drafter.py:169-176) and confirming the absence of a `v_proj` or V-related parameter.
3. It removes the ANE transpose op inside the drafter as well (currently emitted as one MIL `transpose` per V-consuming layer). Audit §2.4 counts 25 drafter transposes; dropping the two on V-cache paths would land the drafter at ~23, freeing a small but real fraction of its ~1.5 ms wall-clock.
4. Option A would require us to re-transpose on the target-side consumers of kv14_v (chunks 3, 4, and merged variants) because those consumers use kv14_v in a matmul that already expects `(..., seq, hd)`. Net: +1 transpose per target consumer, -1 transpose per drafter call. Since target consumers run every decode step and drafter calls run only during speculation, Option A is a net *loss* on the common path.
5. Option C adds a Swift-side CPU copy that the MTP path cannot amortize — every speculative cycle pays it. And it leaves the drafter's internal `kv_v.transpose(-2,-1)` MIL op in place, which is the same amount of data being shuffled on the ANE regardless.

Secondary benefit: aligns the drafter's calling convention with the target's, which removes a class of future bugs (e.g. if we later add a "verify with drafter K cache" path, or swap the drafter for an EAGLE-3 head that uses HF conventions natively).

---

## 7. Patch

### 7.1 `conversion/build_mtp_drafter.py`

File header docstring, lines 28-30 (update the I/O contract documentation):

```diff
-    kv13_v         (1, 1, 256, W) fp16  — target's sliding V
+    kv13_v         (1, 1, W, 256) fp16  — target's sliding V
-    kv14_v         (1, 1, 512, C) fp16  — target's full V
+    kv14_v         (1, 1, C, 512) fp16  — target's full V
```

Line 111-112 (update forward-docstring comments):

```diff
-        kv13_v,         # (1, 1, 256, W)
+        kv13_v,         # (1, 1, W, 256)
         kv14_k,         # (1, 1, C, 512)
-        kv14_v,         # (1, 1, 512, C)
+        kv14_v,         # (1, 1, C, 512)
```

Line 210-212 (the attention kernel — drop the V transpose):

```diff
         # Attn @ V
-        v_t = kv_v.transpose(-2, -1).to(MODEL_DTYPE)  # (1, 1, ctx, hd)
-        out = torch.matmul(attn.float(), v_t.float()).to(MODEL_DTYPE)  # (1, nh, 1, hd)
+        # V arrives as (1, 1, ctx, hd) in target's native layout.
+        v = kv_v.to(MODEL_DTYPE)                                       # (1, 1, ctx, hd)
+        out = torch.matmul(attn.float(), v.float()).to(MODEL_DTYPE)    # (1, nh, 1, hd)
```

Line 322-324 (forward-test sample tensors):

```diff
-        kv13_v = torch.zeros(1, 1, 256, W, dtype=MODEL_DTYPE)
+        kv13_v = torch.zeros(1, 1, W, 256, dtype=MODEL_DTYPE)
         kv14_k = torch.zeros(1, 1, C, 512, dtype=MODEL_DTYPE)
-        kv14_v = torch.zeros(1, 1, 512, C, dtype=MODEL_DTYPE)
+        kv14_v = torch.zeros(1, 1, C, 512, dtype=MODEL_DTYPE)
```

Line 359, 361 (declared input spec):

```diff
-            ct.TensorType(name="kv13_v",       shape=(1, 1, 256, W), dtype=fp16_type),
+            ct.TensorType(name="kv13_v",       shape=(1, 1, W, 256), dtype=fp16_type),
             ct.TensorType(name="kv14_k",       shape=(1, 1, C, 512), dtype=fp16_type),
-            ct.TensorType(name="kv14_v",       shape=(1, 1, 512, C), dtype=fp16_type),
+            ct.TensorType(name="kv14_v",       shape=(1, 1, C, 512), dtype=fp16_type),
```

### 7.2 `conversion/mtp_drafter_model.py` (parity reference)

Line 144-146 (matching change in the PyTorch reference, so `test_mtp_parity.py` still compares apples-to-apples):

```diff
-        # attn @ V: V is (B, 1, D, ctx) → need (B, 1, ctx, D)
-        v_t = v.transpose(-2, -1)  # (1, 1, ctx, D)
-        out = torch.matmul(attn.float(), v_t.float()).to(x.dtype)  # (B, H, 1, D)
+        # V is already (B, 1, ctx, D) in target's native layout — no transpose.
+        out = torch.matmul(attn.float(), v.float()).to(x.dtype)  # (B, H, 1, D)
```

Docstring/comment updates at `:113-114` and `:134-135` to reflect the new convention.

Line 582-585 (forward-test tensors):

```diff
-        kv13_v = torch.randn(B, 1, 256, ctx)
+        kv13_v = torch.randn(B, 1, ctx, 256)
         kv14_k = torch.randn(B, 1, ctx, 512)
-        kv14_v = torch.randn(B, 1, 512, ctx)
+        kv14_v = torch.randn(B, 1, ctx, 512)
```

### 7.3 `conversion/test_mtp_local.py`

Delete `to_drafter_v` (lines 157-165). Replace the two callers (line 168, 170) with `to_drafter_k`:

```diff
-        kv13_v = to_drafter_v(v13, 256, ctx)
+        kv13_v = to_drafter_k(v13, 256, ctx)
         kv14_k = to_drafter_k(k14, 512, ctx * 4)
-        kv14_v = to_drafter_v(v14, 512, ctx * 4)
+        kv14_v = to_drafter_k(v14, 512, ctx * 4)
```

### 7.4 `conversion/test_mtp_parity.py`

Update zero-KV fixture shapes at lines 89, 91 to match the new contract (swap the last two dims). One-liner each.

### 7.5 Swift side

No change. `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` already passes `lastKV13V` / `lastKV14V` through without transpose, which is exactly what the patched drafter expects. The two lines in `MtpDraftSource.swift:68,70` that document V shapes as `(1,1,256,W)` / `(1,1,512,C)` should be updated to `(1,1,W,256)` / `(1,1,C,512)` — cosmetic.

---

## 8. Validation procedure

1. **Unit parity** — run `python conversion/test_mtp_parity.py` after patching. Zero-KV drafter call should match the TFLite reference within the pre-existing 0.82 cosine envelope. (Target: no regression vs. current; we are not improving parity here, only changing I/O shape.)

2. **Rebuild drafter** — `python conversion/build_mtp_drafter.py --ckpt output/mtp_probe/mtp_drafter.pt --output conversion/output/iphone_8k/mtp_drafter.mlpackage --palettize-int4 --sliding-window 512 --context-length 8192`. Expect ~38 MB output, same size as current.

3. **Structural audit** — re-run the audit script that produced `docs/MLPACKAGE_STRUCTURE_AUDIT.md §2.4`. Expected drafter op count diff: `transpose` 25 → 23 (two V-transposes removed, one per attention block that reads V — one SWA shared across 3 layers + one full). No change to `matmul`, `softmax`, `topk`.

4. **On-device smoke** — deploy to iPhone 17 Pro via the CoreMLLLMChat example (MTP is behind `mtpEnabled`; set to true for this test only; set acceptance fallback threshold to 0 to force the path). Run a ~100-token decode with a fixed seed prompt. Expected: no `MultiArray shape ... does not match the shape ... specified in the model description` error (which is what we would see today if MTP were re-enabled).

5. **Perf check** — with `SpecProfile` instrumentation (already in `MtpSpeculativeEngine.swift:103, 140`), compare `draftMs` per burst before/after across 50 bursts. Expected: 0.1-0.3 ms reduction per burst (removing the in-drafter ANE transpose on V). No change in `verifyMs` since target is untouched.

6. **Regression on target** — explicit check that the target path (`ChunkedEngine.predictStep`) is byte-identical before/after. Since we did not touch target code or target mlpackages, this should be trivially true; confirm with `tok/s` baseline at ctx=2048 (current: ~27.2 tok/s, `docs/BASELINE_SPEED_AUDIT.md` if present). Tolerance: ±0.5 tok/s (measurement noise).

7. **Roll forward** — once 1-6 pass, delete `transposeLastTwoDims` cruft (already gone from main, but confirm it is not lurking in `.claude/worktrees/`) and update `docs/MLPACKAGE_STRUCTURE_AUDIT.md §4.4` to mark the item resolved.

Rollback: if any of 1-6 fails, revert the drafter mlpackage (git + re-copy the previous `mtp_drafter.mlpackage` to device). No target-side changes to undo.

---

## 9. Summary

- Root cause: drafter V-layout was inherited from TFLite's pre-transposed convention and never reconciled with the target's HF-style `(B, 1, ctx, D)` export.
- Producer fix (Option A) has blast radius over ~15 converter files and all 4 chunk mlpackages; also net-negative on op count because target consumers would need to re-transpose internally.
- Swift seam (Option C) ships fast but locks in 0.1-0.5 ms per burst and leaves the internal ANE transpose in the drafter unchanged.
- Recommended: Option B (~10 LOC in `build_mtp_drafter.py` + parity reference + parity test fixture, single 38 MB drafter rebuild, no target-side code changes). Validation procedure above; expected perf delta per burst is 0.1-0.3 ms with no correctness risk because the weight tensors involved have no axis dependency on the V layout.
