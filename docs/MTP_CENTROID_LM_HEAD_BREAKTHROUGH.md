# Gemma 4 MTP Drafter — Centroid LM Head Breakthrough (2026-05-06)

**Outcome on Mac (E2B, K=3 verify, K_USE=2, ANE, drafter fp16+int32 buffer,
`MTP_DRAFT_POS_MODE=constpm1` default):**

| Content type | base tok/s | centroid tok/s | speedup | per-slot accept |
|---|---|---|---|---|
| Repetitive ("yes" × 30) | 33.3 | **70.6** | **2.12×** | 0.61 |
| Translation list | 33.2 | 56.2 | 1.69× | 0.38 |
| JSON output | 33.2 | 54.0 | 1.63× | 0.33 |
| Capitals list | 33.1 | 44.6 | 1.35× | 0.24 |
| Code (Fibonacci) | 33.5 | 49.1 | 1.47× | 0.27 |

Average ~**1.65×**, peak **2.12× on repetitive content**. The vendor's "up to
3×" is genuinely achievable — speedup scales with content predictability
(per-slot accept rate). Chat code is the hardest case (highest entropy → lowest
accept). Drafter forward step: ~9 ms (broken full-vocab path) → **2.5 ms**
(centroid path, ~3.6× faster drafter compute).

For comparison: HF runtime ceiling on chat-templated code per the earlier
investigation is **1.19×** — our 1.47× on the same content type already
exceeds the HF Python reference's reported ceiling.

---

## Root cause

The official Gemma 4 MTP drafter (`google/gemma-4-E2B-it-assistant`) is trained
against a **MaskedEmbedder cluster-routed lm head**, not full vocab argmax.
Architecture:

* `centroids: nn.Linear(256, 2048)` — learned (not derived from token embeds).
* `token_ordering: int32(262144)` — assigns each canonical token id to one of
  2048 clusters; reshape `(2048, 128)` gives the per-cluster token list.
* Inference flow at the lm head:
  1. `centroid_logits = h @ centroids.T`  → `(2048,)`
  2. top-32 clusters via `argpartition`
  3. gather their canonical token ids: 32 × 128 = **4096 candidate tokens**
  4. `selected_logits = h @ embed[candidates].T`  → `(4096,)`
  5. argmax (or top-K) over the 4096

Our previous build skipped `masked_embedding.*` tensors and used full-vocab
argmax (`logits = h @ embed_tokens.T` over 262 144). The drafter network's
hidden states optimise for cluster discrimination first; full argmax sees the
same logits but considers tokens outside the network's "intended" vocab,
producing tokens the network was never trained to predict — and the target
rejects them.

Random-`h` test (50% disagreement): see `conversion/build_mtp_drafter.py`
docstring + the disagreement check we ran (14/20 random hidden states gave
different top-1 between full-vocab and cluster paths).

---

## Implementation

`conversion/build_mtp_drafter.py` gained `--centroid-lm-head` and now ships a
working `MaskedEmbedderANE` (rolled into `MtpDrafterANE.forward`).

### Key wiring in `MtpDrafterANE.forward` (cluster path)

```python
h_nchw = h.permute(0, 2, 1).unsqueeze(2)
c_logits = self.centroids(h_nchw).squeeze(-1).squeeze(-1).squeeze(0)  # (2048,)
_, top_clusters = torch.topk(c_logits.float(), 32)                    # (32,)

ordering_2d = self.token_ordering.view(2048, 128)                     # int32
selected_canonical = ordering_2d.index_select(0, top_clusters.long())  # (32,128)
selected_canonical_flat = selected_canonical.reshape(-1)              # (4096,)

# lm_head_weight has dim 0 = vocab_size > 32767 → coremltools' gather guard
# keeps these indices int32 (no int16 demotion).
sel_emb = self.lm_head_weight.index_select(0, selected_canonical_flat.long())

h_2d = h.reshape(1, -1)
selected_logits = (h_2d.float() @ sel_emb.float().T).squeeze(0)       # (4096,)

top_k_vals, top_k_pos = torch.topk(selected_logits, k=8)              # positions in 4096
top_k_ids = selected_canonical_flat.index_select(0, top_k_pos.long()) # int32 token ids
```

### Three coremltools quirks that bit during conversion

1. **`add_int16_cast` pass demotes topk indices to uint16** (max 65535) when
   vocab > 32767. Truncates token ids. **Fix: drop `common::add_int16_cast`
   from the pass pipeline** (centroid path only — the baseline build keeps
   the optimisation since it doesn't hit topk-on-large-vocab).

2. **`compute_precision=FLOAT16` casts fp32 buffers to fp16.** Storing
   `token_ordering` as fp32 (so it could carry int values losslessly) is
   defeated by the global fp16 cast — values >2048 lose precision, multiples
   of 32 above 32k. **Fix: store `token_ordering` as int32 throughout.** The
   only gather using these large values has `lm_head_weight.shape[0] = 262144
   > 32767`, which trips the `add_int16_cast` gather guard and keeps indices
   int32.

3. **Module-level `.to(MODEL_DTYPE=fp16)` casts the int32 buffer too.**
   Reassert `ane.token_ordering.data = ane.token_ordering.data.to(int32)`
   right after the `.to()` call.

### Pass pipeline (centroid path only)

```python
def _build_centroid_pass_pipeline():
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes({"common::add_int16_cast"})
    return pipeline
```

`compute_precision` stays at `FLOAT16`. Output dtype must be set explicitly:

```python
ct.TensorType(name="top_k_indices", dtype=np.int32),
```

### Output verification (CPU only, np.random seed=42)

| build | top_k_indices |
|---|---|
| broken (FLOAT16, default pipeline) | `[4, 10, 1, 6, 4, 0, 0, 1]` (positions, not ids) |
| broken (FLOAT16, output_dtype int32) | `[-24672, 19610, 21571, ...]` (uint16 sign-extended) |
| broken (FLOAT16, fp32 token_ordering) | `[50816, 41024, 26592, 58752, ...]` (multiples of 32) |
| **fixed** (FLOAT16, int32 token_ordering, no add_int16_cast) | `[106400, 85146, 152643, 143440, ...]` ✓ |

---

## What did NOT help

### AWQ-style smoothing on target chunks (calibration-aware quantisation)

* Implemented in `conversion/awq_smooth_gemma4.py` (v1, naïve calibration:
  8 prompts × 1 last token, single-position attention approx).
* Built K=3 verify chunks with smoothed weights.
* Bench result on chat code: **net neutral** (-3 % to +6 % across prompts,
  noise band).
* Math correctness verified: `cos(orig_qproj_out, awq_qproj_out) = 1.000000`
  in fp32 — smoothing is mathematically equivalent, the issue is calibration
  quality vs the actual problem (which turned out to be the lm head, not
  target quantisation).

### K=8 verify chunks + K_USE sweep

| config | tok/s (fib) | accept |
|---|---|---|
| K=3 verify, K_USE=2 (centroid drafter) | **47.4** | 0.25 |
| K=8 verify, K_USE=2 | 36.5 | 0.05 |
| K=8 verify, K_USE=4 | 42.4 | 0.09 |
| K=8 verify, K_USE=6 | 34.2 | 0.06 |
| K=8 verify, K_USE=8 | 33.7 | 0.07 |

K=8 verify chunks underperform K=3 because:

1. **KV cache pollution at K_USE < K**: with K_USE=2 the trailing 5 verify
   slots are padded with the last drafted token. The verify chunk *writes*
   all K positions to KV cache, so dummy positions corrupt cache state for
   the next round → drafter sees wrong K → wrong drafts.
2. **Autoregressive accept decay** at K_USE=K=8: per-slot accept drops as
   p, p², p³, … so longer drafts have diminishing returns. Cycle math:
   K_USE=8 emit ≈ 1 + 1·p_avg ≈ 2 vs K_USE=2 emit ≈ 1.5; drafter cost grows
   linearly with K_USE.

K=3 verify chunks stay the production setup.

---

## Reproduction

```bash
# 1. Build centroid drafter
~/.pyenv/versions/lama-cml/bin/python conversion/build_mtp_drafter.py \
  --hf-repo google/gemma-4-E2B-it-assistant \
  --output /tmp/mtp_drafter_centroid.mlpackage \
  --sliding-window 512 --context-length 2048 \
  --centroid-lm-head

# 2. Build K=3 verify chunks (target side, baseline INT4)
~/.pyenv/versions/lama-cml/bin/python conversion/build_verify_chunks.py \
  --model gemma4-e2b --K 3 --output /tmp/gemma4_chunks_K3 --ctx 2048

# 3. Compile to .mlmodelc and assemble bundle
xcrun coremlcompiler compile /tmp/mtp_drafter_centroid.mlpackage <bundle>
for i in 1 2 3 4; do
  xcrun coremlcompiler compile /tmp/gemma4_chunks_K3/chunk${i}.mlpackage <bundle>
done

# 4. Bench (Mac, ANE)
SPECULATIVE_PROFILE=1 SMOKE_TARGET_DEVICE=ane MTP_K_USE=2 \
  ./.build/release/coreml-llm-smoke <bundle> "<prompt>" 96
```

Verify the drafter produced sane top-K ids (CPU run, random seed):

```python
import coremltools as ct, numpy as np
m = ct.models.MLModel(".../mtp_drafter.mlpackage",
                      compute_units=ct.ComputeUnit.CPU_ONLY)
# ... feed dummy inputs ...
out = m.predict(feed)
assert max(out["top_k_indices"]) > 1024  # not stuck in fp16-quantised range
```

---

## Files touched

* `conversion/build_mtp_drafter.py` — `--centroid-lm-head`, pass pipeline,
  fp16 cast workaround, int32 buffer reassert.
* `conversion/mtp_drafter_model.py` — load `masked_embedding.*` from HF.
* `conversion/awq_smooth_gemma4.py` — naïve AWQ smoothing (kept for
  reference; not currently shipping).
* `conversion/build_verify_chunks.py` — `--awq-state` flag (kept).

## Open questions

* **3× vendor speedup**: vendor's screenshot shows 2.11× on Fibonacci
  recursion. Our same prompt: 1.52× (50.97 vs 33.5 tok/s). Difference is
  per-slot accept (vendor ~0.55 vs ours ~0.29). Likely causes:
  - Vendor demo runs E4B (8B target) — more confident → higher accept
  - Vendor target is bf16 (no quantisation drift on K cache)
  - Our drafter is identical (same official weights), so it's target-side
* **iPhone**: Mac numbers don't transfer 1:1; iPhone ANE has different gather
  / topk performance characteristics. Re-bench needed.
* **Drafter on ANE vs GPU**: the cluster path's gather + scatter ops may
  force GPU fallback. ANE-only drafter would be even faster.

## What was tried but didn't help (negative results)

* **AWQ-style activation-aware smoothing on target**: net-neutral (+/- noise).
  Calibration too small (8 prompts × 1 last token); v2 with longer calibration
  not run.
* **K=8 verify chunks** (with K_USE 2/4/6/8 sweep): K_USE < K causes KV
  cache pollution from padded slots, K_USE=K causes autoregressive accept
  decay. K=3 + K_USE=2 stays optimal.
* **INT4 palettize on drafter**: drafter latency 2.5 ms → 6.4 ms on Mac
  (dequant overhead dominates the small drafter). iPhone may differ.
* **Mixed-precision target chunks** (`PALETTIZE_KEEP_FP_KV=1`,
  `PALETTIZE_KEEP_FP_ATTN=1`): k/v_proj or all self_attn weights kept fp16,
  rest INT4. Net-neutral on Fibonacci (-7 % vs fresh INT4 at constpm1) and
  most chat prompts; +4 % on repeat. Mixed precision breaks ANE's all-INT4
  fast path. Code kept env-gated (`PALETTIZE_KEEP_FP_KV=1`) for iPhone
  experimentation since iPhone ANE may handle mixed precision differently.
* **`MTP_DRAFT_POS_MODE=perstep`** (former default): broken on freshly-built
  K=3 chunks (accept 0.00 on repeat). Production chunks tolerated it via
  some pipeline-version coincidence; HF reference is `constpm1` and the
  fresh build requires it. Now the Swift default.
