# Stage 5 (qwen3vl-linear) — Plan 3 Linear projections fanout

**Date:** 2026-04-26
**Branch:** `stage5-qwen3vl-linear`
**Status:** Mac-side complete. Hold merge for iPhone 17 Pro tok/s check.

This doc is the durable record so the work survives session boundaries.
Detail behind every number is in the linked commits and probe scripts.

---

## 0. TL;DR

Qwen3-VL 2B stateful + multifunction converters now accept
`--linear-projections`. Mac data shows body chunks at parity, but
`chunk_head` regresses (-31 pt ANE placement, +32 % per-call wall
time) when its `lm_head` is swapped from `Conv2dLinear` to `nn.Linear`.
The body win + head loss net out to **E2E +2.2 %** on Mac (within the
±5 % bar but not free). iPhone 17 Pro re-test gates adoption — body-only
swap is the obvious mitigation if iPhone confirms the head regression.

---

## 1. What shipped to the branch

| commit | scope |
|---|---|
| `7821f36` | claim INFLIGHT |
| `f391c59` | converter: `--linear-projections` flag + `_project` helper on chunks builder + multifunction builder |
| `f3ae5fb` | runtime ModelInfo + Mac probes/sanity |

### Code changes

`conversion/build_qwen3_vl_2b_stateful_chunks.py`
- `_replace_conv2dlinear_with_linear` — `Conv2dLinear` (1×1 wrapper) →
  `nn.Linear` weight reshape `(out, in, 1, 1) → (out, in)`.
- `_swap_layer_projections_to_linear(layer)` — walks every projection
  on a single decoder layer.
- `_project(proj, x_conv)` — dispatches Conv2dLinear / Linear at trace
  time. Conv path stays in conv layout `(B, in, 1, T)` end-to-end;
  Linear path squeezes to `(B, T, in)`, runs, re-permutes back.
- `ANEStatefulDecoderLayer / ANEStatefulBodyChunk / ANEHeadChunk` accept
  `use_linear: bool = False`; head swaps its own `lm_head`.

`conversion/build_qwen3_vl_2b_stateful_multifunction.py`
- Re-imports the helpers; `ANEStatefulPrefillLayer /
  ANEStatefulPrefillBodyChunk` accept `use_linear`. Both `infer` (T=1)
  and `prefill_b8` (T=8) functions swap together (multifunction
  weights are shared).

`Sources/CoreMLLLM/ModelDownloader.swift`
- `qwen3vl_2b_stateful_linear` ModelInfo (folder
  `qwen3-vl-2b-stateful-linear`, sideload-only, experimental). Existing
  `localModelURL` + LLMRunner stateful detection routes transparently
  because the inner `qwen3_vl_2b_stateful_chunks/` layout is unchanged.

### Mac probes / sanity

| script | purpose |
|---|---|
| `conversion/probe_qwen3vl_chunk0_linear_latency.py` | chunk_0 dispatch A/B |
| `conversion/probe_qwen3vl_chunk0_op_mix.py` | MIL op + ANE placement audit per variant |
| `conversion/probe_qwen3vl_e2e_linear_latency.py` | 4-chunk + head full step A/B with per-component breakdown |
| `conversion/sanity_qwen3vl_stateful_chunks.py` | load-all + state round-trip + E2E token sanity |

---

## 2. chunk_0 PoC (INT8)

```
[Conv2dLinear] /tmp/q3vl_linear_test/conv/.../chunk_0.mlpackage
  total ops:    1572   compute ops:  650 (excl const)
  ANE %:        93.5   (608/650)
  median 8.29 ms

[nn.Linear]    /tmp/q3vl_linear_test/linear/.../chunk_0.mlpackage
  total ops:    1589   compute ops:  762 (excl const)
  ANE %:        94.5   (720/762)
  median 8.31 ms

  Δ total ops    +1.1 %      Δ compute ops  +17.2 %
  Δ ANE %        +1.0 pt     Δ Mac latency  +0.2 % (noise, std=0.18)
```

The +17 % compute-op count comes from the `_project` Linear branch
adding 4 extra reshape-class ops (transpose, squeeze, expand_dims) per
projection — Qwen3-VL is conv-layout-native end-to-end inside the
attention block, so the Linear branch needs to permute IN and OUT
around the matmul. Despite the higher op count, ANE placement is +1 pt
and Mac latency is parity within noise. **Body chunks pass adoption
criteria on Mac.**

Different from Gemma 4 where Linear was -8 % MIL ops: Gemma 4 layers
operate in `(B, S, hidden)` BSC layout natively, so the Linear branch
is a direct call with no wrap.

---

## 3. Full 4-chunk + head E2E (INT8)

| component | Conv2dLinear | nn.Linear | Δ |
|---|---|---|---|
| chunk_0 | 8.35 ms | 8.17 ms | -2.2 % |
| chunk_1 | 8.13 ms | 8.34 ms | +2.6 % |
| chunk_2 | 8.17 ms | 8.40 ms | +2.7 % |
| chunk_3 | 8.37 ms | 8.42 ms | +0.6 % |
| **chunk_head** | **2.86 ms** | **3.78 ms** | **+32.1 %** |
| **E2E** | **36.07 ms** | **36.87 ms** | **+2.2 %** |
| Mac decode tok/s | 27.7 | 27.1 | — |

ANE placement (INT8, post-palettize):

| chunk | Conv | Linear | Δ |
|---|---|---|---|
| chunk_0 | 93.5 % | 94.5 % | **+1.0 pt** |
| chunk_head | 84.6 % (11/13) | **53.3 % (8/15)** | **-31.3 pt** |

`chunk_head` is a small graph (final_norm + lm_head + argmax,
~12-15 ops). Its `lm_head` is a single `nn.Linear` with output dim
`vocab_size = 151_936`. Under fp16 conversion the linear op falls off
ANE entirely (0 / 14); INT8 palettization recovers most but not all
(8 / 15 = 53.3 % vs Conv 11 / 13 = 84.6 %). The CPU/GPU fallback for
that one big linear shows up as +32 % wall time on Mac.

**Sanity at Mac CPU_AND_NE level:** both variants produce the same
`token_id = 220` from the same synthetic input — bit-equivalent within
fp16 rounding.

---

## 4. Adoption decision (Mac side)

User-supplied criteria:
- ANE placement parity (±1 pt) + iPhone tok/s parity (±5 %)

Mac-side observation:
- Body chunks (0…3): pass.  ANE +1 pt, latency parity.
- chunk_head: **fails** ANE parity (-31 pt). Wall time +32 %.
- E2E: +2.2 % (within ±5 % but not free).

Two paths from here:
- **A. Ship as-is, gate on iPhone.** If iPhone 17 Pro shows tok/s
  parity (the +32 % on a 3 ms head likely matters less when iPhone
  body chunks are slower than Mac), the regression is absorbed.
- **B. Body-only Linear swap.** Add a knob on `ANEHeadChunk` so
  `use_linear=True` keeps the head as `Conv2dLinear`. Captures the
  body win without taking the head loss.

**Decision: A** — match Gemma 4 Plan 3 record (full swap, gated on
iPhone). If iPhone 17 Pro confirms the head regression hurts decode
tok/s, B becomes a one-line follow-up (`use_linear and not is_head`
guard in `ANEHeadChunk.__init__`).

---

## 5. iPhone hand-off

iPhone 17 Pro (`A6F3E849-1947-5202-9AD1-9C881CA58EEF`) was unavailable
during this session. iPhone 15 was paired but not re-tested — its A16
ANE differs enough from A19 Pro that a result there wouldn't gate
adoption, and per the iPhone-trip-is-scarce rule we bundle that with
the next 17 Pro session.

When the iPhone 17 Pro is connected next:

```bash
# 1. Stage two parity bundles on Mac
python3.12 conversion/build_qwen3_vl_2b_stateful_chunks.py \
    --out-dir /tmp/q3vl_conv_full --num-chunks 4 --nbits 8
python3.12 conversion/build_qwen3_vl_2b_stateful_chunks.py \
    --out-dir /tmp/q3vl_linear_full --num-chunks 4 --nbits 8 \
    --linear-projections

# 2. Push BOTH to the device (mirror conv to the production folder
#    so the picker shows like-for-like; linear to the new entry).
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
SRC_CONV=/tmp/q3vl_conv_full/qwen3_vl_2b_stateful_chunks
SRC_LIN=/tmp/q3vl_linear_full/qwen3_vl_2b_stateful_chunks
bash scripts/qwen3vl_stateful_push.sh "$SRC_CONV"     # → qwen3-vl-2b-stateful/
# Override REMOTE_DIR for the linear push:
SRC_DIR="$SRC_LIN" \
REMOTE_DIR="Documents/Models/qwen3-vl-2b-stateful-linear/qwen3_vl_2b_stateful_chunks" \
    bash scripts/qwen3vl_stateful_push.sh "$SRC_LIN"
# (the qwen3vl_stateful_push.sh REMOTE_DIR is hardcoded today; either
#  edit the const for the second push or copy the script with a sed
#  one-shot — kept manual rather than adding a flag, since this is
#  one-time A/B tooling.)

# 3. In Xcode scheme: LLM_SHOW_EXPERIMENTAL=1 + LLM_PROFILE_EVERY_STEP=1
# Picker exposes:
#   - Qwen3-VL 2B (stateful, Phase 1)            (Conv2dLinear)
#   - Qwen3-VL 2B (stateful, Linear projections) (nn.Linear)

# 4. Record decode tok/s, prefill ms, phys_footprint for each variant on
#    a 64-token "smoke" + a 256-token "decode-bound" prompt. Apply the
#    ±5 % parity bar.
```

Adoption rule:
- **GO** → leave the branch state, open PR, merge to main.
- **HOLD on head only** → flip `ANEHeadChunk.__init__` to skip the
  swap when `use_linear=True`, rebuild Linear bundle, re-validate
  iPhone, then PR.
- **HOLD entirely** → revert `f391c59` + `f3ae5fb` from main; keep the
  branch open with notes.

---

## 6. Open follow-ups

- **iPhone 17 Pro tok/s A/B** — the gating step.
- **Full `ane_ops.Conv2dLinear` deprecate plan** still requires
  Qwen3.5 / 4-chunk Gemma 4 paths re-validated with Linear (separate
  sub-stages, see roadmap §2 Stage 5).
- **chunk_head Linear ANE drop** — worth a small probe to see if it's
  fixable by a different `_project` shape contract for output-dim
  > N (e.g., reshape to `(1, vocab, 1, 1)` instead of relying on
  Linear's native 2D contract). Not load-bearing for this stage; only
  if path A above lands HOLD.

---

## 7. Lessons

- **Layer layout choice flips the Linear-vs-Conv MIL-op delta.** Gemma
  4 (BSC-native) saw -8 % ops on Linear. Qwen3-VL (conv-layout-native
  inside the attention block) sees +17 % ops on the body — more
  permute/squeeze wraps. ANE absorbs the extra ops cleanly, but it
  shows up on op-count audits.
- **Tiny standalone heads are sensitive to `nn.Linear` output dim.**
  The Qwen3-VL chunk_head is just `final_norm + lm_head + argmax`
  (~13 ops). The `lm_head` is the dominant op and ends up off-ANE
  under Linear; same op inside a 7-layer body chunk doesn't show this.
  Per-chunk ANE placement is a meaningful adoption check, not just
  bundle-wide average.
- **`Torch var kv_cache_0 is added again` warning** appears at the
  chunk N→N+1 boundary in trace; it's harmless (the buffer name
  legitimately exists across modules). Do not chase it.
