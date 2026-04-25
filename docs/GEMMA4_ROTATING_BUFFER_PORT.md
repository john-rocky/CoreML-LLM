# Porting the Bonsai Mask-Based Rotating Buffer to Gemma 4

**Last updated:** 2026-04-24

This document analyzes whether the circular mask-based rotating KV buffer
technique proven on Ternary-Bonsai-1.7B (see `DECODE_STATE_LAYOUTS.md` §3)
is worth porting to Gemma 4 E2B / E4B. Short answer: **yes, for the 7
full-attention layers** — that's where Gemma 4's decode-time bottleneck sits
and where the technique maps cleanly.

## Where Gemma 4 is today

From `conversion/models/gemma4.py:70-77` and `gemma4_swa_chunks.py`:

- **E2B**: 35 layers = 28 sliding + 7 full. `layer_types[i] == "full_attention"`
  for i ∈ {4, 9, 14, 19, 24, 29, 34} (every 5th layer).
- **E4B**: 42 layers = 35 sliding + 7 full. Same 1-in-5 cadence.
- **Sliding layers** (`W=512`, `head_dim=256`): use shift-based update
  `K_new = cat([K_cache[:, :, 1:, :], k], dim=2)`
  (`gemma4_swa_chunks.py:105-108`). Ships on iOS ANE without ANEC -14.
- **Full-attention layers** (`W=ctx`, `head_dim=512`): use mask-based update
  `K_new = K_cache * (1 - update_mask) + k * update_mask`
  with state buffer sized to full context.
- **KV sharing**: layers 15-34 read from L13 (sliding producer) and L14
  (full producer) via explicit I/O tensors, not `ct.StateType`
  (`gemma4_swa_chunks.py:112-129`).

## Where Gemma 4 hurts

From `docs/SPEED_8K.md` and `docs/HANDOFF.md:145`:

- **ctx=2048 → 31.4 tok/s** decode, iPhone 17 Pro ANE
- **ctx=8192 → 14.9 tok/s** — 2.1× regression from 2K to 8K
- Chunk2 (the chunk containing the 7 full-attention layers) measures
  **~2.96 ms / full layer** at ctx=8192 vs **1.5–1.7 ms / sliding layer**.
  Full-attn state read/write is ~2× the cost of a sliding-W=512 state read
  per step.
- chunk2 is where the budget goes. Reducing full-attn per-step cost has
  the largest leverage on ctx=8K tok/s.

## Application vectors

### A. Convert 7 full-attention layers to mask-based rotating SWA (recommended trial)

**What**: Replace the `ctx`-sized state buffer on each full layer with a
W-sized rotating buffer (W configurable, default 1024). Write slot =
`pos % W`, attention over W slots. Same mask-based blend op already used
for the write — just smaller state.

**Expected speedup** (back-of-envelope): full-layer per-step cost scales
with state_length. 8192 → 1024 = 8× reduction in state reads/softmax. At
2.96 ms/layer × 7 layers = 20.7 ms → ~2.6 ms total. chunk2 budget drops by
~18 ms/step. At the 14.9 tok/s / 67 ms baseline, 18 ms saved → ~25 tok/s
(+68%) at ctx=8192. That's the single biggest lever available.

**Quality risk**: full-attention layers retain long-range context. Converting
them to SWA discards tokens older than W. Precedent: `gemma4_swa_wfa.py`
attempted this exact semantic change (with shift-based update, W=2048) and
was **shelved for quality regression** on prompts that need attention beyond
the window (see `docs/EXPERIMENTS.md` "WFA section").

**Two things make this worth trying again**:

1. **Attention sinks** (StreamingLLM-style). Reserve the first 4 slots of
   the W-sized buffer permanently for the first 4 prompt positions. The
   full-attention layers regain a global anchor at the cost of 4 slots
   (~0.4% of W=1024). Mask-based rotating trivially supports this: just
   fix `update_mask` to never write to slots 0-3 once positions 0-3 are
   captured. WFA didn't have this — so its quality regression is not
   directly predictive of our ceiling.
2. **Mask-based vs shift-based is strictly better for ANEC**: even if we
   end up wanting W=2048 (matching WFA) or ctx=4096, mask-based pattern
   works on Qwen3 + Stateful + tied (proven). Shift-based doesn't (ANEC -14).
   So this gives future flexibility Gemma 4 doesn't currently have.

**Code changes** (small, localized):

- `conversion/models/gemma4_swa_chunks.py:99-101` — the existing mask-based
  write for full-attention. Change the state buffer shape from
  `(1, num_kv_heads, ctx, head_dim)` to `(1, num_kv_heads, W, head_dim)`.
- Host-side (Swift `Gemma4Chunk2.swift` or equivalent): update_mask becomes
  `(1, 1, W, 1)` sized, write index = `pos % W`; causal_mask becomes
  `(1, 1, 1, W)` and the valid-slot fill logic handles wraparound.
- (Optional) Sink retention: first 4 slots are reserved. Host sets
  update_mask[0, 0, s, 0] = 0 for s ∈ {0,1,2,3} when `pos >= 4`.
- Reuse `conversion/build_bonsai_17b_decode_chunks.py` helpers
  (`_decode_layer_step`) — the op pattern is identical, just different
  state size. Or copy the pattern into Gemma 4's wrapper directly.

**Risk-mitigation plan**:

1. Gate the behavior behind a `--full-layer-window W` CLI flag (default None =
   keep current ctx-sized full attention).
2. Quality sweep on Gemma 4 E2B at ctx=2K, 4K, 8K with W ∈ {512, 1024, 2048,
   4096}. Use the Qwen3.5-2B acceptance prompts (factual, multilingual,
   code-switch) + long-context retrieval probes (needle in haystack).
3. Ship only if quality regression < measurable threshold AND speedup
   materializes.

### B. Convert 28 sliding layers from shift-based to mask-based

**What**: Replace `gemma4_swa_chunks.py:105-108` shift-based `cat` with
mask-based rotating write. Semantics identical, op pattern different.

**Expected speedup**: zero to marginal. Sliding layers already run at
~1.5-1.7 ms/layer on ctx=8K, which is mostly the GQA + MLP, not the state
write. The shift pattern already ships, so ANEC accepts it for Gemma 4's
specific config.

**Worth doing anyway?** Three reasons it might:

1. **Future-proofing**: when Apple changes ANEC lowering in an iOS update,
   the shift-based path on Gemma 4 could become another ANEC -14 surprise.
   Mask-based has strictly more coverage (Qwen3 + Gemma 4 both).
2. **Consistency**: `gemma4_lite_chunks.py` already uses mask-based for all
   layers (sliding + full) — converging the shipping variant removes a
   divergence.
3. **Measurement first**: convert one sliding layer, trace, measure. If
   mask-based is faster by any margin, switch. If not, leave it.

**Verdict**: low priority. Do it after (A) if (A) ships. Or leave indefinitely
if Gemma 4 ships don't break.

### C. Extend sliding window from 512 to 1024+

**What**: Bump `config.sliding_window` from 512 to 1024 or 2048, keep shift
pattern. Gains quality (sliding layers see more context per step), costs
per-step time (scales linearly with W).

**Expected impact**: at W=1024 the sliding layers' state read doubles,
costing ~1 ms/layer × 28 layers = ~28 ms/step extra. That drops tok/s from
14.9 → ~11 at ctx=8K. Not worth it unless quality measurements show a
meaningful retrieval gain.

**Verdict**: not a rotating-buffer port, orthogonal. Don't conflate.

### D. Apply our full Bonsai pipeline to Qwen3-4B / Qwen3-8B

Out of scope for this doc but worth flagging: the entire mask-based rotating
pipeline works directly on larger Qwen3 variants (4B, 8B) with no
architectural changes. Register the model in `conversion/config.py`, adjust
`--split-at` based on parameter count, and rebuild. Gemma 4 is the harder
port; Qwen3-4B is the easier extension.

## Minimum viable experiment

Only (A) is a meaningful decode-time change. Scope:

1. **New CLI flag on `gemma4_swa_chunks.py`** (or new file
   `build_gemma4_full_rotating.py` if surgery in-place is too invasive):
   `--full-layer-window W`. Default None → keep ctx-sized full attention.
2. **Runtime state buffer for full layers** becomes `(1, nkv_full, W, 512)`
   instead of `(1, nkv_full, ctx, 512)`. (`gemma4_swa_chunks.py` around
   the `SWAChunk2` constructor / state-tensor declarations.)
3. **Optional sink retention**: reserve slots 0-3, never overwrite after
   pos 3.
4. **Swift side**: chunk2's caller builds `update_mask` and `causal_mask`
   sized to W, with `write_slot = pos % W` and wrap-around causal.
5. **Parity harness**: adapt `conversion/bonsai_reference_oracle.py` pattern
   to Gemma 4 — compare first-5-token greedy from HF vs our model. Not for
   quality eval (long-context retrieval is a separate eval), but for
   "did I wire up the cache index right."

Estimated effort: 1–2 days for a prototype, 2–3 days including an iPhone
tok/s measurement and a minimal quality sweep (factual + multilingual +
one needle-in-haystack prompt). If the sweep shows quality cliff, stop and
revisit with sinks or hybrid (e.g. keep 2 of 7 full layers unchanged, apply
rotating to the other 5).

## What NOT to do

- **Do not swap shift→mask on sliding layers without measurement** (vector B).
  Low upside, risk of ANEC surprise.
- **Do not enlarge the sliding window** (vector C) hoping for speed. That
  makes things slower; it's a quality knob, not a speed one.
- **Do not remove the full-attention layers entirely** (i.e. all-sliding
  Gemma 4). Precedent: WFA. Quality shelved. Don't re-run that experiment
  without sinks + a different W.

## References

- `docs/DECODE_STATE_LAYOUTS.md` — the decode-path knowledge base this port
  is based on.
- `docs/TERNARY_BONSAI.md` §SWA — the proven pattern for Qwen3-class.
- `conversion/build_bonsai_17b_decode_chunks.py` — the reference build
  with `--sliding-window W`. The `_decode_layer_step` function there is
  copy-pasteable into Gemma 4's per-layer decode.
- `conversion/models/gemma4_swa_chunks.py:99-108` — current full-attention
  and sliding write patterns, side-by-side.
- `conversion/models/gemma4_swa_wfa.py` — prior full→SWA attempt (shift-based,
  no sinks) that was shelved. Read its header comments first.
- `docs/SPEED_8K.md` — the measurement baseline for chunk2's
  2.96 ms/full-layer figure.
- `docs/EXPERIMENTS.md` WFA section — quality regression narrative.
- `docs/HANDOFF.md:145` — ctx=8K / 14.9 tok/s shipping number.
