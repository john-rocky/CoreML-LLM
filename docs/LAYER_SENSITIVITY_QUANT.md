# R16 — Per-Layer Quantization Sensitivity for Gemma 4 E2B

Date: 2026-04-16  •  Branch: `research/conversion-deep-dive`  •  Author: research round 3

Related docs: `EXPERIMENTS.md` (W2/W3 rejection), `CONVERSION_AUDIT_2026_04_15.md`, `GEMMA4_ANE_REWRITES.md`, `ANE_CONVERSION_RECIPE_2026.md`, `GPU_WHY_FAST.md`.

---

## 1. Executive verdict

**Pursue — but as a one-week offline research spike, not a shipping-path change yet.**

Our 2026-04-13 rejection of W2A16 / W3A16 concluded "the post-training cliff is between 3 and 4 bits." That conclusion is true *for uniform* palettization. It is **not true for mixed-precision** palettization: the same Gemma 4 E2B that outputs gibberish at uniform 2-bit almost certainly tolerates 2-bit on a subset of modules (late-layer `down_proj`, per-layer-embeddings, and some `gate_proj`) while the sensitive modules (`embed_tokens`, L0–L2 attention, `post_ffn_norm`, logit softcap constants) stay at 4–6 bits. coremltools has the per-op plumbing (`op_type_configs`, `op_name_configs` on `OptimizationConfig`) to do this without any custom MIL passes, so the risk is calibration-design, not tool-support. The payoff is bounded: an achievable average of ~3.2–3.4 bits means **~800 MB saved off the current 4-bit 4-chunk set** (~3.2 GB today) and a proportional per-step weight-bandwidth reduction of ~17–20 % — worth roughly +3–5 tok/s on top of whatever chunk-consolidation and D-1 wiring deliver. The PLE tensor alone is 2.3 B params and dominates the budget; if PLE survives at 2-bit the total savings are ~560 MB from one tensor.

Defer shipping until (a) the sensitivity sweep is complete, (b) an on-device speed test confirms the mixed-bit `.mlpackage` actually runs on ANE without a bank-conflict regression, and (c) a perplexity gate on a fixed eval set passes within 5 % of the 4-bit baseline. If any of those fail, this path converts to "shelved, numeric reason recorded" in `EXPERIMENTS.md`.

---

## 2. Sensitivity metric selection

Five candidates, ranked by cost / fidelity for our 35-layer case.

| Metric | Cost | Fidelity | Verdict |
|---|---|---|---|
| Weight-magnitude stats per layer (max, std, kurtosis) | free (one pass over state_dict) | weak — misses activation interaction | use as pre-screen only |
| Activation-magnitude stats (mean / max abs per channel, on calibration tokens) | ~10 min on Mac | medium — captures dynamic range | **use as primary input** |
| AWQ-style salience (top-1 % activation channels pinned to higher bits) | ~30 min | good — directly motivates per-channel bit allocation | **use for allocator** |
| Hessian / GPTQ sensitivity | ~8 GPU-hours on a 3090 | best | **defer** — too expensive for the iteration we want |
| Per-module perplexity delta (quantize one module to 2-bit, hold rest at 4-bit, measure ΔPPL on WikiText-2) | ~350 conversions × 1 min fp-eval = ~6 h | highest signal, end-to-end | **use as final arbiter** on a shortlist of 40 modules |

Recommended pipeline:

1. **Pre-screen with activation-magnitude stats.** Captures the intuition behind both AWQ and llama.cpp's imatrix (Σ Act² per column) without any GPU. Run the fp16 reference decoder on ~512 tokens of calibration text (WikiText-2 + a few chat prompts) and record, per projection: mean |activation|, max |activation|, 99.9-th percentile. Columns with outliers > 10× the median are the "sensitive" columns in AWQ's sense.
2. **Rank modules** by a composite score `sensitivity = log(max_abs / median_abs) + 0.25 · log(param_count)` — heavier penalty for high-dynamic-range tensors, mild penalty for small tensors we can afford to leave at 4-bit.
3. **Shortlist 40** candidate modules for the PPL sweep (full 350 is overkill; pick the 10 highest-ranked per tier: embed/norm/attn/mlp).
4. **PPL sweep** — quantize each shortlisted module individually to 2-bit, rest to 4-bit, measure PPL delta on a 2 k-token eval. Tokenizer + chat-template exactly match shipping path.

We re-use the existing `conversion/smoke_w2_quality.py` harness (already does Mac-CPU autoregressive generation) as the evaluation loop; swap the three qualitative prompts for a quantitative PPL loop on a fixed 2 k-token slice.

---

## 3. Gemma 4 E2B sensitivity tiers (a-priori, to be verified)

From architecture facts (memory) + analogy to every published mixed-precision LLM recipe. This is our starting hypothesis; §2 step 4 validates it.

### Tier S — never below 4-bit (often 6)
- `embed_tokens` (and the tied lm-head read). Tie means any quantization error doubles — it enters both the prompt path and the final logits. Apple and the GPTQ literature both pin embeddings at higher bits.
- Logit softcap constants (`tanh(x/30) * 30`). Not weights we quantize, but the `30.0` scalars must stay fp16 — flagged only because some allocators sweep into scalar consts.
- Final `norm.weight` (one vector, 1536 floats). 4-bit of a 1536-vector saves nothing and creates a clamp artifact on every step.
- QK-norm scales (per head, per layer). Small parameter count, high functional impact — keep at 4-bit minimum.

### Tier A — 4-bit floor, PPL-check 3-bit
- L0–L2 attention projections (q / k / v / o). Early layers carry the most input-side information; the GPTQ paper and AWQ both show 4-bit there is the cliff edge.
- L33–L34 MLP `down_proj`. Last layers feed logits; larger quantization error here attacks the output distribution directly.
- Sliding vs global attention in the global layers (`L14` owns `kv_full` read by L15–L34). This single layer's K/V error propagates across 20 downstream layers.

### Tier B — 3-bit target, PPL-check 2-bit
- Mid-stack (L5–L28) `q_proj`, `k_proj`, `v_proj`, `o_proj`. Typical LLM pattern: the middle 70 % of attention blocks tolerate one step lower.
- L5–L28 MLP `gate_proj`. Gating is relatively robust — many recipes quantize `gate` harder than `up`.

### Tier C — 2-bit target
- L15–L34 MLP `up_proj` **and** `down_proj` on the double-wide (2× intermediate) portion. Gemma 4's late-layer MLP is `6144 → 12288 → 1536` per layer, which is where bulk params live. Aggressive compression here is where the size win concentrates. (Hypothesis: empirically the doubled width supplies enough representational redundancy that 2-bit survives — contradicts the uniform-W2 failure because the uniform failure dropped *every* projection, including the sensitive ones.)
- `per_layer_input_gate.weight` and `per_layer_projection.weight` (Conv2d 1×1 per layer, total 35 × 2 × (1536 × 256)). Small individually, cumulative ~53 MB at fp16 — 2-bit buys ~40 MB and the projection is followed by a norm, which absorbs rounding.

### Tier D — 2-bit or lower if it holds
- `embed_tokens_per_layer` (PLE). 262 144 × 35 × 256 = **2.349 B params**, which at uniform fp16 is 4.4 GB and at 4-bit is 1.15 GB. This single tensor dominates on-disk size. Embedding lookups are the quantization-friendliest operation in a transformer — the full value of the entry is consumed directly, no matmul amplification of error. Strong prior that 2-bit (or even 1-bit with per-row scale) survives. **Highest-leverage single experiment in this whole doc.**

---

## 4. PLE-specific deep dive

`embed_tokens_per_layer` is the odd-one-out. Two reasons it deserves a separate analysis from everything else:

**(1) It dominates the weight budget.** At fp16 it's 4.4 GB — larger than the entire Gemma 4 E2B decoder. At current 4-bit it's 1.15 GB. No other single tensor in the model is close. Dropping PLE from 4 → 2 bits saves ~576 MB. Dropping from 4 → 1 bit (with per-row fp16 scale, total ~288 MB of effective storage) saves ~860 MB. Either would, by itself, exceed the size-reduction from every other tier combined.

**(2) Its function is an indexed lookup, not a matmul.** A given token ID selects a single row (35 × 256 = 8960 fp16 values). That row is then added / scaled / projected into the main residual stream via `per_layer_input_gate` and `per_layer_projection` — both of which are *not* quantized in the 4-bit fp16 shipping path. The quantization error on a PLE row is bounded by one codeword quantization step; it enters the residual stream *once*, at one position, with no cross-token accumulation. This is structurally different from quantizing a matmul weight, where a single column's error multiplies against every token's activation.

### PLE sub-experiments (highest-leverage)

- **Experiment PLE-A**: uniform 2-bit palettization of `embed_tokens_per_layer` only, rest of model 4-bit. Expected size Δ: −576 MB. Expected PPL Δ: < 0.1 (hypothesis).
- **Experiment PLE-B**: per-row 2-bit with fp16 per-row scale (coremltools `per_grouped_channel` with `group_size=vocab_size` axis flipped). Higher quality than uniform at same nbits.
- **Experiment PLE-C**: rank-k factorisation of PLE before quantization. PLE is `V × (L × d)` with `V=262144`, `L × d = 8960`. Likely low-rank: most token embeddings are linear combinations of ~512 "eigen-embeddings." Factor as `V × r + r × (L × d)`, then 4-bit the small r-wide factor. At `r=512` storage becomes `262144 × 512 × 0.5 B + 512 × 8960 × 0.5 B = 67 MB + 2.3 MB = 69 MB` — 94 % reduction vs current 4-bit PLE. Quality: depends on low-rank fidelity; needs SVD-reconstruction PPL check before committing.
- **Experiment PLE-D**: product quantization of PLE rows. Split 8960-d row into 70 sub-vectors of 128 dims, k-means to 256 codes each → 70 bytes per row, 17 MB total. Extreme compression. Quality depends on sub-vector clustering fidelity.

The four sub-experiments stack from conservative to aggressive. Start with PLE-A — if PPL holds, we have 576 MB back with ~10 lines of `OpPalettizerConfig` change and a one-line `op_name_configs` entry pointing at `embed_tokens_per_layer`.

---

## 5. IMatrix-style calibration for Core ML

llama.cpp's imatrix records `Σ(activation²)` per column per tensor across a calibration corpus, then feeds those importance scores into the IQ-quant family to allocate bits / group sizes where they reduce total loss most. The methodology is well-tested at sub-4-bit precisions (IQ2, IQ3) that uniform k-means cannot reach.

Core ML has no native imatrix equivalent. coremltools' `OpPalettizerConfig` accepts `nbits`, `granularity`, `group_size`, `cluster_dim`, and a `lut_dtype` — nothing driven by activation statistics. But we can emulate the *effect* cheaply with two ingredients we already have:

### Ingredient 1 — activation trace on the reference decoder
The `conversion/smoke_w2_quality.py` harness already runs the fp16 decoder on Mac CPU. Extend it to register forward hooks on every `nn.Linear` / `nn.Conv2d` that records:

- `sum_sq[c]` — running sum of squared pre-matmul activations on input channel `c`
- `max_abs[c]` — running max of absolute activation on channel `c`
- `n_tokens` — count (for normalization)

Save as a `.npz` indexed by the PyTorch module path (matches naming in `conversion/models/gemma4.py`). ~5 minutes of compute on 512 calibration tokens.

### Ingredient 2 — translate importance into per-op palettization config
For each module:

```
importance_score = (sum_sq / n_tokens) ** 0.5        # RMS activation per input channel
sensitivity = importance_score.max() / importance_score.median()
```

Decision tree:

- If `sensitivity > 5`: outlier channels present. Use smaller `group_size` so scales adapt (e.g., 16 instead of 32) OR upgrade the whole module one bit.
- If `sensitivity < 2`: flat activation profile. Safe to palettize at 2-bit with group_size=64 (lower overhead).
- Otherwise: default 4-bit, group_size=32 (today's recipe).

This is a *reduced* imatrix — we don't get per-weight bit allocation like IQ-quants, but coremltools' per-op granularity is all we can express downstream anyway. The gain is module-level: a sensitivity-aware mapping from module → (nbits, group_size).

### Ingredient 3 — assemble `OptimizationConfig`
coremltools (verified 2026-04-16) accepts:

```python
cfg = OptimizationConfig(
    global_config=OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=32),
    op_type_configs={
        # no type-level overrides by default
    },
    op_name_configs={
        "embed_tokens_per_layer": OpPalettizerConfig(nbits=2, granularity="per_grouped_channel", group_size=64),
        "layers_25_mlp_down_proj": OpPalettizerConfig(nbits=2, granularity="per_grouped_channel", group_size=16),
        # ... 40-entry table from allocator
    },
)
```

Names correspond to MIL op names after conversion — not to PyTorch module names. A brief reconnaissance pass on one chunk is needed to extract the MIL-side naming before the `op_name_configs` dict is built. `mlmodel.get_spec()` plus a walk of the MIL program gives us the mapping.

---

## 6. Implementation plan

Target branch: `feat/per-layer-palette` off `research/conversion-deep-dive`. Do **not** merge until §7 gates pass.

**Phase 1 — sensitivity collection (2 days)**
- `conversion/collect_sensitivity.py`: forward-hook harness, produces `sensitivity.npz`.
- Calibration corpus: 512 tokens from WikiText-2 + 128 tokens from a held-out chat template + 128 from code. Save the exact token IDs in-repo so the corpus is reproducible.
- Output: one table of `(module_name, mean_act, max_act, sensitivity_score, param_count)` for all 350+ quantizable modules.

**Phase 2 — allocator + PPL shortlist (1 day)**
- `conversion/allocate_bits.py`: takes `sensitivity.npz`, emits a candidate `op_name_configs` dict and a shortlist of 40 modules for per-module PPL sweep.
- Greedy allocator: sort by `sensitivity * param_count`, assign bits top-down until budget average (e.g., 3.2 bits) is hit.
- PPL harness: fp16 reference vs one-module-quantized model, 2 k-token eval, cosine-sim on logits as a cheap proxy for PPL if fp16 eval is too slow on Mac CPU.

**Phase 3 — per-op conversion pipeline (2 days)**
- Extend `conversion/exporter.py :: _quantize_model` with a `mode="mixed"` branch that reads the allocator output and builds `OptimizationConfig(op_name_configs=...)`.
- Name-mapping helper: PyTorch `module.fqn` → MIL op name. Expected pattern: MIL `linear` / `conv` ops inherit a sanitised form of the PyTorch FQN. Small reconnaissance notebook to confirm on chunk2.
- Fall-back behaviour: if an `op_name` in the allocator doesn't match any MIL op (e.g., the op was fused into a FusedQKV), the call must hard-error, not silently skip.

**Phase 4 — on-device validation (2 days)**
- Convert one chunk (chunk2, smallest risk) with the mixed config. Size diff vs 4-bit baseline.
- `MLComputePlan.deviceUsage` audit: all `constexpr_lut_to_dense` ops should remain on ANE. If the compiler pushes mixed-bit LUT ops to CPU, this path is dead — the whole gain would be eaten by CPU fallback latency.
- Tok/s measurement on iPhone 17 Pro, compared to the uniform-4-bit chunk2.
- Generation-quality smoke test (same 3 prompts as `smoke_w2_quality.py`), gated — reject cliff if any prompt returns gibberish.

**Phase 5 — full 4-chunk shipping (conditional, 1 day)**
- Only if Phase 4 tok/s is neutral-or-positive and quality passes. Apply the allocator to all 4 chunks, regenerate, publish to HF.

---

## 7. Risk register

**R1 — per-op LUT triggers ANE bank conflict.** The shipping path today has every chunk's `constexpr_lut_to_dense` ops homogeneous in bit width. Mixed widths mean mixed LUT layouts in memory; the ANE compiler might pack them into the same bank and serialize access. Mitigation: Phase 4 tok/s measurement is the gate. If we see a slowdown, the acceptable bit-width set shrinks to `{4}` with only `group_size` varying, which recovers *some* flexibility without mixing nbits.

**R2 — MIL op-name mismatch after optimization passes.** Name-based op config is fragile: a fusion pass that merges q/k/v into a FusedQKV renames the operands, and the allocator's `layers_7_q_proj` key goes stale. Mitigation: run the allocator *after* all fusion passes (i.e., against the final MIL program right before serialisation) and do not keep the mapping cached across builds.

**R3 — calibration corpus overfit.** llama.cpp discussion #5263 warns imatrix overfits to calibration text. Mitigation: mix corpus (wiki + chat + code) and do not include any evaluation prompts in calibration.

**R4 — 2-bit still fails on Tier B modules.** Our prior W2 rejection was *uniform* W2; some Tier B modules may still be genuinely W2-incompatible. Mitigation: the allocator is budget-aware — if Tier B fails, fall back to 3-bit on those modules, re-check average bit budget, ship whatever average we actually achieve. The paper result we want is "heterogeneous beat uniform at same average bits," not "2-bit ships."

**R5 — QAT is the only way to sub-4 bits.** Apple's 3B model uses 2-bit *QAT*. The prior W2 failure suggests our post-training ceiling is above 2 bits on average. Mitigation: the whole premise of this doc is that per-layer variance lets us average ~3.2 bits *without touching QAT* — if the 3.2-bit average fails PPL, the next step is QAT on just the Tier C modules (a narrower QAT scope than Apple's whole-model QAT, since most modules stay at 4-bit PTQ).

**R6 — compute budget.** 350 × 1-min conversion + eval = ~6 hours. Mac Studio M4 Max handles it. No cloud compute needed unless we escalate to full GPTQ Hessian sensitivity (then one 3090-hour per chunk).

---

## 8. Projected savings

Back-of-envelope, assuming the hypothesised tier assignment survives the PPL sweep intact.

### Size

| Component | fp16 | 4-bit uniform (today) | Mixed (hypothesised) |
|---|---|---|---|
| `embed_tokens` (tied) | ~1 GB | ~256 MB | ~256 MB (Tier S kept at 4) |
| `embed_tokens_per_layer` | ~4.4 GB | ~1.15 GB | ~576 MB (2-bit) |
| Attention proj (35 × 4 modules) | ~650 MB | ~163 MB | ~140 MB (mostly 3-bit) |
| MLP proj (35 × 3 modules, L15-L34 double-wide) | ~2.1 GB | ~528 MB | ~340 MB (mix of 2/3-bit) |
| Norms + scalars | ~1 MB | ~1 MB | ~1 MB (kept fp16) |
| Per-layer Conv2d 1×1 | ~53 MB | ~13 MB | ~7 MB (2-bit) |
| **Total** | ~8.2 GB | ~2.12 GB | **~1.32 GB** |

That's **~800 MB saved** vs today's 4-bit, or a 38 % size reduction on top of the existing palettization.

### Speed

Bandwidth-bound decode throughput scales roughly with 1 / weight-bytes-read-per-step. Today: per step we dequantize ~2 GB effective weights (PLE rows + projections the step touches). Mixed at average ~3.2 bits: ~1.6 GB effective. Expected per-step bandwidth Δ: −20 %, giving +4–5 tok/s on a ~22 tok/s baseline (decode, 2K ctx). This is additive with chunk consolidation and D-1 fused-QKV.

Risk-adjusted: expect +2–3 tok/s after bank-conflict losses (R1) and Tier B fallbacks (R4).

---

## 9. What this doc commits to, and what it doesn't

**Commits to**:
- A reproducible sensitivity sweep methodology (§2, §5) that we can run again on Gemma 4 E4B / any future drop-in.
- A validated prior (§3) that names which modules are Tier A/B/C/D ahead of data, so the PPL sweep is a *confirmation* step, not open-ended search.
- An implementation path (§6) that re-uses `ct.optimize.coreml.palettize_weights` unchanged — no custom passes, no fork of coremltools.
- A numeric gate (§7 R1, R4) for shelving this experiment if ANE bank conflict or Tier B failures wipe out the expected gain.

**Does not commit to**:
- A ship decision. Shipping requires Phase 4 data.
- QAT. If post-training mixed-precision at ~3.2-bit average fails, QAT is a separate 1–2 week project with GPU-time requirements that don't fit in this round.
- Changing the 4-chunk split or any other axis. This is orthogonal to chunk consolidation, D-1 fused modules, and the EAGLE-3 drafter — can land in parallel.

---

## Sources

- [coremltools palettization overview (granularities, nbits support)](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)
- [coremltools palettization API (op_type_configs, op_name_configs)](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-api.html)
- [llama.cpp imatrix README (Σ(Act²) formula, Stats fields)](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- [llama.cpp quantize README (IQ-quant family)](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Apple Intelligence Foundation Language Models tech report (arXiv 2507.13575)](https://arxiv.org/abs/2507.13575)
- [Importance matrix calculation PR #4861 (ikawrakow)](https://github.com/ggml-org/llama.cpp/pull/4861)
- [imatrix overfitting discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)
- Internal: `docs/EXPERIMENTS.md` (W2/W3 rejection, 2026-04-13), `conversion/smoke_w2_quality.py`, `conversion/exporter.py :: _quantize_model`, `conversion/models/gemma4.py` (PLE structure).
