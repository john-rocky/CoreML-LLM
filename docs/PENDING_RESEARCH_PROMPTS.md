# Pending Research Prompts — FULL POWER (timed out 2026-04-16)

Each prompt below is self-contained and at original full scope. Paste one per
fresh Claude Code session with model=Opus. They timed out when run 9-parallel
in one session; a fresh session per prompt should clear the timeout.

## Branch & session setup (do this FIRST in every new session)

Repo: `/Users/majimadaisuke/Downloads/CoreML-LLM/`

ALL 5 prompts below must commit onto the same branch:
**`research/conversion-deep-dive`** (pushed to origin).

Before pasting any prompt, run:
```
cd /Users/majimadaisuke/Downloads/CoreML-LLM
git checkout research/conversion-deep-dive
git pull --ff-only
```

After the research doc lands, commit and push:
```
git add docs/<new_doc>.md
git -c commit.gpgsign=false commit -m "docs: <prompt letter> — <short title>"
git push
```

## Integration phase (after all 5 complete)

Open a separate Claude Code session after all 5 docs land. Paste:

> Integrate every research doc on branch `research/conversion-deep-dive` under
> `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/` into a single action-ready
> roadmap. Read all pre-conv + round-2 + round-3 docs (roughly 20+ files).
> Resolve contradictions, deduplicate overlapping findings, rank all
> optimizations by Δtok/s / LOC × risk, and produce one unified document
> `docs/INTEGRATED_ROADMAP.md` covering: (1) immediate wins (≤ 1 day each),
> (2) medium-term work (1-week each), (3) deferred / conditional, (4) rejected
> with primary-source rationale. Also decompose into Issue/PR-sized work units
> the user can execute one at a time. Use Read/Grep — do not re-research. If
> two docs disagree, prefer the newer one (dated 2026-04-16 or later) and
> flag the disagreement.

The integration agent must NOT spawn further research — it's purely a
synthesis pass over existing docs.

Already-existing reference docs (agents should read these for context):
- `docs/CONVERSION_AUDIT_2026_04_15.md` — 19-item conversion audit
- `docs/GEMMA4_ANE_REWRITES.md` — 8 concrete PyTorch rewrites
- `docs/ANE_CONVERSION_RECIPE_2026.md` — Apple ml-ane-transformers + ct 9.0 recipe
- `docs/MIL_PASSES_ADDITIONAL.md` — 73 MIL passes audited
- `docs/ANE_NEW_TECHNIQUES_2026.md` — 2025-2026 techniques
- `docs/MLPACKAGE_STRUCTURE_AUDIT.md` — compiled-graph analysis
- `docs/GPU_WHY_FAST.md` — Metal vs ANE structural analysis
- `docs/IMPLEMENTATION_LOG_2026_04_15.md` — what's shipped / deferred
- `docs/D1_WIRING_PATCH_PLAN.md` — FusedQKV/GateUp wiring plan
- `docs/D3_V_LAYOUT_ANALYSIS.md` — drafter V-layout fix
- `docs/D4_NCHW_FEASIBILITY.md` — NCHW end-to-end feasibility
- `docs/D5_STATEFUL_KV_IOS26.md` — stateful KV stay-rejected
- `docs/APPLE_2026_ROADMAP_WATCH.md` — WWDC 2026 survey

Gemma 4 E2B facts (from prior research):
- hidden_size=1536, intermediate_size=6144, 35 layers
- num_attention_heads=8, num_kv_heads=1
- head_dim=256 sliding / 512 global, 5:1 local:global pattern
- L0-L14 own K/V; L15-L34 KV-share reading kv13 (sliding) / kv14 (global)
- Sandwich norm: 4 RMSNorm per layer (input, post_attn, pre_ffn, post_ffn)
- QK-norm: per-head RMSNorm on Q and K
- RoPE: sliding theta=10000, global theta=1M with partial_rotary_factor=0.25
- Activation: GELU-tanh (NOT SiLU)
- Logit softcap: tanh(x/30) * 30
- Tied word embeddings
- Per-layer embeddings (PLE): 262144 × 35×256 = 2.349B params (!)
- Layer scalar (scalar multiplier per layer, 35 scalars)
- Per-layer input gate + projection + post-norm (3 ops × 35 layers = 105 aux ops)
- Double-wide MLP for L15-L34 (intermediate *= 2)

---

## Prompt A — R9: MIL/ANE op complete catalog

Produce a complete, cited catalog of coremltools 9.0 MIL operations with
ANE/GPU/CPU placement behavior. Goal: identify ANE-native ops the CoreML-LLM
project is NOT currently using that could replace slower equivalents.

Target repo: `/Users/majimadaisuke/Downloads/CoreML-LLM/`. Current decoder uses
Conv2d, ANERMSNorm (cat+layer_norm trick), manual softmax decomposition,
F.layer_norm, elementwise ops. Existing MIL op usage per
`docs/MLPACKAGE_STRUCTURE_AUDIT.md`: conv, matmul, transpose, reshape,
reduce_max/sum, sub, exp, real_div, slice_by_index, gather_along_axis, topk,
cast, add/mul, concat, gelu variants.

Research method:

1. **Clone coremltools** (`git clone https://github.com/apple/coremltools /tmp/ct9` at latest 9.x tag). Read:
   - `coremltools/converters/mil/mil/ops/defs/` — canonical op definitions by category (tensor_operation, activation, elementwise_binary/unary, normalization, linear, conv, pool, reduction, rnn, image_resizing, random, sort, optical_flow, scatter_gather, tensor_transformation, control_flow)
   - For each op def, extract: canonical name, supported dtypes, input/output rank constraints, iOS deployment target minimum, and any docstring hints about backend placement

2. **Identify ANE-placement rules**:
   - Look at `coremltools/converters/mil/backend/mil/passes/` for passes that annotate ops with target placement
   - Look at `coremltools/models/utils.py` for `MLComputePlan` API — what ops does Apple label as "NeuralEngine" vs "GPU" vs "CPU"
   - Read `docs/MLPACKAGE_STRUCTURE_AUDIT.md` and `docs/ANE_CONVERSION_RECIPE_2026.md` for empirically observed placements

3. **Cross-reference with Orion paper** (arXiv 2603.06728) — their op-level measurements identify which ops have dedicated ANE hardware vs are emulated:
   - matmul emulated via Conv1x1 (3x penalty)
   - LayerNorm native ANE kernel
   - Softmax — ?
   - SiLU/GELU — ?
   - Scaled dot product attention — ?
   - Argmax, topk — ?
   - Dynamic slice, gather — ?

4. **Identify ops the project could USE but doesn't**:
   - `softmax` (native MIL) instead of decomposed softmax (currently cat+layer_norm style)
   - `scaled_dot_product_attention` op (iOS17+) — would fuse Q@K→softmax→@V on ANE in one op
   - `layer_norm` with right config to replace the `cat([x,-x])` trick
   - `fused_scaled_dot_product_attention` variant if exists
   - `range_1d` for RoPE position indexing
   - `linspace` for constant tables
   - `gelu` with `approximate_mode='tanh'` (Gemma's GELU-tanh)
   - Any other op that maps to a dedicated ANE kernel we're missing

5. **Identify ops the project uses but shouldn't**:
   - `gather_along_axis` (non-ANE per audit)
   - `scatter_*` (non-ANE)
   - `dynamic_reshape` with non-constant shape (non-ANE)
   - `band_part` / `tril` / `triu` variants if runtime-shaped
   - Any op that falls to CPU silently

6. **iOS 26 / coremltools 9.0 new ops**:
   - Release notes for 9.0 — Int8 I/O, model state read/write
   - Compare op registries between 8.x and 9.0

OUTPUT: `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/MIL_OP_CATALOG.md`

Sections:
1. Executive summary — top 5 "missing wins": ANE-native ops we aren't using that could replace slower current ops
2. Complete op catalog table: name, category, min iOS, ANE-native (Y/N/Unknown), notes
3. Currently-used ops in the Gemma 4 pipeline (from audit)
4. Ops worth adopting with specific call sites
5. Ops to eliminate with replacement strategy
6. iOS 26 / ct9.0 new ops
7. References

2500-4500 words. Use Read/Grep/Bash on coremltools source. Cite file:line for every op.

---

## Prompt B — R11: Algebraic rewrite exploration

Explore algebraic reformulations of Gemma 4 E2B operations that could produce
faster ANE execution while preserving numerical equivalence (or acceptable
error bounds).

Context: `/Users/majimadaisuke/Downloads/CoreML-LLM/`. Current Gemma 4 E2B
decoder has specific operations that might have cheaper mathematical
formulations when mapped to ANE's Conv1x1-biased datapath. The standard CoreML
conversion treats operations as black boxes; algebraic rewrites can reshape
the math before conversion.

Focus areas:

1. **RMSNorm re-derivation**:
   - Current: `ANERMSNorm` uses the trick `LayerNorm([x, -x])[:hidden] * scale` — this needs cat + layer_norm + slice + mul = 4 ops
   - Alternative A: direct `rsqrt(mean(x²) + ε)` — ANE may or may not have rsqrt native. Check.
   - Alternative B: learned approximation — replace RMSNorm with a scale-only multiplicator where the scale is the running statistic from training. Loses adaptivity but saves compute. Quantify error.
   - Alternative C: group_norm (cheaper on some backends) — confirm ANE has group_norm native
   - Alternative D: instance_norm across spatial=1 — equivalent math, maybe different MIL lowering
   - Derive and compare each for fp16 numerical stability

2. **Attention re-derivation**:
   - Current (per `gemma4_swa_chunks.py`): manual softmax with cat+layer_norm pattern, Q@K^T fp16 with effective scale=1.0 to avoid overflow
   - Alternative: decompose `attn = softmax(QK^T/sqrt(d))V` differently:
     - Scale K instead of output: `(Q) @ (K/sqrt(d))^T` — wait, this is what scale=1.0 avoids (overflow)
     - Log-sum-exp form: `attn_logits = QK^T`, `max_l = max(attn_logits)`, `attn = exp(attn_logits - max_l) / sum(exp)` — this is what we already do
     - Online attention (FlashAttention style) — can we tile in MIL? Probably not with current API, but worth confirming
   - **KEY QUESTION**: can we use coremltools' `scaled_dot_product_attention` MIL op (iOS 17+) to fuse the whole attention block into one op? Does it accept Gemma 4's effective scale=1.0? Does it run on ANE or force GPU? Cite primary evidence.
   - **QK-norm absorption**: `q_norm(q_proj(x)) = scale_q * rms_normalize(q_proj(x))`. If q_norm is a per-head RMSNorm on head_dim, can we pre-divide q_proj weights by the expected norm magnitude at initialization? Loses per-token adaptive normalization. Error bound?

3. **Logit softcap re-derivation**:
   - Current: `logits = tanh(raw / 30) * 30`
   - Alternative A: pre-scale lm_head weights by 1/30, then `tanh(scaled) * 30` — one less div
   - Alternative B: approximate `tanh(x)` with clipping: `clip(x, -C, C) * normalization` — much cheaper, but changes the softmax distribution. Quantify KL divergence on a typical logit distribution.
   - Alternative C: the softcap's purpose is to prevent extreme logits; a min/max clip achieves the same gradient-bound effect with 1 op instead of 3

4. **Per-layer embedding handling**:
   - Current architecture: `embed_tokens_per_layer = nn.Embedding(262144, 35*256)` and `per_layer_model_projection = nn.Conv2d(hidden, 35*256, 1)`. The "per-layer input" is a sum/gate of the per-layer embedding lookup and the Conv projection of hidden state.
   - This is ~2.3B params (262144 × 8960). It's an enormous weight.
   - Alternative A: low-rank factor — `PLE ≈ U @ V^T` where `U: 262144 × r`, `V: 8960 × r`. For r=128, total is 262144×128 + 8960×128 ≈ 35M (98% reduction). Retraining required, but quality impact on r?
   - Alternative B: quantize PLE harder (INT2 or INT3) — 80% of the weight storage goes to PLE. Aggressive quantization here gives more headroom than compressing the transformer.
   - Alternative C: PLE-free variant — does Gemma 4 really need per-layer embeddings? What's the ablation?
   - Alternative D: runtime streaming — load only the PLE slices for the current token, keep the rest on disk

5. **RoPE partial rotation optimization**:
   - Global attention uses head_dim=512 but only partial_rotary_factor=0.25 → only first 128 dims rotate, the other 384 dims pass through
   - Current implementation likely computes full 512-dim sin/cos tables — can we compute only 128?
   - The "pass through" 384 dims don't need cos/sin multiplication — can we split the tensor and only apply RoPE to first 128?
   - Saves: ~384 * seq_len elementwise ops per step per attention head

6. **layer_scalar absorption**:
   - Each layer has a scalar multiplier. 35 scalars, 35 multiplies per token
   - These multiply the layer output. Absorb into down_proj (last Conv of FFN)?
   - Or: fold into post_feedforward_layernorm scale?

7. **Sandwich norm redundancy**:
   - post_attention_layernorm: applied to attention output BEFORE residual add
   - post_feedforward_layernorm: applied to MLP output BEFORE residual add
   - These wrap the residual contribution. Pre-norm is applied to the skip path.
   - Claim: given scale is only learnable ~1.0, post_norm may be a no-op for convergence — but this is an ablation, not a rewrite.
   - Better: post_norm scale absorbed into the preceding Conv (o_proj, down_proj). This is like A6 from prior work but on different norms.

8. **Validation**:
   - For each rewrite, quantify: cosine similarity vs reference on a held-out activation, perplexity delta on WikiText-2, top-1 token match on 256 decode steps
   - Which rewrites are TRULY lossless vs approximations with bounded error?

OUTPUT: `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/GEMMA4_ALGEBRAIC_REWRITES.md`

Target 3500-5500 words. Each rewrite gets: math derivation, op count before/after, lossless-or-approximate label, error bound estimate, implementation sketch (PyTorch), validation plan, recommended apply order.

---

## Prompt C — R12: Cross-chunk + boundary optimization

Audit and optimize cross-chunk redundancy and chunk boundary choices in the
Gemma 4 E2B CoreML pipeline at `/Users/majimadaisuke/Downloads/CoreML-LLM/`.

Current chunking (per `conversion/models/gemma4_swa_chunks.py`):
- chunk1: L0-7 (8 layers)
- chunk2: L8-14 (7 layers, owns KV for kv13/kv14)
- chunk3: L15-24 (10 layers, KV-shared)
- chunk4: L25-34 (10 layers, KV-shared) + norm + lm_head + argmax

There's also merged variants (merged_chunk1, merged_chunk12, merged_chunk34,
merged_full) per `gemma4_swa_merged*.py`. Total 4+ chunk layouts coexist. Goal:
reduce inter-chunk data traffic and find boundary choices that fit ANE's 32MB
SRAM budget optimally.

Research tasks:

1. **Cross-chunk redundancy audit**:
   - List every input to every chunk (from build_merged_chunks.py and build_verify_chunks.py input specs)
   - Identify which inputs are SHARED across chunks (identical bytes passed to multiple chunks each token):
     - RoPE sin/cos tables (cos_s, sin_s, cos_f, sin_f) — likely passed to every chunk
     - Causal masks (causal_mask_full, causal_mask_sliding, update_mask) — likely passed to every chunk
     - per_layer_combined / per_layer_raw — passed as input
   - For each shared input: shape × dtype = bytes per token. Quantify the cross-chunk transfer volume per decode step.

2. **Constant folding opportunity**:
   - Any input that's constant across all tokens for a given context length: bake as model constant, remove from inputs
   - RoPE cos/sin tables for context_length=8192 are ~8192*128*2 bytes = 2MB — bakable
   - Causal masks are purely functions of position, could be indexed from a constant table
   - Trade-off: constant bytes live in the mlpackage weights (gets quantized? palettized?)
   - If baked: one less input per chunk × 4 chunks × per-decode I/O setup cost

3. **Per-layer embedding chunk placement**:
   - PLE (per_layer_combined) is currently computed up-front and passed as input to all chunks
   - Alternative: split PLE into per-chunk slices, each chunk only receives its own slice
   - Current: `(1, 1, nlayers * pld) = (1, 1, 35*256) = (1, 1, 8960)` per token, 17.5 KB per chunk × 4 chunks
   - Alternative: chunk1 gets slices for L0-7 (2048 bytes), chunk2 gets L8-14 (1792), etc.

4. **Hidden state inter-chunk size**:
   - chunk output -> chunk input: `hidden_states` of shape `(1, 1, hidden) = (1, 1, 1536)` = 3 KB per inter-chunk boundary
   - With 4 chunks, 3 boundaries = 9 KB per decode step
   - Negligible vs KV bandwidth but still — any compression? Probably not worth it.

5. **KV cache cross-chunk pattern**:
   - chunk2 emits K_sliding_out, V_sliding_out, K_full_out, V_full_out, kv13_k, kv13_v, kv14_k, kv14_v
   - chunks 3/4 consume kv13_k/v and kv14_k/v (read-only)
   - The kv13/kv14 emit path: chunk2 writes, chunks 3/4 read — is the same Swift buffer reused zero-copy?
   - Check `Sources/CoreMLLLM/ChunkedEngine.swift` for the buffer lifecycle — are any copies unnecessary?

6. **ANE SRAM budget per chunk**:
   - Orion: 32 MB SRAM, ~30% throughput drop when working set exceeds
   - Per-chunk working set = weights (in palettized form, fp16 resident while compute) + activations + KV subsets
   - Gemma 4 E2B weights: 1.5GB fp16, palettized INT4 = ~380 MB. Per chunk = ~95 MB. Way more than 32 MB → SRAM is a streaming cache, not resident.
   - The relevant SRAM fit is per-layer working set: input act + Q/K/V temp + attn matrix + output act = ~1.5-3 MB per layer. Fits easily.
   - SO: SRAM is NOT the binding constraint for chunk sizing. Dispatch count is.

7. **Optimal chunk count reconsideration**:
   - Given SRAM isn't binding, the only reason to chunk is ANE per-compilation size limits and dispatch amortization
   - 1-chunk merged_full exists (`MergedChunk1`) but is experimental ("35 layers may exceed the ANE per-function stability ceiling")
   - 2-chunk merged (MergedChunk12+34) is safer
   - Question: what's the ACTUAL ANE ceiling? Per audit, 35 layers is near the ceiling. Is it hit on iPhone 17 Pro / iOS 26? Test or reason from Orion's numbers.

8. **Boundary layer choice**:
   - Current split has L14/L15 as the KV-share boundary, which is a natural line
   - Alternative: split on a dispatch-cost-optimal boundary. E.g., chunks 3&4 (both KV-shared) have less total work than chunks 1&2 (own KV). Shift boundary to balance wall-clock.
   - Compute wall-clock per layer from prior bench data (`docs/BASELINE_SPEED_AUDIT.md` if available)

9. **New chunk design proposal**:
   - 2-chunk (merged): chunks 1 (L0-14 own KV) + chunk 2 (L15-34 shared KV + lm_head). Already exists.
   - 3-chunk asymmetric: chunk1 (L0-14 own KV), chunk2 (L15-24 shared KV), chunk3 (L25-34 shared KV + lm_head). Might better balance compute.
   - Ask: what does the current 2-chunk variant cost vs 4-chunk? Is there existing measurement?

10. **Specific recommendations**:
    - RoPE/mask baking as constants: +X tok/s estimate
    - PLE chunk-splitting: +Y tok/s estimate
    - Optimal chunk boundary: +Z tok/s estimate

OUTPUT: `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/CROSS_CHUNK_OPTIMIZATION.md`

Target 2500-4000 words. Use Read/Grep on build_*.py and chunked model files. Quantify every claim with bytes/ms/ops where possible.

---

## Prompt D — R13: Gemma 4 structured LayerSkip

Design a Gemma 4 E2B specific LayerSkip / early-exit speculative decoding
scheme that exploits the KV-share structure and sandwich-norm architecture.

Context: `/Users/majimadaisuke/Downloads/CoreML-LLM/`. Gemma 4 E2B has a
natural early-exit boundary at L14/L15 because:
- L0-L14 compute their own Q, K, V
- L15-L34 are KV-shared — they have Q projection but no K/V projection; they read kv13 (sliding) and kv14 (global) from L13/L14
- The post-L14 stack is architecturally "a prediction head over frozen KV"

Existing doc `docs/FUNDAMENTAL_UNTRIED.md §4` sketches LayerSkip for Gemma 4 at
1.4-1.6x speculative speedup but was never implemented. Deepen it into an
implementation-ready design.

Research tasks:

1. **Natural early-exit candidates**:
   - **L14 boundary**: exit after chunk2, use the hidden state as a draft prediction via a trained "early head" on top of L14. Verifier: chunks 3+4 run on accepted drafts.
   - **L7 boundary**: within chunk1. Much shallower (20% of layers). Likely low accept rate but very fast draft.
   - **L24 boundary**: after chunk3, deeper draft, higher accept rate expected. Less speedup.
   - **Dynamic multi-exit**: draft at multiple depths, use confidence to choose

2. **Draft head design**:
   - The early-exit point has hidden_size=1536 states
   - Need to produce logits over vocab=262144
   - Options:
     - Reuse the full lm_head (tied embedding) — but it's designed to work with norm+softcap
     - A fresh small classifier: `Linear(1536, 262144)` — too big (~400MB) unless we prune vocab
     - Shared-vocab early head: predict only a frequent subset (e.g., top 16K tokens) then fall back
     - Pre-norm + lm_head + softcap applied at early layer — arithmetically valid? Test.

3. **Training the early head**:
   - Supervised training from the model's own outputs on a corpus
   - Collect (hidden_state_at_L14, token_at_L_full) pairs from a rollout
   - Train just the early head (rest frozen) — hours on a single GPU
   - Use the existing EAGLE-3 training infrastructure if compatible (`conversion/train_eagle3_draft.ipynb`)

4. **On-device pipeline**:
   - Decode step: run chunk1 + chunk2 to get L14 hidden state
   - Early head: produce K-token draft (just top-1 or top-K)
   - Verify: run chunks 3+4 on the K draft tokens using Q=K speculative verification (already exists in project as `verify_qK`)
   - Accept/reject per standard rejection sampling

5. **Expected acceptance rates**:
   - Literature on LayerSkip generically: 30-60% accept on standard models
   - Gemma 4 specifically: unknown, depends on KV-share structure helping or hurting
   - Best case: 50% → 1.4x speedup; worst case: 20% → ≤1x regression

6. **Integration with existing spec-decode infrastructure**:
   - Existing verify_qK in `build_verify_chunks.py` — reuse?
   - How does this interact with MTP drafter (existing)? Mutually exclusive, or both run as a union with accept-the-longer?
   - SuffixDecoding (existing, demoted) — union with early-exit?
   - Prompt Lookup Decoding — existing in PR #36 per CONVERSION_AUDIT

7. **Additional Gemma 4 exploits**:
   - **Double-wide MLP** (L15-L34 have 2x intermediate). These 20 layers do MORE FFN work than L0-L14. Early exit skips a big chunk of compute.
   - **Sandwich norm boundary**: post_feedforward_layernorm of L14 could be the natural "clean" output state for exit
   - **layer_scalar** on each layer: what's the learned distribution? If layer_scalar is near-1 for most layers but varies significantly across the stack, we might find "skippable" layers mid-stack

8. **Alternative: structured layer dropout**:
   - Skip random subset of layers at inference, verify
   - Different accept pattern from early exit

9. **Alternative: layer caching for repeat tokens**:
   - If the same token is decoded repeatedly (punctuation, spaces), cache the hidden states through the network and short-circuit
   - Cache hit rate estimation

10. **Concrete implementation plan**:
    - Week 1: training data collection (rollout target, extract L14 hidden states, log target token)
    - Week 2: train early head (1 epoch on ShareGPT or similar)
    - Week 3: CoreML conversion of early head
    - Week 4: Swift integration in SpeculativeLoop (extend existing spec-decode harness)
    - Week 5: measurement, tuning exit depth
    - Week 6: composition with MTP drafter if parallel accept makes sense

OUTPUT: `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/GEMMA4_LAYERSKIP_DESIGN.md`

Sections: motivation, architecture, training plan, on-device pipeline, expected speedup with honest confidence intervals, integration plan, risks, 6-week roadmap.

Target 2500-4500 words. Research: read `docs/FUNDAMENTAL_UNTRIED.md`, `docs/ANE_OPTIMIZATION_SURVEY.md`, existing verify_qK path, existing MTP infrastructure, EAGLE-3 infrastructure. Don't reinvent what's in `docs/EAGLE3_INTEGRATION_STATE.md`.

---

## Prompt E — R16: Per-layer quantization sensitivity

Design a per-layer (and per-module) quantization sensitivity analysis for
Gemma 4 E2B that could enable heterogeneous bit-width palettization — lower
bits on robust layers, higher bits on sensitive ones — without the W2/W3
gibberish cliff experienced in post-training uniform palettization.

Context: `/Users/majimadaisuke/Downloads/CoreML-LLM/`. Prior attempts
documented in `docs/EXPERIMENTS.md` and memory: W2A16 and W3A16 post-training
palettization produced gibberish (even though W4 works). The assumption was
"the cliff is between 3-bit and 4-bit uniformly." But Apple's Foundation
Models 2025 paper and Llama.cpp IQ-quant work show the cliff is heterogeneous
— some layers tolerate much lower bits than others.

Research tasks:

1. **Identify per-layer sensitivity metrics**:
   - Weight magnitude distribution (mean, max, std) per layer — outliers indicate sensitivity
   - Activation magnitude statistics per layer
   - Hessian-based sensitivity (GPTQ-style) — expensive but accurate
   - AWQ-style salience: which weights correspond to large activation channels?
   - Simple heuristic: run the model with each layer quantized to 2/3 bits individually, measure perplexity delta
   - llama.cpp imatrix (importance matrix) methodology — can we adapt?

2. **Gemma 4 E2B architecture tiers (candidate sensitivity ordering)**:
   - **Highest sensitivity (keep 4-6 bits)**: embed_tokens (and tied lm_head), early attention layers (L0-L2), final norm scale, logit softcap
   - **Medium (4 bits)**: attention projections overall (q, k, v, o)
   - **Lower (2-3 bits)**: middle FFN layers (gate, up, down) — typical LLM pattern
   - **Weight-size-dominant**: per-layer embeddings (2.3B params!) — if we can palettize these at 2-bit acceptably, huge win

3. **Tool-side capabilities**:
   - coremltools 8.1+: `OpPalettizerConfig` with `per_op` config — can specify per-op bit-widths
   - coremltools 9.0: `per_block` granularity for finer-grained scales (smaller blocks = better quality at same bits)
   - MLX group-wise quantization recipes
   - QAT via Apple's `ct.optimize.torch` for true recovery from 2-bit — but requires days of GPU training

4. **Practical sensitivity sweep design**:
   - Baseline: fully-fp16 reference decoder
   - For each of N candidate modules (35 layers × ~10 projection types = ~350 modules): quantize that module to 2 bits, keep rest at 4, measure PPL delta on WikiText-2 or similar
   - Rank modules by PPL sensitivity
   - Assign bits budget: e.g., 50% of modules at 4-bit, 30% at 3-bit, 20% at 2-bit → average ~3.3 bits, better than uniform 4-bit on size
   - Compare to uniform 4-bit baseline

5. **Per-layer embedding-specific (unique to Gemma 4)**:
   - `embed_tokens_per_layer` is 2.3B params. A single bit saved here = 285 MB saved.
   - Uniform 4-bit PLE: ~1.15 GB
   - 2-bit PLE: ~570 MB
   - Is PLE quality-robust to 2-bit? Likely yes — embedding lookups are less sensitive than computation weights. But test.
   - Alternative: factor PLE as low-rank matrix + residual at 4-bit, reducing total size below uniform 2-bit

6. **IMatrix workflow adaptation**:
   - llama.cpp's `imatrix` computes per-weight importance from activation traces on a calibration set
   - coremltools doesn't have imatrix natively but we could implement:
     - Run fp16 model on calibration tokens, log activation magnitudes per weight column
     - Use log as a weight-importance mask
     - Set `OpPalettizerConfig`'s `granularity` and `group_size` based on importance — sensitive columns get smaller groups (finer scales)

7. **Implementation plan**:
   - Phase 1: sensitivity sweep script (offline, Mac or Colab) — 1 week
   - Phase 2: optimal bit assignment via greedy allocator — 2 days
   - Phase 3: conversion with per-op config — 3 days
   - Phase 4: on-device tok/s + perplexity validation — 2 days

8. **Risk**:
   - Per-op config might fail coremltools conversion (untested at this scale)
   - Block-sparse bits create memory bank conflicts on ANE — may hurt speed even if smaller
   - Sensitivity sweep is 350+ conversions × PPL eval — expensive compute

OUTPUT: `/Users/majimadaisuke/Downloads/CoreML-LLM/docs/LAYER_SENSITIVITY_QUANT.md`

Sections:
1. Executive verdict: pursue / defer / reject, one paragraph
2. Sensitivity metric selection
3. Per-layer candidate tiers for Gemma 4 (with rationale)
4. PLE-specific deep dive (highest weight share)
5. IMatrix-like calibration approach
6. Implementation plan with PyTorch + coremltools sketch
7. Projected savings (size + tok/s via fewer dequantize cycles)

Target 2500-4500 words. Use Read on existing conversion config, WebFetch on Apple Foundation Models paper (arXiv 2507.13575) and llama.cpp imatrix docs.

---

## Usage

Open a new Claude Code session (Opus model), paste ONE prompt verbatim, let it
run. After completion, the doc lands under `docs/`. When all 5 complete, come
back to the main conversation for integration.
