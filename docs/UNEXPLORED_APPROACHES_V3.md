# Unexplored Approaches — Round 4 (exhaustive sweep)

Findings from a 6-way parallel search (2026-04-13) covering: cutting-edge
speculative decoding (2025-2026), Chinese/East Asian mobile NPU research,
Apple patents + WWDC, cross-domain ANE tricks (vision/audio/diffusion),
exotic lossless decode methods, and novel KV/attention compression.

Filtered against everything in `SPEED_8K.md`, `ALTERNATIVE_APPROACHES.md`,
`UNEXPLORED_APPROACHES.md`, `UNEXPLORED_APPROACHES_V2.md`, and
`FUNDAMENTAL_UNTRIED.md`. Only genuinely new-to-this-repo items below.

---

## A. Speculative decoding — training-free, lossless methods not yet listed

### A1. ESP — Embedding-Space Probing (March 2026)

arXiv 2603.17942. Training-free, lossless, **no draft model**. Inserts
mask tokens from the LLM's own embedding space to probe future-token
predictions in parallel. Builds a speculative tree from the probes,
verifies via standard tree attention. The static tree structure (top-K
pruning with fixed fan-out) is ANE-compilable via EnumeratedShapes.
Requires zero extra parameters.

- ANE: static-shape tree verify, mask-token insertion is fixed-width.
- Expected: workload-dependent. No published small-model numbers yet.
- Cost: ~3 days. Needs Q=K verifier (G1).

### A2. SSSD — Simply-Scalable Speculative Decoding (Nov 2024, revised Jan 2026)

arXiv 2411.05894. Training-free, lossless. Builds a trie-based n-gram
cache on-the-fly from generated tokens + prompt. CPU-side lookup (pure
dictionary, ~negligible latency), fixed-width tree verification on
target. Reported **up to 2.9× over AR decoding**. Outperforms
training-based methods under domain shift.

- ANE: CPU trie + static-shape ANE tree verify. <2 MB memory.
- Expected: ~1.5–2× on small models (2-4B). Zero-risk.
- Cost: ~2 days Swift.
- Relationship to SuffixDecoding (FUNDAMENTAL_UNTRIED §1): SSSD uses a
  trie cache; SuffixDecoding uses a suffix tree. Different data
  structures for the same goal (n-gram draft from history). Composable
  — union their candidates into one K-wide slate.

### A3. Token Recycling (Aug 2024, revised May 2025)

arXiv 2408.08696. Training-free, lossless. Stores previously-seen
candidate tokens in a lightweight **adjacency matrix** (<2 MB), retrieves
draft trees via BFS. **~2× speedup across all model sizes**, 31% better
than prior training-free methods. CPU-side adjacency lookup + ANE
static-shape tree verify.

- ANE: fixed-budget candidate set, no dynamic shapes.
- Expected: 1.3–1.6× on 2-4B.
- Cost: ~2 days. Composes with EAGLE-3 (recycled tokens as fallback
  when draft has low confidence).

### A4. Sequoia — hardware-aware optimal tree topology (NeurIPS 2024)

arXiv 2402.12374. Not a drafting method itself — a **tree shape
optimizer** that uses dynamic programming to find the optimal
verification tree topology given: (a) a draft model's acceptance rates,
(b) a hardware latency budget. Up to **+33% more accepted tokens per
step** vs flat speculation.

- ANE: tree shape is computed offline and is static at inference time.
- Expected: +20–33% on top of any speculative method (EAGLE-3,
  SuffixDecoding, SSSD, etc.).
- Cost: ~1 day. Offline DP, pure algorithmic. Apply to the verify pass
  of whichever speculative method ships first.

### A5. Traversal Verification (May 2025)

arXiv 2505.12398. Lossless (proven identical to target distribution). A
**drop-in replacement for top-down tree verification** that uses
leaf-to-root traversal, preserving subsequences that standard
verification discards. Increases accepted tokens per step with zero
model change. Pure CPU-side post-processing of the verify output.

- ANE: does not touch the forward pass.
- Expected: +10–20% accepted tokens on top of any tree-based spec.
- Cost: ~0.5 day. Pure Swift logic change in verify loop.

### A6. Staged Speculative Decoding (Spector & Re, 2023)

arXiv 2308.04623. Training-free, lossless. Nested cascade: cheapest
model proposes K tokens → medium model verifies/extends → full target
verifies final set. **Maps directly to Gemma 4's chunk boundary**:
chunk1-2 (L0–14) as stage 1, chunk1-4 as stage 2. Zero extra model
parameters.

- ANE: each stage has a fixed draft length, static shapes.
- Expected: 1.3–1.8× on 2-4B.
- Cost: ~2 days. Chunk1-2 already exist; just needs a small projection
  head on L14's output (hours of training on existing EAGLE-3 corpus).
- Relationship to FUNDAMENTAL_UNTRIED §4 (LayerSkip): Staged Spec is
  the formal framework; LayerSkip-on-KV-share is the Gemma-4-specific
  instantiation. This paper provides the verification algorithm.

---

## B. ANE micro-optimizations from vision/audio/diffusion domains

### B1. exp2-based softmax (from Apple ml-stable-diffusion)

ANE has a **native EXP2 instruction** but not EXP. Apple's own
`ml-stable-diffusion/attention.py` converts `exp(x)` to
`exp2(x * (1/log(2)))`. Our `ane_softmax` in `conversion/ane_ops.py`
uses `torch.exp` — if CoreML lowers this to the non-native `exp` op,
switching to `torch.exp2` with the `1/ln(2)` scale factor could yield
a free speedup on the ANE side.

- Cost: 2 LoC change in `ane_ops.py`. Reconvert.
- Risk: zero (mathematically identical).
- Expected: small (0–5%) but free.

### B2. MLP tile reshape (B,C,8,8)

Vision community insight: MLP/FFN layers run **~50% faster** with a
`(B,C,8,8)` tile layout instead of `(B,C,1,S)` on ANE. The trick:
reshape to `(B,C,8,8)` before FFN Conv2d ops, reshape back to
`(B,C,1,64)` for attention. Untried in any LLM-on-ANE project we found.

- Cost: ~20 LoC in `gemma4_swa_chunks.py` MLP section. Reconvert.
- Risk: numerical equivalence needs verify (just reshapes, should be
  exact).
- Expected: potentially significant on MLP-bound layers (which are
  ~60% of per-layer compute). Needs measurement.

### B3. Split-einsum V2 per-head attention (from Apple ml-stable-diffusion)

Apple's V2 attention splits multi-head attention into **independent
single-head operations** and further **chunks the query sequence into
512-token blocks** to keep attention weight matrices in L2 residency.
Our LLM attention currently uses a batched multi-head tensor.

- Applicable to: prefill (where Q-length is large). Not useful for
  Q=1 decode.
- Cost: ~1 day converter change for prefill chunks only.
- Expected: better ANE utilization for prefill → faster TTFT.

### B4. ANE pipeline prewarming (from WhisperKit)

WhisperKit's decoder runs **4 dummy predictions × 16 steps** before
the first real generation. Without this, the first call is **~7× slower**
due to lazy ANE pipeline setup. Zero-cost trick overlapped with other
init.

- Cost: 10 LoC in `ChunkedEngine.load()`.
- Risk: zero (dummy predictions discarded).
- Expected: eliminates first-token latency spike. No throughput gain.

### B5. WhisperKit MLState → 45% latency reduction (evidence for FUNDAMENTAL_UNTRIED §2)

WhisperKit's paper (arXiv 2507.10860) reports that adopting CoreML
`MLState` for the Whisper decoder's KV cache reduced per-step latency
from **8.4ms to 4.6ms (45% reduction)**. This is the first published
measurement of MLState's effect on a real autoregressive decoder on
ANE. Strengthens the case for FUNDAMENTAL_UNTRIED §2.

---

## C. Quantization & weight optimization

### C1. Mixed-bit palettization (from Apple ml-stable-diffusion)

Apple's MBP assigns **per-layer bit-widths from {1,2,4,6,8}** based on
PSNR sensitivity analysis. ANE hardware natively accelerates all these
bit-widths. Sensitive layers (attention output projections) stay at
6-8 bit; insensitive FFN layers drop to 2-4 bit. Average ~4 bit with
minimal quality loss. See `apple/coreml-stable-diffusion-mixed-bit-palettization`
on HuggingFace.

- Extends FUNDAMENTAL_UNTRIED §3 (W2A16): instead of uniform 2-bit,
  use a per-layer sensitivity sweep to find the optimal bit allocation.
- Cost: ~1-2 days. `coremltools.optimize.coreml` supports per-layer
  config.
- Expected: better quality/size tradeoff than uniform W2 or W4.

---

## D. Attention & KV optimization

### D1. SparQ Attention (Dec 2023)

arXiv 2312.04985. Training-free, near-lossless. Uses the query vector
to approximate which K entries will have high attention scores, then
fetches only **top-r keys** for the full dot product. The sparse access
pattern can be mapped to a **fixed top-r gather** operation — ANE-
friendly if r is fixed at compile time.

- ANE: fixed-r gather + reduced matmul. Static shapes.
- Expected: meaningful for full-attn layers at 8K where K=8192. Reads
  only r=512 of 8192 entries → ~16× bandwidth saving on Q@K.
- Cost: ~2 days. Needs converter change + reconvert.
- Risk: near-lossless (~0.1% quality drop per paper). Not bit-identical.

### D2. LUT-based softmax (from llm.npu, ASPLOS 2025)

arXiv 2407.05858. Replace `exp(x)` in softmax with a **lookup table**.
On mobile NPU, this gives **up to 19× speedup** for the softmax op
specifically. The LUT is a fixed 256-entry table covering the FP16
input range, with linear interpolation.

- Extends B1 (exp2 softmax): LUT is more aggressive. May trade
  precision for speed.
- ANE: the lookup is a gather from a constant table — ANE-native.
- Cost: ~1 day. Small converter change.
- Risk: precision depends on table resolution. Needs quality gate.

### D3. TransMLA — post-training MLA retrofit (NeurIPS 2025 Spotlight)

arXiv 2502.07864 / github.com/MuLabPKU/TransMLA. Retrofits DeepSeek's
MLA (Multi-head Latent Attention) onto any GQA model post-training.
Compresses KV cache by **68–93%** with minimal accuracy loss. At 93%
compression on 8K context: **10.6× inference speedup** reported.

- Already mentioned in SPEED_8K.md as "MHA→MLA retrofit (Tier C,
  requires fine-tune)" — but TransMLA is a specific, concrete
  implementation that makes the path much clearer. The paper provides
  the actual conversion recipe.
- Cost: ~2-3 days conversion + QLoRA recovery.
- This would reduce 8K KV from ~48 MB to ~3-7 MB, making KV cache
  bandwidth almost free.

---

## E. Compound strategies

### E1. Speculative stack: SuffixDecoding → SSSD → Token Recycling → EAGLE-3

All four produce candidate continuations consumed by one shared Q=K
verifier. At each step, **union** their candidates into a single K-wide
slate (first-match-wins ranking). This is how vLLM's "ngram" speculator
operates (merges prompt-lookup with streaming n-gram). Coverage is
additive, latency is not multiplicative.

Apply **Sequoia** (A4) to optimize the tree shape for the combined
candidate pool, and **Traversal Verification** (A5) to extract more
accepted tokens from each verification pass. These two are pure
algorithmic improvements that compound with any draft source.

### E2. ANE micro-opt stack: exp2 + MLP tile + prewarming

All three are independent, zero-risk, zero-training changes that can
ship in a single reconversion pass:
- exp2 softmax: 2 LoC
- MLP (B,C,8,8) tile: 20 LoC
- Prewarming: 10 LoC Swift (no reconversion)

Combined expected: 5–15% free throughput + eliminated first-token spike.

---

## Summary table

| # | Technique | Type | Training | Lossless | ANE static | Expected | Source |
|---|---|---|---|---|---|---|---|
| A1 | ESP | speculative | none | yes | yes | unknown (2026 paper) | 2603.17942 |
| A2 | SSSD | speculative | none | yes | yes | 1.5–2.9× | 2411.05894 |
| A3 | Token Recycling | speculative | none | yes | yes | 1.3–1.6× | 2408.08696 |
| A4 | Sequoia | tree optimizer | none | yes | yes | +20–33% on any spec | 2402.12374 |
| A5 | Traversal Verify | verify algo | none | yes | n/a (CPU) | +10–20% accepted | 2505.12398 |
| A6 | Staged Spec | speculative | hours | yes | yes | 1.3–1.8× | 2308.04623 |
| B1 | exp2 softmax | ANE micro | none | exact | yes | 0–5% | ml-stable-diffusion |
| B2 | MLP tile (8,8) | ANE micro | none | exact | yes | up to 50% on FFN | vision community |
| B3 | Split-einsum V2 | ANE micro | none | exact | yes | prefill speedup | ml-stable-diffusion |
| B4 | Prewarming | ANE micro | none | exact | n/a | first-token fix | WhisperKit |
| B5 | MLState evidence | evidence | — | — | — | 45% latency drop | 2507.10860 |
| C1 | Mixed-bit palette | quantization | none | near | yes | better q/size | ml-stable-diffusion |
| D1 | SparQ Attention | attention | none | near | yes | 8K full-attn win | 2312.04985 |
| D2 | LUT softmax | attention | none | near | yes | up to 19× on op | 2407.05858 |
| D3 | TransMLA | KV compress | QLoRA | near | yes | 10.6× at 93% KV | 2502.07864 |

---

## Recommended sequencing (additive to FUNDAMENTAL_UNTRIED.md)

1. **Immediate (zero-risk, no reconversion)**: B4 prewarming (10 LoC Swift)
2. **Next reconversion pass**: B1 exp2 softmax + B2 MLP tile reshape (compound)
3. **With Q=K verifier**: A4 Sequoia + A5 Traversal Verify (pure algorithmic, compounds with any spec method)
4. **First speculative deployment**: A2 SSSD or A3 Token Recycling (training-free, ~2 days each, stack with SuffixDecoding from FUNDAMENTAL_UNTRIED)
5. **After EAGLE-3**: apply Sequoia tree optimization + Traversal Verify to EAGLE-3's tree
6. **Quality-tolerant path**: D1 SparQ Attention for 8K full-attn layers, C1 mixed-bit palettization

---

## References

- [ESP: Embedding-Space Probing (arXiv 2603.17942)](https://arxiv.org/abs/2603.17942)
- [SSSD (arXiv 2411.05894)](https://arxiv.org/abs/2411.05894)
- [Token Recycling (arXiv 2408.08696)](https://arxiv.org/abs/2408.08696)
- [Sequoia (arXiv 2402.12374)](https://arxiv.org/abs/2402.12374)
- [Traversal Verification (arXiv 2505.12398)](https://arxiv.org/abs/2505.12398)
- [Staged Speculative Decoding (arXiv 2308.04623)](https://arxiv.org/abs/2308.04623)
- [Apple ml-stable-diffusion (SPLIT_EINSUM, exp2, MBP)](https://github.com/apple/ml-stable-diffusion)
- [Apple — Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [WhisperKit (arXiv 2507.10860)](https://arxiv.org/abs/2507.10860)
- [SparQ Attention (arXiv 2312.04985)](https://arxiv.org/abs/2312.04985)
- [llm.npu (arXiv 2407.05858)](https://arxiv.org/abs/2407.05858)
- [TransMLA (arXiv 2502.07864)](https://arxiv.org/abs/2502.07864)
- [GliDe with CaPE (arXiv 2402.16785)](https://arxiv.org/abs/2402.16785)
- [shadowAttn (arXiv 2508.16703)](https://arxiv.org/abs/2508.16703)
- [PowerInfer-2 (arXiv 2406.06282)](https://arxiv.org/abs/2406.06282)
- [Qwen 3.5 Small — Gated DeltaNet](https://github.com/QwenLM/Qwen3.5)
- [sd.npu (arXiv 2510.15312)](https://arxiv.org/abs/2510.15312)
- [SpecAttn Co-Design (arXiv 2602.07223)](https://arxiv.org/abs/2602.07223)
- [SpargeAttention (arXiv 2502.18137)](https://arxiv.org/abs/2502.18137)
