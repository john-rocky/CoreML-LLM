# Deep-research findings (2026-05-13 evening)

6 parallel research agents investigated my blind spots:
1. Recent SpecDec papers (2025 H2 - 2026 H1)
2. Apple WWDC 2025/26 + iOS 26 / Foundation Models
3. Competing iPhone/Apple Silicon LLM projects
4. ANE 18 / A19 Pro specs
5. Lightweight drafter training methods
6. Undocumented CoreML / ANE empirical tricks

Below is the SYNTHESIZED high-conviction shortlist with proper citations.
Replaces the speculative padding in the prior vault.

---

## Tier S — Highest conviction, ready to implement

### S1. **MLComputePlan instrumentation** — foundational, 0 risk

iOS 17.4 / macOS 14.4+ public API exposes per-op cost + device assignment.
We currently GUESS what runs on ANE vs CPU. With this we can verify whether
LM head fell back to CPU, whether layer norm split off ANE, etc.

Citation: [Apple docs](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw/estimatedcostofmlprogramoperation:),
[freedomtan/coreml_modelc_profling](https://github.com/freedomtan/coreml_modelc_profling).

```swift
@available(iOS 17.4, macOS 14.4, *)
func dumpComputePlan(_ url: URL) async throws {
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
    if case .program(let p) = plan.modelStructure {
        for fn in p.functions {
            for op in fn.value.block.operations {
                let cost = plan.estimatedCost(of: op)?.weight ?? 0
                let dev  = plan.deviceUsage(for: op)?.preferred ?? .cpu
                print("\(fn.key)/\(op.operatorName) cost=\(cost) dev=\(dev)")
            }
        }
    }
}
```

**Action**: 30 lines Swift, gate behind `LLM_DUMP_COMPUTE_PLAN=1`. **Every
subsequent lever needs this to validate "did it stay ANE?"**

### S2. **Ping-pong IOSurface outputBackings for verify chunks** — re-enable backings safely

We disabled verify backings (`5b68fb3`) after iPhone "pixel buffer locked"
crash. Root cause documented: reading `.dataPointer` on a backed array
locks the CVPixelBuffer, next prediction errors. The ANEMLL pattern is
**two parallel IOSurface backing sets, alternating per cycle**.

Citation: [Apple dev forum 735698](https://developer.apple.com/forums/thread/735698),
[ANEMLL ping-pong](https://github.com/Anemll/Anemll/blob/main/docs/anemll-dedup.md),
[hollance/CoreMLHelpers](https://github.com/hollance/CoreMLHelpers).

Expected gain: **+5-8% tok/s** (15ms saved per verify cycle from 4 chunks
× 3-5ms IOSurface marshalling).

**Action**: 4-8h Swift change. `verifyOutBackingsA[1..4]` and
`verifyOutBackingsB[1..4]`, alternate using `cycleParity`.

### S3. **Argmax-in-LM-head fusion (decode chunk4)** — bandwidth saver

Currently chunk4 emits full fp16 logits (262144 × 2 bytes = 524 KB / token)
which marshalls back via IOSurface for argmax. Fuse `reduce_argmax + reduce_max`
into the MIL graph after `lm_head` conv → return `(Int32 idx, fp16 max_val)`
= 8 bytes / token.

Citation: [smpanaro coreml-llm-cli NOTES.md](https://github.com/smpanaro/more-ane-transformers/blob/main/src/experiments/NOTES.md)
— "moving final token selection pre-lm_head dropped gpt2-medium from 98 ms
to 52 ms". [ANEMLL --argmax](https://github.com/Anemll/Anemll).

Expected gain: **+3-5% tok/s** decode (~2 MB/token DMA elimination).

**Action**: 1 day. Modify `conversion/build_gemma4_3way.py` chunk4
to emit `(token_id, token_logit)` instead of `logits_fp16`. **Decode
only — verify chunk still needs full logits for FLy K=16 acceptance.**

### S4. **Self-distillation drafter from INT4 target** — Path B alternative

Currently centroid drafter is trained against **fp16 Gemma 4 E2B**. Our
target is **INT4-palettized**. The drafter is misaligned. arxiv 2505.22179
explicitly notes nobody has trained a drafter against quantized targets.

Recipe:
1. Overnight Mac: run `coreml-llm-smoke` to generate 100k continuations
   from our actual INT4 CoreML target. Cheap (we already have the runtime).
2. Train EAGLE-3 drafter head (or similar) on `(prompt, target-INT4-sampled,
   target top-k logits)` tuples with LK Loss (arxiv 2602.23881).
3. Cost: ~8-12 H100h = ½ GPU-day. Open code: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE).

Expected gain: **narrative acc 0.25 → 0.40+ = narrative +20-30%**. This
is the missing Path B in cheap form.

Citations:
- [LK Losses arxiv 2602.23881](https://arxiv.org/abs/2602.23881) — direct
  accept-rate optimization, +7.7-8.2 % on similar targets.
- [SpecDec-meets-Quant arxiv 2505.22179](https://arxiv.org/html/2505.22179v1)
  — quantized-target alignment is open ground.
- [EAGLE-3 arxiv 2503.01840](https://arxiv.org/abs/2503.01840) — training-time
  test rollout, 1 day single host.

**Action**: requires GPU access (or rented A100). Was previously blocked by
"訓練禁止" — but cost is ≤1 GPU-day, not the 1 GPU-week the user assumed.
Worth re-asking the user.

### S5. **TALON adaptive token tree** — pure Swift refactor of FLy

Our FLy K=16 currently uses a fixed-shape tree. TALON (arxiv 2601.07353,
Jan 2026) dynamically expands the tree based on drafter per-step confidence
under a fixed token budget. Claims up to 5x e2e (vs AR baseline, not vs
fixed-tree SD — so actual delta on top of our FLy K=16 is smaller).

Citation: [TALON arxiv 2601.07353](https://arxiv.org/abs/2601.07353).

Expected gain on top of our FLy K=16: **+5-15%** code, neutral narrative.
Pure runtime swap, no training.

**Action**: 1-3 day Swift change to `MtpSpeculativeEngine`. Replace the
flat top-K comparison with adaptive expansion guided by per-step drafter
softmax. No model rebuild.

---

## Tier A — High conviction, empirical / env only

### A1. **ANE working-set 32 MB SRAM ceiling** — verify cycle root cause?

iPhone M4 ANE measured: ~32 MB SRAM, working sets > that drop throughput
~30%. iPhone 17 Pro SLC is also 32 MB. **Our verify cycle (K=3 batched
across 4 INT4 chunks) likely exceeds 32 MB** → DRAM spill on every cycle
→ structural 50% heavier than Mac M-series (which has bigger SLC).

Citation: [maderix M4 ANE substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615),
[Orion paper arxiv 2603.06728](https://arxiv.org/html/2603.06728v1).

**Action**: probe working-set size per chunk via MLComputePlan + Instruments.
If confirmed, the path is to shrink per-cycle working set — verify K=2
(rebuild) or per-K serialized verify.

### A2. **smpanaro 4-mlmodelc ANE residency ceiling** — known capacity

Empirical: **iPhone ANE VM ~3.75 GB / ~4 model cap**. Our 4 decode + 4
prefill = 8 mlmodelc; prefill not always resident. Centroid drafter (149 MB)
makes 5 resident at decode time. **Risk: drafter may silently CPU-fallback
under memory pressure.**

Citation: [smpanaro NOTES.md](https://github.com/smpanaro/more-ane-transformers/blob/main/src/experiments/NOTES.md).

**Action**: MLComputePlan run on drafter confirms ANE residency. If not,
either drop a chunk size or add explicit `MLResidencyHint`.

### A3. **Apple Foundation Models framework adapter toolkit** — public 48M drafter recipe

Apple ships a public adapter training recipe for their on-device 3B model:
[foundation-models-adapter](https://developer.apple.com/apple-intelligence/foundation-models-adapter/).
**Cannot directly reuse** (targets Apple's 3B), but `examples/train_draft_model.py`
+ `export_fmadapter` is the Apple-blessed recipe for drafter training. Reading
it tells us exactly how Apple trains their 48M drafter (60-80% acceptance,
2-4× speedup).

Citation: [Apple Intelligence Foundation Language Models tech report,
arxiv 2507.13575](https://arxiv.org/abs/2507.13575).

**Action**: read the recipe, adapt to Gemma 4 E2B. Same hyperparams /
loss / data mix template.

### A4. **MTP_VERBOSE_SETUP + cold→warm drafter curve probe** — R3 from prior vault

Today saw drafter cold=81ms vs warm=1.8ms (44× slower cold). AutoBench
warmup (`bc5b04a`) is 24 tokens — may be too short to fully warm. Plot
draft= time across cycles 0-20 to characterise the curve.

**Action**: env-only iPhone bench, ~15 min. Already covered in
`docs/IPHONE_LEVER_VAULT_2026_05_14.md` R3.

---

## Tier B — Promising but unverified

### B1. **TurboQuant WHT-rotated KV cache** — INT2/3-bit lossless KV

`atomic-llama-cpp-turboquant` fork claims 2/3/4-bit KV with Walsh-Hadamard
rotation + PolarQuant, ~6.4× compression at 2-bit, lossless. Plus a "TurboFlash"
Metal kernel and Gemma 4 MTP integration claiming 85-88% accept.

Citation: [AtomicBot/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant).

**Caveat**: single party (fork), claims need verification. Worth a Mac
empirical test before iPhone.

Expected gain (if real): **KV memory -50%**, ANE working set drops below
32 MB → A1 root cause solved → **+10-20% decode** potential.

**Action**: Mac empirical first. Fact-check claims by re-running their bench
recipe. ~1 day Python.

### B2. **Variational Speculative Decoding (VSD)** — stacks on EAGLE-3

arxiv 2602.05774. Reformulates drafter training as ELBO maximization on
sequence acceptance. Stacks ON TOP of EAGLE-3 / HASS / GRIFFIN with +7-9%
each. Single RTX PRO 6000 for drafter training.

**Action**: blocked on S4 (need drafter retrain anyway). Use VSD loss if
we do S4.

### B3. **Mamba / SSM drafter** — long-shot for ANE

arxiv 2506.01206 — 130M Mamba drafter, ~16 GPU-h training. Linear-time
recurrent inference = structurally faster per draft step.

**Caveat**: Mamba `selective_scan` op not validated on CoreML/ANE. Could
require custom kernel. High risk, high reward.

**Action**: feasibility-check via coremltools conversion of a 130M Mamba.
~1 day before committing training time.

### B4. **`async let` overlap of decode-N+1-chunk1 prep with verify-N-chunk4**

HF blog (fguzman82) shows iPhone ANE supports 2-way concurrency cleanly.
Our verify chunk4 + next decode chunk1 are data-independent (next decode
needs verify's accepted token, but **prep** of next inputs can run while
verify is still finishing).

Citation: [HF blog async batch prediction](https://huggingface.co/blog/fguzman82/coreml-async-batch-prediction).

Expected: **+1-3% tok/s** by hiding ~1ms overlap per cycle.

**Action**: Swift change. Two MLModel handles on decode_chunk1 (share
weights via mmap, no extra memory). ~4h.

---

## Tier C — Audit-only / defensive

### C1. **ANE Field Guide hard rules** — avoid silent ANE eviction

[skyfallsin field guide](https://github.com/skyfallsin/apple-neural-engine-field-guide):
* MIL `tile` op poisons ANE state per process
* N-broadcast `mul` unreliable; use same-shape mul
* `IOSurface W < 32` doesn't bind to ANE
* Runtime-weight conv fails 0x1d
* Dispatch latency = 119 µs + bytes / 78 GB/s

**Action**: audit our MIL for tile / broadcast-mul. Defensive only.

### C2. **GELU tanh-approx** — already safe

Gemma 4 uses GELU-pytorch_tanh which is ANE-fast. **Audit-only confirmed
OK.**

### C3. **Stateful tensor width-32 alignment** — ANE 18 quirk

Apple Developer Forums 2026: stateful tensors must have width = multiple
of 32 to land on ANE. Our `kv13_v` shape mismatch (iPhone expected
(1,1,256,512) vs decode emitted (1,1,512,256)) is consistent with **ANE
compiler folding transpose into 32-aligned shape**.

**Action**: defensive — ensure all stateful K/V tensor widths are 32-aligned
in any new build.

---

## Confirmed dead-ends (don't pursue)

| lever | reason | source |
|---|---|---|
| Tree verify (MTP_TREE_VERIFY=1) | -20 % yes-yes, no free-form lift | code comment + memory |
| MARS / CSD lossy verify | output incoherent | code comment + R1 agent |
| Apple block-shared KV (AFM trick) | blocked by Gemma 4 E2B `attention_k_eq_v=false` | memory `project_gemma4_k_eq_v_false` |
| Apple Foundation Models inference reuse | closed loop with their 3B, not Gemma | R2 agent |
| MLX speculative decoding | GPU/Metal only, no ANE | R3 agent |
| LiteRT-LM 56 tok/s | GPU/Metal only, no ANE path on iOS | R3 agent |
| llama.cpp ANE backend | doesn't exist yet | R3 agent |
| Adaptive K_USE | Mac +3.5 % vs static +13.8 % | code comment + memory |
| iPhone Game Mode for LLM | no public API | R4 agent |
| `experimentalMLE5EngineUsage` private key | App Store rejection risk | R6 agent |

---

## Honest revised expectations

Replaces the prior 53-lever "+10-20% optimistic" estimate with sourced
estimates per lever:

| lever | conviction | est gain | cost |
|---|---|---|---|
| S1 MLComputePlan | foundational | 0 (unlocks others) | 1h |
| S2 ping-pong outputBackings | high | +5-8% | 4-8h |
| S3 argmax-in-LM-head | high | +3-5% decode | 1d |
| S4 INT4-target self-distill drafter | high | narrative +20-30% | ½ GPU-day |
| S5 TALON adaptive tree | high | +5-15% code | 1-3d |
| A1 32MB SRAM probe | high (probing) | unknown, root-cause | 1d |
| A4 cold→warm probe | high (probing) | warmup tuning | 15min |
| B1 TurboQuant WHT-KV | medium | +10-20% IF real | 1d Mac + 1d iPhone |
| B2 VSD loss | blocked on S4 | +7-9% additive | included in S4 |
| B4 async let overlap | medium | +1-3% | 4h |

### Realistic combined upper bound (Tier S all wins)

* Code: today 50 (warm K_USE=1) → **65-75 tok/s** (+30-50%) if S2+S3+S5
  stack as designed.
* Narrative: today 31 (T=1 parity, MTP bails) → **40-45 tok/s** (+30-45%)
  if S4 drafter retrain works AND Tier S levers help on the MTP-engaged path.

These are **honest, sourced, not padding**.

### What's NOT in scope here

- **Smaller iPhone CPU win** (<+3%): I'd ignore — within noise.
- **Speculative claims without citations**: previously L26-L46 padding.
- **Production polish (L43 prompt classifier)**: real UX win, but no tok/s
  headline gain.

---

## Recommended order (replaces prior vault Tier 1-6)

If user authorises drafter retrain (S4): **S4 in parallel with S1+S2+S3
in another track.** S4 takes 12-24 hours wall clock; in the meantime ship
S1-S3.

If drafter retrain not authorised: **S1 → S2 → S3 → S5 → A1 → B1**. 1-week
budget gets ~+15-20% on top of today's 40/50/31 baseline.

Skip the prior vault's L16-L46 (padded ideas). The 8 levers above (S1-S5,
A1, B1, B4) are the actual high-conviction set.
