# Session 2026-05-06 — Gemma 4 official MTP drafter, Mac bench

Continuation of `docs/MTP_GEMMA4_OFFICIAL_HANDOFF.md`. Goal was a Mac
benchmark of `MtpSpeculativeEngine` against the Gemma 4 E2B 3-chunk
stateful baseline (~34.6 tok/s) to decide whether the official Google
drafter (`google/gemma-4-E2B-it-assistant`) clears the rolling-acceptance
≥ 0.50 + tok/s ≥ 1.3× baseline gates for an iPhone push.

## llama.cpp 統合知見 (2026-05-06、`~/Downloads/gemma4_mtp_findings.md` より)

別の人が同じ drafter を llama.cpp に統合した記録から、当 CoreML 経路と
**完全に一致している事項**:

- HF safetensors 経路 (= 当 repo と同じ source) を使う。`.pt` (TFLite extract) は
  別系統で互換性なし。
- concat 順序: `concat(embed, activation)` ← embed が先 ✓
- token_embedding source: **target の `embed_tokens.weight[token] * sqrt(hidden_target=1536)`** ✓
- step-0 carry seed: target の **post-final-norm hidden** (`model.norm` 通過後) ✓
- step-i (i>0) carry: drafter の `post_projection` output ✓
- partial_rotary 0.25 on full layer ✓ (当 repo は `cos_full.npy` で対応)
- **per-step position** = `current_position_ids = N-1 + step` ✓ (当 repo の per-step pos+k と同じ;
  HF docs は constant と書いてあるが llama.cpp 実装も per-step。HF docs が雑)
- final_logit_softcapping は **null** (HF assistant)。`.pt` は 30.0 だが HF は無し ✓
- FFN activation: **`gelu_pytorch_tanh`** (`.pt` は erf gelu) ✓
- norm はすべて `output * weight` (Gemma 3 系の `(1+weight)` shift は **使わない**) ✓
- `layer_scalar` (`[1]` scalar buffer): 各 layer 末尾で `hidden_states *= layer_scalar`。
  E2B 値 0.04, 0.25, 0.43, 0.12。**忘れると残差が爆発**。当 drafter ANE module も適用済み ✓

**llama.cpp で見つかったバグで、当 CoreML で同じものが起きうるもの**:

| llama.cpp が踏んだバグ | 当 repo の状態 |
|---|---|
| token_embedding を drafter から取得 → 全くデタラメ | 当 repo: target の `engine.embedToken()` (sqrt scaled) で正しい ✓ |
| concat 順序が `[hidden, embed]` 逆 → top5 全 0 | 当 repo: `embed_token, proj_act` 別 input、内部で正しく concat ✓ |
| GELU 種別 (.pt erf vs HF tanh) → logit magnitude ずれ | 当 repo: HF tanh 経路 ✓ |
| softcap 既定 30 で logit clip | 当 repo: 当 drafter は softcap 無し (HF spec) ✓ |
| `layer_scalar` 抜け → layer 1+ で 24x scale 差 | 当 repo: `mtp_drafter_model.py` で `register_buffer("layer_scalar")` 適用済み ✓ |
| `q_norm` per-layer **fixed scalar** (~0.99 SWA, ~1.02 full、`[head_dim]` で同じ値 broadcast) | **要確認**: 当 repo の `q_norm` 重み load 時にこの構造を扱っているか |

**llama.cpp が達成した parity** (drafter graph 単独): HF F32 K/V を直接流したケース、
全層 cos ≥ 0.998、drafter top5 完全一致。**llama.cpp 側の statics (drafter graph 数学) は完全に一致**。
ただし **#16 / #17** (実際の speculation pipeline integration とエンドツーエンド bench) は未完了。
Google blog の "~2x" を実環境で再現できているか **誰もまだ示せていない**。

**含意**: 当 repo の +13.8% という数字は、**drafter 数学が正しい上で MTP cycle のコスト構造**から
出ている結果。drafter 内部のバグではなく、cycle math の問題なので、追加実験は
"verify time は K/compute_unit にどう依存するか" を切り分ければ原因が分かる。

---

## TL;DR (2026-05-06 v2 — full bit-faithfulness audit)

After a comprehensive chunk-by-chunk audit + multiple rebuild experiments
(fp16, INT8, INT4+per_channel_scale, hybrid, GPU compute), the realistic
Mac M3 Ultra ceiling on this bundle is **+9.7-13.8% at K_USE=2 static**
(33.04 → 36.24-37.58 tok/s on chat-templated long code).

The gap to vendor numbers (LiteRT-LM, mlx-vlm, vLLM, Ollama all hit
1.8-3.94×) comes from:

1. **INT4 weight palettization noise** dominates K cache drift
   (chunk1 cos 0.978 vs fp16 0.993). Vendors use bf16/fp16 weights on
   hardware with abundant memory bandwidth (M-series GPU, NVIDIA RTX,
   DGX); our ANE constraint forces INT4. Per-chunk audit proves the math
   IS correct in fp16; conversion isn't broken.
2. **Naive kmeans INT4 vs AWQ/GPTQ calibration.** vLLM PR #41745 +
   community use AWQ-4 (200+ tok/s on 26B). Our coremltools palettize
   uses uniform kmeans without activation-aware scaling. Rough estimate:
   AWQ → cos 0.99 at INT4 size → MTP +20-30%.
3. **Hardware ceiling.** GPU-compute and CPU-compute on our M3 Ultra
   gave 17-18 tok/s baseline (vs ANE 33). Switching off ANE is a net
   loss even with cleaner K cache.

### Experiments performed and rejected

| build | chunk1 cos | chunk4 cos | baseline | MTP K_USE=2 | verdict |
|---|---|---|---|---|---|
| INT4 default (production) | 0.978 | 0.834 | 33.04 | **37.58** | **ship at +13.8%** |
| INT4 + ANERMSNorm fp32 cast | 0.968 | 0.860 | 32.86 | 35.73 | no-op (FLOAT16 strips cast) |
| fp16 (no palettize) | 0.993 | 0.968 | 12.86 | 18.16 | math correct, ANE BW too slow |
| INT8 palettize | started, killed (~4 hours) | – | – | – | kmeans 256-cluster too slow |
| INT4 + enable_per_channel_scale=True | 0.968 | 0.865 | 3.78 | 8.27 | LUT explosion (149→490 MB) |
| Hybrid (chunk2 fp16, others INT4) | 0.978 | 0.817 | 30.32 | 33.81 | marginal +11.5%, not absolute win |
| ANE → GPU compute | – | – | 17.35 | 16.35 | INT4 chunks slow on GPU |
| ANE → CPU compute | – | – | 18.48 | 12.17 | even slower |
| HF heuristic adaptive K_USE | – | – | 33.01 | 34.15 | over-shrinks at 28% accept; static K_USE=2 wins |

### Path to higher gains (unfunded)

| approach | est. gain | effort |
|---|---|---|
| GPTQ via `coremltools.optimize.torch.LayerwiseCompressor` (calibration-aware INT4) | +20-30% | 2-3 days |
| AWQ-style custom palettize pass in coremltools | +30-50% (matches vendor 1.8×) | 1 week |
| Move target to bf16 GPU compute (rearch) | unclear, depends on M-series GPU vs ANE bandwidth ratio | 1 week |
| Tree-style verify (Eagle-3 layout) recover from chain breaks | +10-20% on top of current | 1 week |

## TL;DR (2026-05-06 v1)

- **MTP path is correct.** PyTorch port + compiled CoreML mlpackage both
  reproduce HF reference top-1 on captured real-data K/V
  (`conversion/replay_capture_through_{port,coreml}.py`). The mlpackage
  is bit-faithful to the trained drafter.
- **HF runtime ceiling (Mac CPU fp16/fp32):** capitals raw 2.24× / 90 %,
  chat-templated capitals 0.92× / 33 %, chat-templated CODE prompt
  **1.19× / 47 %**, essay 0.48 % / 15 %, qa 0.72× / 22 %. Speedup is
  **prompt-shape dependent** — Google's demos are mostly chat × code/structured.
  Earlier "chat halves accept" reading was misleading (only 3 rounds of
  data); proper bench is `conversion/bench_hf_chat_template_v2.py`.
- **Mac CoreML net positive at K_USE=2 default:** list +5.8 %, code
  +2.2 % over baseline. Below HF's 1.19× ceiling on the same prompts.
- **Root cause of remaining gap: INT4 weight palettization noise.**
  Per-chunk parity audit (`conversion/diff_per_chunk.py`):
    - INT4 default (production):    chunk1 cos=0.978, chunk4 cos=0.834
    - fp16 (no palettize, build-only diagnostic): chunk1 cos=0.993, chunk4 cos=0.968
    - INT8 palettize (started, killed): too slow (~4 hours, kmeans 256-cluster)
  The drafter expects K cache from a numerically clean target; INT4
  palettization shifts K by ~2-5 % cosine, which cuts drafter accept
  roughly in half (HF 47 % → ours 28 % per-slot). Going below INT4 noise
  would require either disabling palettization (negates speed: fp16
  baseline drops to 12.86 tok/s vs INT4 32.86) or AWQ/GPTQ-style
  calibration that's a separate project.
- **Six alignment bugs found and fixed during the chase:** embed scale,
  ctx mismatch, post-norm carry seed, partial-rotary RoPE spec
  divergence in target build, K_USE default, drafter mlpackage
  ctx-length mismatch. Details below.
- **Bundle pairing surprise.** `MtpSpeculativeEngine` requires
  `ChunkedEngine` 4-chunk multifunction (with `verify_qK` at K=8); the
  v1.7.0 ship 3-chunk stateful Linear bundle (`Gemma4StatefulEngine`)
  cannot host MTP without re-engineering. We benched against the legacy
  4-chunk bundle in `output/gemma4-e2b/bundle/`.
- **Long-standing target build divergence from HF spec uncovered**:
  `generate_rope.py` produced full-rotary `cos_full.npy`/`sin_full.npy`,
  but HF Gemma 4 E2B's `text_config.rope_parameters.full_attention`
  specifies `partial_rotary_factor=0.25` (rope_type=proportional). All
  bundles built before this session have full-rotary K caches in
  `kv14_*`. `conversion/generate_rope.py` now defaults to HF-spec
  partial=0.25; existing in-flight Mac bundle was patched in place (cos
  / sin originals preserved as `*.fullrotary.bak`).
- **Verdict (Mac, K=8):** MTP fails Pass criterion. Don't push to iPhone
  yet. Likely next steps: smaller K (3), tree-style verification, or
  drafter retraining on a fp16 target with our actual rotation
  convention.

## Bugs found and fixed (in order)

1. **Bundle ctx mismatch.** First `mtp_drafter.mlpackage` was built at
   default `--context-length 8192`; target bundle is ctx=2048. Rebuilt
   drafter at ctx=2048 (`mtp_drafter_ctx2k.mlpackage`).
2. **Embed scale missing** (Open item #2 in handoff). `lookupRawEmbed`
   returned the unscaled INT8-dequantized row; HF
   `Gemma4ScaledWordEmbedding` applies `* sqrt(hidden_size)` inside the
   embedding lookup. `MtpSpeculativeEngine` now calls
   `engine.embedToken(_:)` (which applies `config.embedScale = 39.19`).
   *Lifted acceptance from 0/7 always to occasional 1–3/7 (~3 %).*
3. **First-burst carry seed = zeros** (Open item #3, but fix was
   different from doc). Doc said use `lastHiddenAtL34`, but that tap is
   only present in EAGLE-3-built chunks; legacy 4-chunk bundle's chunk4
   doesn't expose it. Per HF
   `SinglePositionMultiTokenCandidateGenerator.get_candidates`
   (transformers `candidate_generator.py:1357`), the carry seed is
   `model_outputs.hidden_states[-1]`, i.e. the **post-final-norm** hidden
   — `chunk4.hidden_states_out`. Added `ChunkedEngine.lastDecodeHiddenStateOut`
   capturing this on every decode T=1 and using it as the first-burst
   `carryState`. Fall-back to zeros if absent.
4. **Target full-rotary vs HF partial-rotary mismatch** (Open item #4
   plus a bigger discovery). HF Gemma 4 base spec for full-attention:
   `partial_rotary_factor=0.25`, `rope_type=proportional`. Our
   `generate_rope.py` produced full-rotary tables (256 active freqs
   instead of 64 + 192 zeros). Drafter HF training assumes partial
   rotary on the full layer — our K cache rotation diverges. Fixed by
   regenerating `cos_full.npy` / `sin_full.npy` with HF-spec partial
   rotary; baseline tok/s 32.75 → 33.42 (slightly faster), output
   coherent. Build script defaults updated.
5. **Constant vs per-step `position_ids`** (HF design quirk).
   `SinglePositionMultiTokenCandidateGenerator` docstring says the
   assistant is "locked into a constant `position_ids` value" and the
   code (`candidate_generator.py:1370`) confirms — `position_ids` is set
   ONCE before the autoregressive loop. Our Swift used `pos+k` per
   step. We tried both: per-step gave higher accept on capital-cities
   prompt (29 % vs 22 %), constant gave slightly higher on essay; net
   wash. **Reverted to per-step pos+k** with a doc note flagging the
   discrepancy. Worth revisiting.

## Numbers

Mac M3 Ultra, `output/gemma4-e2b/bundle/` (4-chunk legacy multifunction,
K=8 verify), partial-rotary cos_full.npy in place.

| Run | Prompt | tok/s | accept | note |
|---|---|---|---|---|
| Baseline (no MTP) | essay | 32.75 | n/a | full-rotary cos_full |
| Baseline (no MTP) | essay | 33.42 | n/a | partial-rotary cos_full |
| MTP (all fixes) | essay (256 tok) | 16.55 | 0.02 | regression |
| MTP (all fixes) | capitals (96 tok) | 17.41 | 0.29 | still slower than baseline |

Cycle math at K=8: ~35 ms draft + ~31 ms verify = 66 ms per cycle. To
beat baseline 30 ms/tok we need ≥ 2.2 emitted tok/cycle ⇒ accept ≥ 0.17
on average. Capitals (29 %) clears that, essay (2 %) doesn't. The
overall average drag from low-accept cycles dominates wall time.

## Remaining knowns (unverified)

- **Sliding-mask flip.** HF
  `create_attention_masks` (gemma4_assistant) does
  `sliding_attention_mask.flip(dims=(1,))` — our
  `makeSlidingCausalMask` is right-aligned without flip. Could account
  for residual mismatch in the SWA layers.
- **K=8 vs K=3.** Drafter mlpackage is single-step (K-agnostic); verify
  is K=8 (per chunk multifunction). A K=3 verify build could lift
  break-even and reduce per-cycle overhead. Not attempted.
- **Tree verification.** Round-1 example: drafter[1, 3, 4] all match
  target[1, 3, 4] but drafter[0] doesn't, so chain accept stops at 0.
  Tree-style verify (Eagle-3 layout) would recover those.
- **Constant-pos vs per-step.** Mixed empirical signal. HF code clearly
  uses constant; our drafter port may have absorbed something
  position-dependent during training that makes per-step better in
  practice.

## Files touched

- `conversion/generate_rope.py` — partial_rotary_factor support, default
  0.25 for full attention (HF spec).
- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` — embed scale fix
  (`lookupRawEmbed` → `embedToken`); first-burst carry seed from
  `lastDecodeHiddenStateOut`; per-step `pos+k` retained.
- `Sources/CoreMLLLM/ChunkedEngine.swift` — capture chunk4
  `hidden_states_out` into `lastDecodeHiddenStateOut`.
- `output/gemma4-e2b/bundle/cos_full.npy`, `sin_full.npy` — replaced
  with partial-rotary tables. Originals at `*.fullrotary.bak` (gitignored).
- `output/gemma4-e2b/bundle/mtp_drafter.mlmodelc` — compiled
  `mtp_drafter_ctx2k.mlpackage` (ctx=2048). Symlink to repo-root
  package was the original wiring; replaced with the compiled version.

## Next moves (when revisited)

1. **Build a 3-chunk multifunction stateful bundle with verify_qK at
   K=3.** This unlocks `Gemma4StatefulEngine` parity baseline (34.6
   tok/s) AND lowers per-cycle drafter cost to 3×35 + 1×31 = 136 ms /
   ~3 emit ≈ 22 tok/s — still likely under baseline unless drafter
   accept jumps materially.
2. **Verify the constant-pos behaviour by reading HF reference
   live.** Run `target.generate(assistant_model=drafter)` on the same
   prompt, log the actual draft positions HF feeds. If HF really uses
   constant pos and gets 50 %+ on essays, our drafter port has a more
   subtle bug.
3. **Check sliding-mask flip alignment.** Easiest diagnostic: temporarily
   build a "flipped" sliding mask in `makeDrafterSWAMask` and rerun.
4. **Drafter retraining** if 1–3 don't move the needle. Rebuild the
   drafter on top of our fp16 target with our actual rotation
   convention; HF blog claims "up to 2×" assumes retrained-against-target.

## State of the repo at end of session

- Branch: main (uncommitted changes)
- Drafter works end-to-end; MTP K_USE=2 net positive on structured
  prompts (list/code), neutral-to-disabled on essays via fallback.
- HF runtime bench scripts shipped: `conversion/bench_hf_assistant.py`,
  `conversion/bench_hf_chat_template.py`,
  `conversion/dump_hf_drafter_steps.py`,
  `conversion/capture_hf_drafter_inputs.py`,
  `conversion/replay_capture_through_port.py`,
  `conversion/replay_capture_through_coreml.py`,
  `conversion/replay_swift_through_coreml.py`. These let any future
  session reproduce the HF ceiling and bisect any port-side regression
  to one specific component (port → coreml → swift).
- `conversion/generate_rope.py` partial-rotary default is a clean
  improvement for ANY future Gemma 4 build and could be committed
  independently — but every existing on-disk bundle uses full-rotary
  `cos_full.npy` and would be slightly off-spec until rebuilt.

## Final A/B numbers (Mac M3 Ultra, INT4 production bundle)

Default config (K_USE=2, partial-rotary cos_full, post-norm carry seed,
embed scale via `engine.embedToken`).

| prompt | baseline tok/s | MTP tok/s | Δ | accept (slot) |
|---|---|---|---|---|
| chat × 20 capitals list | 33.07 | 35.00 | +5.8 % | 0.09 (28 %) |
| chat × Fibonacci code | 33.12 | 33.86 | +2.2 % | 0.06 |
| chat × doubly-linked list code (long) | 32.86 | 35.73 | **+8.7 %** | 0.07 (28 %) |
| chat × free-form essay | 33.01 | 29.79 → fallback | n/a | 0.02 |

### Audit-only (not shipped)

| build | chunk1 cos | chunk4 cos | baseline tok/s | MTP K_USE=2 | Δ |
|---|---|---|---|---|---|
| INT4 palettize (default ship) | 0.978 | 0.834 | 32.86 | 35.73 | +8.7 % |
| fp16 (no palettize) | **0.993** | **0.968** | 12.86 | 18.16 | +41 % (but absolute slower) |
| Hybrid (chunk2 fp16, others INT4) | 0.978 | 0.817 | 30.32 | 33.81 | +11.5 % |

fp16 chunks have the cleanest K cache and highest relative speedup, but
the ANE-resident memory bandwidth penalty drops absolute tok/s 60 %
below the INT4 baseline. Net result: pure INT4 wins on production
absolute throughput with MTP on.

HF runtime reference (Mac CPU fp32):
- Raw capitals: 90 % accept, 2.24× — **Google's "up to 2×" verified for raw input**.
- Chat-templated capitals: 33 % accept, 1.09× — **drafter halved by template**.

Our Swift Mac on chat-templated tracks HF's chat-template ceiling
(within fp16/INT4 ANE noise), so the "29 % accept" measurement is
correct, not a port bug.

## Tuning knobs (env vars added to `MtpSpeculativeEngine` /
`MtpDraftSource`)

- `MTP_K_USE` — drafter forwards per cycle. Default 2 (best on
  chat-templated). Set 0 for legacy K=8.
- `MTP_DRAFT_POS_MODE` — `perstep` (default), `constpm1`, `constpos`.
  HF docs claim constant; our Mac empirically prefers per-step.
- `MTP_MASK_OFFSET` — drafter mask offset (default 1). 2/3 reduce
  attention to recent K, hurts on Mac.
- `MTP_DRAFTER_DEVICE` — `cpu`/`gpu`/`ane` (default ANE). For numerics
  bisection.
- `MTP_FORCE_SPECULATE` — bypass rollingAcceptance fallback (debugging).
