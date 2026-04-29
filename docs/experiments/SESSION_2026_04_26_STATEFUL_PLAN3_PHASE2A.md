# Stateful Gemma 4 + Plan 3 Linear migration + Phase 2a cross-turn KV reuse

**Sessions:** 2026-04-25 (PM) → 2026-04-26 (early AM)
**Outcome:** Three independent wins shipped to `main`. Stateful path now
production-quality, Plan 3 GO at iPhone scale, Phase 2a delivers ~95 %
TTFT reduction on multi-turn chat.

This doc is the durable record so the work doesn't get lost. Detail
behind every number is in the linked commits and probe scripts.

---

## 0. TL;DR

| theme | result |
|---|---|
| Gemma 4 E2B stateful chunks (Phase 1) build | Mac OK, ANE 90-94 % per chunk, predict round-trip verified |
| `coremltools 9` wins to dig | 5 ranked, Top-2 = `canonicalize_inplace_pattern` (the actual error -14 fix) + `linear` activation-quant |
| Plan 3 (`Conv2d-1×1` wrapper → `nn.Linear` native) | **GO at production scale**. Mac E2E parity, iPhone Turn-2 long-decode parity |
| iPhone runtime wiring (`Gemma4StatefulEngine` host) | LLMRunner detection + manual prompt construct + skipSet, both bundles deployable |
| Phase 2a (cross-turn KV reuse via LCP match) | **Multi-turn TTFT 7.3 s → 0.34 s (-95 %)** |
| cml9 GPU lowprec hint (vision encoder) | **HOLD** — Mac bit-exact output, no shader flip; runtime change captures nothing |

---

## 1. Stateful chunks build (`build_gemma4_e2b_stateful_chunks.py`)

cml9 + iOS18 makes the dual-state pattern work that hit error -14 in
2026-04-13 / 04-15. Confirmed by full 4-chunk build:

| chunk | layers | size | ANE placement |
|---|---|---|---|
| 1 (own KV, computes PLE) | L0-7 | 148.7 MB | 93.6 % (1071/1144) |
| 2 (own KV, emits kv13/14) | L8-14 | 128.1 MB | 93.6 % (919/982) |
| 3 (stateless, KV-shared) | L15-24 | 310.5 MB | 92.2 % (824/894) |
| 4 (stateless + lm_head) | L25-34 | 502.8 MB | 91.3 % (832/911) |

Mac sanity: `conversion/sanity_stateful_chunks.py` chains all four,
verifies state round-trip on chunk_1 (Δ hidden = 2.45 between pos=0 /
pos=1 with same state) and produces a valid `token_id`.

---

## 2. cml9 dig — Top-5 ranked (full report `docs/SESSION_2026_04_25_RESIDUAL.md` neighbour)

1. **`canonicalize_inplace_pattern` + `prefer_state_in_downstream`** — the
   pass set that recovered the dual-state build. Already firing, no flag
   to flip.
2. **PR #2577: `linear` activation-quant native** — clears the way to
   drop `ane_ops.Conv2dLinear`. Validated end-to-end (this session).
3. `guard_negative_gather_indices` — automatic, no action.
4. `AllowLowPrecisionAccumulationOnGPU` runtime hint — **HOLD** (§7).
5. `torchao.dequantize_affine` recognition — converter cleanup, low ROI.

---

## 3. Plan 3 — Linear vs Conv2d migration

### Probe ladder

| scale | env | Conv2d | Linear | Δ |
|---|---|---|---|---|
| 5-op micro | Mac | 1.02 ms | 0.86 ms | -16 % (small-N noise) |
| 5-layer fp16 | Mac (MBA) | 5.78 ms | 5.89 ms | +1.9 % (noise) |
| 5-layer + W4 | Mac (MBA) | **1.80** | **2.18** | **+21 %** (synthetic anomaly) |
| chunk_1 + W4 | Mac | 4.06 ms | 4.10 ms | +0.9 % |
| 4-chunk E2E + W4 | Mac | 23.48 ms | **23.24 ms** | **-1.0 %** |
| **Turn-2 long decode (256/237 tok)** | **iPhone 17 Pro** | **40.0 tok/s** | **39.9 tok/s** | **-0.25 %** |

### Verdict

**MBA's 5-layer +21 % gap was a synthetic-probe artifact**, not silicon.
At chunk_1 / 4-chunk / iPhone scale it disappears. Linear is parity on
iPhone, ANE placement parity (90-94 % both), MIL ops -8 % on Linear
(no `Conv2d-1×1` permute/squeeze wrappers).

### Migration status

- `--linear-projections` flag on `build_gemma4_e2b_stateful_chunks.py`
  (opt-in).
- `_project` helper in `conversion/models/gemma4_swa_stateful_chunks.py`
  dispatches Conv2d / Linear at trace time.
- **`ane_ops.Conv2dLinear` NOT deleted** — used by Qwen3-VL / Qwen3.5 /
  4-chunk Gemma 4 converters too. Full deletion needs those paths
  re-validated; a separate PR.

### Probe scripts

- `conversion/probe_linear_vs_conv2d.py` — micro 5-op
- `conversion/probe_linear_vs_conv2d_5layer.py` — 5-layer (MBA)
- `conversion/probe_linear_vs_conv2d_attn_w4.py` — 5-layer + attn + W4 (MBA)
- `conversion/probe_chunk1_linear_w4_latency.py` — 35-layer chunk_1 + W4
- `conversion/probe_e2e_linear_latency.py` — full 4-chunk E2E + W4

---

## 4. Runtime wiring (`Gemma4StatefulEngine` host)

The upstream `Gemma4StatefulEngine.swift` (Phase 1, commit `e47e22b`)
shipped with no LLMRunner detection. This session lands the picker plumbing.

`Sources/CoreMLLLM/ModelDownloader.swift`:
- `gemma4e2bStateful` (Conv2d) + `gemma4e2bStatefulLinear` (Plan 3 partner)
  ModelInfo entries, both `LLM_SHOW_EXPERIMENTAL=1` gated, sideload-only.
- `localModelURL` recognises `gemma4_e2b_stateful_chunks/` subdir.

`Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift`:
- Stateful detection block dispatches to `Gemma4StatefulEngine`.
- **Manual Gemma 4 prompt construct** (NOT `applyChatTemplate`):
  `<bos><|turn>user\n…<turn|>\n<|turn>model\n` — Gemma 4 uses
  `<|turn>` / `<turn|>` (id 105 / 106) where Gemma 2/3 used
  `<start_of_turn>` / `<end_of_turn>` (literal text differs, IDs match).
- `skipSet = [1, 105, 106]` filters control tokens from the chat bubble
  before the engine breaks on the next-iteration EOS check.

### Bundle assembly

- `scripts/assemble_gemma4_stateful_bundle.sh` — single variant
- `scripts/assemble_gemma4_stateful_ab.sh` — both variants (Plan 3 A/B)

### Bug saga (re-record so we don't repeat)

1. **`model_config.json` `context_length: 2048` from staging** —
   chunks were built at `--ctx 512` (gemma4-e2b registry default).
   Mismatch crashed mask allocation. Fixed by patching the JSON to 512
   per bundle. Lesson: rebuild chunks at production ctx (2048) when
   shipping for real, OR keep this patch step in the assemble script.
2. **`devicectl device copy to --destination <file>` is destructive** —
   replaces the destination's PARENT directory with just the source
   file's tree. Erased the 4-chunk + sidecar contents twice during the
   tokenizer_config / model_config patches. Always push at directory
   level. Re-pushing the full ~3.7 GB bundle was the only recovery.
3. **`tokenizer_config.json` ships with no `chat_template`** —
   `applyChatTemplate` failed silently and we fell back to plain
   `tok.encode(text:)`, producing degenerate "こんにちは こんにちは…"
   loops. The fix is the manual construct in §4 — matches
   `CoreMLLLM.swift:1515` (production Gemma 4 path).
4. **My initial chat template assumed Gemma 2/3 markers** —
   `<start_of_turn>` BPE-decomposes for Gemma 4 (its actual markers are
   `<|turn>` / `<turn|>`). Output included literal `<endof_turn>` text
   leak. Removed the embedded Gemma 2/3 template from
   `tokenizer_config.json`; the LLMRunner manual construct uses the
   right markers.

### Output quality after fixes

Both Conv2d and Linear stateful variants produce production-equivalent
output:

> こんにちは！何かお手伝いできることはありますか？☺️

Diverges from production 4-chunk only at low-confidence argmax positions
(stateful's ring + slice_update KV layout vs production's shift cache
yield slightly different fp16 rounding). Behaviour is "stateful-family
characteristic", not Linear-specific.

---

## 5. Phase 2a — cross-turn KV reuse

`Sources/CoreMLLLM/Gemma4StatefulEngine.swift` (commit `bdac795`):

- New fields: `persistedState1` / `persistedState2` (`MLState?`),
  `persistedInputIds` (`[Int32]`), `persistedPosition` (`Int`).
- `resetPersistedState()` public method, called from
  - `load(modelDirectory:)` (state binds to MLModel instance, drop on
    re-load to avoid dangle)
  - `LLMRunner.resetConversation()` (chat clear)
- `generate()` LCP scan: if `persistedInputIds` is a strict prefix of
  `inputIds` and both states still bound → resume from `resumeAt = LCP`,
  prefill only `[resumeAt, inputIds.count)`. Otherwise allocate fresh
  `makeState()` on chunk_1 / chunk_2. After generate, persist
  `inputIds + decoded.dropLast()` (the last decoded token's "feed" step
  never ran).

### iPhone 17 Pro Linear validation

| | Turn 1 (fresh) | Turn 2 (resume) |
|---|---|---|
| prefill tokens | 10 | 13 [resumed L=265] |
| prefill ms | 271 | 336 |
| decode tokens | 11 (EOS) | 20 |
| decode tok/s | 32.3 | 43.0 |
| (without resume — would have been) | — | 278 tok @ 38 tok/s ≈ 7.3 s |
| **TTFT delta** | — | **0.34 s, -95 %** |

Decode tok/s unchanged by Phase 2a. Output unchanged (state correctness).

### Phase 2b (NOT in this session)

Multifunction `prefill_b8` / `prefill_b16` for first-turn TTFT.
Converter-side work:
- emit MIL functions per batch size with `T=N` shapes for hidden /
  per_layer_combined / KV writes
- engine adds `prefill_bN` function dispatch on input length
- bundle re-deploy with multifunction `.mlpackage`
Estimated 2-3 days. Reference: Qwen3-VL v1.5.0 multifunction commit
`e9feb2a`.

---

## 6. Reproduction steps

### Build chunks (Mac, ≈ 10 minutes)

```bash
# default Conv2d wrapper (matches existing production stateful)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/gemma4-e2b-stateful \
    --hf-dir output/gemma4-e2b/hf_model

# Plan 3 Linear native
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/gemma4-e2b-stateful-linear \
    --hf-dir output/gemma4-e2b/hf_model \
    --linear-projections
```

Both produce 4 `chunk_{1..4}.mlpackage` + `_audit_ane` printout.

### Mac sanity (ANE % + state round-trip)

```bash
python3.12 conversion/sanity_stateful_chunks.py
# → loads 4 chunks, predicts state-round-trip, asserts token_id valid.
```

### Mac latency probe (Plan 3 A/B)

```bash
python3.12 conversion/probe_e2e_linear_latency.py
# → 4-chunk E2E latency comparison Conv2d vs Linear, both W4
```

### Bundle + iPhone deploy

```bash
./scripts/assemble_gemma4_stateful_ab.sh
# → build/gemma4_stateful_ab/{conv,linear}/gemma4_e2b_stateful_chunks/

DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
for v in conv linear; do
  case $v in
    conv)   dst=gemma4-e2b-stateful ;;
    linear) dst=gemma4-e2b-stateful-linear ;;
  esac
  xcrun devicectl device copy to --device $DEVICE \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source build/gemma4_stateful_ab/$v \
    --destination Documents/Models/$dst
done
```

### Xcode run

Scheme env: `LLM_SHOW_EXPERIMENTAL=1` + `LLM_PROFILE_EVERY_STEP=1`.

Picker exposes `Gemma 4 E2B (stateful, MLState)` (Conv2d) and
`Gemma 4 E2B (stateful, Linear projections)`. Both produce
production-equivalent output.

`[Gemma4Stateful] prefill X tok in … | decode Y tok in …` log line on
each generate; on the 2nd+ turn with prefix-extending prompt, expect
`[resumed L=N]` and a much shorter prefill block.

### Reset hooks

- Chat clear → `LLMRunner.resetConversation()` →
  `gemma4StatefulEngine?.resetPersistedState()`.
- Picker switch → engine instance dropped via `loadModel`'s reset block.

---

## 7. cml9 GPU lowprec hint — HOLD

`Sources/CoreMLLLM/CoreMLLLM.swift` working-tree change (3 sites:
vision / video vision / audio model configurations) was pending iPhone
A/B validation. MBA-side Mac probe (`conversion/probe_vision_lowprec_gpu.py`,
not committed yet) returned:

| | A (default) | B (hint=true) | A2 sanity rerun |
|---|---|---|---|
| median | 104.44 ms | 104.83 ms | 105.38 ms |
| Δ | — | +0.38 % (noise) | — |
| `max_abs_diff` | — | **0.0** (bit-exact) | — |

The bit-exact output is decisive: GPU dispatches the same shader in
both cases. cml9 has no fp32→fp16 toggle to flip on this prebuild
`.mlmodelc`; the compiler already chose fp16 accumulation everywhere it
was safe. **Runtime change drops** — no commit.

iPhone re-test would only matter if the device-side compiler picked
a different shader, which is structurally unlikely.

---

## 8. Commits landed

| commit | scope |
|---|---|
| `b5fef64` (2026-04-25 18:19) | `probe(cml9): linear-vs-conv2d + stateful Mac sanity` |
| `72d30b3` (2026-04-25 18:39, MBA) | `probe(cml9): linear vs conv2d 5-layer + attn+w4` |
| `7c9cfea` (2026-04-26 03:45) | `feat(gemma4): stateful runtime wiring + Plan 3 Linear opt-in` |
| `bdac795` (2026-04-26 04:??) | `feat(gemma4): stateful Phase 2a — cross-turn KV reuse via LCP match` |

---

## 9. Open items (next sessions)

- **Phase 2b** (multifunction prefill_bN) — see §5. Reference Qwen3-VL
  v1.5.0 commit `e9feb2a`.
- **lm-split A/B** — `stash@{0}` from earlier. Bundles already pushed to
  `Documents/Models/gemma4-e2b-lmsplit-{baseline,8,16}` on the iPhone.
  Decision: not gated on stateful work; orthogonal probe.
- **Long-context residual probe at T=2048** — `docs/SESSION_2026_04_25_RESIDUAL.md`
  notes the residual stream stays well under fp16 overflow at observed
  context, but real production T re-measure is cheap.
- **Plan 3 full migration** (delete `ane_ops.Conv2dLinear`) — needs
  Qwen3-VL / Qwen3.5 / 4-chunk Gemma 4 converter paths re-validated
  with Linear, then a single PR drops the wrapper class.

---

## 10. Lessons / non-obvious things to remember

- **`devicectl device copy to --destination <file>` is destructive at
  the parent directory level.** Always copy directories.
- **Gemma 4 turn markers are `<|turn>` / `<turn|>`** (literal text),
  Gemma 2/3 chat templates with `<start_of_turn>` BPE-decompose and
  produce degenerate output.
- **Manual prompt construct + `tok.encode(text:)` matches the
  production CoreMLLLM.swift path** (line 1515 region) and works
  reliably; `applyChatTemplate` fails silently when the bundle has no
  `chat_template`.
- **Stateful chunks output diverges slightly from production 4-chunk**
  due to ring-buffer + slice_update vs shift-based KV layout. Within
  acceptable quality, not a Linear-vs-Conv2d effect.
- **MBA-scale (5-layer) probe results don't always extrapolate** —
  the +21 % W4 gap was a synthetic-probe artifact, gone at chunk_1 and
  4-chunk-E2E scale.
- **`MLState` handles bind to specific `MLModel` instances.** Drop
  persisted state on `load()` or model swap, otherwise the handle
  dangles.
