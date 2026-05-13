# Gemma 4 E2B Mac deployment — two paths

Updated 2026-05-09. Both validated end-to-end.

## TL;DR

| path | absolute decode | MTP gain | when to use |
|---|---|---|---|
| **A. Our Mac ANE INT4** | 32 tok/s baseline → 40-43 tok/s MTP-on | **1.26-1.34×** (mostly memory-bound) | shipping target == iPhone (CoreML/ANE), where Metal-GPU is unavailable |
| **B. LiteRT-LM Mac GPU (Metal)** | 143 tok/s baseline (no MTP gain on this backend) | 1.006× (compute-bound) | absolute speed on Mac, prototyping, dev tooling |

Mac GPU is in the *fast / compute-bound* regime where MTP cannot pay off (drafter overhead ≈ per-token cost). Mobile GPUs (Android/iPhone) are in the slow-enough regime where MTP > 2× helps. Mac ANE is in the slow-memory regime where MTP also helps, just at a lower ratio.

## Path A — our Mac ANE CoreML Swift path

Bundle: `output/gemma4-e2b/bundle_diff_logits/`
- `chunk{1,2,3,4}.mlmodelc` (postnorm_attn flavor: fp16 attention + INT4 FFN, K=3 verify, ctx=2048)
- `mtp_drafter.mlmodelc` (HF gemma-4-E2B-it-assistant, INT4 palettize)
- `prefill_chunk{1,2,3,4}.mlmodelc`
- `cos_*.npy`, `sin_*.npy`, embed table, per-layer norm/proj

Smoke run:
```bash
swift run -c release coreml-llm-smoke \
  output/gemma4-e2b/bundle_diff_logits "$PROMPT" 256
```

MTP-on greedy (default speculation gate disabled, force engagement):
```bash
SPECULATIVE_PROFILE=1 LLM_EAGLE3_DISABLE=1 MTP_FORCE_SPECULATE=1 \
  swift run -c release coreml-llm-smoke <bundle> "$PROMPT" 256
```

MTP-off baseline:
```bash
LLM_EAGLE3_DISABLE=1 LLM_MTP_DISABLE=1 \
  swift run -c release coreml-llm-smoke <bundle> "$PROMPT" 256
```

Cold-start measured (per `docs/SESSION_2026_05_09_AWQ_NOGO.md` follow-ups):
- translate: MTP-off 31.95 → MTP-on 42.67 = **1.336×** ✅ (1.30+)
- code_class: MTP-off 31.84 → MTP-on 40.13 = 1.260× (just below 1.30)
- markdown_table: MTP-off 31.79 → MTP-on 41.60 = **1.316×** ✅

Bistability caveat: cold-start window = first 2 iters per ANE cache state. Subsequent iters degrade to ~1.08× until ANE cache is wiped + reprime. See `executorch#16492`-style symptom; per-query cold start matches real user pattern.

## Path B — LiteRT-LM official Mac GPU

Install:
```bash
pip install litert-lm
```

Run with Gemma 4 E2B:
```bash
LITERTLM=~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/*/gemma-4-E2B-it.litertlm

# Benchmark
litert-lm benchmark "$LITERTLM" --backend gpu \
  --enable-speculative-decoding=auto \
  -p 256 -d 256

# Interactive
litert-lm run "$LITERTLM" --backend gpu \
  --enable-speculative-decoding=auto \
  --prompt "$PROMPT"
```

(Auto-download from HF: `--from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm`)

Mac measured (3-iter avg, decode 256 tokens):
- Mac GPU MTP-OFF: 143.6 tok/s
- Mac GPU MTP-ON: 144.5 tok/s
- speedup: 1.006× (negligible — Mac GPU is too fast for MTP to help)
- Mac CPU MTP-OFF: 37.2 tok/s
- Mac CPU MTP-ON: 37.2 tok/s
- speedup: 1.000× (LiteRT-LM Mac CPU MTP not implemented)

LiteRT-LM v0.11.0 release note explicitly markets ">2× faster decode speeds on **mobile GPUs**" — Mac GPU is excluded.

## When to pick which

- **Targeting iPhone production**: Path A. CoreML/ANE pipeline ships via App Store; LiteRT-LM iOS runs but distribution is C++ library.
- **Targeting Mac as dev/desktop**: Path B (143 tok/s vs 40 tok/s, ~3.5× faster absolute).
- **Need MTP-on benchmarks for paper / talk**: Path A on Mac (only Mac path that benefits from MTP).
- **Need Apple-native multimodal (image / audio / video)**: Path A (E4B multimodal already shipped, see `project_e4b_multimodal_shipped.md`).

## Open follow-ups

- iPhone empirical: `LiteRT-LM iPhone GPU` vs `our iPhone ANE` decode tok/s, MTP gain.
- Per `project_drafter_structurally_dead.md`, LiteRT extracted drafter is dead on our W4A16. Path B uses Google's co-trained drafter natively, so this isn't blocking.
- Drafter retraining (Path B in roadmap) deferred — see `project_mtp_pivot.md`. Realistic lift from current 1.26× on code is +0.01-0.04× per latest analysis (small tail).
