# Qwen3.5-0.8B Phase 4 — Next Session Handoff

Paste the "Prompt to start the session" block below into a new Claude Code session in this repo. Memory (`~/.claude/projects/-Users-majimadaisuke-Downloads-CoreML-LLM/memory/`) auto-loads the Phase 2a/2b/3ab summaries, so you don't need to re-explain context.

---

## Prompt to start the session

> Qwen3.5-0.8B (Gated DeltaNet) のCoreML化、Phase 4を進めたい。Phase 0-3bは完了済み、`research/qwen3_5-phase2a` ブランチと PR #109 に全資料がある。memoryの `qwen35_phase*_result.md` を参照してから着手して。
>
> まず最初に、現状レビュー:
> 1. `research/qwen3_5-phase2a` が origin/main にマージされてるか確認
> 2. されていなければ merge してから作業開始 (新ブランチは origin/main ベース)
> 3. されていれば fresh branch を main から切る
>
> Phase 4の順序:
> - (4a) 24層 fp16 フル converter: `conversion/test_qwen3_5_stack_trace.py` を24層に拡張 + embed_tokens + final RMSNorm + lm_head。parity cos ≥ 0.998 vs HF logits (Phase 1 oracle `qwen3_5_reference_logits.pt` と比較)、ANE placement ≥ 99% を確認。
> - (4b) INT4 palettization: `conversion/models/gemma4.py` のレシピを移植。projection weights (in_proj_qkv, in_proj_z, gate/up/down, q/k/v/o_proj) を palettize。parity cos ≥ 0.995。
> - (4c) V-weight reorder trick (llama.cpp PR #19468): 変換時に V重みを reorder して runtime `repeat_interleave` を省く。0.8B は num_v/num_k = 1 なので no-op の可能性あり、要確認。
> - (4d) HF upload: `mlboydaisuke/qwen3.5-0.8b-coreml`、text-only タグ。
> - (4e) Swift側: `Sources/CoreMLLLM/` に Qwen3.5 ローダ追加、ModelDownloader エントリ、iOS picker case、README 行。実機ベンチはユーザーがiPhone 17 Proで。
>
> 4aから順に。各ステップ完了時に 1 PR (セッションの慣例)。INT4とHF upload (4b, 4d) はユーザー承認必須。

---

## What's already proven (do not redo)

| Phase | Result | File |
|---|---|---|
| 1 | prefill vs recurrent cos=0.999998 | `conversion/qwen3_5_reference_oracle.py` |
| 2a | decode step 100% ANE, cos=1.000000 | `conversion/test_qwen3_5_decode_trace.py` |
| 2b | chunked prefill 99.91% ANE at seq=2048, cos=1.000+ | `conversion/test_qwen3_5_prefill_trace.py` |
| 3a | full_attention 100% ANE, cos=1.000 | `conversion/test_qwen3_5_full_attention_trace.py` |
| 3b | 4-layer stack 99.63% ANE, cos=1.000003 | `conversion/test_qwen3_5_stack_trace.py` |

The `DecoderLayer` and `MiniStack` classes in `test_qwen3_5_stack_trace.py` are the starting point for the 24-layer converter — just extend `NUM_LAYERS=4` to `24`, add `embed_tokens + final_norm + lm_head`, capture parity with `qwen3_5_reference_logits.pt`.

## Trace gotchas (already encountered, don't rediscover)

- Use fixed `self.S` not `B, S, H = x.shape` unpacking — coremltools aten::Int cast trap
- Precompute causal mask as `register_buffer`, not `torch.triu` inside forward
- Drop `tensor.dtype` and `.to(dtype)` inside forward
- 5D matmul `(B, H, NC, CS, CS)` → CPU; reshape to 3D `(B*H*NC, CS, CS)` + `torch.bmm` → ANE
- HF's 63-iter Gauss-elim inner loop → replaced with Neumann iteration `T_{k+1} = I + L @ T_k` (CS steps, numerically stable; repeated-squaring `(I+L)(I+L²)...(I+L³²)` was tried and broke parity due to intermediate L^32 magnitudes)
- `mb.cumsum` always lands on CPU — expect 1 cumsum per linear_attention layer, doesn't contaminate neighbors
- Torch 2.11 on Apple Silicon works but prints "not tested" warning — ignore

## Known constraints (from CLAUDE.md / user memory)

- **Do not include "claude" in commit messages or committer**
- **Do not commit CoreML models or build artifacts**
- **iOS builds are tested by user on iPhone 17 Pro** — don't attempt to build in CI
- Code comments and UI text in English
- Research branch pattern: one session = one PR off origin/main

## Fallback trigger (still active)

If Phase 4a full-stack parity ever drops below cos≥0.995 and the cause is not trivially fixable, **pivot to Qwen3-0.6B** (plain transformer, 80% success probability for full integration). Research unknown resolved. The Gated DeltaNet primitives are proven, so failure at 4a would point to stacking / weight-mapping bugs, which should be debuggable rather than fundamental.

## Resources

- PR #109: https://github.com/john-rocky/CoreML-LLM/pull/109
- Roadmap: `docs/QWEN35_ROADMAP.md`
- Phase 1 oracle (fp16 logits, 128MB, 10 prompts): `conversion/qwen3_5_reference_logits.pt`
- HF text-only loader: `Qwen3_5ForCausalLM(config=Qwen3_5TextConfig.from_pretrained("Qwen/Qwen3.5-0.8B"), torch_dtype=torch.float32)` — no patches needed
- Gemma 4 INT4 palettization reference: look at existing `conversion/models/gemma4.py` and `conversion/convert.py`
- llama.cpp Qwen3.5 conversion: https://github.com/ggml-org/llama.cpp/pull/19468 (V-weight reorder)
