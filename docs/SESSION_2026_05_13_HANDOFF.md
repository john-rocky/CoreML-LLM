# Next-Session Handoff Prompt — iPhone Gemma 4 E2B 1.5× via L12 Subset LM Head

Paste the following block into the next Claude Code session as the FIRST
user message. It primes the assistant with all relevant state and gives a
concrete starting task.

---

## Prompt to paste

> iPhone Gemma 4 E2B の lossless 1.5× 高速化を継続する。前セッション (2026-05-13)
> までで以下が確定:
>
> 1. **訓練禁止**。Mac で empirical に 1.5× を出してから iPhone push する方針。
> 2. **16 lever 検証完了** — 詳細 `docs/IPHONE_SPEEDUP_LEVER_INVENTORY_2026_05_13.md`。
>    現状 iPhone 1.16× lossless free-form が ceiling。FLy top-K=16 + L5 async +
>    INT4 drafter + ANE compute + never-bail で shipped 済み。
> 3. **残る唯一の証拠ベース 1.5× 路線 = L12 Subset LM head**:
>    - chunk4 の 600M-param LM head matmul (~7-10ms iPhone) をスキップ、
>      Swift 側で 1024-候補 sparse matmul に置換
>    - Math 投影: chunk4 -7ms → cycle 47→40ms → 1.38×、+L5 → 1.85×
>    - Mac empirical 確認済: chunk4 full 11.4ms → subset 8.4ms (-3ms = -26%)
> 4. **L12 Python side 完了**, Swift side 残:
>    - `output/gemma4-e2b/chunks_subset/chunk4_subset.mlmodelc` (311 MB built)
>    - `output/gemma4-e2b/lm_head_fp16.bin` (768 MB extracted, V=262144 × H=1536 fp16 row-major)
>    - `conversion/models/gemma4_swa_chunks.py` に `SWAVerifyChunk4Subset` class
>    - `conversion/build_chunk4_subset.py` (`--ctx 2048` 必須)
>    - `conversion/extract_lm_head.py`
>
> **やること**: Swift 統合 (8-12h 想定):
> 1. `ChunkedEngine.swift` で `chunk4_subset.mlmodelc` を opt-in load
>    (env `MTP_SUBSET_LM_HEAD=1` 等)
> 2. `lm_head_fp16.bin` を起動時に `Data` または `MLMultiArray` でロード
> 3. `verifyCandidates` の variant: `verifyCandidatesSubset(tokens:candidateIds:)`
>    - 既存 chunks 1-3 + chunk4_subset → `normed_hidden`
>    - Swift で gather: `lm_head[candidate_ids] (M, H)`
>    - sparse matmul: `normed_hidden @ gathered.T` via Accelerate vDSP_mmul
>    - argmax over M candidates → 候補 index → token ID
> 4. `MtpSpeculativeEngine` で候補セット構築:
>    - drafter top-K (top-32 × K-1 = 64 tokens、すでに drafterTopKByStep に集めてる)
>    - PLD top-K via `PromptLookupDraft.propose(history:, ngramSize:2/1, maxDraftLen:8)`
>    - 直近 emit 30 tokens
>    - Top-N 高頻度 English tokens (~900、ハードコード or `.bin` file)
>    - dedupe + cap 1024
> 5. Low-confidence fallback: max sparse logit < 閾値 (≈15.0) なら full chunk4 再実行
> 6. Mac smoke で **subset argmax == full argmax** (lossless 検証)
> 7. iPhone deploy + bench: `Say yes 30 times.` (yes-yes) と `What is your favourite hobby and why?`
>    (free-form) で UI tok/s 計測
>
> **読む順**:
> 1. `docs/IPHONE_GEMMA4_SPEEDUP_MASTER_2026_05_13.md` (全体)
> 2. `docs/SUBSET_LM_HEAD_PROGRESS_2026_05_13.md` (L12 詳細)
> 3. `docs/IPHONE_SPEEDUP_LEVER_INVENTORY_2026_05_13.md` (試行履歴)
>
> **制約再確認**:
> - 訓練しない
> - 出力品質を Mac で確認してから iPhone push
> - 「成果出るまで止めるな」 = math 上 1.5× 投影できるまで作業継続
> - iPhone test は Mac で gain 確証後のみ
>
> 現状: `git status` で modified ファイル確認、iPhone 17 Pro に L5+FLy-16+never-bail
> 版が deploy 済み。

---

## Quick-start commands for the next session

```bash
# 1. Inspect current state
cd /Users/majimadaisuke/Downloads/workspace/CoreML-LLM
git status
ls -la output/gemma4-e2b/chunks_subset/
ls -la output/gemma4-e2b/lm_head_fp16.bin

# 2. Rebuild chunk4_subset if needed (~3 min)
PYENV_VERSION=lama-cml python conversion/build_chunk4_subset.py --K 3 --ctx 2048 \
  --output output/gemma4-e2b/chunks_subset
xcrun coremlcompiler compile output/gemma4-e2b/chunks_subset/chunk4_subset.mlpackage \
  output/gemma4-e2b/chunks_subset/

# 3. Mac smoke test (after Swift integration done)
SPECULATIVE_PROFILE=1 LLM_MTP_ENABLE=1 MTP_FLY_TOPK=16 MTP_SUBSET_LM_HEAD=1 \
  .build/release/coreml-llm-smoke output/gemma4-e2b/bundle_mac_logits \
  "What is your favourite hobby and why?" 120

# 4. iPhone deploy
cd Examples/CoreMLLLMChat
xcodebuild -project CoreMLLLMChat.xcodeproj -scheme CoreMLLLMChat \
  -configuration Release -destination 'platform=iOS,id=00008150-0018713A0207801C' \
  -derivedDataPath /tmp/CoreMLLLMChat-build build
xcrun devicectl device install app --device A6F3E849-1947-5202-9AD1-9C881CA58EEF \
  /tmp/CoreMLLLMChat-build/Build/Products/Release-iphoneos/CoreMLLLMChat.app

# 5. Push subset chunk4 + LM head bin to iPhone (if Swift loads from app sandbox)
# ... add to push_gemma4_e2b_bundle.sh
```

## Risk register for Swift integration

| Risk | Mitigation |
|---|---|
| Swift sparse matmul too slow (>5 ms) | Profile vDSP_mmul vs BNNS; try INT4-quantized buffer |
| Candidate set misses target argmax (>5%) | Add fallback to full chunk4; expand candidate top-N |
| chunk4_subset's normed_hidden inaccurate vs original | Mac smoke: compare subset argmax vs chunk4's argmax token-by-token on hobby prompt; require 100% match |
| iPhone savings < 5 ms | Accept actual measurement; if <3 ms, L12 not worth shipping |
| LM head buffer too big for iPhone (768 MB) | Use INT4 palettize buffer (~200 MB) with Swift dequant in gather step |
| Async L5 + subset interfere | Disable async first, validate subset alone, then re-enable |

## Decision tree if L12 fails

```
L12 Swift integration complete?
├─ YES → iPhone bench
│   ├─ ≥1.5× lossless: ship as default
│   ├─ 1.3-1.5×: ship as opt-in, document as best-effort
│   └─ <1.3×: revert, document why (likely iPhone savings < 3 ms)
└─ NO → blocked
    ├─ Swift sparse matmul perf: try BNNS, AMX, Metal compute shader
    ├─ Candidate coverage: empirically tune set composition
    └─ Last resort: training (Path B Self-Distillation MTP arxiv 2602.06019, 1 GPU-week)
```
