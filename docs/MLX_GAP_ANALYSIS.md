# MLX vs CoreML-LLM 総合ギャップ分析

**Date:** 2026-04-19
**Source:** `ml-explore/mlx` upstream snapshot (cloned 2026-04-19) vs current `Sources/CoreMLLLM`
**Method:** 3 並列 Explore エージェントによる両コードベース全走査
**Goal:** **1 秒でも高速化する**（tok/s 数値天井ではなく、体感応答時間の短縮が唯一の評価軸）
**Purpose:** 上記ゴールの下で、秒数 ROI が最大のカードを特定する。`docs/FUNDAMENTAL_UNTRIED.md`, `docs/MAC_FIRST_EXECUTION_PLAN.md`, `docs/BASELINE_SPEED_AUDIT.md` の裏付け資料。

---

## 一行結論

> 目標を「tok/s 上限」ではなく「**1 秒でも縮める**」に置き換えると、秒数/工数の最大 ROI は **GPU prefill (MLX-Swift, `PRIORITY_ROADMAP.md` item #27)**: TTFT 13s → ~1s、**1 生成ごとに −12 秒**の単独最大カード、工数 7–10 日。次点は **prefix / system-prompt KV cache** (turn 2+ の TTFT をさらに圧縮、0.5–1 日)。decode 側の秒短縮は **Metal-LLM 全面移植** が最大（典型 200-tok 応答で −5〜10 秒）だが数週〜数ヶ月。**ANE Python wiring (#1/#3)** は D1b / PR #17 / chunk consolidation の track record が示す通り期待値 ±1〜2 秒 / iPhone 遠征 1 回分で、TTFT カードを使い切った後の中小規模な賭け。Metal-LLM は「tok/s を破る唯一のルート」ではなく「**decode の秒を最大に削る手段**」と位置づけ直す。

### 秒数 ROI ランク表（1 秒ゴール基準、典型 prompt=2K / 出力 200 tok で見積）

| 順位 | カード | 節約秒数 | 工数 | リスク | 種別 |
|---|---|---|---|---|---|
| 1 | **GPU prefill (MLX-Swift)** | TTFT **−12 s / gen** | 7–10 日 | M（MLX-Swift iOS 前例薄） | TTFT |
| 2 | **Prefix KV disk cache** 走行内有効化 | 再生成で TTFT **−3〜12 s** | 0.5–1 日 | L | TTFT |
| 3 | **System-prompt KV cache** (turn 2+) | turn 2+ TTFT **−2〜5 s** | 0.5 日 | L | TTFT |
| 4 | **Metal-LLM decode 全面移植** | 200 tok 応答で **−5〜10 s** | 週–月 | M-H（移植規模 + 電力/配布） | decode |
| 5 | **ANE #3 sliding padding drop** | 200 tok で **−0.5〜1 s** | 30 LoC + iPhone 1 回 | M | decode |
| 6 | **ANE #1 softmax swap** | 200 tok で **−1〜2 s** 想定（ただし PR #17 で 5.5× 遅延の近縁失敗） | 2 LoC + iPhone 1 回 | M-H | decode |
| 7 | **出力バッファ pooling + prewarm** | **−0.1〜0.5 s / gen** | 0.5–1 日 | L | ランタイム |
| 8 | **`reshapeFrequency=.infrequent` hint** | 1 step **−0.5 ms** ≈ 200 tok で **−0.1 s** | 0.5 日 | L | ランタイム |

**運用則**: 上から順に実装。**1〜3 は独立で合算可能**、TTFT だけで最大 **−15〜20 秒/gen** が現実的射程。decode 側（4 以降）はここを埋めた後の追加施策。5/6 は Mac parity ≠ iPhone 成功の事例が 3 回あるため、iPhone 遠征 1 回をコストに計上して判断。

---

## 0. 根本的アーキテクチャ差分（これが全てを決めている）

| 観点 | MLX | CoreML-LLM |
|---|---|---|
| 実行モデル | C++コア + Metal lazy graph + ランタイム融合 (`compile()`) | CoreML 静的グラフ + Swift オーケストレーション |
| カーネル所有権 | 自前 Metal カーネル 50+ 個（`mlx/backend/metal/kernels/*.metal`） | **Metal カーネル 0 個**（全て ANE Compiler 任せ） |
| 融合戦略 | eval 時に DAG を辿って kernel fusion + splat | mlpackage 変換時に完全静的、以後融合不可 |
| 形状 | 動的 | 完全静的（chunk 境界で固定） |
| ディスパッチ遅延 | MTLCommandQueue（〜10–50μs） | ANE DART 下限 **2.3 ms/dispatch** |

CoreML-LLM は「CoreML に乗った瞬間に MLX が持つ最適化余地をすべて失う」設計。`docs/FUNDAMENTAL_UNTRIED.md` で Metal-LLM Phase 3 が critical path とされているのは正しい。

---

## 1. カーネル面で MLX が持っていて CoreML-LLM に存在しないもの

### 1.1 SDPA（注意計算）

MLX 側:
- `sdpa_vector` — decode、qL ≤ 8、**GQA 組込、causal、bool/float mask、attention sinks**
- `sdpa_vector_2pass_1/2` — N > 1024 の long-context split-K、数値安定 logsumexp
- `sdpa_full` — prefill、head_dim ∈ {64, 80, 128}、タイル bq × bk 自動
- `steel_attention_nax` — M4+ Pro/Max NAX 専用、bq=64 / bk=32

CoreML-LLM 側:
- 注意は `matmul + softmax + matmul` の 3op 分離。`docs/INTEGRATED_ROADMAP.md` #1 の ane_softmax 置換すら未着手。
- 融合 SDPA なし、vector パス/prefill パスの分離なし、NAX 非対応。

### 1.2 Quantized Matmul

MLX 側: int4 / int8 / **fp4(e2m1) / fp8(e4m3) / fp8(e8m0) / mxfp / nax**、group = 32/64/128、QMV / QMM / QVM、**split-K、gather、segmented、quad variant**。

CoreML-LLM 側: CoreML 変換時に `int4 palettized, group=32` 1 種のみ。実行時可変量子化なし、fp4/fp8 皆無、gather/segmented matmul 皆無。

### 1.3 残りの融合カーネル

| Op | MLX (Metal) | CoreML-LLM |
|---|---|---|
| RMSNorm | 1 kernel（fp16 → fp32 reduction、weight scale 融合） | CoreML 分解。音声系は **GPU RMSNorm 壊れて CPU 実装**（`AudioProcessor`） |
| LayerNorm | 1 kernel（mean/var/affine 一発） | CoreML 分解 |
| RoPE | traditional/modern、T=1 fast path、freqs 事前計算入力対応 | CoreML 内で生成（T=1 特化なし） |
| Softmax | block/looped 2 パス、precise-fp32 reduction | ANE softmax は未最適（#1 で差し替え予定） |
| GEMV | simdgroup tile、safe/unsafe variants | 存在せず |
| Steel GEMM | 全デバイス tile 自動（g/p/d 分類）、splitk/fused/gather/segmented | 存在せず |

---

## 2. KV Cache で決定的に負けている

| 機能 | MLX | CoreML-LLM |
|---|---|---|
| 保持形式 | Metal resident buffer | MLMultiArray × 4（sliding/full × chunk1,2） |
| 量子化 | なし（MLX も非対応） | なし（INT8 試行 → ANE 内部で fp16 に dequant、0 ゲイン） |
| Paged | **なし**（両者同じ欠点） | なし |
| Prefix reuse | ユーザ実装任せ | ディスク冷キャッシュのみ、走行中 reuse なし |
| Padding | 無駄なし | **head_dim 256 → 512 パッド**（28 層、+5–10% 放置中） |
| Sliding window sinks | `sinks` buffer 正式サポート | 未対応 |
| Stateful API | 静的グラフなので自然 | `MLState` 試行 → ANE error -14 で頓挫 |

MLX にもない paged KV / prefix-reuse-tree は両者共通の弱点だが、**sinks と padding は即座に取れる差**。

---

## 3. グラフ最適化・コンパイル

MLX:
- `compile()` で最大 depth = 11、arrays = 24 の DAG を単一 Metal kernel に融合。
- JIT 経路（unary / binary / copy / softmax / reduce）＋ 事前コンパイル経路（SDPA / RoPE / quant）の使い分け。
- `function_constant` で条件分岐をカーネル内で潰す。
- `custom_kernel.cpp` でユーザ定義 Metal を VJP / vmap 対応のまま差し込める「脱出ハッチ」。

CoreML-LLM:
- 融合は `coremltools` 変換時の一度きり、ランタイム改変不可。
- CoreML がサポートしない融合（ex: fused QKV、QK^T as Conv1x1、RMSNorm + Linear）は **Python 側で手動 wiring**（`docs/D1_WIRING_PATCH_PLAN.md`）するしかない。しかもまだ landed していない。
- カスタムレイヤ脱出ハッチなし（CoreML Custom Layer は ANE を諦める＝意味がない）。

---

## 4. メモリ・実行スケジューリング

MLX:
- **Residency Set**（macOS 15+ / MTL3）で重みを GPU VRAM pinning → 7B 規模で効く。
- Allocator LRU プール + GC 閾値 + resource_limit（MTL オブジェクト数）。
- Fast fence（CPU-GPU 共有 atomic counter、kernel 境界で sleep 不要）。
- 複数 stream を同一 queue で async 実行。

CoreML-LLM:
- **Residency/pinning なし**（CoreML が内部で持っているが制御不可）。
- **Chunk 直列化**（1 チャンク完了 → 次チャンク）。`docs/BASELINE_SPEED_AUDIT.md` で **これが最大のボトルネック** と明記（1.5–2.0× 天井）。パイプライニング未実装。
- dispatch 重畳不可（ANE DART が 2.3 ms 床を強制）。

---

## 5. Benchmark / 計測インフラ

MLX: `benchmarks/python/sdpa_bench, rms_norm_bench, rope_bench, gather_qmm_bench, segmented_mm_bench, compile_bench` — 40 iter × 5 warmup、融合 ON/OFF の統計的比較が標準装備。

CoreML-LLM: `eval/` と `Sources/accept-rate-bench` は accept-rate と wall-clock tok/s 中心。**Op 単位の micro-bench が存在しない**（どの op が遅いかの一次資料がない）。MLX は 1 op 単位で速度が見える文化。

---

## 6. その他の差

| 領域 | MLX | CoreML-LLM |
|---|---|---|
| IO | safetensors + GGUF 直読（llama.cpp 互換） | HF DL → coremltools 変換必須、GGUF 非対応 |
| 量子化実行時変更 | `to_quantized()` で動的 | 変換時固定 |
| 多重精度 | fp32 / fp16 / bf16 / complex64 | fp16 中心、bf16 取り扱い不安定 |
| sampling | MLX 外（利用側実装） | `ArgmaxInModel`, softcap, 投機デコード有り（**CoreML-LLM 唯一の優位**） |
| Speculative | primitives なし | MTP / CrossVocab / DrafterUnion 実装済（**CoreML-LLM の強み**） |
| Multimodal | なし | Image / Video / Audio processor 実装（**CoreML-LLM の強み**） |

---

## 7. 優先度別「盗むべきもの」

### S（最大 ROI、Metal-LLM Phase 3 の核）

1. **SDPA を MLX `sdpa_vector` + `sdpa_vector_2pass` 移植**。GQA / causal / mask / sinks 全部入り。decode 単体で ANE 全体 > Metal SDPA になる可能性がこのカーネルに集約されている。
2. **Residency Set による重み GPU pinning**（`mlx/backend/metal/resident.cpp`）。Metal に落とす以上、ここでサボると PCIe dominated になる。
3. **Kernel fusion policy**（`mlx/compile.cpp` の depth = 11 / arrays = 24 の fuse ルール）。Metal-LLM forward を書くときの設計原則そのもの。

### A（取れる差、数%単位で積める）

4. **RMSNorm / LayerNorm 1-pass Metal**（`rms_norm.metal`）。音声系 CPU 実装の置き換えとしてもすぐ効く。
5. **RoPE T=1 fast path**（`rope.metal` の single-element variant）。decode の seq=1 で効く。
6. **Steel GEMM tile auto-tune** の方針（device classification 'g'/'p'/'d'）。M-series ごとの tile 切替。
7. **Quantized QMV + fused bias**（`quantized.metal` の `axpby` variant）。
8. **fp8 / fp4 量子化** の導入検討（CoreML ではムリ、Metal-LLM ならできる）。

### B（小さいが確実）

9. **function_constant による条件分岐潰し**パターン（SDPA の `has_mask / do_causal / bool_mask / float_mask / has_sinks / align_Q / align_K`）。
10. **Fast fence**（`fence.cpp` macOS 15+ atomic counter）。CPU-GPU 同期遅延を消す。
11. **Benchmark harness** を MLX 形式（40 iter × 5 warmup、op 単位）にして **どこが遅いかの定量化**。

### ただし注意

- **Paged KV / prefix-tree reuse は MLX にも無い**。両者同時に埋めるべき共通の谷。
- MLX は **batch > 1 前提**の最適化（gather / segmented mm）が多い。CoreML-LLM は batch=1 固定なので **そのまま効くとは限らない**。
- CoreML-LLM が持っている **speculative / multimodal / ANE 接続** は MLX 側にない資産。Metal-LLM に移る際もここは保持すべき。

---

## 関連 docs

- `docs/FUNDAMENTAL_UNTRIED.md` — Metal-LLM pivot の戦略論
- `docs/MAC_FIRST_EXECUTION_PLAN.md` — Mac 側検証を先にやる方針
- `docs/BASELINE_SPEED_AUDIT.md` — chunk 直列化が最大ボトルネックの根拠
- `docs/INTEGRATED_ROADMAP.md` — Python 側 wiring 最適化（#1, #3, #5, #11 等）
- `docs/LITERT_RUNTIME_ANALYSIS.md` — 比較対象である LiteRT 側の構造
- `docs/ANE_OPTIMIZATION_SURVEY.md` — ANE を諦めない最適化の棚卸し
