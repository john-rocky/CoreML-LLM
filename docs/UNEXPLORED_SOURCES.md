# Unexplored Sources — Deep Dive (2026-04-14)

既存ドキュメント全体を横断精査し、まだ当たっていない有望なソースを特定。さらに論文の中身、PC 向けローカル推論エンジンの実装パターン、Google LiteRT-LM のソースコード、Apple の最新研究まで踏み込んで、**具体的に何が使えるか**を評価した。

---

## Part A: PC ローカル推論エンジンからの具体的知見

### A.1 KV キャッシュ量子化 — llama.cpp TurboQuant パターン

llama.cpp は KV キャッシュ自体を量子化する TurboQuant を実装している。

- **K は量子化に強く、V は弱い**（非対称）。K=3bit / V=4bit が品質/メモリの最適点
- **Residual Window**: 直近 128–256 トークンは FP16 のまま、古いコンテキストだけ量子化
- Llama 3 70B @ 128K で KV メモリ 27.3 GB → 5.6 GB（80% 削減）

**ANE への適用**: 現在 KV キャッシュは全て FP16 MLMultiArray。Sliding Window (512) の直近 128 位置を FP16 のまま、残りを INT8 に量子化すれば KV メモリ ~50% 削減。`EmbeddingLookup.swift` に既にある vDSP INT8→FP16 デコード基盤をそのまま転用可能。Global Attention (8192) 層でも同様に適用すれば、8K コンテキストのメモリ圧を大幅に緩和。

- **参考**: arXiv (TurboQuant)、llama.cpp `ggml-quants.c`

### A.2 重要度ベース混合精度 — llama.cpp IQ 量子化

llama.cpp の IQ (Importance Matrix) 量子化は、キャリブレーションデータで各重みの出力への影響度を計測し、影響度の高い重みにより多くの量子化レベルを割り当てる。

- `llama-imatrix` で重要度行列を生成 → 量子化時に参照
- IQ2_XXS (2.06 bpw) が同じビット幅の標準 Q2_K を品質で上回る
- **K-quants**: ブロック内でサブグループごとに異なるビット配分

**ANE への適用**: 現在 INT4 palettization は全レイヤー均一。coremltools の `OpPalettizerConfig` は per-op config をサポートしているため、以下の混合精度が可能:

```
embedding / lm_head 周辺: 6-bit (品質影響大)
attention projection (q/k/v/o): 4-bit (現状維持)
FFN (gate/up/down): 4-bit or 3-bit (品質影響小)
```

MLX の WWDC 2025 セッションでも同じパターン（embedding/lm_head を 6-bit、残りを 4-bit）が推奨されている。

- **参考**: llama.cpp `examples/imatrix/`, MLX convert docs

### A.3 System Prompt KV キャッシュ永続化 — llama.cpp / vLLM / Ollama 共通パターン

3つの主要エンジン全てが実装している最適化: system prompt の KV 状態を保存し、会話ターンをまたいで再利用。

- **llama.cpp**: `--prompt-cache` フラグでバイナリ保存、プレフィックスマッチで再利用
- **vLLM**: ハッシュベースの Automatic Prefix Caching — トークン列のハッシュで KV ブロックを特定
- **Ollama 0.19+**: KV キャッシュがセッション間で永続化、インテリジェントなチェックポイント + eviction

**ANE への適用**: IOSurface-backed CVPixelBuffer は既にメモリ永続。system prompt prefill 後に `currentPosition` を保存し、次ターンでは prefill をスキップして保存位置から decode を再開するだけ。実装量 ~20 行。TTFT を会話2ターン目以降で 2–5× 改善可能。

### A.4 PagedAttention の本質 — vLLM

vLLM の PagedAttention は KV キャッシュを 16 トークン単位のブロックに分割し、ページテーブルで管理。メモリ無駄は <4%（連続確保だと 60–80%）。

**ANE への直接適用は不可**（ANE は連続 [1,C,1,S] テンソルを要求）。ただし「遅延確保」のコンセプトは有用: 8192 位置の global attention バッファを起動時に全確保せず、コンテキスト成長に応じて段階的に拡張（memcpy で大きなバッファに移動）。メモリ圧が高い iPhone で有効。

### A.5 N-gram 投機的デコーディング — llama.cpp 実装

llama.cpp は `--spec-ngram` フラグでドラフトモデル不要の投機的デコーディングを実装。

- 生成済みトークン履歴から n-gram パターンマッチ
- `--spec-ngram-size-n 12`（検索ウィンドウ）、`--spec-ngram-size-m 48`（ドラフト長）
- CPU のみ、追加メモリゼロ

**ANE への適用**: → Part B.1 Prompt Lookup Decoding として詳述。

### A.6 SRAM 32 MB 崖 — Orion 論文の実測データ

Orion 論文（arXiv 2603.06728）の ANE マイクロアーキテクチャ実測:

| 指標 | 値 |
|------|------|
| ANE コア数 | 16 |
| オンチップ SRAM | 32 MB |
| 実効 FP16 スループット | ~19 TFLOPS |
| ディスパッチオーバーヘッド | ~0.095 ms/call |
| SRAM 超過時スループット低下 | ~30% |
| 最適 op グラフ深度 | 16–64 ops で 94% ANE 利用率 |

**具体的示唆**: 各チャンクの per-prefill-token ワーキングセットを計算し 32 MB 以内に収める。Gemma 4 の hidden_dim=2560 の場合、attention の Q*K^T 中間テンソルだけで `2560 * seq_len * 2 bytes * num_heads`。`prefillN` パラメータを SRAM 予算から逆算してチューニングすべき。

---

## Part B: 論文 Deep Dive — 具体的適用性評価

### B.1 Prompt Lookup Decoding — ゼロコスト即日実装

**リポジトリ**: github.com/apoorvumang/prompt-lookup-decoding
**HuggingFace 統合**: PR #27775、`PromptLookupCandidateGenerator` クラス

#### アルゴリズム（実装 ~20 行）

```python
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    for ngram_size in range(max_ngram_size, 0, -1):   # 最大 n-gram から試行
        ngram = input_ids[0, -ngram_size:]             # 末尾 N トークン
        windows = input_ids.unfold(dim=1, size=ngram_size, step=1)
        matches = (windows == ngram).all(dim=2)
        match_indices = matches.nonzero()[1]
        for idx in match_indices:
            start = idx + ngram_size
            end = start + num_pred_tokens
            if end <= input_ids.shape[1]:
                return input_ids[0, start:end]         # マッチ後の続きを返す
    return torch.tensor([])
```

#### ベンチマーク

| タスク | 高速化 |
|--------|--------|
| 要約 | 2.4× |
| 文脈 QA | 2.4× |
| コード補完 | 最大効果 |
| 自由会話 (Turn 1) | 効果なし |

#### ANE 適用性: **最高 — 即日実装可能**

Swift で ~50 行。CPU で n-gram マッチ → 候補トークン列を生成 → ChunkedEngine の verify モードで一括検証。モデル変更不要、追加メモリゼロ、失敗時は通常デコードにフォールバック。

**MTP ドラフターと排他ではなく併用可能**: まず Prompt Lookup でマッチを試み、マッチなしなら MTP ドラフターに fallback。

### B.2 DistillSpec — EAGLE-3 Blocker 1 への直接対処

**論文**: arXiv 2310.08461 (ICLR 2024, Google DeepMind)

#### 蒸留の具体的手法

4 つの divergence 関数を体系的に評価:
- Forward KL: `KL(p_target || p_draft)`
- Reverse KL: `KL(p_draft || p_target)`
- Jensen-Shannon Divergence (JSD)
- Total Variation Distance (TVD)

**重要な発見: 最適な divergence はタスクとデコーディング戦略に依存。**
- 要約 (greedy): JSD + mixed on-policy データが最良
- 数学 (greedy): FKL + draft-only on-policy データが最良
- サンプリング (T=1): RKL + target 生成データが最良

**On-policy データ生成が鍵**: ドラフトモデル自身で訓練データを生成（off-policy より常に優れる）。

#### 高速化実績

- 標準 speculative decoding 比 **10–45% 高速化**
- 未知タスクへの転移: 平均 26% 高速化
- Model garden シナリオ: 6–10× レイテンシ削減（XSum 6.4×、GSM8K 10.7×）

#### オープンソース

公式コードなし。後続研究 **AdaSPEC** (github.com/yuezhouhu/adaspec) が関連実装を公開。

#### ANE 適用性: **中〜高 — 訓練フェーズのみ GPU 必要**

蒸留は GPU 上の訓練時テクニック。蒸留済み重みは通常の CoreML エクスポートで ANE に載る。ただし **EAGLE-3 → MTP へのピボットが進行中**であり、MTP ドラフターが高 acceptance を示せば DistillSpec は不要になる可能性。MTP の acceptance 率が低かった場合の保険として調査価値あり。

### B.3 QuIP# — Sub-4-bit 量子化の再挑戦

**論文**: arXiv 2402.04396 (ICML 2024)
**リポジトリ**: github.com/Cornell-RelaxML/quip-sharp

#### Incoherence Processing の具体的アルゴリズム

1. ランダム符号ベクトル `SV`, `SU` を {+/-1}^n で生成
2. Fast Walsh-Hadamard 変換で重み行列を incoherent 基底に変換:
   ```
   W_hat = Had(diag(SU) * Had(diag(SV) * W^T)^T)
   ```
3. E8 格子 codebook（65,536 エントリ、8 次元）で量子化
4. 推論時は逆変換: `y = Had(SU * decompress_multiply(W_hat, C, Had(SV * x)))`

#### 品質（WikiText2 perplexity）

| モデル | ビット幅 | QuIP# PPL | FP16 PPL |
|--------|----------|-----------|----------|
| Llama 2 70B | 2-bit | 4.16 | 3.32 |
| Llama 2 70B | 3-bit | 3.56 | 3.32 |
| Llama 2 13B | 2-bit | 5.74 | — |

**3-bit QuIP# が 4-bit QuIP# を上回る**（文献初の結果）。

#### ANE 適用性: **低 — 直接実装は非現実的**

- Hadamard 変換は CoreML/ANE のネイティブ op にない（行列乗算で代替すると効率ゼロ）
- E8 codebook lookup は gather/scatter op が必要（ANE は gather 非対応）
- 8 次元ベクトル量子化は Conv2d レイアウトと非互換

**代替案**: coremltools の `OpPalettizerConfig(nbits=2)` によるネイティブ 2-bit palettization を再テスト。失敗済みの W2 はナイーブな k-means だが、**重要度ベース混合精度**（A.2）と組み合わせれば品質改善の余地あり:
- 感度の高いレイヤー（early attention, final norm）を 4-bit 維持
- 感度の低いレイヤー（中間 FFN）のみ 2-bit
- → 実質 ~3-bit 平均で品質とサイズのバランスを取る

### B.4 CLLMs (Consistency LLMs) — Jacobi 反復並列デコード

**論文**: arXiv 2403.00835 (ICML 2024)
**リポジトリ**: github.com/hao-ai-lab/Consistency_LLM

#### Jacobi 反復の具体的手順

1. n トークンブロックをランダム初期化: `y = [y1, y2, ..., yn]`
2. LLM に因果マスク付きで n トークンを**一括**入力
3. 全 n 位置の予測を同時取得
4. `y_new == y_old` なら収束（固定点）、そうでなければ step 2 へ

**バニラ Jacobi は 1.05× しか出ない**（AR 訓練された LLM は前のトークンが間違っていると正しく予測できない）。Consistency training で固定点収束を加速。

#### 訓練コスト

| データセット | 訓練時間 (8x A100) |
|-------------|-------------------|
| Spider | 2 時間 |
| GSM8K | 12 時間 |
| ShareGPT | 30 時間 |

#### ANE 適用性: **低 — アーキテクチャ的に不適合**

- Jacobi 反復は**毎ステップで n トークン入力**が必要 → ChunkedEngine は single-token decode に最適化されており、可変長 Jacobi ブロックは prefill パス（decode より大幅に遅い）を使うことになる
- ANE では single-token decode と multi-token decode のレイテンシ差が GPU ほど大きくない → 並列化の利得が小さい
- ベースモデル自体の fine-tuning が必要

### B.5 MInference — 動的スパースアテンション

**論文**: arXiv 2407.02490 (NeurIPS 2024 Spotlight)
**リポジトリ**: github.com/microsoft/MInference

#### 3 つのスパースパターン

- **A-Shape**: 先頭トークン（BOS/system prompt）+ ローカルウィンドウに集中。静的パターン。
- **Vertical-Slash**: 特定の「重要な」列 + 対角帯。半動的（top-k 列を per-input で検出）。
- **Block-Sparse**: ブロック単位でクラスタリング。最も動的。

#### コンテキスト長と効果

| コンテキスト長 | 高速化 |
|---------------|--------|
| 100K | 1.8× |
| 300K | 4.1× |
| 1M | 10× |
| **8K 以下** | **効果なし〜マイナス** |

#### ANE 適用性: **非常に低**

- **Prefill only** の手法（decode には適用不可）
- 8K コンテキストではスパースインデックス構築のオーバーヘッドが利得を上回る
- カスタム GPU カーネル (Triton) が必要、ANE にはプログラマブルカーネルインターフェースなし
- **唯一の示唆**: A-shape パターンの知見は既存の SWA + Global Attention 設計を検証するのみ

---

## Part C: Google LiteRT-LM ソースコード Deep Dive

### C.1 LiteRT-LM ベンチマーク — ANE との比較

Google 公式ベンチマーク（prefill 1024 tok, decode 256 tok, context 2048）:

**Gemma 4 E2B (2.58 GB)**

| デバイス | バックエンド | Decode tok/s | TTFT | メモリ |
|----------|-------------|-------------|------|--------|
| iPhone 17 Pro | CPU | 25.0 | 1.9s | 607 MB |
| iPhone 17 Pro | **GPU** | **56.5** | 0.3s | 1450 MB |
| Samsung S26 Ultra | GPU | 52.1 | 0.3s | 676 MB |
| MacBook Pro M4 Max | GPU | 160.2 | 0.1s | 1623 MB |

**比較**: CoreML/ANE パイプラインは iPhone 15 Pro で ~11 tok/s。Google の iPhone 17 Pro GPU は 56.5 tok/s — ハードウェア世代差を考慮しても **GPU パスが decode で圧倒的**。

**重要**: これらのベンチマークには **MTP/投機的デコーディングは含まれていない**。Google は MTP の acceptance 率や on-device 高速化の数値を**一切公開していない**。

### C.2 MTP ドラフターのデコードループ — LiteRT-LM ソースから確認

`runtime/executor/llm_litert_compiled_model_executor.cc` から:

1. **Prefill 後の初回 decode**: 通常 decode → logits + `activations`（ターゲットの最終隠れ状態）を取得 → `mtp_drafter_->Draft()` 呼び出し
2. **2 回目以降**: ドラフターは前ステップの `projected_activations` を保持。ターゲット decode なしでドラフト生成可能
3. **トークン組み立て**: 検証済みトークンを先頭に付加、末尾トークンは次イテレーションの入力として保留

### C.3 MTP ドラフターの activations 入力 — 謎が解決

`llm_litert_mtp_drafter.cc` の `RunDraftingLoop` から確認:

```
activations = concat(
    embedding_of_current_token[1536],     ← EmbeddingLookupManager で取得
    projected_activations_from_prev_step[1536]  ← ドラフター自身の出力
)
→ 3072 次元の入力
```

**EAGLE の `[h, e(t+1)]` パターンと同一構造**。`projected_activations` は 1 ステップ前のドラフター出力がフィードバックされる。初回はターゲットの decode 出力から取得。

### C.4 ドラフターの KV キャッシュ戦略

- ドラフターは **Q-only** — K/V projection を持たず、ターゲットの KV キャッシュ (`kv13`, `kv14`) を直接読む
- **別途ドラフター KV キャッシュは不要** → メモリ追加ゼロ
- ダブルバッファリング: `input_kv_cache_buffers_` と `output_kv_cache_buffers_` をスワップ（GPU バックエンドが同一バッファの読み書きを許可しないため）
- NPU パス: `CommitVerifiedKVCache()` で検証成功後のみ永続状態を更新

### C.5 Google の最適化テクニック

- **KV キャッシュを Conv Weight レイアウトで保存**: attention の K@V は数学的に conv と等価。KV を conv weight 形式で保存することでフォーマット変換オーバーヘッドを排除 → **CoreML の Conv2dLinear アプローチと同じ発想**
- **Embedding を mmap**: ディスクから直接メモリマップ、RAM にロードしない → E2B で 607 MB の低メモリ数値の理由
- **Prefill/Decode で重みと KV キャッシュを共有**: 別コンテキストだが物理メモリは同一
- **Dequantize を Conv に融合**: decode フェーズでは INT4 重みのデコードを conv オペレーションに統合

### C.6 MTP Acceptance 率の推定

Google は公式数値を未公開。ただし以下から推定:

- **Thoughtworks EAGLE3 for Gemma 4 31B** (サーバー): 訓練時 acceptance 0.75–0.82。ただしバックエンド不一致（HF Transformers で訓練、SGLang で推論）で隠れ状態が最大 32% 乖離、acceptance が ~13% に低下
- Google の MTP ドラフターはこの問題を回避（Google 自身の推論ランタイムに対して訓練）

**妥当な推定**: 会話タスクで 60–80% acceptance、コードタスクで 40–60%。1.3–1.6× の on-device 高速化。

---

## Part D: Apple の最新研究からの知見

### D.1 Apple Foundation Models 2025 — 本番アーキテクチャ

**論文**: arXiv 2507.13575

Apple の on-device モデル（AFMTextV7, 3.18B params）の本番構成:

| 要素 | 仕様 |
|------|------|
| 総レイヤー数 | 56（2 セグメント） |
| Segment 0 | 35 layers, フル QKV attention |
| Segment 1 | 21 layers, **Q-only attention** — K/V 投影なし、Seg 0 最終層の KV を再利用 |
| Hidden dim | 2048 |
| Attention | 3 local SWA (window=4096) + 1 global NoPE の繰り返し |
| 量子化 | **2-bit QAT** (learnable weight clipping) |
| Embedding 量子化 | 4-bit |
| KV キャッシュ | 8-bit + low-rank adapter で品質回復 |
| 本番サイズ | ~1.0–1.1 GB |

**Speculative Decoding ドラフター (48.77M params)**:
- 12 transformer layers, hidden=256, 8 heads
- FFN: 3× expansion (256 → 768)
- 共有語彙 (153,600)
- 4–8 候補トークン生成、1 パスで検証
- **60–80% acceptance rate**
- **2–4× 推論高速化**
- iPhone 15 Pro: **30 tok/s** (投機的デコーディング前)

#### Gemma 4 への具体的示唆

1. **KV-Share の拡張**: Apple は最後 37.5% のレイヤーで KV を再利用。Gemma 4 の KV-Share (L15-34 が KV13/14 を読む) は既に同じパターンだが、Apple は **Q-only** まで踏み込んでいる（K/V projection 自体を削除）。ドラフターでは既にこれを実現（MTP ドラフターは Q-only）。
2. **8-bit KV + LoRA 回復**: KV キャッシュを INT8 に量子化し、品質劣化を低ランクアダプターで回復。→ A.1 の KV 量子化と組み合わせ可能。
3. **ドラフターサイズの参考値**: Apple のドラフターは 48.77M params (ベースの 1.5%)。Google の MTP ドラフターは ~2M params body + 44 MB total (embedding 含む)。どちらもごく小さい。

### D.2 Apple ReDrafter — オープンソース実装

**リポジトリ**: github.com/apple/ml-recurrent-drafter（PyTorch + MLX 実装公開）

#### アーキテクチャ

```python
# recurrent_drafting/modeling_drafter.py
input = concat(context[hidden_size], state[hidden_size])  # 2*hidden_size
proj = Linear(2*hidden_size, exit_dim)                     # optional
lm_head = N × ResBlock(Linear + SiLU residual) → Linear(exit_dim, vocab_size)
state_update = silu(rnn_w(embedding(prev_token)) + rnn_u(state))  # GRU-like
```

- ビームサーチでドラフト: `beam_width × beam_length` 候補を生成
- Dynamic tree attention で prefix を重複排除してから検証

#### Apple Silicon での実測

| デバイス | 高速化 | 最適 beam 幅 |
|----------|--------|-------------|
| M1 Max | 1.37× | beam=1, length=4–5 |
| M2 Ultra | 1.52× | beam=3, length=4–5 |
| H100 (CUDA) | 4× | beam=50+ |

**Apple Silicon では beam 幅が狭い** — GPU の並列計算力が限定的なため、広い beam は逆効果。ANE でも同様の傾向が予想される。

### D.3 Mirror Speculative Decoding — GPU+ANE 並列パターン

**論文**: arXiv 2510.13161

#### コアコンセプト

ドラフトとターゲットを**直列ではなく並列**に実行:

1. ターゲットが前半レイヤーを処理開始
2. Early-exit 層でターゲットが top-k 候補を emit
3. ドラフトが**同時に**全候補のブランチ rollout を開始
4. ターゲットが後半レイヤーを処理（ドラフトと並列）
5. 両方完了後、受理判定

**レイテンシモデル**: `T_Mirror = T_target(前半) + max{T_target(後半), T_draft(γ)} + T_verify`

**結果**: 14B–66B モデルで 2.8–5.8× 高速化、EAGLE-3 比 +30%。

#### ANE への適用可能性

- ドラフトを **GPU (Metal)** で、ターゲット検証を **ANE** で並列実行
- 通信はトークン ID のみ（マイクロ秒オーダー）
- GPU ドラフトのコストは ANE 検証時間内に収まれば**実質ゼロ**
- **熱的にも有利**: GPU と ANE に負荷を分散

**ただし Gemma 4 E2B (4B) ではターゲットのレイテンシが短すぎ**、ドラフト計算を隠しきれない可能性。より大きなモデル (E4B 以上) で効果的。

### D.4 coremltools 8.3 / 9.0 の新機能

**coremltools 8.3** (2025-04-29):
- **ANE 上の top-k レジデンシ改善** — in-model argmax に直接関連
- `scaled_dot_product_attention_sliced_q` パス: **ANE で 34% 高速化、45% メモリ削減**（長シーケンス）
- MLModelBenchmarker / MLModelInspector / MLModelValidator ユーティリティ
- リモートデバイスベンチマーク/デバッグ

**coremltools 9.0** (2025-11-10):
- iOS 26 / macOS 26 デプロイメントターゲット
- Int8 モデル入出力サポート
- Model state の read/write API
- PyTorch 2.7 + ExecuTorch 0.5 サポート

**MLState の ANE 対応**: coremltools 9.0 でも明示的な ANE 対応の言及なし。explicit KV I/O アプローチは引き続き必要。

### D.5 Core AI フレームワーク — WWDC 2026 予告

iOS 27 で CoreML を置き換える「Core AI」フレームワークが発表予定（2026 年 6 月）。
- ニューラルエンジン + ユニファイドメモリの活用強化
- サードパーティ AI モデルの統合標準化（MCP 経由の可能性）
- **技術的詳細は未公開** — WWDC セッション待ち

---

## Part E: 総合優先度評価

### Tier 1: 即日〜1 週間で実装、リスク低

| 施策 | 工数 | 期待効果 | リスク |
|------|------|----------|--------|
| Prompt Lookup Decoding (B.1) | 0.5 日 | 要約/QA で 2.4×、自由会話で 0× | ゼロ |
| System Prompt KV 永続化 (A.3) | 0.5 日 | 2 ターン目以降 TTFT 2–5× 改善 | ゼロ |
| 重要度ベース混合精度量子化 (A.2) | 1–2 日 | 品質改善 or サイズ 20–30% 削減 | 低 |

### Tier 2: 1–2 週間、MTP/投機的デコーディング直結

| 施策 | 工数 | 期待効果 | リスク |
|------|------|----------|--------|
| KV キャッシュ INT8 量子化 (A.1) | 2–3 日 | KV メモリ 50% 削減 → 長コンテキスト対応 | 品質検証必要 |
| DistillSpec (B.2) | 2–3 日 | EAGLE-3 acceptance 0%→50%+ | MTP ピボット中なら不要 |
| coremltools SDPA sliced_q パス (D.4) | 1 日 | ANE 34% 高速化、45% メモリ削減 | 低 |

### Tier 3: 中期、アーキテクチャ変更

| 施策 | 工数 | 期待効果 | リスク |
|------|------|----------|--------|
| Mirror SD (GPU draft + ANE verify) (D.3) | 3–5 日 | EAGLE-3 比 +30%、熱分散 | E2B では効果限定的 |
| Mixed-bit palettization (感度別 2/4/6-bit) (B.3 代替) | 2–3 日 | モデルサイズ 30–40% 削減 | 品質検証必要 |
| Apple 式 KV INT8 + LoRA 回復 (D.1) | 3–5 日 | KV 75% 削減 + 品質維持 | LoRA 訓練必要 |

### 却下・優先度低

| 施策 | 理由 |
|------|------|
| QuIP# / AQLM | Hadamard/gather op が ANE 非対応。ネイティブ 2-bit palettization + 混合精度で代替 |
| CLLMs | Jacobi 反復が ANE single-token decode と不適合 |
| MInference | 8K コンテキストでは効果なし、カスタムカーネル必要 |

---

## 参考ソース一覧

### 論文
- DistillSpec: arXiv 2310.08461 (ICLR 2024)
- QuIP#: arXiv 2402.04396 (ICML 2024)
- CLLMs: arXiv 2403.00835 (ICML 2024)
- MInference: arXiv 2407.02490 (NeurIPS 2024 Spotlight)
- Apple Foundation Models 2025: arXiv 2507.13575
- Apple ReDrafter: arXiv 2403.09919
- Mirror Speculative Decoding: arXiv 2510.13161
- Orion (ANE Reverse Engineering): arXiv 2603.06728
- DFlash: arXiv 2602.06036
- TurboQuant (KV Cache Quantization): sotaaz.com/post/turboquant-practical-en
- KVSwap (Disk Offload): arXiv 2511.11907
- EAGLE3 for Gemma 4 (Thoughtworks): huggingface.co/blog/lujangusface/tw-eagle3-gemma4

### リポジトリ
- apple/ml-recurrent-drafter: github.com/apple/ml-recurrent-drafter
- Prompt Lookup Decoding: github.com/apoorvumang/prompt-lookup-decoding
- AdaSPEC (DistillSpec 後続): github.com/yuezhouhu/adaspec
- QuIP#: github.com/Cornell-RelaxML/quip-sharp
- CLLMs: github.com/hao-ai-lab/Consistency_LLM
- MInference: github.com/microsoft/MInference
- DFlash MLX: github.com/Aryagm/dflash-mlx
- LiteRT-LM: github.com/google-ai-edge/LiteRT-LM
- Orion: github.com/mechramc/Orion

### ベンチマーク・記事
- Gemma 4 E2B LiteRT-LM: huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
- Ollama MLX Backend: ollama.com/blog/mlx
- Apple ML Research 2025 Updates: machinelearning.apple.com/research/apple-foundation-models-2025-updates
- Apple On-Device Llama 3.1: machinelearning.apple.com/research/core-ml-on-device-llm
- MLX on M5: machinelearning.apple.com/research/exploring-llms-mlx-m5
- Gemma 4 MTP Discussion: huggingface.co/google/gemma-4-E4B-it/discussions/5
