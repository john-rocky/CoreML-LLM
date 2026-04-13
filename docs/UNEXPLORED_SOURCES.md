# Unexplored Sources — Gap Analysis (2026-04-14)

既存ドキュメント全体（ANE_OPTIMIZATION_SURVEY, LITERT_*_ANALYSIS, UNEXPLORED_APPROACHES V1–V5, FUNDAMENTAL_UNTRIED, EXPERIMENTS, PRIORITY_ROADMAP, SPEED_8K, EAGLE3_INTEGRATION_STATE, MTP_PATH_A_FINDINGS 等）を横断精査し、まだ当たっていない有望なソースを特定した。

---

## 1. 現在のブロッカーに直結するソース

### 1.1 DistillSpec — 蒸留によるドラフト/ターゲット分布アライメント

- **論文**: arXiv 2310.08461
- **対処するブロッカー**: EAGLE-3 Blocker 1（ドラフト/ターゲット分布不一致）
- **概要**: ターゲットモデルの出力分布に合わせてドラフトモデルを KL-divergence 最小化で蒸留する。EAGLE-3 の現在の問題は `use_cache=False`（KV-Share 無し）で訓練したドラフトが、KV-Share 有りの推論時分布と乖離していること。DistillSpec なら KV-Share 有りの推論分布を直接ターゲットとして蒸留できるため、フルリトレーニングよりコストが低い可能性がある。
- **期待効果**: acceptance 率を 0% → 50%+ に引き上げ
- **工数見積**: 2–3 日（蒸留スクリプト + 既存 EAGLE-3 weight ベース）
- **優先度**: 最高 — EAGLE-3 の最大ブロッカーを直接解決

### 1.2 Online Speculative Decoding — 推論時ドラフト適応

- **論文**: arXiv 2310.07177
- **概要**: 推論中にドラフトモデルをオンライン更新し、ドメイン固有の入力パターンに自動適応する。acceptance 率が低いドラフターでも、使用中に改善される。
- **対処するブロッカー**: EAGLE-3 / MTP ドラフターの汎用 acceptance 率が低い場合の救済策
- **期待効果**: +15–30% acceptance 率改善（論文値）
- **工数見積**: 3–4 日（on-device fine-tuning 基盤が必要）
- **優先度**: 中 — MTP が高 acceptance を示せば不要、保険的位置づけ

### 1.3 CLLMs (Consistency Large Language Models)

- **論文**: arXiv 2403.00835
- **概要**: Jacobi 反復ベースの並列デコーディング。ドラフトモデル不要で 2.4–3.4× 高速化を報告。Lookahead Decoding（調査済み）の改良版だが、consistency training により収束を大幅に加速。
- **EAGLE-3/MTP との関係**: ドラフトモデルが全て失敗した場合の完全な代替パス
- **工数見積**: 3–4 日（Jacobi ループの CoreML 実装 + consistency loss での fine-tuning）
- **優先度**: 中 — ドラフトモデル系が軌道に乗れば不要

---

## 2. 量子化の失敗（W2/W3）を覆す手法

W2/W3 の post-training palettization は品質崩壊を確認済み（2026-04-13）。Apple QAT 以外に sub-4-bit を実現する手法が存在する。

### 2.1 QuIP# — Incoherence Processing + Lattice Codebooks

- **論文**: arXiv 2402.04396
- **概要**: 重み行列を incoherent 基底に変換してから量子化することで、2-bit でも品質を維持。E8 lattice codebook により、ナイーブな k-means palettization（失敗済み）とは根本的に異なるアプローチ。
- **W2 失敗との差異**: 失敗した W2 は 4 codewords の post-training palettization。QuIP# は数学的に最適な codebook + incoherence 変換を組み合わせ。
- **期待効果**: モデルサイズ 50% 削減（INT4 比）、品質は INT4 の 95%+ を維持
- **工数見積**: 2–3 日（coremltools との統合が要調査）
- **優先度**: 高

### 2.2 AQLM (Additive Quantization for LLMs)

- **論文**: arXiv 2401.06118
- **概要**: Additive quantization（複数 codebook の加算）により 2-bit で競争力のある品質。QuIP# と異なるアプローチだが同等の結果。
- **期待効果**: QuIP# と同等
- **工数見積**: 2–3 日
- **優先度**: 高（QuIP# と並行評価が理想）

---

## 3. ANE Prefill 高速化の未調査ソース

### 3.1 MInference — 動的スパースアテンション

- **論文**: arXiv 2407.02490 (Microsoft)
- **概要**: 長コンテキスト prefill を動的スパースアテンションで高速化。A-shape / Vertical-Slash / Block-Sparse の 3 パターンをヘッドごとに自動検出し、不要な attention 計算をスキップ。
- **既存計画との関係**: GPU Prefill（Phase 5）と組み合わせ可能。GPU 上でスパース prefill を行えば TTFT をさらに短縮。
- **期待効果**: prefill 計算量 50–80% 削減（コンテキスト長依存）
- **工数見積**: 3–4 日（パターン検出 + スパースマスク生成）
- **優先度**: 中 — 8K prefill が UX ボトルネックの場合に有効

---

## 4. ゼロコスト即効性のある手法

### 4.1 Prompt Lookup Decoding

- **実装**: HuggingFace transformers 標準搭載（`prompt_lookup_num_tokens` パラメータ）
- **概要**: 入力プロンプト内の n-gram マッチでドラフトトークンを生成。ドラフトモデル不要、学習不要、CPU のみで動作。会話の継続・定型応答・コード補完で特に有効。
- **SuffixDecoding との差異**: SuffixDecoding は過去の全生成テキストから trie を構築するが、Prompt Lookup は現在のプロンプトのみ参照。trie 構築コスト（SuffixDecoding の blocker だった 1–2s insert）が発生しない。
- **期待効果**: 1.2–1.6×（入力が繰り返しを含む場合）
- **工数見積**: 0.5 日（Swift で n-gram matcher 実装）
- **優先度**: 高 — リスクゼロ、即日実装可能

---

## 5. Google AIEdgeGallery の謎を解くソース

### 5.1 MediaPipe LLM Inference API / LiteRT 内部実装

- **リポジトリ**: github.com/google-ai-edge/mediapipe
- **調査ポイント**: AIEdgeGallery の速度の秘密を解明するため、以下を読み解く:
  - GPU delegate + XNNPACK の最適化手法
  - LiteRT 側のデコードループ実装（バッチ検証、KV キャッシュ戦略）
  - MTP ドラフターとの統合パターン
- **MTP 抽出との関係**: ドラフターの抽出は完了（44.3 MB）。次に必要なのはデコードループ側の実装パターン。
- **優先度**: 高 — MTP Swift wiring の実装指針として

### 5.2 Gemma 4 Technical Report (Google DeepMind)

- **調査ポイント**: MTP の訓練手法の詳細（損失関数、共有 KV の扱い、acceptance 率の公式値）。ドラフター実装のリファレンスとして、公式の期待 acceptance 率がわかれば目標設定が明確になる。

---

## 6. 最近の注目論文（2025–2026、既存ドキュメント未掲載）

### 6.1 Mixture of Depths

- **論文**: arXiv 2404.02258
- **概要**: トークンごとに動的に計算深度を決定。「簡単な」トークンは浅い層でスキップ。LayerSkip（調査済み）と似ているが、トークン単位の動的判断が異なる。
- **KV-Share との相性**: L15 以降のスキップは KV 書き込みも省略できるため、KV-Share アーキテクチャと自然に組み合わさる。
- **工数見積**: 3–4 日（router の訓練 + chunk 内スキップロジック）
- **優先度**: 低 — 訓練が必要

### 6.2 BiTA (Bi-directional Tuning for Lossless Acceleration)

- **論文**: arXiv 2401.10774
- **概要**: Medusa 系の改良。Bidirectional attention により品質を維持しつつ多トークン予測。
- **Medusa 失敗との関係**: Medusa は 1.3% acceptance で壊滅的だったが、BiTA は異なるアーキテクチャで改善の可能性。ただし Gemma 4 での検証が必要。
- **工数見積**: 3–4 日
- **優先度**: 低 — MTP/EAGLE-3 が優先

### 6.3 MLC-LLM (mlc.ai)

- **リポジトリ**: github.com/mlc-ai/mlc-llm
- **概要**: Apache TVM 系のモバイル最適化コンパイラ。ANE ではなく GPU/CPU 向けだが、グラフレベル最適化のパターン（op fusion, layout transformation）に ANE 向けのヒントがある可能性。
- **優先度**: 低 — 直接使えないが最適化パターンの参考

---

## 優先度サマリー

| 優先度 | ソース | 対処する課題 | 工数 |
|--------|--------|-------------|------|
| **最高** | DistillSpec | EAGLE-3 Blocker 1（分布不一致） | 2–3 日 |
| **高** | Prompt Lookup Decoding | 即効性のある投機的デコーディング | 0.5 日 |
| **高** | QuIP# / AQLM | W2/W3 品質崩壊の解決 | 2–3 日 |
| **高** | MediaPipe LLM / LiteRT 内部 | MTP Swift wiring の実装指針 | 1–2 日 |
| **中** | MInference | prefill 高速化 | 3–4 日 |
| **中** | Online Speculative Decoding | ドラフト acceptance 率改善 | 3–4 日 |
| **中** | CLLMs | ドラフトモデル不要の代替パス | 3–4 日 |
| **低** | Mixture of Depths | 動的計算スキップ | 3–4 日 |
| **低** | BiTA | Medusa 改良版 | 3–4 日 |
| **低** | MLC-LLM | 最適化パターン参考 | — |
