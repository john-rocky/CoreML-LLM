# Hire the author

CoreML-LLM is built and maintained by Daisuke Majima. If you need help shipping on-device LLMs on Apple silicon, this page is the entry point.

## What I work on

- **CoreML conversion** — HuggingFace / PyTorch → ANE-resident `.mlpackage` / `.mlmodelc`, including the awkward bits: chunking, KV cache layout, RoPE, INT8 / INT4 / W8A8 quantization, multifunction prefill, MLState.
- **ANE performance** — getting models off CPU and onto the Neural Engine, then keeping them there. Per-layer attribution, dispatch reduction, op-level rewrites for ANE-friendliness.
- **iOS / macOS integration** — Swift Package design, background download, prompt caching, multimodal pipelines (image / video / audio), function calling, embeddings.
- **Architectural advisory** — given a model and a target device, what is achievable, what is not, and where the cliffs are.

## Track record

This repository is the public portion of the work. Highlighted milestones:

<!-- TODO_USER_FILL — pick 3–5 items from your release history that you most want to be remembered for. Suggested seed list:
- Gemma 4 E2B 3-chunk decode on iPhone A19 Pro (34.2 tok/s, ANE-resident, multimodal preserved)
- Qwen3-VL 2B stateful: 4 s → 125 ms 2nd-turn TTFT via cross-turn KV reuse
- Qwen3.5 0.8B (first hybrid SSM+attention LLM on CoreML, 99.9 % ANE)
- LFM2.5 350M conversion pipeline (Liquid AI hybrid attn + short-conv, ANE)
- Models Zoo iOS app shipping all of the above
-->

- TODO_USER_FILL — case study 1
- TODO_USER_FILL — case study 2
- TODO_USER_FILL — case study 3

## Engagement models

<!-- TODO_USER_FILL — pick the ones you actually want to offer. Examples:
- Fixed-scope conversion: "I will deliver an ANE-resident CoreML build of model X with an iOS sample app, in N weeks, for ¥Y."
- Hourly / daily advisory.
- Monthly retainer.
- Code review of an existing CoreML / ANE pipeline.
-->

- TODO_USER_FILL — engagement option 1
- TODO_USER_FILL — engagement option 2

## Contact

- Email: <!-- TODO_USER_FILL: e.g. samuraibrothersmail@gmail.com -->
- X / Twitter: <!-- TODO_USER_FILL: @handle and link -->
- LinkedIn: <!-- TODO_USER_FILL -->
- GitHub: [@john-rocky](https://github.com/john-rocky)

When emailing, please include:

1. Target model (HF repo URL is fine) and target device tier.
2. Whether on-device inference is a hard requirement, or whether server-side fallback is acceptable.
3. Latency / memory budget per request.
4. Timeline and licensing constraints on the model weights.

## 日本語

Apple Neural Engine (ANE) で動く on-device LLM の設計・変換・最適化を業務で請けています。

- 対応領域：HuggingFace モデルの CoreML 変換 (chunked, INT8/INT4/W8A8, MLState 等)、ANE への配置、iOS / macOS 統合、マルチモーダル (画像 / 動画 / 音声)、function calling、embeddings、アプリ実装まで。
- 形態：<!-- TODO_USER_FILL: 業務委託 / 顧問 / 単発レビュー / 月額リテイナー など -->
- 言語：日本語・英語
- 連絡先：<!-- TODO_USER_FILL --> （上記 Contact を参照）

「このモデルをこの端末でこの速度で動かしたい」という具体的な要件があるほど、見積もりは早く返せます。
