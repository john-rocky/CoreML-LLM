# Documentation

| Topic | File |
|---|---|
| Architecture, ANE optimizations, project structure, MLX comparison | [ARCHITECTURE.md](ARCHITECTURE.md) |
| HF conversion, ANE tricks, INT4/INT8/W8A8 rationale | [CONVERSION.md](CONVERSION.md) |
| Adding a new architecture | [ADDING_MODELS.md](ADDING_MODELS.md) |
| Benchmark methodology (tok/s, ANE %, memory) | [BENCHMARKING.md](BENCHMARKING.md) |
| 3-chunk decode (+8.2 %) | [THREE_CHUNK_MAC_BENCH.md](THREE_CHUNK_MAC_BENCH.md) |
| `.mlpackage` vs `.mlmodelc`, format gotchas | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Image pipeline | [MULTIMODAL.md](MULTIMODAL.md) |
| Video pipeline | [VIDEO_PHASE2_CONTINUATION.md](VIDEO_PHASE2_CONTINUATION.md) |
| Audio pipeline | [AUDIO.md](AUDIO.md) |
| 8K context roadmap, ANE-compat matrix | [SPEED_8K.md](SPEED_8K.md) |
| FunctionGemma I/O contract | [FUNCTIONGEMMA.md](FUNCTIONGEMMA.md) |
| EmbeddingGemma I/O contract, Matryoshka recipe | [EMBEDDINGGEMMA.md](EMBEDDINGGEMMA.md) |
| LFM2.5 conversion (ChatML, ANE dual-state workaround, fp16 padding drift) | [LFM2_CONVERSION_FINDINGS.md](LFM2_CONVERSION_FINDINGS.md) |
| Research background, competitive landscape | [RESEARCH.md](RESEARCH.md) |
| Decision log (WFA, Flash, W8A8, Medusa, EAGLE-3, SDPA fusion, KV alias, Topology I) | [EXPERIMENTS.md](EXPERIMENTS.md) |
