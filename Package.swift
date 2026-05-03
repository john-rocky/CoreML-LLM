// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreMLLLM",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "CoreMLLLM", targets: ["CoreMLLLM"]),
        .executable(name: "video-test", targets: ["VideoTest"]),
        .executable(name: "accept-rate-bench", targets: ["AcceptRateBench"]),
        .executable(name: "coreml-llm-smoke", targets: ["CoreMLLLMSmoke"]),
        .executable(name: "union-bitexact", targets: ["UnionBitExact"]),
        .executable(name: "determinism-oracle", targets: ["DeterminismOracle"]),
        .executable(name: "verify-k8-probe", targets: ["VerifyK8Probe"]),
        .executable(name: "ane-residency-gate", targets: ["AneResidencyGate"]),
        .executable(name: "gemma4mm-smoke", targets: ["Gemma4MMSmoke"]),
        // Standalone samples for the two Gemma-3-based models. These live in
        // the same package on purpose — a LocalAIKit-style wrapper can depend
        // on the `CoreMLLLM` library and use `FunctionGemma` / `EmbeddingGemma`
        // / `Gemma3BundleDownloader` directly, without pulling the sample CLIs.
        .executable(name: "functiongemma-demo", targets: ["FunctionGemmaDemo"]),
        .executable(name: "embeddinggemma-demo", targets: ["EmbeddingGemmaDemo"]),
    ],
    dependencies: [
        // Range widened to 1.0.x: mlx-swift-examples caps swift-transformers at
        // <1.1.0, so any consumer that also pulls MLX deadlocks if we require
        // 1.1+ here. 1.0.x already exposes the `Tokenizers` product with the
        // `Tokenizer` protocol + `AutoTokenizer.from(modelFolder:)` API that
        // CoreMLLLM uses, so 1.0.x is source-compatible with 1.1.x here.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "CoreMLLLM",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "VideoTest",
            dependencies: ["CoreMLLLM"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "CoreMLLLMTests",
            dependencies: ["CoreMLLLM"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "CoreMLLLMSmoke",
            dependencies: ["CoreMLLLM"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Mac-only bench that measures offline draft-source accept rate. Runs
        // the shipping CoreMLLLM pipeline on a prompt corpus via oracle replay
        // at temperature = 0. See docs/MAC_FIRST_EXECUTION_PLAN.md §A1.
        .executableTarget(
            name: "AcceptRateBench",
            dependencies: ["CoreMLLLM"],
            path: "Sources/accept-rate-bench",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Mac-only bit-exact verifier for DrafterUnion (Phase B Task 1).
        // Runs each prompt twice (serial vs union) and asserts the
        // emitted token streams match — this gates the iPhone trip.
        .executableTarget(
            name: "UnionBitExact",
            dependencies: ["CoreMLLLM"],
            path: "Sources/union-bitexact",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Mac-only CoreML determinism oracle (quality-gate PoC). Runs a
        // tiny fixed corpus at argmax (all drafters off) and compares token
        // IDs against a committed oracle. Catches silent regressions in the
        // shipped decode path. See docs/QUALITY_GATE.md.
        .executableTarget(
            name: "DeterminismOracle",
            dependencies: ["CoreMLLLM"],
            path: "Sources/determinism-oracle",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // LookAhead K=8 probe harness — measures pure verify_qK=8 wall-clock
        // on ANE. Go / no-go gate before committing to full LookAhead impl.
        // See docs/LOOKAHEAD_PROBE_HANDOFF.md.
        .executableTarget(
            name: "VerifyK8Probe",
            dependencies: ["CoreMLLLM"],
            path: "Sources/verify-k8-probe",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Mac smoke test for Gemma4StatefulMultimodalEngine — text-only
        // generate to catch engine bugs without an iPhone trip.
        .executableTarget(
            name: "Gemma4MMSmoke",
            dependencies: [
                "CoreMLLLM",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/gemma4mm-smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // FunctionGemma-270M standalone CLI. Does NOT combine with Gemma 4 —
        // multi-model orchestration belongs in the LocalAIKit wrapper.
        .executableTarget(
            name: "FunctionGemmaDemo",
            dependencies: ["CoreMLLLM"],
            path: "Sources/functiongemma-demo",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // EmbeddingGemma-300M standalone CLI. Same rationale as above.
        .executableTarget(
            name: "EmbeddingGemmaDemo",
            dependencies: ["CoreMLLLM"],
            path: "Sources/embeddinggemma-demo",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // T2 ANE residency CI gate. Loads each chunk{1..4}.mlpackage in
        // a model directory, queries MLComputePlan, and exits non-zero if
        // any chunk's ANE op fraction drops below the threshold (default
        // 99.5%). Optionally writes/diffs a JSON baseline.
        .executableTarget(
            name: "AneResidencyGate",
            dependencies: ["CoreMLLLM"],
            path: "Sources/ane-residency-gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
