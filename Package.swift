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
        .executable(name: "ane-residency-gate", targets: ["AneResidencyGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
    ],
    targets: [
        .target(
            name: "CoreMLLLM",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
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
        // T2: ANE residency CI gate. Loads each chunk{1..4}.mlpackage in
        // a model directory, queries MLComputePlan, and exits non-zero if
        // any chunk's ANE op fraction drops below the threshold (default
        // 99.5%). Also writes a JSON baseline so PRs can be diffed.
        // See docs/LITERT_PERF_ADOPTIONS.md §T2.
        .executableTarget(
            name: "AneResidencyGate",
            dependencies: ["CoreMLLLM"],
            path: "Sources/ane-residency-gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
