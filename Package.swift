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
        .executable(name: "accept-rate-bench", targets: ["AcceptRateBench"]),
        .executable(name: "coreml-llm-smoke", targets: ["CoreMLLLMSmoke"]),
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
    ]
)
