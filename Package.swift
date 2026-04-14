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
            name: "MtpMacCheck",
            dependencies: ["CoreMLLLM"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
