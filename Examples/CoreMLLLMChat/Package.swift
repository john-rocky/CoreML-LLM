// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CoreMLLLMChat",
    platforms: [.iOS(.v18), .macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
    ],
    targets: [
        .executableTarget(
            name: "CoreMLLLMChat",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "CoreMLLLMChat"
        ),
    ]
)
