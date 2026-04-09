import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Downloads and caches CoreML models with resume support.
@Observable
final class ModelDownloader: NSObject {
    var isDownloading = false
    var progress: Double = 0
    var status = ""
    var availableModels: [ModelInfo] = ModelInfo.defaults
    var refreshTrigger = 0  // Increment to force UI refresh

    private let fileManager = FileManager.default
    private var session: URLSession!
    private var downloadContinuation: CheckedContinuation<URL, Error>?
    private var totalBytesForAllFiles: Int64 = 0
    private var completedBytesForPreviousFiles: Int64 = 0

    struct ModelInfo: Identifiable {
        let id: String
        let name: String
        let size: String
        let downloadURL: String
        let folderName: String

        static let defaults: [ModelInfo] = [
            ModelInfo(id: "gemma4-e2b", name: "Gemma 4 E2B", size: "2.5 GB",
                      downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/main",
                      folderName: "gemma4-e2b"),
            ModelInfo(id: "qwen2.5-0.5b", name: "Qwen2.5 0.5B (Text)", size: "309 MB",
                      downloadURL: "https://github.com/john-rocky/CoreML-LLM/releases/download/v0.1.0/qwen2.5-0.5b-coreml.zip",
                      folderName: "qwen2.5-0.5b"),
        ]
    }

    // File list with estimated sizes for progress weighting
    private struct DownloadFile {
        let remotePath: String
        let localPath: String
        let estimatedSize: Int64  // bytes
    }

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 7200
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func isDownloaded(_ model: ModelInfo) -> Bool { localModelURL(for: model) != nil }

    func hasFiles(_ model: ModelInfo) -> Bool {
        fileManager.fileExists(atPath: modelsDirectory.appendingPathComponent(model.folderName).path)
    }

    func localModelURL(for model: ModelInfo) -> URL? {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        // Chunked
        let chunk1 = dir.appendingPathComponent("chunk1.mlmodelc")
        if fileManager.fileExists(atPath: chunk1.appendingPathComponent("weights/weight.bin").path) {
            return chunk1
        }
        // Monolithic .mlmodelc
        let modelc = dir.appendingPathComponent("model.mlmodelc")
        if fileManager.fileExists(atPath: modelc.appendingPathComponent("weights/weight.bin").path) {
            return modelc
        }
        // .mlpackage
        let pkg = dir.appendingPathComponent("model.mlpackage")
        if fileManager.fileExists(atPath: pkg.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin").path) {
            return pkg
        }
        return nil
    }

    func download(_ model: ModelInfo) async throws -> URL {
        if let existing = localModelURL(for: model) { return existing }

        isDownloading = true
        progress = 0
        status = "Starting..."

        #if os(iOS)
        var bgTask: UIBackgroundTaskIdentifier = .invalid
        bgTask = UIApplication.shared.beginBackgroundTask {
            UIApplication.shared.endBackgroundTask(bgTask)
            bgTask = .invalid
        }
        #endif

        defer {
            isDownloading = false
            #if os(iOS)
            if bgTask != .invalid { UIApplication.shared.endBackgroundTask(bgTask) }
            #endif
        }

        let destDir = modelsDirectory.appendingPathComponent(model.folderName)
        try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)

        if model.downloadURL.contains("huggingface.co") {
            try await downloadFromHuggingFace(model, to: destDir)
        } else {
            // Clean start for ZIP
            try? fileManager.removeItem(at: destDir)
            try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)
            guard let url = URL(string: model.downloadURL) else { throw DownloadError.invalidURL }
            let tempZip = try await downloadSingleFile(url)
            status = "Extracting..."
            try unzipFile(tempZip, to: destDir)
            try? fileManager.removeItem(at: tempZip)
        }

        guard let result = localModelURL(for: model) else { throw DownloadError.extractionFailed }
        status = "Ready"
        progress = 1.0
        return result
    }

    func delete(_ model: ModelInfo) throws {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        if fileManager.fileExists(atPath: dir.path) { try fileManager.removeItem(at: dir) }
        refreshTrigger += 1
    }

    // MARK: - HuggingFace Download with Resume

    private func downloadFromHuggingFace(_ model: ModelInfo, to destDir: URL) async throws {
        let base = model.downloadURL

        var files: [DownloadFile]

        // Lite chunks: 2 chunks (chunk1 = 0.49GB, chunk2 = 0.85GB) + external PLE
        files = [
            .init(remotePath: "lite-chunks/chunk1.mlmodelc/weights/weight.bin", localPath: "chunk1.mlmodelc/weights/weight.bin", estimatedSize: 490_000_000),
            .init(remotePath: "lite-chunks/chunk1.mlmodelc/coremldata.bin", localPath: "chunk1.mlmodelc/coremldata.bin", estimatedSize: 500_000),
            .init(remotePath: "lite-chunks/chunk1.mlmodelc/model.mil", localPath: "chunk1.mlmodelc/model.mil", estimatedSize: 100_000),
            .init(remotePath: "lite-chunks/chunk1.mlmodelc/analytics/coremldata.bin", localPath: "chunk1.mlmodelc/analytics/coremldata.bin", estimatedSize: 1_000),
            .init(remotePath: "lite-chunks/chunk2.mlmodelc/weights/weight.bin", localPath: "chunk2.mlmodelc/weights/weight.bin", estimatedSize: 850_000_000),
            .init(remotePath: "lite-chunks/chunk2.mlmodelc/coremldata.bin", localPath: "chunk2.mlmodelc/coremldata.bin", estimatedSize: 500_000),
            .init(remotePath: "lite-chunks/chunk2.mlmodelc/model.mil", localPath: "chunk2.mlmodelc/model.mil", estimatedSize: 100_000),
            .init(remotePath: "lite-chunks/chunk2.mlmodelc/analytics/coremldata.bin", localPath: "chunk2.mlmodelc/analytics/coremldata.bin", estimatedSize: 1_000),
            .init(remotePath: "lite-chunks/model_config.json", localPath: "model_config.json", estimatedSize: 1_000),
            // Tokenizer
            .init(remotePath: "hf_model/tokenizer.json", localPath: "hf_model/tokenizer.json", estimatedSize: 30_000_000),
            .init(remotePath: "hf_model/tokenizer_config.json", localPath: "hf_model/tokenizer_config.json", estimatedSize: 5_000),
            .init(remotePath: "hf_model/config.json", localPath: "hf_model/config.json", estimatedSize: 5_000),
            // External embeddings (memory-mapped)
            .init(remotePath: "embed_tokens_q8.bin", localPath: "embed_tokens_q8.bin", estimatedSize: 402_653_184),
            .init(remotePath: "embed_tokens_scales.bin", localPath: "embed_tokens_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "embed_tokens_per_layer_q8.bin", localPath: "embed_tokens_per_layer_q8.bin", estimatedSize: 2_348_810_240),
            .init(remotePath: "embed_tokens_per_layer_scales.bin", localPath: "embed_tokens_per_layer_scales.bin", estimatedSize: 524_288),
            // Per-layer projection
            .init(remotePath: "per_layer_projection.bin", localPath: "per_layer_projection.bin", estimatedSize: 27_525_120),
            .init(remotePath: "per_layer_norm_weight.bin", localPath: "per_layer_norm_weight.bin", estimatedSize: 1_024),
            // Vision model for multimodal (lazy-loaded)
            .init(remotePath: "vision.mlmodelc/weights/weight.bin", localPath: "vision.mlmodelc/weights/weight.bin", estimatedSize: 320_000_000),
            .init(remotePath: "vision.mlmodelc/coremldata.bin", localPath: "vision.mlmodelc/coremldata.bin", estimatedSize: 200_000),
            .init(remotePath: "vision.mlmodelc/model.mil", localPath: "vision.mlmodelc/model.mil", estimatedSize: 50_000),
            .init(remotePath: "vision.mlmodelc/metadata.json", localPath: "vision.mlmodelc/metadata.json", estimatedSize: 1_000),
            .init(remotePath: "vision.mlmodelc/analytics/coremldata.bin", localPath: "vision.mlmodelc/analytics/coremldata.bin", estimatedSize: 1_000),
        ]

        totalBytesForAllFiles = files.reduce(0) { $0 + $1.estimatedSize }
        completedBytesForPreviousFiles = 0

        for file in files {
            let destFile = destDir.appendingPathComponent(file.localPath)

            // Resume: skip if file already exists with correct size
            if fileManager.fileExists(atPath: destFile.path) {
                let attrs = try? fileManager.attributesOfItem(atPath: destFile.path)
                let existingSize = attrs?[.size] as? Int64 ?? 0
                if existingSize > 0 {
                    completedBytesForPreviousFiles += existingSize
                    progress = Double(completedBytesForPreviousFiles) / Double(totalBytesForAllFiles)
                    continue  // Already downloaded
                }
            }

            let fileName = (file.localPath as NSString).lastPathComponent
            status = "Downloading \(fileName)..."

            guard let url = URL(string: "\(base)/\(file.remotePath)") else { continue }
            try fileManager.createDirectory(at: destFile.deletingLastPathComponent(), withIntermediateDirectories: true)

            let tempFile = try await downloadSingleFile(url)
            try? fileManager.removeItem(at: destFile)
            try fileManager.moveItem(at: tempFile, to: destFile)

            let downloadedSize = (try? fileManager.attributesOfItem(atPath: destFile.path))?[.size] as? Int64 ?? file.estimatedSize
            completedBytesForPreviousFiles += downloadedSize
            progress = Double(completedBytesForPreviousFiles) / Double(totalBytesForAllFiles)
        }
    }

    // MARK: - Single File Download

    private func downloadSingleFile(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.downloadContinuation = continuation
            let task = session.downloadTask(with: url)
            task.resume()
        }
    }

    // MARK: - ZIP

    private func unzipFile(_ zipURL: URL, to destDir: URL) throws {
        #if targetEnvironment(simulator) || os(macOS)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = ["-xk", zipURL.path, destDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        #else
        try extractZipNative(from: zipURL, to: destDir)
        #endif
    }

    #if !targetEnvironment(simulator) && !os(macOS)
    private func extractZipNative(from zipURL: URL, to destDir: URL) throws {
        let data = try Data(contentsOf: zipURL)
        guard data.count > 22 else { throw DownloadError.extractionFailed }
        var eocdOffset = data.count - 22
        while eocdOffset >= 0 {
            if data[eocdOffset] == 0x50 && data[eocdOffset+1] == 0x4B &&
               data[eocdOffset+2] == 0x05 && data[eocdOffset+3] == 0x06 { break }
            eocdOffset -= 1
        }
        guard eocdOffset >= 0 else { throw DownloadError.extractionFailed }
        let cdOffset = Int(data[eocdOffset+16..<eocdOffset+20].withUnsafeBytes { $0.load(as: UInt32.self) })
        let cdCount = Int(data[eocdOffset+10..<eocdOffset+12].withUnsafeBytes { $0.load(as: UInt16.self) })
        var pos = cdOffset
        for _ in 0..<cdCount {
            guard data[pos] == 0x50, data[pos+1] == 0x4B else { break }
            let uncompSize = Int(data[pos+24..<pos+28].withUnsafeBytes { $0.load(as: UInt32.self) })
            let nameLen = Int(data[pos+28..<pos+30].withUnsafeBytes { $0.load(as: UInt16.self) })
            let extraLen = Int(data[pos+30..<pos+32].withUnsafeBytes { $0.load(as: UInt16.self) })
            let commentLen = Int(data[pos+32..<pos+34].withUnsafeBytes { $0.load(as: UInt16.self) })
            let localOffset = Int(data[pos+42..<pos+46].withUnsafeBytes { $0.load(as: UInt32.self) })
            let name = String(data: data[pos+46..<pos+46+nameLen], encoding: .utf8) ?? ""
            let destPath = destDir.appendingPathComponent(name)
            if name.hasSuffix("/") {
                try fileManager.createDirectory(at: destPath, withIntermediateDirectories: true)
            } else {
                try fileManager.createDirectory(at: destPath.deletingLastPathComponent(), withIntermediateDirectories: true)
                let lnl = Int(data[localOffset+26..<localOffset+28].withUnsafeBytes { $0.load(as: UInt16.self) })
                let lel = Int(data[localOffset+28..<localOffset+30].withUnsafeBytes { $0.load(as: UInt16.self) })
                let ds = localOffset + 30 + lnl + lel
                try Data(data[ds..<ds+uncompSize]).write(to: destPath)
            }
            pos += 46 + nameLen + extraLen + commentLen
        }
    }
    #endif

    private var modelsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("Models")
    }
}

// MARK: - URLSession Delegate

extension ModelDownloader: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let dest = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try fileManager.moveItem(at: location, to: dest)
            downloadContinuation?.resume(returning: dest)
        } catch {
            downloadContinuation?.resume(throwing: error)
        }
        downloadContinuation = nil
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let currentTotal = completedBytesForPreviousFiles + totalBytesWritten
        let overallProgress = Double(currentTotal) / Double(max(totalBytesForAllFiles, 1))

        let mbDone = Double(currentTotal) / 1_000_000
        let mbTotal = Double(totalBytesForAllFiles) / 1_000_000

        Task { @MainActor in
            self.progress = min(overallProgress, 0.99)
            self.status = String(format: "%.0f / %.0f MB", mbDone, mbTotal)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            downloadContinuation?.resume(throwing: error)
            downloadContinuation = nil
        }
    }
}

enum DownloadError: LocalizedError {
    case invalidURL, extractionFailed
    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid download URL"
        case .extractionFailed: return "Failed to extract model"
        }
    }
}
