import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Downloads and caches CoreML models with background download support.
@Observable
final class ModelDownloader: NSObject {
    var isDownloading = false
    var progress: Double = 0
    var status = ""
    var availableModels: [ModelInfo] = ModelInfo.defaults

    private let fileManager = FileManager.default
    private var backgroundSession: URLSession!
    private var downloadContinuation: CheckedContinuation<URL, Error>?
    private var currentFileProgress: (completed: Int, total: Int) = (0, 1)

    struct ModelInfo: Identifiable {
        let id: String
        let name: String
        let size: String
        let downloadURL: String
        let folderName: String

        static let defaults: [ModelInfo] = [
            ModelInfo(
                id: "gemma4-e2b",
                name: "Gemma 4 E2B (Multimodal)",
                size: "2.7 GB",
                downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/main",
                folderName: "gemma4-e2b"
            ),
            ModelInfo(
                id: "qwen2.5-0.5b",
                name: "Qwen2.5 0.5B (Text)",
                size: "309 MB",
                downloadURL: "https://github.com/john-rocky/CoreML-LLM/releases/download/v0.1.0/qwen2.5-0.5b-coreml.zip",
                folderName: "qwen2.5-0.5b"
            ),
        ]
    }

    override init() {
        super.init()
        let config = URLSessionConfiguration.background(withIdentifier: "com.coreml-llm.download")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        backgroundSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func isDownloaded(_ model: ModelInfo) -> Bool {
        localModelURL(for: model) != nil
    }

    func localModelURL(for model: ModelInfo) -> URL? {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        let pkg = dir.appendingPathComponent("model.mlpackage")
        let weight = dir.appendingPathComponent("model.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
        // Only consider downloaded if weight.bin exists (prevents partial download issues)
        guard fileManager.fileExists(atPath: pkg.path),
              fileManager.fileExists(atPath: weight.path) else { return nil }
        return pkg
    }

    func download(_ model: ModelInfo) async throws -> URL {
        if let existing = localModelURL(for: model) { return existing }

        isDownloading = true
        progress = 0
        status = "Downloading \(model.name)..."
        defer { isDownloading = false }

        // Clean up any partial download
        let destDir = modelsDirectory.appendingPathComponent(model.folderName)
        try? fileManager.removeItem(at: destDir)
        try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)

        if model.downloadURL.contains("huggingface.co") {
            try await downloadFromHuggingFace(model, to: destDir)
        } else {
            guard let url = URL(string: model.downloadURL) else {
                throw DownloadError.invalidURL
            }
            let tempZip = try await downloadSingleFile(url)
            status = "Extracting..."
            try unzipFile(tempZip, to: destDir)
            try? fileManager.removeItem(at: tempZip)
        }

        guard let result = localModelURL(for: model) else {
            throw DownloadError.extractionFailed
        }

        status = "Ready"
        progress = 1.0
        return result
    }

    func delete(_ model: ModelInfo) throws {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        if fileManager.fileExists(atPath: dir.path) {
            try fileManager.removeItem(at: dir)
        }
    }

    // MARK: - HuggingFace Download

    private func downloadFromHuggingFace(_ model: ModelInfo, to destDir: URL) async throws {
        let base = model.downloadURL
        var files: [(String, String)] = [
            ("model.mlpackage/Manifest.json", "model.mlpackage/Manifest.json"),
            ("model.mlpackage/Data/com.apple.CoreML/model.mlmodel", "model.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            ("model.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "model.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            ("model_config.json", "model_config.json"),
            ("hf_model/tokenizer.json", "hf_model/tokenizer.json"),
        ]

        if model.id.contains("gemma") {
            files += [
                ("vision.mlpackage/Manifest.json", "vision.mlpackage/Manifest.json"),
                ("vision.mlpackage/Data/com.apple.CoreML/model.mlmodel", "vision.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
                ("vision.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "vision.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            ]
        }

        currentFileProgress = (0, files.count)

        for (i, (remotePath, localPath)) in files.enumerated() {
            let fileName = (localPath as NSString).lastPathComponent
            status = "Downloading \(fileName) (\(i+1)/\(files.count))..."
            currentFileProgress = (i, files.count)

            guard let url = URL(string: "\(base)/\(remotePath)") else { continue }
            let destFile = destDir.appendingPathComponent(localPath)
            try fileManager.createDirectory(at: destFile.deletingLastPathComponent(), withIntermediateDirectories: true)

            let tempFile = try await downloadSingleFile(url)
            try? fileManager.removeItem(at: destFile)
            try fileManager.moveItem(at: tempFile, to: destFile)

            progress = Double(i + 1) / Double(files.count)
        }
    }

    // MARK: - Background Download

    private func downloadSingleFile(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.downloadContinuation = continuation
            let task = backgroundSession.downloadTask(with: url)
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
                let localNameLen = Int(data[localOffset+26..<localOffset+28].withUnsafeBytes { $0.load(as: UInt16.self) })
                let localExtraLen = Int(data[localOffset+28..<localOffset+30].withUnsafeBytes { $0.load(as: UInt16.self) })
                let dataStart = localOffset + 30 + localNameLen + localExtraLen
                try Data(data[dataStart..<dataStart+uncompSize]).write(to: destPath)
            }
            pos += 46 + nameLen + extraLen + commentLen
        }
    }
    #endif

    private var modelsDirectory: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("Models")
    }
}

// MARK: - URLSession Delegate (Background Download)

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
        guard totalBytesExpectedToWrite > 0 else { return }
        let fileProgress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        let (completed, total) = currentFileProgress
        let overallProgress = (Double(completed) + fileProgress) / Double(total)
        Task { @MainActor in
            self.progress = overallProgress
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
