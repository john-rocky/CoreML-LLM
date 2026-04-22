import Foundation

/// Lightweight HuggingFace-snapshot downloader for FunctionGemma and
/// EmbeddingGemma bundles.
///
/// Separate from the heavy `ModelDownloader` (which is tied to the sample
/// app's UI: @Observable, background URLSession, pause/resume). This one
/// is a plain async function — programmatic-friendly for LocalAIKit or any
/// library consumer that just wants a "give me the bundle at this local
/// directory" call.
///
/// Gated HuggingFace models require the caller to pass an `hfToken` (the
/// `hf_...` read token issued from https://huggingface.co/settings/tokens).
/// Anonymous public repos work without a token.
public enum Gemma3BundleDownloader {

    /// Canonical file lists for each bundle. Centralised here so the API
    /// surface stays `download(modelId:into:)` without the caller needing
    /// to know the HF layout.
    public enum Model: String, Sendable {
        case functionGemma270m = "functiongemma-270m"
        case embeddingGemma300m = "embeddinggemma-300m"

        /// Default HuggingFace repo for each bundle. Override via the
        /// `download(customRepo:…)` entry point if you re-host the bundle.
        public var defaultRepo: String {
            switch self {
            case .functionGemma270m:  return "mlboydaisuke/functiongemma-270m-coreml"
            case .embeddingGemma300m: return "mlboydaisuke/embeddinggemma-300m-coreml"
            }
        }

        /// Files that must be present for the bundle to load. Paths are
        /// relative to the repo root (on HF) and to the local directory
        /// (after download).
        public var bundleFiles: [String] {
            switch self {
            case .functionGemma270m:
                // Compiled mlmodelc path layout (preferred to save the
                // on-device `MLModel.compileModel` step). Falls back to
                // the .mlpackage if the host re-uploads the raw package.
                return [
                    "model.mlmodelc/weights/weight.bin",
                    "model.mlmodelc/coremldata.bin",
                    "model.mlmodelc/model.mil",
                    "model.mlmodelc/metadata.json",
                    "model.mlmodelc/analytics/coremldata.bin",
                    "model_config.json",
                    "hf_model/tokenizer.json",
                    "hf_model/tokenizer_config.json",
                    "hf_model/config.json",
                    "hf_model/special_tokens_map.json",
                    "hf_model/chat_template.jinja",
                    "cos_sliding.npy", "sin_sliding.npy",
                    "cos_full.npy", "sin_full.npy",
                ]
            case .embeddingGemma300m:
                return [
                    "encoder.mlmodelc/weights/weight.bin",
                    "encoder.mlmodelc/coremldata.bin",
                    "encoder.mlmodelc/model.mil",
                    "encoder.mlmodelc/metadata.json",
                    "encoder.mlmodelc/analytics/coremldata.bin",
                    "model_config.json",
                    "hf_model/tokenizer.json",
                    "hf_model/tokenizer_config.json",
                    "hf_model/config.json",
                    "hf_model/special_tokens_map.json",
                ]
            }
        }

        /// Files that may legitimately be missing from some uploads
        /// (descriptive / optional). A 404 on these is not fatal.
        public var optionalFiles: Set<String> {
            [
                "model.mlmodelc/metadata.json",
                "model.mlmodelc/analytics/coremldata.bin",
                "encoder.mlmodelc/metadata.json",
                "encoder.mlmodelc/analytics/coremldata.bin",
                "hf_model/chat_template.jinja",
                "hf_model/special_tokens_map.json",
            ]
        }
    }

    public struct Progress: Sendable {
        public let bytesReceived: Int64
        public let bytesTotal: Int64
        public let currentFile: String
    }

    public enum Error: Swift.Error, LocalizedError {
        case httpStatus(Int, url: String, body: String)
        case missingLocalFile(String)
        public var errorDescription: String? {
            switch self {
            case .httpStatus(let s, let u, let b):
                return "HTTP \(s) fetching \(u) — \(b.prefix(120))"
            case .missingLocalFile(let p):
                return "Missing local file after download: \(p)"
            }
        }
    }

    // MARK: - Public entry points

    /// Download `model` from its default HF repo into `directory/<model>/`.
    /// Skips files already present on disk. Returns the local bundle URL.
    @discardableResult
    public static func download(
        _ model: Model,
        into directory: URL,
        hfToken: String? = nil,
        onProgress: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        try await download(
            customRepo: model.defaultRepo,
            bundleFiles: model.bundleFiles,
            optionalFiles: model.optionalFiles,
            folderName: model.rawValue,
            into: directory,
            hfToken: hfToken,
            onProgress: onProgress)
    }

    /// Download an arbitrary HF snapshot by `customRepo` + explicit file list.
    /// Useful for custom re-hosts or private repos.
    @discardableResult
    public static func download(
        customRepo repo: String,
        bundleFiles: [String],
        optionalFiles: Set<String> = [],
        folderName: String,
        into directory: URL,
        hfToken: String? = nil,
        onProgress: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let bundle = directory.appendingPathComponent(folderName)
        try FileManager.default.createDirectory(
            at: bundle, withIntermediateDirectories: true)

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 7200
        let session = URLSession(configuration: config)

        var totalBytes: Int64 = 0
        var totalReceived: Int64 = 0

        for relativePath in bundleFiles {
            let localURL = bundle.appendingPathComponent(relativePath)
            if FileManager.default.fileExists(atPath: localURL.path),
               let sz = (try? FileManager.default.attributesOfItem(atPath: localURL.path))?[.size] as? Int64,
               sz > 0 {
                totalBytes += sz
                totalReceived += sz
                onProgress?(Progress(
                    bytesReceived: totalReceived, bytesTotal: totalBytes,
                    currentFile: relativePath))
                continue
            }

            try FileManager.default.createDirectory(
                at: localURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)

            let remoteURL = "https://huggingface.co/\(repo)/resolve/main/\(relativePath)"
            var request = URLRequest(url: URL(string: remoteURL)!)
            if let hfToken {
                request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
            }

            let (tmpURL, response) = try await session.download(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
                if http.statusCode == 404 && optionalFiles.contains(relativePath) {
                    try? FileManager.default.removeItem(at: tmpURL)
                    continue
                }
                let body = (try? String(contentsOf: tmpURL, encoding: .utf8)) ?? ""
                try? FileManager.default.removeItem(at: tmpURL)
                throw Error.httpStatus(http.statusCode, url: remoteURL, body: body)
            }

            try? FileManager.default.removeItem(at: localURL)
            try FileManager.default.moveItem(at: tmpURL, to: localURL)

            if let sz = (try? FileManager.default.attributesOfItem(atPath: localURL.path))?[.size] as? Int64 {
                totalBytes += sz
                totalReceived += sz
            }
            onProgress?(Progress(
                bytesReceived: totalReceived, bytesTotal: totalBytes,
                currentFile: relativePath))
        }

        // Sanity check: all non-optional files must exist.
        for relativePath in bundleFiles where !optionalFiles.contains(relativePath) {
            let p = bundle.appendingPathComponent(relativePath).path
            guard FileManager.default.fileExists(atPath: p) else {
                throw Error.missingLocalFile(relativePath)
            }
        }

        return bundle
    }

    /// Return the local bundle URL if all required files are already on disk
    /// under `directory/<model>/`, else `nil`.
    public static func localBundle(
        _ model: Model, under directory: URL
    ) -> URL? {
        let bundle = directory.appendingPathComponent(model.rawValue)
        let fm = FileManager.default
        for rel in model.bundleFiles where !model.optionalFiles.contains(rel) {
            if !fm.fileExists(atPath: bundle.appendingPathComponent(rel).path) {
                return nil
            }
        }
        return bundle
    }
}
