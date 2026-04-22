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
                    // Batched prefill mlpackage (T=32). ~10× faster prompt
                    // ingestion; `Gemma3FunctionGemma.load` picks it up
                    // automatically when present.
                    "prefill_t32.mlmodelc/weights/weight.bin",
                    "prefill_t32.mlmodelc/coremldata.bin",
                    "prefill_t32.mlmodelc/model.mil",
                    "prefill_t32.mlmodelc/metadata.json",
                    "prefill_t32.mlmodelc/analytics/coremldata.bin",
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
                "prefill_t32.mlmodelc/weights/weight.bin",
                "prefill_t32.mlmodelc/coremldata.bin",
                "prefill_t32.mlmodelc/model.mil",
                "prefill_t32.mlmodelc/metadata.json",
                "prefill_t32.mlmodelc/analytics/coremldata.bin",
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

        public init(bytesReceived: Int64, bytesTotal: Int64, currentFile: String) {
            self.bytesReceived = bytesReceived
            self.bytesTotal = bytesTotal
            self.currentFile = currentFile
        }
    }

    /// Per-file expected sizes for the default repos, so the UI can show a
    /// realistic "bytesTotal" (and non-zero percentage) before any HEAD
    /// request completes. Values are rounded from the actual HF-hosted
    /// artifact sizes; Content-Length from the server overrides at download
    /// time. Missing entries fall back to 1 MB placeholder.
    static let defaultFileSizes: [String: Int64] = [
        // FunctionGemma (INT8-quantized, decode + prefill_t32)
        "model.mlmodelc/weights/weight.bin":         422_000_000,
        "model.mlmodelc/coremldata.bin":                   1_000,
        "model.mlmodelc/model.mil":                      400_000,
        "model.mlmodelc/metadata.json":                    8_000,
        "model.mlmodelc/analytics/coremldata.bin":           250,
        "prefill_t32.mlmodelc/weights/weight.bin":   422_000_000,
        "prefill_t32.mlmodelc/coremldata.bin":             1_000,
        "prefill_t32.mlmodelc/model.mil":                450_000,
        "prefill_t32.mlmodelc/metadata.json":              8_000,
        "prefill_t32.mlmodelc/analytics/coremldata.bin":     250,
        "cos_sliding.npy":                             2_097_280,
        "sin_sliding.npy":                             2_097_280,
        "cos_full.npy":                                2_097_280,
        "sin_full.npy":                                2_097_280,
        // EmbeddingGemma (INT8)
        "encoder.mlmodelc/weights/weight.bin":       295_000_000,
        "encoder.mlmodelc/coremldata.bin":                  1_000,
        "encoder.mlmodelc/model.mil":                     700_000,
        "encoder.mlmodelc/metadata.json":                   8_000,
        "encoder.mlmodelc/analytics/coremldata.bin":          250,
        // Tokenizer (shared between both)
        "hf_model/tokenizer.json":                    33_000_000,
        "hf_model/tokenizer_config.json":                 80_000,
        "hf_model/config.json":                            1_000,
        "hf_model/special_tokens_map.json":                  900,
        "hf_model/chat_template.jinja":                    2_000,
        "hf_model/added_tokens.json":                      3_000,
        "model_config.json":                               1_500,
    ]

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
    ///
    /// Progress is reported **incrementally during each file** (via
    /// `URLSessionDownloadDelegate.didWriteData`), so even a single 800 MB
    /// weight.bin updates the UI byte-by-byte. `bytesTotal` is pre-seeded
    /// from `defaultFileSizes` so the percentage is meaningful before any
    /// network round-trip completes.
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

        // Pre-seed total bytes from known sizes so the UI shows a real
        // denominator from the very first progress update.
        var estimatedTotal: Int64 = 0
        for rel in bundleFiles {
            estimatedTotal += defaultFileSizes[rel] ?? 1_000_000
        }

        // Bytes already on disk (skipped files) + bytes downloaded so far.
        let progressState = ProgressState(totalEstimated: estimatedTotal)

        for relativePath in bundleFiles {
            let localURL = bundle.appendingPathComponent(relativePath)

            // Already present → count and skip.
            if FileManager.default.fileExists(atPath: localURL.path),
               let sz = (try? FileManager.default.attributesOfItem(atPath: localURL.path))?[.size] as? Int64,
               sz > 0 {
                await progressState.addCompleted(bytes: sz)
                let snap = await progressState.snapshot(currentFile: relativePath)
                onProgress?(snap)
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

            // Tell the caller which file we're starting BEFORE the bytes flow,
            // so the UI can show the filename immediately.
            onProgress?(await progressState.snapshot(currentFile: relativePath))

            let (tmpURL, response) = try await downloadWithProgress(
                request: request,
                file: relativePath,
                state: progressState,
                onUpdate: { snap in onProgress?(snap) }
            )
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
                await progressState.commitFile(bytes: sz)
            }
            onProgress?(await progressState.snapshot(currentFile: relativePath))
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

    /// Actor guarding the incremental counters so `didWriteData` callbacks
    /// from the download delegate can update progress from any queue without
    /// racing the main async flow.
    private actor ProgressState {
        private var totalEstimated: Int64
        private var completedBytes: Int64 = 0        // bytes from files fully done
        private var inFlightBytes: Int64 = 0         // bytes downloaded so far on current file

        init(totalEstimated: Int64) {
            self.totalEstimated = totalEstimated
        }

        func addCompleted(bytes: Int64) { completedBytes += bytes }

        func updateInFlight(bytes: Int64, expectedTotal: Int64) {
            inFlightBytes = bytes
            // If the server gave us an accurate Content-Length that's bigger
            // than our estimate, bump the denominator so percentage doesn't
            // exceed 100%.
            if expectedTotal > 0 {
                let roomLeft = totalEstimated - completedBytes
                if expectedTotal > roomLeft {
                    totalEstimated += (expectedTotal - roomLeft)
                }
            }
        }

        func commitFile(bytes: Int64) {
            completedBytes += bytes
            inFlightBytes = 0
        }

        func snapshot(currentFile: String) -> Progress {
            let received = completedBytes + inFlightBytes
            let total = max(totalEstimated, received)
            return Progress(bytesReceived: received, bytesTotal: total,
                            currentFile: currentFile)
        }
    }

    /// Classic `downloadTask` + delegate bridged to async via a single
    /// continuation. We hit this route (rather than the simpler
    /// `session.download(for:delegate:)` async API) because empirically the
    /// async variant silently skips the `didWriteData` calls on macOS/iOS
    /// builds we tested against, so the caller only sees 0% until each file
    /// finishes. The classic task + session delegate is well-trodden and
    /// delivers incremental callbacks on a predictable schedule.
    private static func downloadWithProgress(
        request: URLRequest,
        file: String,
        state: ProgressState,
        onUpdate: @escaping (Progress) -> Void
    ) async throws -> (URL, URLResponse) {
        final class Delegate: NSObject, URLSessionDownloadDelegate {
            let file: String
            let state: ProgressState
            let onUpdate: (Progress) -> Void
            var continuation: CheckedContinuation<(URL, URLResponse), Swift.Error>?
            private var lastUpdate = Date.distantPast
            private var movedURL: URL?

            init(file: String, state: ProgressState,
                 onUpdate: @escaping (Progress) -> Void) {
                self.file = file
                self.state = state
                self.onUpdate = onUpdate
            }

            func urlSession(_ session: URLSession,
                            downloadTask: URLSessionDownloadTask,
                            didWriteData bytesWritten: Int64,
                            totalBytesWritten: Int64,
                            totalBytesExpectedToWrite: Int64) {
                let now = Date()
                guard now.timeIntervalSince(lastUpdate) > 0.1 else { return }
                lastUpdate = now
                let f = file
                Task {
                    await state.updateInFlight(
                        bytes: totalBytesWritten,
                        expectedTotal: totalBytesExpectedToWrite)
                    let snap = await state.snapshot(currentFile: f)
                    onUpdate(snap)
                }
            }

            func urlSession(_ session: URLSession,
                            downloadTask: URLSessionDownloadTask,
                            didFinishDownloadingTo location: URL) {
                // The tmp file at `location` is deleted when this delegate
                // call returns. Move it somewhere stable so the caller's
                // `moveItem` succeeds.
                let stable = FileManager.default.temporaryDirectory
                    .appendingPathComponent("gemma3dl-\(UUID().uuidString)")
                try? FileManager.default.moveItem(at: location, to: stable)
                movedURL = stable
            }

            func urlSession(_ session: URLSession, task: URLSessionTask,
                            didCompleteWithError error: (any Swift.Error)?) {
                guard let cont = continuation else { return }
                continuation = nil
                if let error {
                    cont.resume(throwing: error)
                    return
                }
                guard let moved = movedURL, let resp = task.response else {
                    cont.resume(throwing: URLError(.badServerResponse))
                    return
                }
                cont.resume(returning: (moved, resp))
            }
        }

        let delegate = Delegate(file: file, state: state, onUpdate: onUpdate)
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 7200
        config.waitsForConnectivity = true
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                delegate.continuation = cont
                let task = session.downloadTask(with: request)
                task.resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
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
