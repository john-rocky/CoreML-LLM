import CoreML
import Foundation
import HuggingFace
import Tokenizers

/// Runtime for Perplexity's pplx-embed (a bidirectional Qwen3 encoder → masked
/// mean pool → tanh int8 quantize). Exposes the official pplx-embed contract:
///
///   plain   : `[String] -> [[Int8]]`        (1024-d int8 per text)
///   context : `[[String]] -> [[[Int8]]]`    (per-document late chunking;
///             per-chunk 1024-d int8)
///
/// Each call also exposes the `binary` (+1/-1 Float) and `ubinary` (packed
/// UInt8[dim/8]) variants.
///
/// Output-format design decision. The underlying mlpackage emits native int8
/// already (the `int8` output IS the deliverable; it's readable in Swift via
/// the dtype-agnostic NSNumber subscript even though the Python CoreML bridge
/// can't read int8 on macOS 26). We derive the other two formats directly from
/// the int8 vector:
///
///   binary[i]  = int8[i] >= 0 ? +1 : -1
///   ubinary    = packbits(int8[i] >= 0)
///
/// This is bit-exact with the reference `st_quantize` everywhere except the
/// measure-zero x≈0 case: the reference branches on the raw pre-tanh value
/// `x >= 0`, whereas we branch on the rounded int8. Since `round(tanh(x)*127)`
/// is 0 only in a tiny neighbourhood of x=0 and is otherwise sign-faithful,
/// the int8-derived sign agrees with the raw sign except when |x| is so small
/// it rounds to int8 0 — there we map 0 to the `>= 0` (positive) branch to
/// match the reference's tie direction. For strictly bit-exact binary/ubinary
/// against a `pooled_fp16`-output model, build with
/// `--output-mode pooled_fp16` and apply all three quantizers in Swift; we ship
/// the int8-derived path because it needs only one model and one forward pass.
///
/// I/O contract of the underlying mlpackages (from build_pplx_embed_bundle.py):
///   plain:
///     input_ids       (1, L)   int32
///     attention_mask  (1, L)   fp16   (1.0 valid, 0.0 pad)
///     → embedding     (1, 1024) int8
///   context:
///     input_ids       (1, L)   int32
///     attention_mask  (1, L)   fp16
///     pool_matrix     (32, L)  fp16   (row k = 1/n_k over chunk k's span)
///     → embedding     (32, 1024) int8 (only first n_chunks rows are valid)
public final class PplxEmbed {

    /// The three published pplx-embed output formats.
    public enum Format: String, Sendable {
        case int8
        case binary
        case ubinary
    }

    /// Per-bundle config parsed from model_config.json.
    public struct BucketConfig: Sendable {
        public let maxSeqLen: Int    // for dynamic, the RangeDim upper bound
        public let embedDim: Int
        public let variant: String   // "plain" | "context"
        public let dynamic: Bool     // flexible RangeDim model (GPU; the >max-bucket catch-all)
        public let url: URL
    }

    public static let embedDim = 1024
    public static let nMaxChunks = 32

    private let tokenizer: Tokenizer
    private let sepTokenId: Int
    private let computeUnits: MLComputeUnits

    /// Fixed ANE buckets, sorted ascending by maxSeqLen.
    private let buckets: [BucketConfig]
    /// Optional flexible RangeDim catch-all for inputs larger than the biggest
    /// fixed bucket. Runs on the GPU (flexible shapes force CPU fallback on ANE),
    /// non-padded (actual length). nil if no dynamic bundle was provided.
    private let dynamicBucket: BucketConfig?
    private let variant: String

    /// Lazily compiled+loaded fixed-bucket models, keyed by bucket maxSeqLen.
    private var loaded: [Int: MLModel] = [:]
    /// Lazily loaded dynamic model (GPU).
    private var dynamicModel: MLModel?
    private let lock = NSLock()

    private init(tokenizer: Tokenizer, sepTokenId: Int, buckets: [BucketConfig],
                 dynamicBucket: BucketConfig?, variant: String, computeUnits: MLComputeUnits) {
        self.tokenizer = tokenizer
        self.sepTokenId = sepTokenId
        self.buckets = buckets
        self.dynamicBucket = dynamicBucket
        self.variant = variant
        self.computeUnits = computeUnits
    }

    // MARK: - Loading

    /// Load a pplx-embed bundle.
    ///
    /// `bundleDir` may be either:
    ///   * a directory of bucket subdirectories (e.g. `output/pplx-embed/`
    ///     containing `L512-int8/`, `L1024-int8/`, …) — all int8 buckets are
    ///     discovered and used for token-length-based bucket selection, or
    ///   * a single bucket directory directly containing `encoder.mlpackage`
    ///     (e.g. `output/pplx-embed-context/L512-int8/`).
    ///
    /// Models are compiled + loaded lazily on first use per bucket; the
    /// tokenizer is loaded eagerly from the first bucket's `hf_model/`.
    public static func load(
        bundleDir: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> PplxEmbed {
        let fm = FileManager.default
        var buckets: [BucketConfig] = []

        // A single bucket dir directly contains encoder.mlpackage / .mlmodelc.
        let isSingle = fm.fileExists(atPath: bundleDir.appendingPathComponent("encoder.mlpackage").path)
            || fm.fileExists(atPath: bundleDir.appendingPathComponent("encoder.mlmodelc").path)

        if isSingle {
            if let c = parseBucket(at: bundleDir) { buckets.append(c) }
        } else {
            let entries = (try? fm.contentsOfDirectory(at: bundleDir,
                includingPropertiesForKeys: nil)) ?? []
            for e in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard (try? e.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
                else { continue }
                if let c = parseBucket(at: e) { buckets.append(c) }
            }
        }

        guard !buckets.isEmpty else {
            throw CoreMLLLMError.modelNotFound(
                "no pplx-embed bucket with encoder.mlpackage/.mlmodelc under \(bundleDir.path)")
        }

        // Dominant variant from the fixed buckets (a dynamic-only bundle is plain).
        let variant = (buckets.first { !$0.dynamic } ?? buckets.first!).variant
        let matching = buckets.filter { $0.variant == variant }
        // Fixed ANE buckets (sorted ascending) + at most one dynamic GPU catch-all.
        let fixed = matching.filter { !$0.dynamic }.sorted { $0.maxSeqLen < $1.maxSeqLen }
        let dynamic = matching.first { $0.dynamic }

        let hfDir = (fixed.first ?? dynamic!).url.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: hfDir)
        let sepId = sepTokenId(fromHFDir: hfDir) ?? 151643

        return PplxEmbed(tokenizer: tokenizer, sepTokenId: sepId, buckets: fixed,
                         dynamicBucket: dynamic, variant: variant, computeUnits: computeUnits)
    }

    /// Download selected buckets from a HuggingFace repo, then load them.
    ///
    /// Publishes-as-download path (companion to `conversion/upload_pplx_embed.py`):
    /// the repo holds one subfolder per bucket plus a top-level `manifest.json`
    /// inventory. This fetches the manifest, selects the requested fixed buckets
    /// (+ the dynamic GPU catch-all, if present) for `variant`, and downloads **only
    /// those subfolders' chosen-format files** via the HF Swift Hub client
    /// (`HubClient.downloadSnapshot`, glob-filtered). Crucially, that client uses HF's
    /// **content-addressed cache**: the encoder `weight.bin` is byte-identical across
    /// every bucket, so it is fetched **once by etag** and reused for the rest — native
    /// download dedup (the default 3-bucket pull moves ~1.15 GB, not ~3.5 GB). The
    /// returned snapshot is then handed to the local `load(bundleDir:)` unchanged.
    ///
    /// - Parameters:
    ///   - repo: HF repo id, e.g. `"<account>/pplx-embed-coreml"`.
    ///   - buckets: fixed bucket sizes (L) to fetch, e.g. `[512, 1024, 2048]`.
    ///     The dynamic catch-all (if the repo has one) is always included so long
    ///     inputs still work.
    ///   - into: ignored when `nil` (uses the shared HF cache, enabling cross-call and
    ///     cross-client dedup); pass a directory to download into it instead.
    ///   - variant: `"plain"` or `"context"`.
    ///   - preferCompiled: when the repo ships both formats, download the precompiled
    ///     `.mlmodelc` (no on-device compile) rather than the `.mlpackage`. Only the
    ///     chosen format's weights are fetched per bucket, never both.
    @discardableResult
    public static func load(
        repo: String,
        buckets: [Int] = [512, 1024, 2048],
        into directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String = "plain",
        preferCompiled: Bool = true,
        hfToken: String? = nil,
        onProgress: ((Double) -> Void)? = nil
    ) async throws -> PplxEmbed {
        let manifest = try await fetchManifest(repo: repo, hfToken: hfToken)
        let want = Set(buckets)

        // Select this variant's buckets (requested sizes + any dynamic catch-all) and
        // collect each one's exact chosen-format file paths.
        var matching: [String] = []
        var hasContextSubfolder = false
        for b in manifest.buckets where b.variant == variant {
            guard b.dynamic || want.contains(b.maxSeqLen) else { continue }
            if b.subfolder.hasPrefix("context/") { hasContextSubfolder = true }
            matching.append(contentsOf: b.selectFiles(preferCompiled: preferCompiled))
        }
        guard !matching.isEmpty else {
            throw CoreMLLLMError.modelNotFound(
                "no \(variant) buckets in \(repo) manifest match \(buckets)")
        }

        let client = makeHubClient(hfToken: hfToken)
        let repoID = Repo.ID(stringLiteral: repo)
        let snapshot: URL
        if let directory {
            snapshot = try await client.downloadSnapshot(
                of: repoID, kind: .model, to: directory, matching: matching,
                progressHandler: { p in onProgress?(p.fractionCompleted) })
        } else {
            snapshot = try await client.downloadSnapshot(
                of: repoID, kind: .model, matching: matching,
                progressHandler: { p in onProgress?(p.fractionCompleted) })
        }

        // Context subfolders live under `<repo>/context/`; point load there so the
        // bucket dirs (context/L512-int8/…) are discovered as top-level entries.
        let loadDir = (variant == "context" && hasContextSubfolder)
            ? snapshot.appendingPathComponent("context") : snapshot
        return try await load(bundleDir: loadDir, computeUnits: computeUnits)
    }

    /// Build a `HubClient` (env/anonymous token, or an explicit bearer token).
    private static func makeHubClient(hfToken: String?) -> HubClient {
        if let hfToken, !hfToken.isEmpty {
            return HubClient(host: URL(string: "https://huggingface.co")!, bearerToken: hfToken)
        }
        return HubClient()
    }

    // MARK: - Manifest

    /// One bucket entry parsed from the repo's `manifest.json`.
    private struct ManifestBucket {
        let subfolder: String
        let variant: String
        let dynamic: Bool
        let maxSeqLen: Int
        let formats: [String]   // e.g. ["mlmodelc", "mlpackage"]
        let files: [String]     // exact repo-relative paths (subfolder-prefixed)

        /// Exact file paths to fetch for the chosen format: shared files
        /// (model_config.json, hf_model/…) + only the chosen format's encoder dir. We
        /// pass these to `downloadSnapshot(matching:)` as exact patterns rather than
        /// wildcards — `listFiles(recursive:)` also returns *directory* entries, and a
        /// glob like `encoder.mlmodelc/*` would match (and 404 trying to GET) the
        /// `analytics/`/`weights/` directories.
        func selectFiles(preferCompiled: Bool) -> [String] {
            let preferred = preferCompiled ? "mlmodelc" : "mlpackage"
            let chosen = formats.contains(preferred) ? preferred : (formats.first ?? preferred)
            let otherDir = "\(subfolder)/encoder.\(chosen == "mlmodelc" ? "mlpackage" : "mlmodelc")/"
            return files.filter { !$0.hasPrefix(otherDir) }
        }
    }
    private struct Manifest { let buckets: [ManifestBucket] }

    /// Fetch + parse `manifest.json` from a HF repo.
    private static func fetchManifest(repo: String, hfToken: String?) async throws -> Manifest {
        let urlStr = "https://huggingface.co/\(repo)/resolve/main/manifest.json"
        var req = URLRequest(url: URL(string: urlStr)!)
        if let hfToken { req.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization") }
        let (data, resp) = try await URLSession.shared.data(for: req)
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            throw Gemma3BundleDownloader.Error.httpStatus(
                http.statusCode, url: urlStr, body: String(data: data, encoding: .utf8) ?? "")
        }
        guard let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = j["buckets"] as? [[String: Any]]
        else { throw CoreMLLLMError.modelNotFound("malformed manifest.json in \(repo)") }

        let buckets: [ManifestBucket] = raw.compactMap { e in
            guard let subfolder = e["subfolder"] as? String else { return nil }
            let dynamic = (e["dynamic"] as? Bool) ?? false
            // Fixed buckets carry an integer "bucket"; dynamic uses dynamic_upper/max_seq_len.
            let maxSeqLen = (e["bucket"] as? Int)
                ?? (e["dynamic_upper"] as? Int)
                ?? (e["max_seq_len"] as? Int) ?? 0
            let variant = (e["variant"] as? String) ?? "plain"
            let formats = (e["formats"] as? [String]) ?? ["mlpackage"]
            let fileObjs = (e["files"] as? [[String: Any]]) ?? []
            let files = fileObjs.compactMap { $0["path"] as? String }
            return ManifestBucket(subfolder: subfolder, variant: variant,
                                  dynamic: dynamic, maxSeqLen: maxSeqLen,
                                  formats: formats, files: files)
        }
        return Manifest(buckets: buckets)
    }

    /// Parse a single bucket directory's model_config.json. Only accepts
    /// int8-output buckets (the deliverable format).
    private static func parseBucket(at dir: URL) -> BucketConfig? {
        let fm = FileManager.default
        let hasModel = fm.fileExists(atPath: dir.appendingPathComponent("encoder.mlpackage").path)
            || fm.fileExists(atPath: dir.appendingPathComponent("encoder.mlmodelc").path)
        guard hasModel,
              fm.fileExists(atPath: dir.appendingPathComponent("hf_model").path)
        else { return nil }

        let cfgURL = dir.appendingPathComponent("model_config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        let outputMode = j["output_mode"] as? String ?? "int8"
        guard outputMode == "int8" else { return nil }   // skip pooled_fp16 variants
        // Ship the fp16-weight models only; skip experimental weight-quant bundles
        // (they share output_mode "int8" but would duplicate a bucket size).
        let weightQuant = j["quantization_weights"] as? String ?? "fp16"
        guard weightQuant == "fp16" else { return nil }

        let dynamic = (j["dynamic"] as? Bool) ?? false
        // Fixed bucket: integer "bucket". Dynamic: "bucket" is a string ("1..N");
        // use dynamic_upper as the effective max.
        let maxSeqLen: Int
        if dynamic {
            maxSeqLen = (j["dynamic_upper"] as? Int) ?? (j["max_seq_len"] as? Int) ?? 8192
        } else {
            maxSeqLen = (j["bucket"] as? Int) ?? (j["max_seq_len"] as? Int) ?? 512
        }
        let embedDim = (j["hidden_size"] as? Int) ?? PplxEmbed.embedDim
        let variant = (j["variant"] as? String)
            ?? (dir.path.contains("context") ? "context" : "plain")

        return BucketConfig(maxSeqLen: maxSeqLen, embedDim: embedDim,
                            variant: variant, dynamic: dynamic, url: dir)
    }

    private static func sepTokenId(fromHFDir hfDir: URL) -> Int? {
        // <|endoftext|> id from added_tokens.json (pplx tokenizer: 151643).
        let url = hfDir.appendingPathComponent("added_tokens.json")
        guard let data = try? Data(contentsOf: url),
              let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = j["<|endoftext|>"] as? Int
        else { return nil }
        return id
    }

    private func model(forBucket L: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = loaded[L] { return m }
        guard let cfg = buckets.first(where: { $0.maxSeqLen == L }) else {
            throw CoreMLLLMError.modelNotFound("no loaded bucket for L=\(L)")
        }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        let compiled = cfg.url.appendingPathComponent("encoder.mlmodelc")
        let pkg = cfg.url.appendingPathComponent("encoder.mlpackage")
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            modelURL = compiled
        } else {
            modelURL = try compileSync(pkg)
        }
        let m = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        loaded[L] = m
        return m
    }

    /// Synchronous compile wrapper (MLModel.compileModel is async on newer SDKs
    /// but the legacy throwing overload is sync). Use the sync overload to keep
    /// the model accessor non-async.
    private func compileSync(_ pkg: URL) throws -> URL {
        try MLModel.compileModel(at: pkg)
    }

    /// Load the flexible RangeDim catch-all model on the GPU (flexible shapes
    /// force CPU fallback on the ANE, so this path is GPU-only).
    private func loadDynamicModel(_ cfg: BucketConfig) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = dynamicModel { return m }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let compiled = cfg.url.appendingPathComponent("encoder.mlmodelc")
        let pkg = cfg.url.appendingPathComponent("encoder.mlpackage")
        let url = FileManager.default.fileExists(atPath: compiled.path)
            ? compiled : try compileSync(pkg)
        let m = try MLModel(contentsOf: url, configuration: mlConfig)
        dynamicModel = m
        return m
    }

    /// Pick the smallest bucket whose maxSeqLen >= n; if none, the largest.
    private func bucket(forTokens n: Int) -> BucketConfig {
        for b in buckets where b.maxSeqLen >= n { return b }
        return buckets.last!
    }

    // MARK: - Plain API

    /// Encode texts into 1024-d int8 embeddings (one row per text).
    public func embed(_ texts: [String]) throws -> [[Int8]] {
        try texts.map { try embedOne($0) }
    }

    /// Encode texts and return the requested format.
    /// - int8:    `[[Int8]]` (1024-d)
    /// - binary:  `[[Float]]` (1024-d, +1/-1)
    /// - ubinary: `[[UInt8]]` (128 packed bytes)
    public func embedInt8(_ texts: [String]) throws -> [[Int8]] {
        try embed(texts)
    }

    public func embedBinary(_ texts: [String]) throws -> [[Float]] {
        try embed(texts).map { PplxEmbed.binary(fromInt8: $0) }
    }

    public func embedUBinary(_ texts: [String]) throws -> [[UInt8]] {
        try embed(texts).map { PplxEmbed.ubinary(fromInt8: $0) }
    }

    private func embedOne(_ text: String) throws -> [Int8] {
        var ids = tokenizer.encode(text: text)
        let largestFixed = buckets.last?.maxSeqLen ?? 0

        // Catch-all: inputs larger than the biggest fixed bucket go to the flexible
        // GPU model, non-padded (actual length, capped at the RangeDim upper bound).
        if let dyn = dynamicBucket, ids.count > largestFixed {
            let L = min(ids.count, dyn.maxSeqLen)
            if ids.count > L { ids = Array(ids.prefix(L)) }
            let n = ids.count
            let out = try loadDynamicModel(dyn).prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "input_ids": try makeInputIds(ids, L: n),
                "attention_mask": try makeAttentionMask(n: n, L: n),
            ]))
            return try readPlainRow(out)
        }

        // Fast path: smallest fixed ANE bucket that fits, padded to the bucket.
        let bucket = bucket(forTokens: ids.count)
        let L = bucket.maxSeqLen
        if ids.count > L { ids = Array(ids.prefix(L)) }
        let n = ids.count
        let out = try model(forBucket: L).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": try makeInputIds(ids, L: L),
            "attention_mask": try makeAttentionMask(n: n, L: L),
        ]))
        return try readPlainRow(out)
    }

    /// Read a (1, 1024) int8 "embedding" output into [Int8].
    private func readPlainRow(_ out: MLFeatureProvider) throws -> [Int8] {
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        let d = min(PplxEmbed.embedDim, arr.count)
        var vec = [Int8](repeating: 0, count: d)
        for i in 0..<d { vec[i] = Int8(arr[i].int8Value) }
        return vec
    }

    // MARK: - Context API (late chunking)

    /// Late-chunking context embed. For each document (a list of chunk strings),
    /// returns per-chunk 1024-d int8 embeddings: `[[Int8]]` with one row per
    /// chunk, in order.
    public func embedContext(_ documents: [[String]]) throws -> [[[Int8]]] {
        try documents.map { try embedContextOne($0) }
    }

    public func embedContextBinary(_ documents: [[String]]) throws -> [[[Float]]] {
        try embedContext(documents).map { doc in doc.map { PplxEmbed.binary(fromInt8: $0) } }
    }

    public func embedContextUBinary(_ documents: [[String]]) throws -> [[[UInt8]]] {
        try embedContext(documents).map { doc in doc.map { PplxEmbed.ubinary(fromInt8: $0) } }
    }

    private func embedContextOne(_ chunks: [String]) throws -> [[Int8]] {
        precondition(variant == "context",
                     "embedContext requires a context bundle (variant=context)")
        guard !chunks.isEmpty else { return [] }

        // Join chunks with the sep token, then tokenize the whole window once.
        // The tokenizer adds the literal <|endoftext|> between chunks; we locate
        // its ids among the valid tokens to recover chunk spans.
        let sep = "<|endoftext|>"
        let joined = chunks.joined(separator: sep)
        var ids = tokenizer.encode(text: joined)

        let bucket = bucket(forTokens: ids.count)
        let L = bucket.maxSeqLen
        if ids.count > L { ids = Array(ids.prefix(L)) }
        let n = ids.count

        // Recover chunk spans: [start, sep) (SEP excluded), next start = sep+1,
        // final chunk runs to n.
        var spans: [(Int, Int)] = []
        var start = 0
        for i in 0..<n where ids[i] == sepTokenId {
            spans.append((start, i))
            start = i + 1
        }
        spans.append((start, n))
        // Cap at the model's max chunk count.
        if spans.count > PplxEmbed.nMaxChunks {
            spans = Array(spans.prefix(PplxEmbed.nMaxChunks))
        }
        let nChunks = spans.count

        let inputIds = try makeInputIds(ids, L: L)
        let attn = try makeAttentionMask(n: n, L: L)
        let pool = try makePoolMatrix(spans: spans, L: L)

        let model = try model(forBucket: L)
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIds,
            "attention_mask": attn,
            "pool_matrix": pool,
        ]))
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        // (32, 1024) int8 — read only the first nChunks rows (rest are all-zero).
        let D = PplxEmbed.embedDim
        var result: [[Int8]] = []
        result.reserveCapacity(nChunks)
        for c in 0..<nChunks {
            var row = [Int8](repeating: 0, count: D)
            let base = c * D
            for i in 0..<D { row[i] = Int8(arr[base + i].int8Value) }
            result.append(row)
        }
        return result
    }

    // MARK: - Input builders

    private func makeInputIds(_ ids: [Int], L: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: L)
        for i in 0..<L { p[i] = i < ids.count ? Int32(ids[i]) : 0 }
        return arr
    }

    private func makeAttentionMask(n: Int, L: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: L)
        let one: UInt16 = 0x3C00  // 1.0 in fp16
        for i in 0..<L { p[i] = i < n ? one : 0 }
        return arr
    }

    /// (32, L) fp16 pool matrix; row k = 1/n_k over chunk k's [start,end) span,
    /// unused rows all-zero.
    private func makePoolMatrix(spans: [(Int, Int)], L: Int) throws -> MLMultiArray {
        let rows = PplxEmbed.nMaxChunks
        let arr = try MLMultiArray(shape: [NSNumber(value: rows), NSNumber(value: L)],
                                   dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: rows * L)
        for i in 0..<(rows * L) { p[i] = 0 }
        for (k, span) in spans.enumerated() where k < rows {
            let (s, e) = span
            let count = e - s
            guard count > 0 else { continue }
            let w = float16Bits(Float(1.0) / Float(count))
            let base = k * L
            for col in s..<e { p[base + col] = w }
        }
        return arr
    }

    // MARK: - Format derivation

    /// binary[i] = int8[i] >= 0 ? +1 : -1   (matches reference x>=0 branch).
    public static func binary(fromInt8 v: [Int8]) -> [Float] {
        v.map { $0 >= 0 ? Float(1) : Float(-1) }
    }

    /// ubinary = packbits(int8[i] >= 0), MSB-first per byte (numpy packbits).
    public static func ubinary(fromInt8 v: [Int8]) -> [UInt8] {
        let nBytes = (v.count + 7) / 8
        var out = [UInt8](repeating: 0, count: nBytes)
        for i in 0..<v.count where v[i] >= 0 {
            out[i / 8] |= UInt8(1 << (7 - (i % 8)))
        }
        return out
    }

    /// Float → IEEE-754 binary16 bit pattern (native Float16 round).
    private func float16Bits(_ x: Float) -> UInt16 {
        Float16(x).bitPattern
    }
}
