import Accelerate
import CoreML
import CoreVideo
import Foundation

/// Internal engine for SWA-chunked Gemma 4 E2B inference.
///
/// The model is split into 4 decode chunks + 4 optional prefill chunks:
///   - chunk1: layers 0-7 (7 sliding + 1 full) + PLE projection
///   - chunk2: layers 8-14 (5 sliding + 2 full), outputs shared kv13/kv14
///   - chunk3: layers 15-24 (all KV-shared via kv13/kv14)
///   - chunk4: layers 25-34 (all KV-shared) + RMSNorm + LM head + argmax
///
/// External resources (loaded from disk, not baked into the model):
///   - INT8 quantized embedding tables (embed_tokens + embed_per_layer)
///   - Per-layer projection weight + norm weight (for PLE on CPU/Accelerate)
///   - Pre-computed RoPE cos/sin tables (sliding 256-d, full 512-d)
final class ChunkedEngine {
    // Decode chunks
    private let chunk1: MLModel
    private let chunk2: MLModel
    private let chunk3: MLModel
    private let chunk4: MLModel

    // Prefill chunks (optional; falls back to per-token decode if nil)
    private let prefillChunk1: MLModel?
    private let prefillChunk2: MLModel?
    private let prefillChunk3: MLModel?
    private let prefillChunk4: MLModel?

    // Verify chunks (optional; loaded from multi-function mlpackages via functionName)
    private let verifyChunk1: MLModel?
    private let verifyChunk2: MLModel?
    private let verifyChunk3: MLModel?
    private let verifyChunk4: MLModel?
    let verifyK: Int  // number of draft tokens for verification (0 = no verify)

    // External embeddings
    private let embedTokens: EmbeddingLookup
    private let embedPerLayer: EmbeddingLookup
    private let perLayerProjF32: [Float]
    private let perLayerNormWeight: Data?

    // RoPE tables (memory-mapped numpy .npy files)
    private let cosSlidingTable: Data?
    private let sinSlidingTable: Data?
    private let cosFullTable: Data?
    private let sinFullTable: Data?

    // SWA KV cache buffers (persistent across decode steps, zeroed on reset)
    private var kSliding1: MLMultiArray  // (7, 1, W, maxHd)
    private var vSliding1: MLMultiArray
    private var kFull1: MLMultiArray     // (1, 1, ctx, maxHd)
    private var vFull1: MLMultiArray
    private var kSliding2: MLMultiArray  // (5, 1, W, maxHd)
    private var vSliding2: MLMultiArray
    private var kFull2: MLMultiArray     // (2, 1, ctx, maxHd)
    private var vFull2: MLMultiArray


    // Phase 0e scratch pool: buffers rewritten each decode step instead of
    // freshly allocated. Holds the three largest per-step masks; smaller
    // buffers (RoPE rows, embeddings, plRaw) keep the allocating path since
    // their Foundation overhead is negligible relative to the savings here.
    // All reads/writes happen before a synchronous prediction call, so
    // per-step reuse is race-free.
    private lazy var scratchMaskFull: MLMultiArray = {
        try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: config.contextLength)], dataType: .float16)
    }()
    private lazy var scratchMaskSliding: MLMultiArray = {
        try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: config.slidingWindow)], dataType: .float16)
    }()
    private lazy var scratchUpdateMask: MLMultiArray = {
        try! MLMultiArray(shape: [1, 1, NSNumber(value: config.contextLength), 1], dataType: .float16)
    }()

    let config: ModelConfig
    let prefillN: Int
    var currentPosition: Int = 0

    var hasPrefill: Bool {
        prefillChunk1 != nil && prefillChunk2 != nil
            && prefillChunk3 != nil && prefillChunk4 != nil
    }

    var hasVerify: Bool {
        verifyChunk1 != nil && verifyChunk2 != nil
            && verifyChunk3 != nil && verifyChunk4 != nil
    }

    /// Hidden states from the last verify pass, at all K positions.
    /// Used as MTP drafter carry state — extract at the last accepted position.
    public private(set) var lastVerifyHiddenStates: MLMultiArray?

    /// Last computed kv13/kv14 from chunk2 output (for MTP drafter access).
    private(set) var lastKV13K: MLMultiArray?
    private(set) var lastKV13V: MLMultiArray?
    private(set) var lastKV14K: MLMultiArray?
    private(set) var lastKV14V: MLMultiArray?

    // MARK: - Loading

    static func load(from directory: URL, config: ModelConfig,
                     computeUnits: MLComputeUnits) async throws -> ChunkedEngine {
        // iOS 18+ MLOptimizationHints.specializationStrategy = .fastPrediction
        // trades a longer first-load specialization for shorter per-prediction
        // wall time. Enabled by default — the added load-time cost (~seconds)
        // is amortized across the session, and a shorter ANE busy window per
        // dispatch directly reduces sustained heat per token.
        //
        // Set LLM_FAST_PREDICTION=0 to disable (opt-out).
        //
        // Removed: .reshapeFrequency = .infrequent. Worked stand-alone but
        // combined with LLM_PREFIX_CACHE=1 reproducibly triggered
        // "MILCompilerForANE error: failed to compile ANE model using ANEF"
        // on iPhone 17 Pro (A19 Pro, iOS 26). 1-3% gain not worth the
        // instability; removed entirely rather than left as opt-in to
        // prevent accidental enable.
        let fastPredictionEnabled = ProcessInfo.processInfo.environment["LLM_FAST_PREDICTION"] != "0"
        func applyHints(_ cfg: MLModelConfiguration) {
            guard fastPredictionEnabled else { return }
            if #available(iOS 18.0, macOS 15.0, *) {
                var hints = MLOptimizationHints()
                hints.specializationStrategy = .fastPrediction
                cfg.optimizationHints = hints
            }
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        applyHints(mlConfig)

        // Prefill chunks use GPU for compute-bound batch processing (TTFT win).
        // Decode chunks stay on ANE for bandwidth-bound single-token inference.
        let prefillConfig = MLModelConfiguration()
        let useGPUPrefill = ProcessInfo.processInfo.environment["GPU_PREFILL"] == "1"
        prefillConfig.computeUnits = useGPUPrefill ? .cpuAndGPU : computeUnits
        applyHints(prefillConfig)
        if useGPUPrefill {
            print("[Load] GPU_PREFILL=1 — prefill chunks will use .cpuAndGPU")
        }
        if fastPredictionEnabled {
            print("[Load] MLOptimizationHints.specializationStrategy = .fastPrediction")
        }

        // Self-heal: remove any `prefill_chunk{i}.mlmodelc` directories that
        // lack coremldata.bin. These leak onto disk when an older downloader
        // build copied decode weights into prefill dirs whose metadata 404'd
        // on the remote (e.g. E4B has no prefill on HF). Without cleanup they
        // sit as ~2 GB of zombie weights and the loader's existence probe
        // used to try (and fail) to open them as MLModels.
        for i in 1...4 {
            let prefillDir = directory.appendingPathComponent("prefill_chunk\(i).mlmodelc")
            let coreML = prefillDir.appendingPathComponent("coremldata.bin")
            let fm = FileManager.default
            if fm.fileExists(atPath: prefillDir.path)
                && !fm.fileExists(atPath: coreML.path) {
                print("[Load] Removing stale prefill_chunk\(i).mlmodelc (missing coremldata.bin)")
                try? fm.removeItem(at: prefillDir)
            }
        }

        func findModel(_ name: String) -> URL? {
            // For .mlmodelc we require coremldata.bin alongside the directory
            // — a half-populated directory (e.g. stray prefill_chunk with only
            // weights from an older downloader build) must be treated as
            // "not present" so it doesn't crash the loader.
            let compiled = directory.appendingPathComponent("\(name).mlmodelc")
            let coreML = compiled.appendingPathComponent("coremldata.bin")
            if FileManager.default.fileExists(atPath: coreML.path) { return compiled }
            let pkg = directory.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: pkg.path) { return pkg }
            return nil
        }

        func loadOne(_ name: String, config cfg: MLModelConfiguration) throws -> MLModel {
            guard let url = findModel(name) else {
                throw CoreMLLLMError.modelNotFound(name)
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            let m = try MLModel(contentsOf: url, configuration: cfg)
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print("[Load] \(name) done in \(String(format: "%.1f", dt))s")
            return m
        }

        // Load all chunks in parallel (MLModel(contentsOf:) is thread-safe,
        // ANE compiler can pipeline compilation across chunks)
        let hasPrefillFiles = findModel("prefill_chunk1") != nil
        var c1: MLModel!, c2: MLModel!, c3: MLModel!, c4: MLModel!
        var p1: MLModel?, p2: MLModel?, p3: MLModel?, p4: MLModel?

        let loadT0 = CFAbsoluteTimeGetCurrent()
        try await withThrowingTaskGroup(of: (String, MLModel).self) { group in
            for name in ["chunk1", "chunk2", "chunk3", "chunk4"] {
                group.addTask { (name, try loadOne(name, config: mlConfig)) }
            }
            if hasPrefillFiles {
                for name in ["prefill_chunk1", "prefill_chunk2", "prefill_chunk3", "prefill_chunk4"] {
                    group.addTask { (name, try loadOne(name, config: prefillConfig)) }
                }
            }
            for try await (name, model) in group {
                switch name {
                case "chunk1": c1 = model
                case "chunk2": c2 = model
                case "chunk3": c3 = model
                case "chunk4": c4 = model
                case "prefill_chunk1": p1 = model
                case "prefill_chunk2": p2 = model
                case "prefill_chunk3": p3 = model
                case "prefill_chunk4": p4 = model
                default: break
                }
            }
        }
        let loadDt = CFAbsoluteTimeGetCurrent() - loadT0
        print("[Load] All \(hasPrefillFiles ? 8 : 4) chunks loaded in \(String(format: "%.1f", loadDt))s (parallel)")

        // Load verify functions from multi-function chunks (if available).
        // Multi-function chunks have a "verify_qK" function alongside the
        // default "decode_q1". We detect this by checking if the chunk has
        // the verify function and load it with a separate configuration.
        var v1: MLModel?, v2: MLModel?, v3: MLModel?, v4: MLModel?
        var detectedK = 0
        do {
            let verifyConfig = MLModelConfiguration()
            verifyConfig.computeUnits = computeUnits
            verifyConfig.functionName = "verify_qK"
            applyHints(verifyConfig)

            let verifyT0 = CFAbsoluteTimeGetCurrent()
            try await withThrowingTaskGroup(of: (String, MLModel).self) { group in
                for (name, url) in [("v1", findModel("chunk1")),
                                     ("v2", findModel("chunk2")),
                                     ("v3", findModel("chunk3")),
                                     ("v4", findModel("chunk4"))] {
                    guard let u = url else { continue }
                    group.addTask {
                        let m = try MLModel(contentsOf: u, configuration: verifyConfig)
                        return (name, m)
                    }
                }
                for try await (name, model) in group {
                    switch name {
                    case "v1": v1 = model
                    case "v2": v2 = model
                    case "v3": v3 = model
                    case "v4": v4 = model
                    default: break
                    }
                }
            }
            if v1 != nil && v2 != nil && v3 != nil && v4 != nil {
                // Detect K from verify chunk4's token_ids output shape
                if let desc = v4!.modelDescription.outputDescriptionsByName["token_ids"],
                   let c = desc.multiArrayConstraint, c.shape.count >= 2 {
                    detectedK = c.shape[1].intValue
                }
                let vDt = CFAbsoluteTimeGetCurrent() - verifyT0
                print("[Load] Verify functions loaded (K=\(detectedK)) in \(String(format: "%.1f", vDt))s")
            }
        } catch {
            // Multi-function not available — verify chunks stay nil
            print("[Load] No verify_qK function found, speculative verification disabled")
            v1 = nil; v2 = nil; v3 = nil; v4 = nil
        }

        // Embeddings
        let vocabSize = config.vocabSize
        let hidden = config.hiddenSize
        let nlayers = config.numLayers
        let pld = config.perLayerDim
        let embedTokens = try EmbeddingLookup(
            dataURL: directory.appendingPathComponent("embed_tokens_q8.bin"),
            scalesURL: directory.appendingPathComponent("embed_tokens_scales.bin"),
            vocabSize: vocabSize, dim: hidden, scale: config.embedScale)
        let embedPerLayer = try EmbeddingLookup(
            dataURL: directory.appendingPathComponent("embed_tokens_per_layer_q8.bin"),
            scalesURL: directory.appendingPathComponent("embed_tokens_per_layer_scales.bin"),
            vocabSize: vocabSize, dim: nlayers * pld, scale: config.perLayerEmbedScale)

        // Per-layer projection: convert fp16 → fp32 for Accelerate BLAS
        let projData = try Data(contentsOf: directory.appendingPathComponent("per_layer_projection.bin"),
                                options: .mappedIfSafe)
        let count = nlayers * pld * hidden
        var projF32 = [Float](repeating: 0, count: count)
        projData.withUnsafeBytes { raw in
            let f16Ptr = raw.baseAddress!.assumingMemoryBound(to: UInt16.self)
            // Vectorized fp16→fp32 via Accelerate (vs scalar loop)
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: f16Ptr),
                                    height: 1, width: UInt(count), rowBytes: count * 2)
            projF32.withUnsafeMutableBufferPointer { dst in
                var dstBuf = vImage_Buffer(data: dst.baseAddress!, height: 1,
                                           width: UInt(count), rowBytes: count * 4)
                vImageConvert_Planar16FtoPlanarF(&src, &dstBuf, 0)
            }
        }
        let normWeight = try? Data(contentsOf: directory.appendingPathComponent("per_layer_norm_weight.bin"),
                                   options: .mappedIfSafe)

        // RoPE tables
        let cosS = try? Data(contentsOf: directory.appendingPathComponent("cos_sliding.npy"), options: .mappedIfSafe)
        let sinS = try? Data(contentsOf: directory.appendingPathComponent("sin_sliding.npy"), options: .mappedIfSafe)
        let cosF = try? Data(contentsOf: directory.appendingPathComponent("cos_full.npy"), options: .mappedIfSafe)
        let sinF = try? Data(contentsOf: directory.appendingPathComponent("sin_full.npy"), options: .mappedIfSafe)

        // Prefill N: read from model input shape or default 512
        var prefillN = 512
        if let p1 {
            if let desc = p1.modelDescription.inputDescriptionsByName["hidden_states"],
               let constraint = desc.multiArrayConstraint {
                let shape = constraint.shape
                if shape.count >= 2 { prefillN = shape[1].intValue }
            }
        }

        // Validate context length: every chunk must agree with model_config.json.
        // Mixed 2K / 8K chunk files from different builds are rejected with a clear
        // error so the user knows to re-download a consistent set.
        let configuredCtx = config.contextLength
        for (label, model) in [("chunk1", c1!), ("chunk2", c2!), ("chunk3", c3!), ("chunk4", c4!)] {
            if let desc = model.modelDescription.inputDescriptionsByName["causal_mask_full"],
               let c = desc.multiArrayConstraint,
               let last = c.shape.last?.intValue, last != configuredCtx {
                throw CoreMLLLMError.modelNotFound(
                    "\(label): causal_mask_full expects ctx=\(last) but model_config.json says " +
                    "\(configuredCtx). Delete the model directory and re-download to get a " +
                    "consistent set of chunks.")
            }
        }

        // SWA KV buffers — IOSurface-backed for zero-copy CPU↔ANE transfer.
        // Slot counts (num_sliding_in_chunk / num_full_in_chunk) and num_kv_heads
        // are read from each chunk's input description so E2B (nkv=1, 7/1, 5/2)
        // and E4B (nkv=2, 10/2, 10/2) both allocate the right shapes.
        let maxHd = 512
        let ctx = configuredCtx
        let W = config.slidingWindow
        func ioSurfaceArray(slots: Int, nkv: Int, seqLen: Int) throws -> MLMultiArray {
            let width = maxHd
            let height = slots * nkv * seqLen
            var pixelBuffer: CVPixelBuffer?
            let attrs: [String: Any] = [
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault, width, height,
                kCVPixelFormatType_OneComponent16Half,
                attrs as CFDictionary, &pixelBuffer)
            if status == kCVReturnSuccess, let pb = pixelBuffer {
                CVPixelBufferLockBaseAddress(pb, [])
                memset(CVPixelBufferGetBaseAddress(pb)!, 0, CVPixelBufferGetDataSize(pb))
                CVPixelBufferUnlockBaseAddress(pb, [])
                let shape: [NSNumber] = [NSNumber(value: slots), NSNumber(value: nkv),
                                          NSNumber(value: seqLen), NSNumber(value: maxHd)]
                return try MLMultiArray(pixelBuffer: pb, shape: shape)
            }
            // Fallback to standard allocation
            print("[KV] IOSurface failed for \(slots)x\(nkv)x\(seqLen)x\(maxHd), using standard MLMultiArray")
            let arr = try MLMultiArray(
                shape: [NSNumber(value: slots), NSNumber(value: nkv),
                        NSNumber(value: seqLen), NSNumber(value: maxHd)],
                dataType: .float16)
            memset(arr.dataPointer, 0, slots * nkv * seqLen * maxHd * MemoryLayout<UInt16>.stride)
            return arr
        }

        // Probe the chunk models for expected KV shapes. Shape is (slots, nkv, seqLen, maxHd).
        func kvShape(_ model: MLModel, _ name: String) -> (slots: Int, nkv: Int)? {
            guard let desc = model.modelDescription.inputDescriptionsByName[name],
                  let c = desc.multiArrayConstraint else { return nil }
            let s = c.shape
            guard s.count == 4 else { return nil }
            return (s[0].intValue, s[1].intValue)
        }
        let c1KS = kvShape(c1!, "K_sliding_in") ?? (7, 1)
        let c1KF = kvShape(c1!, "K_full_in")    ?? (1, 1)
        let c2KS = kvShape(c2!, "K_sliding_in") ?? (5, 1)
        let c2KF = kvShape(c2!, "K_full_in")    ?? (2, 1)
        print("[KV] Allocating IOSurface-backed KV cache buffers (ctx=\(ctx)) — " +
              "c1 sliding=\(c1KS.slots)x\(c1KS.nkv) full=\(c1KF.slots)x\(c1KF.nkv), " +
              "c2 sliding=\(c2KS.slots)x\(c2KS.nkv) full=\(c2KF.slots)x\(c2KF.nkv)")

        let engine = try ChunkedEngine(
            chunk1: c1, chunk2: c2, chunk3: c3, chunk4: c4,
            prefillChunk1: p1, prefillChunk2: p2, prefillChunk3: p3, prefillChunk4: p4,
            verifyChunk1: v1, verifyChunk2: v2, verifyChunk3: v3, verifyChunk4: v4,
            verifyK: detectedK,
            embedTokens: embedTokens, embedPerLayer: embedPerLayer,
            perLayerProjF32: projF32, perLayerNormWeight: normWeight,
            cosSlidingTable: cosS, sinSlidingTable: sinS,
            cosFullTable: cosF, sinFullTable: sinF,
            kSliding1: ioSurfaceArray(slots: c1KS.slots, nkv: c1KS.nkv, seqLen: W),
            vSliding1: ioSurfaceArray(slots: c1KS.slots, nkv: c1KS.nkv, seqLen: W),
            kFull1:    ioSurfaceArray(slots: c1KF.slots, nkv: c1KF.nkv, seqLen: ctx),
            vFull1:    ioSurfaceArray(slots: c1KF.slots, nkv: c1KF.nkv, seqLen: ctx),
            kSliding2: ioSurfaceArray(slots: c2KS.slots, nkv: c2KS.nkv, seqLen: W),
            vSliding2: ioSurfaceArray(slots: c2KS.slots, nkv: c2KS.nkv, seqLen: W),
            kFull2:    ioSurfaceArray(slots: c2KF.slots, nkv: c2KF.nkv, seqLen: ctx),
            vFull2:    ioSurfaceArray(slots: c2KF.slots, nkv: c2KF.nkv, seqLen: ctx),
            config: config, prefillN: prefillN)

        // ANE pipeline prewarm (Phase 0b): four dummy decode steps at load
        // time force the ANE compiler to finalize dispatch schedules and
        // resident weight layouts before the first user token arrives —
        // eliminating the ~0.5-1.5s first-token stall. KV cache is reset
        // afterwards so the dummy tokens leave no state behind.
        let warmT0 = CFAbsoluteTimeGetCurrent()
        for i in 0..<4 {
            _ = try engine.predictStep(tokenID: 0, position: i)
        }
        engine.reset()
        print("[Load] ANE prewarm (4 steps) done in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - warmT0))s")

        return engine
    }

    private init(chunk1: MLModel, chunk2: MLModel, chunk3: MLModel, chunk4: MLModel,
                 prefillChunk1: MLModel?, prefillChunk2: MLModel?,
                 prefillChunk3: MLModel?, prefillChunk4: MLModel?,
                 verifyChunk1: MLModel?, verifyChunk2: MLModel?,
                 verifyChunk3: MLModel?, verifyChunk4: MLModel?,
                 verifyK: Int,
                 embedTokens: EmbeddingLookup, embedPerLayer: EmbeddingLookup,
                 perLayerProjF32: [Float], perLayerNormWeight: Data?,
                 cosSlidingTable: Data?, sinSlidingTable: Data?,
                 cosFullTable: Data?, sinFullTable: Data?,
                 kSliding1: MLMultiArray, vSliding1: MLMultiArray,
                 kFull1: MLMultiArray, vFull1: MLMultiArray,
                 kSliding2: MLMultiArray, vSliding2: MLMultiArray,
                 kFull2: MLMultiArray, vFull2: MLMultiArray,
                 config: ModelConfig, prefillN: Int) {
        self.chunk1 = chunk1; self.chunk2 = chunk2
        self.chunk3 = chunk3; self.chunk4 = chunk4
        self.prefillChunk1 = prefillChunk1; self.prefillChunk2 = prefillChunk2
        self.prefillChunk3 = prefillChunk3; self.prefillChunk4 = prefillChunk4
        self.verifyChunk1 = verifyChunk1; self.verifyChunk2 = verifyChunk2
        self.verifyChunk3 = verifyChunk3; self.verifyChunk4 = verifyChunk4
        self.verifyK = verifyK
        self.embedTokens = embedTokens; self.embedPerLayer = embedPerLayer
        self.perLayerProjF32 = perLayerProjF32; self.perLayerNormWeight = perLayerNormWeight
        self.cosSlidingTable = cosSlidingTable; self.sinSlidingTable = sinSlidingTable
        self.cosFullTable = cosFullTable; self.sinFullTable = sinFullTable
        self.kSliding1 = kSliding1; self.vSliding1 = vSliding1
        self.kFull1 = kFull1; self.vFull1 = vFull1
        self.kSliding2 = kSliding2; self.vSliding2 = vSliding2
        self.kFull2 = kFull2; self.vFull2 = vFull2
        self.config = config; self.prefillN = prefillN
    }

    // MARK: - Prefix cache (LLM_PREFIX_CACHE=1)

    /// Disk-backed prefix cache. nil unless `LLM_PREFIX_CACHE=1` was set
    /// when CoreMLLLM was loaded. Owned externally so the same cache can
    /// be shared across reloads of the engine.
    var prefixCache: PrefixCache?

    /// Threshold below which a cache hit is ignored — re-prefilling small
    /// prefixes is cheaper than per-token decode of the delta.
    private let prefixCacheMinHit: Int = 64

    /// Snapshot the 8 persistent KV buffers as a list of Data blobs in the
    /// canonical order: kSliding1, vSliding1, kFull1, vFull1, kSliding2,
    /// vSliding2, kFull2, vFull2.
    func captureKVSnapshot() -> [Data] {
        let bufs = [kSliding1, vSliding1, kFull1, vFull1,
                    kSliding2, vSliding2, kFull2, vFull2]
        return bufs.map { buf in
            let bytes = buf.count * MemoryLayout<UInt16>.stride
            return Data(bytes: buf.dataPointer, count: bytes)
        }
    }

    /// Restore the 8 persistent KV buffers from snapshot data and set
    /// `currentPosition`. Sizes must exactly match the live buffer sizes;
    /// throws if not (cached snapshot from a different model / context len).
    func restoreKVSnapshot(_ blobs: [Data], position: Int) throws {
        let bufs = [kSliding1, vSliding1, kFull1, vFull1,
                    kSliding2, vSliding2, kFull2, vFull2]
        precondition(blobs.count == bufs.count, "snapshot buffer count mismatch")
        for (i, buf) in bufs.enumerated() {
            let expected = buf.count * MemoryLayout<UInt16>.stride
            guard blobs[i].count == expected else {
                throw NSError(
                    domain: "ChunkedEngine", code: 100,
                    userInfo: [NSLocalizedDescriptionKey:
                        "snapshot buffer \(i) size mismatch (got \(blobs[i].count), need \(expected))"])
            }
            blobs[i].withUnsafeBytes { src in
                memcpy(buf.dataPointer, src.baseAddress!, expected)
            }
        }
        currentPosition = position
    }

    // MARK: - Reset

    func reset() {
        for buf in [kSliding1, vSliding1, kFull1, vFull1,
                    kSliding2, vSliding2, kFull2, vFull2] {
            memset(buf.dataPointer, 0, buf.count * MemoryLayout<UInt16>.stride)
        }
        currentPosition = 0
        profileEmbed = 0
        profilePredict = 0
        profileCount = 0
        profileMask = 0
        profileC1 = 0; profileC2 = 0; profileC3 = 0; profileC4 = 0
        profileANEWait = 0; profileCopyBack = 0
    }

    // MARK: - Single-token decode step

    // Profiling accumulators
    private var profileEmbed: Double = 0
    private var profilePredict: Double = 0
    private var profileCount: Int = 0
    // Per-chunk breakdown (includes the chunk's own copyBack cost for KV-holding chunks).
    private var profileMask: Double = 0
    private var profileC1: Double = 0
    private var profileC2: Double = 0
    private var profileC3: Double = 0
    private var profileC4: Double = 0
    // CPU-vs-ANE split: ANE wait = time spent inside chunk.prediction(from:);
    // copyBack = CPU memcpy of KV tensors after each chunk; cpuPrep = remainder
    // (mask/embed/dictionary build). Sum should approximate total wall time.
    private var profileANEWait: Double = 0
    private var profileCopyBack: Double = 0

    // Print [Profile] / [ANE/CPU] every step instead of every 10 steps. Useful
    // for short prompts where the 10-step gate would never fire. Set
    // LLM_PROFILE_EVERY_STEP=1 to enable.
    private let profileEveryStep = ProcessInfo.processInfo.environment["LLM_PROFILE_EVERY_STEP"] == "1"

    // LayerSkip probe: measures early-exit accuracy (chunk3 skipped)
    private let layerSkipProbe = ProcessInfo.processInfo.environment["LAYERSKIP_PROBE"] == "1"
    private var lsProbeTotal: Int = 0
    private var lsProbeMatch: Int = 0

    func predictStep(tokenID: Int, position: Int,
                     imageEmbedding: MLMultiArray? = nil) throws -> Int {
        let ctx = config.contextLength
        let W = config.slidingWindow
        let hidden = config.hiddenSize

        let t0 = CFAbsoluteTimeGetCurrent()
        let hiddenIn: MLMultiArray
        let plRaw: MLMultiArray
        if let imageEmbedding {
            hiddenIn = imageEmbedding
            let totalDim = config.numLayers * config.perLayerDim
            plRaw = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
            memset(plRaw.dataPointer, 0, totalDim * MemoryLayout<UInt16>.stride)
        } else {
            hiddenIn = try embedTokens.lookup(tokenID, shape: [1, 1, NSNumber(value: hidden)])
            plRaw = try lookupPerLayerRaw(tokenID: tokenID)
        }
        let t1 = CFAbsoluteTimeGetCurrent()
        profileEmbed += (t1 - t0)

        let maskFull = try makeCausalMask(position: position, length: ctx)
        let maskSliding = try makeSlidingCausalMask(position: position, W: W)
        let umask = try makeUpdateMask(position: position, length: ctx)
        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable, position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable, position: position, dim: 512)
        let tMask = CFAbsoluteTimeGetCurrent()
        profileMask += (tMask - t1)

        // Chunk 1
        let tC1Start = CFAbsoluteTimeGetCurrent()
        let in1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: kSliding1),
            "V_sliding_in": MLFeatureValue(multiArray: vSliding1),
            "K_full_in": MLFeatureValue(multiArray: kFull1),
            "V_full_in": MLFeatureValue(multiArray: vFull1),
        ])
        let tC1Wait0 = CFAbsoluteTimeGetCurrent()
        let out1 = try chunk1.prediction(from: in1)
        let tC1Wait1 = CFAbsoluteTimeGetCurrent()
        profileANEWait += (tC1Wait1 - tC1Wait0)
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let plc = out1.featureValue(for: "per_layer_combined_out")!.multiArrayValue!
        let tC1Cb0 = CFAbsoluteTimeGetCurrent()
        copyBack(out1, "K_sliding_out", into: kSliding1)
        copyBack(out1, "V_sliding_out", into: vSliding1)
        copyBack(out1, "K_full_out", into: kFull1)
        copyBack(out1, "V_full_out", into: vFull1)
        let tC1End = CFAbsoluteTimeGetCurrent()
        profileCopyBack += (tC1End - tC1Cb0)
        profileC1 += (tC1End - tC1Start)

        // Chunk 2
        let tC2Start = CFAbsoluteTimeGetCurrent()
        let in2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: kSliding2),
            "V_sliding_in": MLFeatureValue(multiArray: vSliding2),
            "K_full_in": MLFeatureValue(multiArray: kFull2),
            "V_full_in": MLFeatureValue(multiArray: vFull2),
        ])
        let tC2Wait0 = CFAbsoluteTimeGetCurrent()
        let out2 = try chunk2.prediction(from: in2)
        let tC2Wait1 = CFAbsoluteTimeGetCurrent()
        profileANEWait += (tC2Wait1 - tC2Wait0)
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let tC2Cb0 = CFAbsoluteTimeGetCurrent()
        copyBack(out2, "K_sliding_out", into: kSliding2)
        copyBack(out2, "V_sliding_out", into: vSliding2)
        copyBack(out2, "K_full_out", into: kFull2)
        copyBack(out2, "V_full_out", into: vFull2)
        let tC2Cb1 = CFAbsoluteTimeGetCurrent()
        profileCopyBack += (tC2Cb1 - tC2Cb0)
        let kv13_k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13_v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14_k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14_v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        lastKV13K = kv13_k; lastKV13V = kv13_v
        lastKV14K = kv14_k; lastKV14V = kv14_v
        let tC2End = CFAbsoluteTimeGetCurrent()
        profileC2 += (tC2End - tC2Start)

        let shared: [String: MLFeatureValue] = [
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13_k), "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k), "kv14_v": MLFeatureValue(multiArray: kv14_v),
        ]

        // Chunk 3
        let tC3Start = CFAbsoluteTimeGetCurrent()
        var d3 = shared; d3["hidden_states"] = MLFeatureValue(multiArray: h2)
        let in3 = try MLDictionaryFeatureProvider(dictionary: d3)
        let tC3Wait0 = CFAbsoluteTimeGetCurrent()
        let h3 = try chunk3.prediction(from: in3)
            .featureValue(for: "hidden_states_out")!.multiArrayValue!
        let tC3End = CFAbsoluteTimeGetCurrent()
        profileANEWait += (tC3End - tC3Wait0)
        profileC3 += (tC3End - tC3Start)

        // Chunk 4
        let tC4Start = CFAbsoluteTimeGetCurrent()
        var d4 = shared; d4["hidden_states"] = MLFeatureValue(multiArray: h3)
        let in4 = try MLDictionaryFeatureProvider(dictionary: d4)
        let tC4Wait0 = CFAbsoluteTimeGetCurrent()
        let out4 = try chunk4.prediction(from: in4)
        let tC4End = CFAbsoluteTimeGetCurrent()
        profileANEWait += (tC4End - tC4Wait0)
        profileC4 += (tC4End - tC4Start)

        // LayerSkip probe: skip chunk3, feed h2 directly to chunk4
        if layerSkipProbe {
            var d4skip = shared; d4skip["hidden_states"] = MLFeatureValue(multiArray: h2)
            let skipOut = try chunk4.prediction(from: MLDictionaryFeatureProvider(dictionary: d4skip))
            let skipToken = skipOut.featureValue(for: "token_id")!.multiArrayValue![0].intValue
            let realToken = out4.featureValue(for: "token_id")!.multiArrayValue![0].intValue
            lsProbeTotal += 1
            if skipToken == realToken { lsProbeMatch += 1 }
            if lsProbeTotal == 1 || lsProbeTotal % 10 == 0 {
                let rate = Double(lsProbeMatch) / Double(lsProbeTotal) * 100
                print(String(format: "[LayerSkip] %d/%d match (%.1f%%) — skip=%d real=%d",
                             lsProbeMatch, lsProbeTotal, rate, skipToken, realToken))
            }
        }

        profilePredict += (CFAbsoluteTimeGetCurrent() - t1)
        profileCount += 1
        if profileCount == 1 || profileCount % 10 == 0 || profileEveryStep {
            let n = Double(profileCount)
            let eMs = profileEmbed / n * 1000
            let pMs = profilePredict / n * 1000
            let mMs = profileMask / n * 1000
            let c1 = profileC1 / n * 1000
            let c2 = profileC2 / n * 1000
            let c3 = profileC3 / n * 1000
            let c4 = profileC4 / n * 1000
            let aneMs = profileANEWait / n * 1000
            let cbMs = profileCopyBack / n * 1000
            let totalMs = eMs + pMs
            let cpuActiveMs = totalMs - aneMs
            let cpuPct = totalMs > 0 ? (cpuActiveMs / totalMs * 100) : 0
            print(String(format:
                "[Profile] emb=%.1fms mask=%.1fms | c1=%.1f c2=%.1f c3=%.1f c4=%.1f " +
                "(sum=%.1fms) | predict=%.1fms total=%.1fms (%.1f tok/s)",
                eMs, mMs, c1, c2, c3, c4, c1 + c2 + c3 + c4,
                pMs, totalMs, 1000.0 / totalMs))
            print(String(format:
                "[ANE/CPU] ANE_wait=%.1fms copyBack=%.1fms cpu_active=%.1fms (%.0f%% CPU)",
                aneMs, cbMs, cpuActiveMs, cpuPct))
        }

        // Best-effort logit stash — zero-cost when chunk4 doesn't
        // expose `token_logit` (the .featureValue call returns nil).
        if let logitsFV = out4.featureValue(for: "token_logit"),
           let arr = logitsFV.multiArrayValue, arr.count > 0 {
            self.lastTokenLogit = Float(truncating: arr[0])
        } else {
            self.lastTokenLogit = nil
        }
        return out4.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    /// Populated by the most recent `predictStep` call when the chunk4
    /// build exposes `token_logit` (the max-logit for the argmax token,
    /// fp16). `nil` when that output is absent. Read by the rich event
    /// stream to surface per-token confidence without an extra dispatch.
    public private(set) var lastTokenLogit: Float?

    // MARK: - Batched prefill (seq=N)

    func runPrefill(tokenIDs: [Int], imageFeatures: MLMultiArray? = nil,
                    imageNumTokens: Int = 256,
                    audioFeatures: MLMultiArray? = nil, audioNumTokens: Int = 50) throws -> Int {
        guard let p1 = prefillChunk1, let p2 = prefillChunk2,
              let p3 = prefillChunk3, let p4 = prefillChunk4 else {
            throw CoreMLLLMError.prefillNotAvailable
        }
        let N = prefillN
        let realLen = tokenIDs.count
        precondition(realLen > 0 && realLen <= N)

        // ---- Prefix cache lookup (text-only) -----------------------------
        // If a previously cached snapshot covers a prefix of this prompt,
        // restore the KV state at that position and per-token decode the
        // delta tokens. Skips the expensive batched prefill entirely.
        // Multimodal prompts (image / audio) bypass the cache to avoid
        // shape/feature mismatches.
        if let cache = prefixCache,
           imageFeatures == nil, audioFeatures == nil,
           let match = cache.longestPrefixMatch(tokenIDs: tokenIDs),
           match.matchLen >= prefixCacheMinHit,
           match.matchLen < tokenIDs.count {
            let cacheT0 = CFAbsoluteTimeGetCurrent()
            do {
                let blobs = try PrefixCache.readBlob(at: match.blobURL,
                                                      expecting: match.entry)
                try restoreKVSnapshot(blobs, position: match.matchLen)
                var nextID = 0
                for i in match.matchLen..<tokenIDs.count {
                    nextID = try predictStep(tokenID: tokenIDs[i], position: i)
                    currentPosition = i + 1
                }
                let dt = CFAbsoluteTimeGetCurrent() - cacheT0
                let delta = tokenIDs.count - match.matchLen
                print("[PrefixCache] HIT match=\(match.matchLen)/\(tokenIDs.count) " +
                      "delta=\(delta) restore+decode=\(String(format: "%.1f", dt*1000))ms")
                return nextID
            } catch {
                print("[PrefixCache] restore failed (\(error)) — full prefill fallback")
                reset()  // restore() may have left partial state
            }
        }
        // ------------------------------------------------------------------

        reset()

        // ---- Pending-Token Skip decision ---------------------------------
        // LiteRT-LM-style optimization: process only the first (realLen-1)
        // prompt tokens through prefill chunks 1+2 (KV writes), skip
        // chunks 3+4 entirely, and let the first decode step consume the
        // stashed last prompt token — writing its KV and sampling the
        // first generated token in one dispatch. Saves roughly half the
        // prefill cost at the price of one decode step.
        //
        // Disabled when:
        //   - realLen < 2 (nothing to pend)
        //   - last prompt token is a multimodal placeholder (predictStep
        //     would need image/audio features we can't reconstruct here)
        //   - multimodal features are present at all (safety margin;
        //     avoids per-token index drift between prefill and decode)
        let IMAGE_TOKEN_ID = 258880
        let AUDIO_TOKEN_ID = 258881
        let VIDEO_TOKEN_ID = 258884
        let pendingSkipEnabled = ProcessInfo.processInfo.environment["LLM_PENDING_TOKEN_SKIP"] == "1"
        let lastPromptTid = tokenIDs[realLen - 1]
        let lastIsMultimodal = lastPromptTid == IMAGE_TOKEN_ID
            || lastPromptTid == AUDIO_TOKEN_ID
            || lastPromptTid == VIDEO_TOKEN_ID
        let hasMultimodalFeatures = imageFeatures != nil || audioFeatures != nil
        let pendingSkip = pendingSkipEnabled && realLen >= 2
            && !lastIsMultimodal && !hasMultimodalFeatures

        let prefillTokens: [Int] = pendingSkip
            ? Array(tokenIDs.prefix(realLen - 1))
            : tokenIDs
        let prefillLen = prefillTokens.count
        // ------------------------------------------------------------------

        let prefillT0 = CFAbsoluteTimeGetCurrent()

        let hiddenIn = try buildPrefillHidden(tokenIDs: prefillTokens, N: N, imageFeatures: imageFeatures,
                                                imageNumTokens: imageNumTokens,
                                                audioFeatures: audioFeatures, audioNumTokens: audioNumTokens)
        let plRaw = try buildPrefillPLR(tokenIDs: prefillTokens, N: N)
        // If the prompt has any vision placeholders, use the
        // vision-group-aware mask so each contiguous run of image/video
        // tokens (= one frame / one image) attends bidirectionally
        // within itself — matching HF's `mm_token_type_ids` behavior.
        let hasVision = prefillTokens.contains { $0 == IMAGE_TOKEN_ID || $0 == VIDEO_TOKEN_ID }
        let causal = hasVision
            ? try makePrefillVisionMask(tokenIDs: prefillTokens, N: N)
            : try makePrefillCausalMask(N: N)
        let cosS = try buildPrefillRoPE(table: cosSlidingTable, N: N, dim: 256)
        let sinS = try buildPrefillRoPE(table: sinSlidingTable, N: N, dim: 256)
        let cosF = try buildPrefillRoPE(table: cosFullTable, N: N, dim: 512)
        let sinF = try buildPrefillRoPE(table: sinFullTable, N: N, dim: 512)
        let lastMask = try makeLastPositionMask(N: N, realLen: prefillLen)

        let prepDt = CFAbsoluteTimeGetCurrent() - prefillT0

        // Prefill chunk 1
        let pc1T0 = CFAbsoluteTimeGetCurrent()
        let out1 = try p1.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
        ]))
        let pc1Dt = CFAbsoluteTimeGetCurrent() - pc1T0
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let plc = out1.featureValue(for: "per_layer_combined_out")!.multiArrayValue!

        // Write KV from chunk1 prefill → decode sliding/full caches
        for (name, slot, kv, hd) in kvMapChunk1Sliding() {
            try writeSlidingFromPrefill(src: out1, name: name, cache: kv, slot: slot,
                                        realLen: prefillLen, hd: hd)
        }
        for (name, slot, kv, hd) in kvMapChunk1Full() {
            try writeFullFromPrefill(src: out1, name: name, cache: kv, slot: slot,
                                     realLen: prefillLen, hd: hd)
        }

        // Prefill chunk 2
        let pc2T0 = CFAbsoluteTimeGetCurrent()
        let out2 = try p2.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
        ]))
        let pc2Dt = CFAbsoluteTimeGetCurrent() - pc2T0
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!

        for (name, slot, kv, hd) in kvMapChunk2Sliding() {
            try writeSlidingFromPrefill(src: out2, name: name, cache: kv, slot: slot,
                                        realLen: prefillLen, hd: hd)
        }
        for (name, slot, kv, hd) in kvMapChunk2Full() {
            try writeFullFromPrefill(src: out2, name: name, cache: kv, slot: slot,
                                     realLen: prefillLen, hd: hd)
        }

        let kv13_k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13_v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14_k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14_v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        lastKV13K = kv13_k; lastKV13V = kv13_v
        lastKV14K = kv14_k; lastKV14V = kv14_v

        let sharedKV: [String: MLFeatureValue] = [
            "kv13_k": MLFeatureValue(multiArray: kv13_k), "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k), "kv14_v": MLFeatureValue(multiArray: kv14_v),
        ]
        let sharedRoPE: [String: MLFeatureValue] = [
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
        ]

        // Pending-Token Skip (LLM_PENDING_TOKEN_SKIP=1): prefill chunks 1+2
        // already wrote KV for positions 0..realLen-2 (prefillLen = realLen-1
        // tokens were passed in). Skip chunks 3+4 of prefill and let the
        // first decode step write KV at position realLen-1 AND sample the
        // first generated token in the same dispatch. Semantically cleaner
        // than PREFILL_BYPASS because there is no redundant KV write at the
        // last position (PREFILL_BYPASS would have prefill c1/c2 and decode
        // c1/c2 both write position realLen-1, idempotent but wasteful).
        if pendingSkip {
            let preDecodeTotal = CFAbsoluteTimeGetCurrent() - prefillT0
            print("[Prefill] PENDING_SKIP prep=\(String(format: "%.1f", prepDt*1000))ms " +
                  "c1=\(String(format: "%.1f", pc1Dt*1000))ms " +
                  "c2=\(String(format: "%.1f", pc2Dt*1000))ms " +
                  "c3/4=skipped " +
                  "prefill=\(String(format: "%.1f", preDecodeTotal*1000))ms " +
                  "(\(prefillLen) prefilled + 1 pending)")
            let nextToken = try predictStep(tokenID: lastPromptTid, position: realLen - 1)
            if let cache = prefixCache {
                do {
                    let blobs = captureKVSnapshot()
                    try cache.store(tokenIDs: tokenIDs, buffers: blobs, position: realLen)
                } catch {
                    print("[PrefixCache] store failed: \(error)")
                }
            }
            return nextToken
        }

        // B1 bypass: chunks 3+4 are KV-shared read-only (no KV writes). For
        // prompt tokens 0..N-2 their hidden-state outputs are discarded; only
        // position N-1 needs chunks 3+4 to produce the first decode token.
        // Skip the prefill chunks 3+4 entirely and use the decode Q=1 path
        // at position N-1 instead. Decode chunks 1+2 re-run for that single
        // position (writes same KV values as prefill — idempotent).
        //
        // Expected saving: -47% prefill time (Apple AFM tech report, "Block 2
        // does not produce any keys or values, the prefill stage is able to
        // bypass all of its computation").
        //
        // Multimodal caveat: predictStep uses embedTokens.lookup for input.
        // Works when the last prompt token is text (chat-template suffix).
        // If the last token is a vision/audio placeholder the text lookup
        // is wrong — bench against non-bypass before shipping multimodal.
        let bypass = ProcessInfo.processInfo.environment["PREFILL_BYPASS"] == "1"
        if bypass {
            let totalPrefill = CFAbsoluteTimeGetCurrent() - prefillT0
            print("[Prefill] BYPASS prep=\(String(format: "%.1f", prepDt*1000))ms " +
                  "c1=\(String(format: "%.1f", pc1Dt*1000))ms " +
                  "c2=\(String(format: "%.1f", pc2Dt*1000))ms " +
                  "c3/4=skipped " +
                  "total=\(String(format: "%.1f", totalPrefill*1000))ms " +
                  "(\(realLen) tokens, \(String(format: "%.0f", Double(realLen)/totalPrefill)) tok/s)")
            let lastTokenID = tokenIDs[realLen - 1]
            return try predictStep(tokenID: lastTokenID, position: realLen - 1)
        }

        // Prefill chunk 3
        let pc3T0 = CFAbsoluteTimeGetCurrent()
        var d3: [String: MLFeatureValue] = [
            "hidden_states": MLFeatureValue(multiArray: h2),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
        ]
        d3.merge(sharedRoPE) { _, b in b }
        d3.merge(sharedKV) { _, b in b }
        let h3 = try p3.prediction(from: MLDictionaryFeatureProvider(dictionary: d3))
            .featureValue(for: "hidden_states_out")!.multiArrayValue!
        let pc3Dt = CFAbsoluteTimeGetCurrent() - pc3T0

        // Prefill chunk 4
        let pc4T0 = CFAbsoluteTimeGetCurrent()
        var d4: [String: MLFeatureValue] = [
            "hidden_states": MLFeatureValue(multiArray: h3),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "last_position_mask": MLFeatureValue(multiArray: lastMask),
        ]
        d4.merge(sharedRoPE) { _, b in b }
        d4.merge(sharedKV) { _, b in b }
        let out4 = try p4.prediction(from: MLDictionaryFeatureProvider(dictionary: d4))
        let pc4Dt = CFAbsoluteTimeGetCurrent() - pc4T0

        let totalPrefill = CFAbsoluteTimeGetCurrent() - prefillT0
        print("[Prefill] prep=\(String(format: "%.1f", prepDt*1000))ms " +
              "c1=\(String(format: "%.1f", pc1Dt*1000))ms " +
              "c2=\(String(format: "%.1f", pc2Dt*1000))ms " +
              "c3=\(String(format: "%.1f", pc3Dt*1000))ms " +
              "c4=\(String(format: "%.1f", pc4Dt*1000))ms " +
              "total=\(String(format: "%.1f", totalPrefill*1000))ms " +
              "(\(realLen) tokens, \(String(format: "%.0f", Double(realLen)/totalPrefill)) tok/s)")

        let nextToken = out4.featureValue(for: "token_id")!.multiArrayValue![0].intValue

        // Snapshot the KV state for future prefix-match hits. Text-only and
        // only when caching is enabled. Multimodal snapshots are skipped
        // (image/audio embeddings aren't part of the cache key contract).
        if let cache = prefixCache, imageFeatures == nil, audioFeatures == nil {
            do {
                let blobs = captureKVSnapshot()
                try cache.store(tokenIDs: tokenIDs, buffers: blobs, position: realLen)
            } catch {
                print("[PrefixCache] store failed: \(error)")
            }
        }

        return nextToken
    }

    // MARK: - Batched speculative verification (Q=K)

    /// Run K draft tokens through the target model in one ANE dispatch per chunk.
    /// KV cache is read-only — no entries are written. Returns the target's argmax
    /// at each of the K positions for comparison against draft proposals.
    ///
    /// - Parameters:
    ///   - tokens: K draft token IDs to verify
    ///   - startPosition: KV cache position of the first draft token
    /// - Returns: Array of K target argmax token IDs
    func verifyCandidates(tokens: [Int32], startPosition: Int) throws -> [Int32] {
        guard hasVerify else {
            throw CoreMLLLMError.predictionFailed
        }
        let K = tokens.count
        precondition(K == verifyK, "verifyCandidates called with \(K) tokens but model expects \(verifyK)")

        let ctx = config.contextLength
        let W = config.slidingWindow
        let hidden = config.hiddenSize

        // Build batched embeddings for K tokens: (1, K, hidden)
        let hiddenIn = try buildVerifyHidden(tokenIDs: tokens.map { Int($0) })
        let plRaw = try buildVerifyPLR(tokenIDs: tokens.map { Int($0) })

        // Causal masks for K query positions
        let maskFull = try makeVerifyCausalMask(startPos: startPosition, K: K, length: ctx)
        let maskSliding = try makeVerifySlidingMask(startPos: startPosition, K: K, W: W)

        // Update indicator for full-attn KV scatter: (1, 1, ctx, K)
        // Column k has 1.0 at position startPosition+k, 0.0 elsewhere
        let updateIndicator = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), NSNumber(value: K)], dataType: .float16)
        let indPtr = updateIndicator.dataPointer.bindMemory(to: UInt16.self, capacity: ctx * K)
        memset(indPtr, 0, ctx * K * 2)
        for k in 0..<K {
            let pos = startPosition + k
            if pos < ctx {
                indPtr[pos * K + k] = 0x3C00  // 1.0 in float16
            }
        }

        // RoPE for K consecutive positions
        let cosS = try lookupRoPEBatch(table: cosSlidingTable, startPos: startPosition, K: K, dim: 256)
        let sinS = try lookupRoPEBatch(table: sinSlidingTable, startPos: startPosition, K: K, dim: 256)
        let cosF = try lookupRoPEBatch(table: cosFullTable, startPos: startPosition, K: K, dim: 512)
        let sinF = try lookupRoPEBatch(table: sinFullTable, startPos: startPosition, K: K, dim: 512)

        // Verify chunk 1: write-through KV
        let out1 = try verifyChunk1!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_indicator": MLFeatureValue(multiArray: updateIndicator),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: kSliding1),
            "V_sliding_in": MLFeatureValue(multiArray: vSliding1),
            "K_full_in": MLFeatureValue(multiArray: kFull1),
            "V_full_in": MLFeatureValue(multiArray: vFull1),
        ]))
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let plc = out1.featureValue(for: "per_layer_combined_out")!.multiArrayValue!
        // Copy back updated KV caches from verify
        copyBack(out1, "K_sliding_out", into: kSliding1)
        copyBack(out1, "V_sliding_out", into: vSliding1)
        copyBack(out1, "K_full_out", into: kFull1)
        copyBack(out1, "V_full_out", into: vFull1)

        // Verify chunk 2: write-through KV, outputs updated kv13/kv14
        let out2 = try verifyChunk2!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_indicator": MLFeatureValue(multiArray: updateIndicator),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: kSliding2),
            "V_sliding_in": MLFeatureValue(multiArray: vSliding2),
            "K_full_in": MLFeatureValue(multiArray: kFull2),
            "V_full_in": MLFeatureValue(multiArray: vFull2),
        ]))
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!
        // Copy back updated KV caches
        copyBack(out2, "K_sliding_out", into: kSliding2)
        copyBack(out2, "V_sliding_out", into: vSliding2)
        copyBack(out2, "K_full_out", into: kFull2)
        copyBack(out2, "V_full_out", into: vFull2)
        let kv13k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        lastKV13K = kv13k; lastKV13V = kv13v
        lastKV14K = kv14k; lastKV14V = kv14v

        let shared: [String: MLFeatureValue] = [
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13k), "kv13_v": MLFeatureValue(multiArray: kv13v),
            "kv14_k": MLFeatureValue(multiArray: kv14k), "kv14_v": MLFeatureValue(multiArray: kv14v),
        ]

        // Verify chunk 3
        var d3 = shared; d3["hidden_states"] = MLFeatureValue(multiArray: h2)
        let h3 = try verifyChunk3!.prediction(from: MLDictionaryFeatureProvider(dictionary: d3))
            .featureValue(for: "hidden_states_out")!.multiArrayValue!

        // Verify chunk 4: returns per-position token IDs (1, K) + hidden_states
        var d4 = shared; d4["hidden_states"] = MLFeatureValue(multiArray: h3)
        let out4 = try verifyChunk4!.prediction(from: MLDictionaryFeatureProvider(dictionary: d4))
        let tokenIds = out4.featureValue(for: "token_ids")!.multiArrayValue!
        // Store hidden_states for MTP drafter carry state
        lastVerifyHiddenStates = out4.featureValue(for: "hidden_states_out")?.multiArrayValue

        // Extract K token IDs from (1, K) int32 output
        var result = [Int32]()
        result.reserveCapacity(K)
        let ptr = tokenIds.dataPointer.bindMemory(to: Int32.self, capacity: K)
        for k in 0..<K {
            result.append(ptr[k])
        }
        return result
    }

    /// Variant of `verifyCandidates` that also returns top-K `(token_id,
    /// logit_fp32)` pairs at each of the K verify positions.
    ///
    /// Requires the verify chunk 4 mlmodelc to expose a `logits_fp16` output of
    /// shape `(1, K, vocab_size)`. The current staging model does NOT expose
    /// this yet — the parallel Track B (`feat/c0-verify-requant`) will re-export
    /// verify chunk 4 with the extra output. Until that lands, this method
    /// throws `CoreMLLLMError.verifyLogitsNotExposed`.
    ///
    /// - Parameters:
    ///   - tokens: K draft token IDs to verify.
    ///   - startPosition: KV cache position of the first draft token.
    ///   - topK: number of (token, logit) pairs to return per position.
    /// - Returns: `(argmax, topK)` where `argmax` is the same K-length array
    ///   that `verifyCandidates` would return, and `topK` is a K-entry array,
    ///   each holding the top-`topK` (tokenID, logit_fp32) pairs at that
    ///   position sorted by descending logit.
    func verifyCandidatesWithLogits(tokens: [Int32], startPosition: Int,
                                    topK: Int = 3) throws
        -> (argmax: [Int32], topK: [[(Int32, Float)]])
    {
        guard hasVerify else {
            throw CoreMLLLMError.predictionFailed
        }
        let K = tokens.count
        precondition(K == verifyK, "verifyCandidatesWithLogits called with \(K) tokens but model expects \(verifyK)")

        let ctx = config.contextLength
        let W = config.slidingWindow

        // Build batched inputs (identical to `verifyCandidates`).
        let hiddenIn = try buildVerifyHidden(tokenIDs: tokens.map { Int($0) })
        let plRaw = try buildVerifyPLR(tokenIDs: tokens.map { Int($0) })
        let maskFull = try makeVerifyCausalMask(startPos: startPosition, K: K, length: ctx)
        let maskSliding = try makeVerifySlidingMask(startPos: startPosition, K: K, W: W)

        let updateIndicator = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), NSNumber(value: K)], dataType: .float16)
        let indPtr = updateIndicator.dataPointer.bindMemory(to: UInt16.self, capacity: ctx * K)
        memset(indPtr, 0, ctx * K * 2)
        for k in 0..<K {
            let pos = startPosition + k
            if pos < ctx {
                indPtr[pos * K + k] = 0x3C00  // 1.0 in float16
            }
        }

        let cosS = try lookupRoPEBatch(table: cosSlidingTable, startPos: startPosition, K: K, dim: 256)
        let sinS = try lookupRoPEBatch(table: sinSlidingTable, startPos: startPosition, K: K, dim: 256)
        let cosF = try lookupRoPEBatch(table: cosFullTable, startPos: startPosition, K: K, dim: 512)
        let sinF = try lookupRoPEBatch(table: sinFullTable, startPos: startPosition, K: K, dim: 512)

        // Verify chunk 1
        let out1 = try verifyChunk1!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_indicator": MLFeatureValue(multiArray: updateIndicator),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: kSliding1),
            "V_sliding_in": MLFeatureValue(multiArray: vSliding1),
            "K_full_in": MLFeatureValue(multiArray: kFull1),
            "V_full_in": MLFeatureValue(multiArray: vFull1),
        ]))
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let plc = out1.featureValue(for: "per_layer_combined_out")!.multiArrayValue!
        copyBack(out1, "K_sliding_out", into: kSliding1)
        copyBack(out1, "V_sliding_out", into: vSliding1)
        copyBack(out1, "K_full_out", into: kFull1)
        copyBack(out1, "V_full_out", into: vFull1)

        // Verify chunk 2
        let out2 = try verifyChunk2!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_indicator": MLFeatureValue(multiArray: updateIndicator),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: kSliding2),
            "V_sliding_in": MLFeatureValue(multiArray: vSliding2),
            "K_full_in": MLFeatureValue(multiArray: kFull2),
            "V_full_in": MLFeatureValue(multiArray: vFull2),
        ]))
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!
        copyBack(out2, "K_sliding_out", into: kSliding2)
        copyBack(out2, "V_sliding_out", into: vSliding2)
        copyBack(out2, "K_full_out", into: kFull2)
        copyBack(out2, "V_full_out", into: vFull2)
        let kv13k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        lastKV13K = kv13k; lastKV13V = kv13v
        lastKV14K = kv14k; lastKV14V = kv14v

        let shared: [String: MLFeatureValue] = [
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13k), "kv13_v": MLFeatureValue(multiArray: kv13v),
            "kv14_k": MLFeatureValue(multiArray: kv14k), "kv14_v": MLFeatureValue(multiArray: kv14v),
        ]

        // Verify chunk 3
        var d3 = shared; d3["hidden_states"] = MLFeatureValue(multiArray: h2)
        let h3 = try verifyChunk3!.prediction(from: MLDictionaryFeatureProvider(dictionary: d3))
            .featureValue(for: "hidden_states_out")!.multiArrayValue!

        // Verify chunk 4
        var d4 = shared; d4["hidden_states"] = MLFeatureValue(multiArray: h3)
        let out4 = try verifyChunk4!.prediction(from: MLDictionaryFeatureProvider(dictionary: d4))
        let tokenIds = out4.featureValue(for: "token_ids")!.multiArrayValue!
        lastVerifyHiddenStates = out4.featureValue(for: "hidden_states_out")?.multiArrayValue

        // Track B gate: verify chunk 4 must expose `logits_fp16` of shape
        // `(1, K, vocab_size)` fp16. Today's staging model only exposes the
        // argmax reduction (`token_ids`).
        guard let logitsFV = out4.featureValue(for: "logits_fp16"),
              let logits = logitsFV.multiArrayValue else {
            throw CoreMLLLMError.verifyLogitsNotExposed
        }

        // Extract argmax first so we match `verifyCandidates`'s return shape
        // exactly (callers combining both can diff).
        var argmax = [Int32]()
        argmax.reserveCapacity(K)
        let tidPtr = tokenIds.dataPointer.bindMemory(to: Int32.self, capacity: K)
        for k in 0..<K { argmax.append(tidPtr[k]) }

        // Extract top-K. Logits are fp16 laid out as (1, K, vocab_size).
        let vocab = config.vocabSize
        let needTopK = max(1, topK)
        precondition(logits.count >= K * vocab,
                     "logits_fp16 output smaller than expected (got \(logits.count), need \(K * vocab))")
        let logitPtr = logits.dataPointer.bindMemory(to: UInt16.self, capacity: K * vocab)

        var topKOut: [[(Int32, Float)]] = []
        topKOut.reserveCapacity(K)
        // Partial selection: linear scan keeping a small sorted buffer.
        // needTopK is tiny (≤ ~10), so O(vocab * needTopK) is fine vs an
        // O(vocab log vocab) full sort.
        for k in 0..<K {
            var best = [(Int32, Float)]()
            best.reserveCapacity(needTopK)
            let rowOffset = k * vocab
            for v in 0..<vocab {
                let logit = Float(Float16(bitPattern: logitPtr[rowOffset + v]))
                if best.count < needTopK {
                    best.append((Int32(v), logit))
                    // Keep sorted descending.
                    best.sort { $0.1 > $1.1 }
                } else if logit > best[needTopK - 1].1 {
                    best[needTopK - 1] = (Int32(v), logit)
                    // Bubble up — tiny list, so insertion-sort scan suffices.
                    var j = needTopK - 1
                    while j > 0 && best[j].1 > best[j - 1].1 {
                        best.swapAt(j, j - 1)
                        j -= 1
                    }
                }
            }
            topKOut.append(best)
        }
        return (argmax, topKOut)
    }

    // MARK: - Verify helpers

    private func buildVerifyHidden(tokenIDs: [Int]) throws -> MLMultiArray {
        let K = tokenIDs.count
        let hidden = config.hiddenSize
        let arr = try MLMultiArray(shape: [1, NSNumber(value: K), NSNumber(value: hidden)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: K * hidden)
        for (k, tid) in tokenIDs.enumerated() {
            let emb = try embedTokens.lookup(tid, shape: [1, 1, NSNumber(value: hidden)])
            let src = emb.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
            memcpy(dst.advanced(by: k * hidden), src, hidden * MemoryLayout<UInt16>.stride)
        }
        return arr
    }

    private func buildVerifyPLR(tokenIDs: [Int]) throws -> MLMultiArray {
        let K = tokenIDs.count
        let totalDim = config.numLayers * config.perLayerDim
        let arr = try MLMultiArray(shape: [1, NSNumber(value: K), NSNumber(value: totalDim)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: K * totalDim)
        memset(dst, 0, K * totalDim * MemoryLayout<UInt16>.stride)
        for (k, tid) in tokenIDs.enumerated() {
            let raw = embedPerLayer.lookupRaw(tid)
            memcpy(dst.advanced(by: k * totalDim), raw, totalDim * MemoryLayout<UInt16>.stride)
        }
        return arr
    }

    private func makeVerifyCausalMask(startPos: Int, K: Int, length: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: K), NSNumber(value: length)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: K * length)
        for q in 0..<K {
            let maxAttend = startPos + q
            for i in 0..<length {
                mp[q * length + i] = i <= maxAttend ? 0 : 0xFC00
            }
        }
        return mask
    }

    private func makeVerifySlidingMask(startPos: Int, K: Int, W: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: K), NSNumber(value: W)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: K * W)
        for q in 0..<K {
            let valid = min(startPos + q + 1, W)
            let start = W - valid
            for i in 0..<W {
                mp[q * W + i] = i >= start ? 0 : 0xFC00
            }
        }
        return mask
    }

    private func lookupRoPEBatch(table: Data?, startPos: Int, K: Int, dim: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: K), NSNumber(value: dim)], dataType: .float16)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: K * dim)
        guard let table else {
            memset(dst, 0, K * dim * MemoryLayout<UInt16>.stride)
            return result
        }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        table.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for k in 0..<K {
                let pos = startPos + k
                let offset = headerSize + pos * rowBytes
                if offset + rowBytes <= table.count {
                    memcpy(dst.advanced(by: k * dim), base.advanced(by: offset), rowBytes)
                } else {
                    memset(dst.advanced(by: k * dim), 0, rowBytes)
                }
            }
        }
        return result
    }

    // MARK: - PLE (monolithic model path, CPU Accelerate)

    func computePerLayerCombined(tokenID: Int, embedding: MLMultiArray) throws -> MLMultiArray {
        let nlayers = config.numLayers
        let pld = config.perLayerDim
        let hidden = config.hiddenSize
        let totalDim = nlayers * pld
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
        let resultPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: totalDim)

        let raw = embedPerLayer.lookupRaw(tokenID)

        let embPtr = embedding.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
        var embF16 = [Float16](repeating: 0, count: hidden)
        var embF32 = [Float](repeating: 0, count: hidden)
        for i in 0..<hidden { embF16[i] = Float16(bitPattern: embPtr[i]) }
        vDSP.convertElements(of: embF16, to: &embF32)

        var proj = [Float](repeating: 0, count: totalDim)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Int32(totalDim), Int32(hidden),
                    config.perLayerProjScale, perLayerProjF32, Int32(hidden),
                    embF32, 1, 0.0, &proj, 1)

        if let normData = perLayerNormWeight {
            normData.withUnsafeBytes { normRaw in
                let normW = normRaw.baseAddress!.assumingMemoryBound(to: Float.self)
                let eps: Float = 1e-6
                for li in 0..<nlayers {
                    let s = li * pld
                    var sumSq: Float = 0
                    proj.withUnsafeBufferPointer { buf in
                        vDSP_svesq(buf.baseAddress! + s, 1, &sumSq, vDSP_Length(pld))
                    }
                    let invRms = 1.0 / sqrtf(sumSq / Float(pld) + eps)
                    for j in 0..<pld { proj[s + j] *= invRms * normW[j] }
                }
            }
        }

        for i in 0..<totalDim {
            let combined = (proj[i] + fp16ToF32(raw[i])) * config.perLayerInputScale
            resultPtr[i] = f32ToFp16(combined)
        }
        return result
    }

    // MARK: - Helpers

    private func copyBack(_ output: MLFeatureProvider, _ name: String, into buf: MLMultiArray) {
        let src = output.featureValue(for: name)!.multiArrayValue!
        memcpy(buf.dataPointer, src.dataPointer, buf.count * MemoryLayout<UInt16>.stride)
    }

    private func lookupPerLayerRaw(tokenID: Int) throws -> MLMultiArray {
        let totalDim = config.numLayers * config.perLayerDim
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
        let raw = embedPerLayer.lookupRaw(tokenID)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: totalDim)
        memcpy(dst, raw, totalDim * MemoryLayout<UInt16>.stride)
        return result
    }

    /// Fill the decode-step full causal mask. Reuses scratchMaskFull when
    /// `length` matches the configured context; falls back to fresh
    /// allocation for verify / custom lengths to keep those call sites
    /// independent of the pooled buffer's lifetime.
    private func makeCausalMask(position: Int, length: Int) throws -> MLMultiArray {
        let mask: MLMultiArray
        if length == config.contextLength {
            mask = scratchMaskFull
        } else {
            mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: length)], dataType: .float16)
        }
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: length)
        for i in 0..<length { mp[i] = i <= position ? 0 : 0xFC00 }
        return mask
    }

    private func makeSlidingCausalMask(position: Int, W: Int) throws -> MLMultiArray {
        let mask: MLMultiArray
        if W == config.slidingWindow {
            mask = scratchMaskSliding
        } else {
            mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: W)], dataType: .float16)
        }
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        let valid = min(position + 1, W)
        let start = W - valid
        for i in 0..<W { mp[i] = i >= start ? 0 : 0xFC00 }
        return mask
    }

    private func makeUpdateMask(position: Int, length: Int) throws -> MLMultiArray {
        let umask: MLMultiArray
        if length == config.contextLength {
            umask = scratchUpdateMask
        } else {
            umask = try MLMultiArray(shape: [1, 1, NSNumber(value: length), 1], dataType: .float16)
        }
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: length)
        memset(up, 0, length * MemoryLayout<UInt16>.stride)
        up[min(position, length - 1)] = 0x3C00
        return umask
    }

    func lookupRoPE(table: Data?, position: Int, dim: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: dim)], dataType: .float16)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        guard let table else { memset(dst, 0, dim * MemoryLayout<UInt16>.stride); return result }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        let offset = headerSize + position * rowBytes
        guard offset + rowBytes <= table.count else { memset(dst, 0, rowBytes); return result }
        _ = table.withUnsafeBytes { raw in memcpy(dst, raw.baseAddress!.advanced(by: offset), rowBytes) }
        return result
    }

    // MARK: - Prefill helpers

    private func buildPrefillHidden(tokenIDs: [Int], N: Int,
                                     imageFeatures: MLMultiArray? = nil,
                                     imageNumTokens: Int = 256,
                                     audioFeatures: MLMultiArray? = nil,
                                     audioNumTokens: Int = 50) throws -> MLMultiArray {
        let IMAGE_TOKEN_ID = 258880
        let AUDIO_TOKEN_ID = 258881
        let VIDEO_TOKEN_ID = 258884
        let hidden = config.hiddenSize
        let arr = try MLMultiArray(shape: [1, NSNumber(value: N), NSNumber(value: hidden)], dataType: .float16)
        memset(arr.dataPointer, 0, N * hidden * MemoryLayout<UInt16>.stride)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N * hidden)
        let imgPtr = imageFeatures?.dataPointer.bindMemory(to: UInt16.self, capacity: imageFeatures?.count ?? 0)
        let audPtr = audioFeatures?.dataPointer.bindMemory(to: UInt16.self, capacity: audioFeatures?.count ?? 0)
        var imageIdx = 0
        var audioIdx = 0
        for (i, tid) in tokenIDs.enumerated() {
            // Image and video share the same `imageFeatures` buffer; the
            // video path concatenates frames into the same per-token
            // (1, N, hidden) layout the image path uses.
            if (tid == IMAGE_TOKEN_ID || tid == VIDEO_TOKEN_ID),
               let fp = imgPtr, imageIdx < imageNumTokens {
                memcpy(dst.advanced(by: i * hidden), fp.advanced(by: imageIdx * hidden),
                       hidden * MemoryLayout<UInt16>.stride)
                imageIdx += 1
            } else if tid == AUDIO_TOKEN_ID, let ap = audPtr, audioIdx < audioNumTokens {
                memcpy(dst.advanced(by: i * hidden), ap.advanced(by: audioIdx * hidden),
                       hidden * MemoryLayout<UInt16>.stride)
                audioIdx += 1
            } else {
                let emb = try embedTokens.lookup(tid, shape: [1, 1, NSNumber(value: hidden)])
                let src = emb.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
                memcpy(dst.advanced(by: i * hidden), src, hidden * MemoryLayout<UInt16>.stride)
            }
        }
        return arr
    }

    private func buildPrefillPLR(tokenIDs: [Int], N: Int) throws -> MLMultiArray {
        let IMAGE_TOKEN_ID = 258880
        let AUDIO_TOKEN_ID = 258881
        let VIDEO_TOKEN_ID = 258884
        let totalDim = config.numLayers * config.perLayerDim
        let arr = try MLMultiArray(shape: [1, NSNumber(value: N), NSNumber(value: totalDim)], dataType: .float16)
        memset(arr.dataPointer, 0, N * totalDim * MemoryLayout<UInt16>.stride)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N * totalDim)
        for (i, tid) in tokenIDs.enumerated() {
            // Multimodal positions get zero PLE — the per_layer_model_projection
            // from hidden_states (vision/audio features) is computed inside
            // chunk1 on ANE. Adding per_layer_raw from a placeholder token
            // corrupts PLE with nonsense.
            if tid == IMAGE_TOKEN_ID || tid == AUDIO_TOKEN_ID || tid == VIDEO_TOKEN_ID {
                continue
            }
            let raw = embedPerLayer.lookupRaw(tid)
            memcpy(dst.advanced(by: i * totalDim), raw, totalDim * MemoryLayout<UInt16>.stride)
        }
        return arr
    }

    /// Prefill causal mask (strict causal). Used for text-only / image
    /// prefills where no per-frame vision grouping is needed.
    private func makePrefillCausalMask(N: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: N)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: N * N)
        for i in 0..<N { for j in 0..<N { mp[i * N + j] = j <= i ? 0 : 0xFC00 } }
        return mask
    }

    /// Vision-group-aware prefill causal mask, matching HF
    /// `create_causal_mask_mapping` + `token_type_ids_mask_function`:
    /// each contiguous run of `<|video|>` (or `<|image|>`) tokens forms a
    /// "vision group" that attends bidirectionally within itself. Between
    /// groups, and between text and vision, standard causal masking
    /// applies. Without this, each video frame's 64 tokens can only see
    /// earlier tokens in the same frame — which robs the model of the 2D
    /// image representation it was trained to build per frame and leads
    /// to the "series of still images" framing seen on-device.
    ///
    /// HF only applies this relaxation to sliding-attention layers. Our
    /// prefill chunks share a single `causal_mask` across sliding+full
    /// layers, so the unmask leaks to full-attention layers too; the
    /// effect is benign (full-attention is already causal → at worst we
    /// unmask a few extra positions inside a vision group that full
    /// attention would have seen later anyway).
    private func makePrefillVisionMask(tokenIDs: [Int], N: Int) throws -> MLMultiArray {
        let IMAGE_TOKEN_ID = 258880
        let VIDEO_TOKEN_ID = 258884
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: N)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: N * N)

        // Group ids: -1 for text/other, 0/1/2/... for each contiguous
        // run of vision placeholder tokens.
        var groupIds = [Int](repeating: -1, count: N)
        var currentGroup = -1
        var prevWasVision = false
        for i in 0..<min(N, tokenIDs.count) {
            let isVision = tokenIDs[i] == IMAGE_TOKEN_ID || tokenIDs[i] == VIDEO_TOKEN_ID
            if isVision {
                if !prevWasVision { currentGroup += 1 }
                groupIds[i] = currentGroup
            }
            prevWasVision = isVision
        }

        // Fill mask: causal by default, unmask pairs that share a vision
        // group so the group attends bidirectionally within itself.
        for i in 0..<N {
            let gi = groupIds[i]
            for j in 0..<N {
                let sameGroup = gi >= 0 && groupIds[j] == gi
                mp[i * N + j] = (j <= i || sameGroup) ? 0 : 0xFC00
            }
        }
        return mask
    }

    private func buildPrefillRoPE(table: Data?, N: Int, dim: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: dim)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N * dim)
        guard let table else { memset(dst, 0, N * dim * MemoryLayout<UInt16>.stride); return arr }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        table.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for p in 0..<N {
                let off = headerSize + p * rowBytes
                if off + rowBytes <= table.count {
                    memcpy(dst.advanced(by: p * dim), base.advanced(by: off), rowBytes)
                } else {
                    memset(dst.advanced(by: p * dim), 0, rowBytes)
                }
            }
        }
        return arr
    }

    private func makeLastPositionMask(N: Int, realLen: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: N), 1], dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N)
        memset(p, 0, N * MemoryLayout<UInt16>.stride)
        p[realLen - 1] = 0x3C00
        return arr
    }

    // MARK: - KV cache write-back from prefill

    private func writeSlidingFromPrefill(src: MLFeatureProvider, name: String,
                                          cache: MLMultiArray, slot: Int,
                                          realLen: Int, hd: Int) throws {
        guard let srcArr = src.featureValue(for: name)?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        let W = config.slidingWindow
        let shape = cache.shape.map { $0.intValue }
        let maxHd = shape[3]
        let slotStride = 1 * W * maxHd
        let dst = cache.dataPointer.bindMemory(to: UInt16.self, capacity: cache.count)
        let s = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
        let startCachePos = W - realLen
        for p in 0..<realLen {
            let srcOff = p * hd
            let dstOff = slot * slotStride + (startCachePos + p) * maxHd
            for j in 0..<hd { dst[dstOff + j] = s[srcOff + j] }
        }
    }

    private func writeFullFromPrefill(src: MLFeatureProvider, name: String,
                                       cache: MLMultiArray, slot: Int,
                                       realLen: Int, hd: Int) throws {
        guard let srcArr = src.featureValue(for: name)?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        let shape = cache.shape.map { $0.intValue }
        let ctx = shape[2]; let maxHd = shape[3]
        let slotStride = 1 * ctx * maxHd
        let dst = cache.dataPointer.bindMemory(to: UInt16.self, capacity: cache.count)
        let s = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
        for p in 0..<realLen {
            let srcOff = p * hd
            let dstOff = slot * slotStride + p * maxHd
            for j in 0..<hd { dst[dstOff + j] = s[srcOff + j] }
        }
    }

    // KV slot mapping: chunk1 outputs K0..K7/V0..V7
    // Sliding slots: L0→0, L1→1, L2→2, L3→3, L5→4, L6→5, L7→6 (skip L4 = full)
    // Full slots: L4→0
    private func kvMapChunk1Sliding() -> [(String, Int, MLMultiArray, Int)] {
        [("K0",0,kSliding1,256),("V0",0,vSliding1,256),
         ("K1",1,kSliding1,256),("V1",1,vSliding1,256),
         ("K2",2,kSliding1,256),("V2",2,vSliding1,256),
         ("K3",3,kSliding1,256),("V3",3,vSliding1,256),
         ("K5",4,kSliding1,256),("V5",4,vSliding1,256),
         ("K6",5,kSliding1,256),("V6",5,vSliding1,256),
         ("K7",6,kSliding1,256),("V7",6,vSliding1,256)]
    }
    private func kvMapChunk1Full() -> [(String, Int, MLMultiArray, Int)] {
        [("K4",0,kFull1,512),("V4",0,vFull1,512)]
    }

    // Chunk2: L8→sliding0, L9→full0, L10→s1, L11→s2, L12→s3, L13→s4, L14→full1
    private func kvMapChunk2Sliding() -> [(String, Int, MLMultiArray, Int)] {
        [("K0",0,kSliding2,256),("V0",0,vSliding2,256),
         ("K2",1,kSliding2,256),("V2",1,vSliding2,256),
         ("K3",2,kSliding2,256),("V3",2,vSliding2,256),
         ("K4",3,kSliding2,256),("V4",3,vSliding2,256),
         ("kv13_k",4,kSliding2,256),("kv13_v",4,vSliding2,256)]
    }
    private func kvMapChunk2Full() -> [(String, Int, MLMultiArray, Int)] {
        [("K1",0,kFull2,512),("V1",0,vFull2,512),
         ("kv14_k",1,kFull2,512),("kv14_v",1,vFull2,512)]
    }

    /// Slice a single feature vector from vision encoder output.
    func sliceFeature(_ features: MLMultiArray, at index: Int) -> MLMultiArray {
        let hs = config.hiddenSize
        let r = try! MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
        let s = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let d = r.dataPointer.bindMemory(to: UInt16.self, capacity: hs)
        memcpy(d, s.advanced(by: index * hs), hs * MemoryLayout<UInt16>.stride)
        return r
    }

    // MARK: - MTP drafter support

    /// Raw (unscaled) token embedding for MTP drafter.
    func lookupRawEmbed(_ tokenID: Int32) throws -> MLMultiArray {
        let hidden = config.hiddenSize
        return try embedTokens.lookupUnscaled(Int(tokenID),
            shape: [1, 1, NSNumber(value: hidden)])
    }

    /// RoPE cos/sin lookups at a specific position (exposed for drafter).
    func lookupCosSWA(position: Int) throws -> MLMultiArray {
        try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
    }
    func lookupSinSWA(position: Int) throws -> MLMultiArray {
        try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
    }
    func lookupCosFull(position: Int) throws -> MLMultiArray {
        try lookupRoPE(table: cosFullTable, position: position, dim: 512)
    }
    func lookupSinFull(position: Int) throws -> MLMultiArray {
        try lookupRoPE(table: sinFullTable, position: position, dim: 512)
    }

    /// Causal masks for drafter (exposed wrappers).
    func makeDrafterSWAMask(position: Int) throws -> MLMultiArray {
        try makeSlidingCausalMask(position: position, W: config.slidingWindow)
    }
    func makeDrafterFullMask(position: Int) throws -> MLMultiArray {
        try makeCausalMask(position: position, length: config.contextLength)
    }
}

// MARK: - SpeculativeTarget conformance

extension ChunkedEngine: SpeculativeTarget {
    public func lastHiddenMulti(at layerIndices: [Int]) throws -> [MLMultiArray] {
        // Placeholder: requires chunk modifications to expose per-layer hidden states.
        // For now, return empty arrays; EAGLE-3 integration will fill this in.
        throw CoreMLLLMError.predictionFailed
    }

    public func verifyCandidates(_ candidates: [Int32], K: Int) throws -> [Int32] {
        return try verifyCandidates(tokens: candidates, startPosition: currentPosition)
    }

    public func commitAccepted(_ tokens: [Int32]) throws {
        // Write-through: verify already wrote KV for all K positions.
        // Rejected entries are masked out by causal mask in future steps.
        // Just advance the position counter.
        currentPosition += tokens.count
    }
}
