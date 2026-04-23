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

    // Prefill chunks (optional; falls back to per-token decode if nil).
    //
    // These are `var` + lock so the deferred-load path (LLM_DEFER_PREFILL=1)
    // can attach them after engine construction. Readers must go through
    // the computed accessors below to avoid torn state during attachment.
    private let prefillLock = NSLock()
    private var _prefillChunk1: MLModel?
    private var _prefillChunk2: MLModel?
    private var _prefillChunk3: MLModel?
    private var _prefillChunk4: MLModel?
    private var _prefillLoadTask: Task<Void, Error>?

    private var prefillChunk1: MLModel? {
        prefillLock.lock(); defer { prefillLock.unlock() }
        return _prefillChunk1
    }
    private var prefillChunk2: MLModel? {
        prefillLock.lock(); defer { prefillLock.unlock() }
        return _prefillChunk2
    }
    private var prefillChunk3: MLModel? {
        prefillLock.lock(); defer { prefillLock.unlock() }
        return _prefillChunk3
    }
    private var prefillChunk4: MLModel? {
        prefillLock.lock(); defer { prefillLock.unlock() }
        return _prefillChunk4
    }

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
    /// Batch width for the prefill graph (e.g. 1024 for v1.2.0+ ships). Read
    /// from the prefill_chunk1 input shape at init when prefill is eagerly
    /// loaded; when LLM_DEFER_PREFILL=1 is active (default) p1 is nil at init
    /// and this stays at its fallback until `attachPrefill` updates it.
    private(set) var prefillN: Int
    var currentPosition: Int = 0

    // EAGLE-3 speculative decoding state (Phase 2B).
    // hidden_at_L{8,17,34} are captured from decode chunks 2/3/4 on each step,
    // then consumed by `SpeculativeLoop` to build the draft's fused hidden. They
    // stay nil until an EAGLE-3-capable decode chunk set is loaded.
    var lastHiddenAtL8:  MLMultiArray?
    var lastHiddenAtL17: MLMultiArray?
    var lastHiddenAtL34: MLMultiArray?

    /// Most recent token produced by `predictStep`. Used to hand the next
    /// `tTokNext` back to `SpeculativeLoop.drawBurst` after a commit burst.
    private(set) var lastArgmaxAfterDecode: Int = 0

    var hasPrefill: Bool {
        prefillLock.lock(); defer { prefillLock.unlock() }
        return _prefillChunk1 != nil && _prefillChunk2 != nil
            && _prefillChunk3 != nil && _prefillChunk4 != nil
    }

    /// Attach prefill chunks after engine construction (LLM_DEFER_PREFILL path).
    /// Set all four at once under lock so `hasPrefill` never observes
    /// a partial attachment. Reads the real `prefillN` from p1's hidden_states
    /// input shape — during init in the deferred-load path p1 is nil, so
    /// `prefillN` held its fallback default and would mismatch the model if
    /// the ship graph is wider than that (e.g. v1.2.0+ N=1024 bundles).
    func attachPrefill(_ p1: MLModel, _ p2: MLModel, _ p3: MLModel, _ p4: MLModel) {
        prefillLock.lock()
        _prefillChunk1 = p1; _prefillChunk2 = p2
        _prefillChunk3 = p3; _prefillChunk4 = p4
        if let hs = p1.modelDescription.inputDescriptionsByName["hidden_states"],
           let constraint = hs.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            if shape.count >= 2 {
                let detected = shape[1]
                if detected != prefillN {
                    print("[Load] prefillN updated via attachPrefill: \(prefillN) → \(detected)")
                    prefillN = detected
                }
            }
        }
        prefillLock.unlock()
    }

    /// Token handle for the background prefill load (if any). Callers can
    /// `try await engine.prefillLoadTask?.value` to block until prefill is
    /// ready — e.g., before shipping a long prompt where fallback per-token
    /// decode would be too slow.
    var prefillLoadTask: Task<Void, Error>? {
        prefillLock.lock(); defer { prefillLock.unlock() }
        return _prefillLoadTask
    }

    fileprivate func setPrefillLoadTask(_ t: Task<Void, Error>) {
        prefillLock.lock()
        _prefillLoadTask = t
        prefillLock.unlock()
    }

    /// Warm the prefill path only. Called by the deferred-load task after
    /// attachPrefill so the first real prefill doesn't pay the cold-compile
    /// hit even when prefill arrived after finalPrewarm.
    ///
    /// Skips if the engine is already mid-conversation (currentPosition > 0)
    /// — warmup resets KV and would corrupt the user's in-flight state.
    /// The first real prefill pays the cold cost in that case.
    public func warmPrefillOnly() throws {
        guard hasPrefill else { return }
        guard currentPosition == 0 else {
            print("[Load] Prefill-only warm skipped (engine in use, pos=\(currentPosition))")
            return
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try? runPrefill(tokenIDs: [0])
        // Transition warm: one decode after the populated-KV prefill.
        for i in 1...2 {
            _ = try? predictStep(tokenID: 0, position: i)
        }
        reset()
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print("[Load] Prefill-only warm done in \(String(format: "%.2f", dt))s")
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

    // 11c write-after-accept: per-T raw K/V slices captured from the last
    // verify call, used by commitAccepted to selectively write only the
    // accepted prefix into persistent kSliding/kFull buffers.
    // Shapes: new_K_sliding_cN (num_slots, 1, K, 256), new_K_full_cN (num_slots, 1, K, 512).
    private var lastNewKSliding1: MLMultiArray?  // (7, 1, K, 256)
    private var lastNewVSliding1: MLMultiArray?
    private var lastNewKFull1: MLMultiArray?     // (1, 1, K, 512)
    private var lastNewVFull1: MLMultiArray?
    private var lastNewKSliding2: MLMultiArray?  // (5, 1, K, 256)
    private var lastNewVSliding2: MLMultiArray?
    private var lastNewKFull2: MLMultiArray?     // (2, 1, K, 512)
    private var lastNewVFull2: MLMultiArray?
    /// Tokens that were the verify input — needed by commitAccepted to detect
    /// which accepted positions match a verified slice (writeable) vs which
    /// need a fresh T=1 step (correction or bonus past the verify range).
    private var lastVerifyInputTokens: [Int32] = []

    /// Test-only KV snapshot. Returns a byte-copy of all 8 persistent KV
    /// buffers, so a test can run two paths and assert byte-identical state.
    internal func _kvSnapshotBytes() -> [String: Data] {
        func snap(_ a: MLMultiArray) -> Data {
            return Data(bytes: a.dataPointer, count: a.count * MemoryLayout<UInt16>.stride)
        }
        return [
            "kSliding1": snap(kSliding1), "vSliding1": snap(vSliding1),
            "kFull1": snap(kFull1), "vFull1": snap(vFull1),
            "kSliding2": snap(kSliding2), "vSliding2": snap(vSliding2),
            "kFull2": snap(kFull2), "vFull2": snap(vFull2),
        ]
    }

    // MARK: - Loading

    static func load(from directory: URL, config: ModelConfig,
                     computeUnits: MLComputeUnits) async throws -> ChunkedEngine {
        // iOS 18+ MLOptimizationHints.specializationStrategy = .fastPrediction
        // trades a longer first-load ANE specialization for (nominally)
        // shorter per-prediction wall time.
        //
        // As of 2026-04-21 device bench on iPhone 17 Pro / A19 Pro / iOS 26,
        // the runtime decode tok/s delta between .fastPrediction on and off
        // was below measurement noise, while the load-time cost was large
        // (multi-second ANECompilerService + P-core burn per chunk × 8
        // chunks, showing up as the hottest moment of the session).
        //
        // Default is now **off**. Set LLM_FAST_PREDICTION=1 to opt back in
        // on hardware/OS combinations where the specialization is worth the
        // load-time heat (older A-series / beta OS builds may differ).
        //
        // Removed: .reshapeFrequency = .infrequent. Worked stand-alone but
        // combined with LLM_PREFIX_CACHE=1 reproducibly triggered
        // "MILCompilerForANE error: failed to compile ANE model using ANEF"
        // on iPhone 17 Pro (A19 Pro, iOS 26). 1-3% gain not worth the
        // instability; removed entirely rather than left as opt-in to
        // prevent accidental enable.
        let fastPredictionEnabled = ProcessInfo.processInfo.environment["LLM_FAST_PREDICTION"] == "1"
        func applyHints(_ cfg: MLModelConfiguration) {
            guard fastPredictionEnabled else { return }
            if #available(iOS 18.0, macOS 15.0, *) {
                var hints = MLOptimizationHints()
                hints.specializationStrategy = .fastPrediction
                cfg.optimizationHints = hints
            }
        }

        // Env override: LLM_COMPUTE_UNITS=cpu|cpuGpu|cpuAne|all — for
        // diagnosing ANE-vs-other-device numerical differences. When a per-
        // device ANE gives very different numerics (e.g., 4-bit palettized
        // weights decoded differently on A-series vs M-series), routing the
        // same .mlmodelc through CPU eliminates the hardware variance at the
        // cost of speed.
        var effectiveUnits = computeUnits
        if let raw = ProcessInfo.processInfo.environment["LLM_COMPUTE_UNITS"]?.lowercased() {
            switch raw {
            case "cpu", "cpuonly":
                effectiveUnits = .cpuOnly
            case "cpugpu", "cpuandgpu":
                effectiveUnits = .cpuAndGPU
            case "cpuane", "cpuandneuralengine":
                effectiveUnits = .cpuAndNeuralEngine
            case "all":
                effectiveUnits = .all
            default:
                break
            }
            print("[Load] LLM_COMPUTE_UNITS=\(raw) → computeUnits=\(effectiveUnits)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = effectiveUnits
        applyHints(mlConfig)

        // Prefill chunks use GPU for compute-bound batch processing (TTFT win).
        // Decode chunks stay on ANE for bandwidth-bound single-token inference.
        let prefillConfig = MLModelConfiguration()
        let useGPUPrefill = ProcessInfo.processInfo.environment["GPU_PREFILL"] == "1"
        prefillConfig.computeUnits = useGPUPrefill ? .cpuAndGPU : effectiveUnits
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

        // Chunk load strategy.
        //
        // Device bench on 2026-04-21 (iPhone 17 Pro / A19 Pro / iOS 26,
        // Gemma 4 E2B, 8 chunks = decode×4 + prefill×4):
        //   all-parallel (8 concurrent):  97.3 s
        //   sequential  (cap = 1):        71.3 s
        //
        // Parallel load is **slower and hotter**. Most likely
        // ANECompilerService serializes compile work internally; naive
        // 8-way TaskGroup seeding just adds scheduling pressure and makes
        // the ANE compile daemon + P-cores all burn at once.
        //
        // Default is now sequential (cap = 1). The original unlimited
        // parallel behavior is preserved as an explicit opt-in via
        // LLM_LOAD_MAX_PARALLEL=0 for hardware/OS combinations where the
        // daemon can actually exploit concurrency.
        //
        // LLM_LOAD_MAX_PARALLEL semantics:
        //   unset   → cap = 1 (sequential, default)
        //   = 0     → unlimited concurrency (legacy fast-burn behavior)
        //   = N > 0 → cap = N concurrent MIL/ANE compilations
        let hasPrefillFiles = findModel("prefill_chunk1") != nil
        var c1: MLModel!, c2: MLModel!, c3: MLModel!, c4: MLModel!
        var p1: MLModel?, p2: MLModel?, p3: MLModel?, p4: MLModel?

        let loadMaxParallel: Int = {
            guard let s = ProcessInfo.processInfo.environment["LLM_LOAD_MAX_PARALLEL"] else {
                return 1   // default: sequential
            }
            return Int(s) ?? 1
        }()
        if ProcessInfo.processInfo.environment["LLM_LOAD_MAX_PARALLEL"] != nil {
            print("[Load] LLM_LOAD_MAX_PARALLEL=\(loadMaxParallel) " +
                  "(override default sequential load)")
        }

        // Default-on: load decode chunks synchronously, then kick off prefill
        // chunks in a background task so the engine is usable for decode-only
        // as soon as c1-c4 are ready. The first user prompt during the load
        // window falls back to per-token decode (slower but interactive) via
        // the `hasPrefill` gate at call sites. Cuts time-to-usable roughly in
        // half (~80s → ~35s on iPhone 17 Pro). Opt-out with LLM_DEFER_PREFILL=0.
        let deferEnv = ProcessInfo.processInfo.environment["LLM_DEFER_PREFILL"]
        let deferPrefill = (deferEnv != "0")
        if deferPrefill && hasPrefillFiles {
            let origin = (deferEnv == "1") ? "env=1" : "default"
            print("[Load] deferred prefill load (\(origin)) — decode chunks foreground, prefill chunks background")
        }
        var chunkWork: [(String, MLModelConfiguration)] = [
            ("chunk1", mlConfig), ("chunk2", mlConfig),
            ("chunk3", mlConfig), ("chunk4", mlConfig),
        ]
        if hasPrefillFiles && !deferPrefill {
            for name in ["prefill_chunk1", "prefill_chunk2",
                         "prefill_chunk3", "prefill_chunk4"] {
                chunkWork.append((name, prefillConfig))
            }
        }

        let loadT0 = CFAbsoluteTimeGetCurrent()
        try await withThrowingTaskGroup(of: (String, MLModel).self) { group in
            let cap = loadMaxParallel > 0
                ? min(loadMaxParallel, chunkWork.count)
                : chunkWork.count
            var idx = 0
            // Seed the initial wave of up to `cap` compilations.
            while idx < cap {
                let (name, cfg) = chunkWork[idx]
                group.addTask { (name, try loadOne(name, config: cfg)) }
                idx += 1
            }
            // Drain completed tasks and refill one-for-one. When
            // cap == chunkWork.count this is equivalent to the original
            // all-at-once launch (all tasks seeded, none refilled).
            while let (name, model) = try await group.next() {
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
                if idx < chunkWork.count {
                    let (nextName, nextCfg) = chunkWork[idx]
                    group.addTask {
                        (nextName, try loadOne(nextName, config: nextCfg))
                    }
                    idx += 1
                }
            }
        }
        let loadDt = CFAbsoluteTimeGetCurrent() - loadT0
        let loadMode: String
        switch loadMaxParallel {
        case 0:  loadMode = "parallel"
        case 1:  loadMode = "sequential"
        default: loadMode = "cap=\(loadMaxParallel)"
        }
        let loadedCount = chunkWork.count
        let totalCount = hasPrefillFiles ? 8 : 4
        let deferredSuffix = (deferPrefill && hasPrefillFiles)
            ? " (\(totalCount - loadedCount) prefill deferred to bg)"
            : ""
        print("[Load] \(loadedCount)/\(totalCount) chunks loaded in " +
              "\(String(format: "%.1f", loadDt))s (\(loadMode))\(deferredSuffix)")

        // Load verify functions from multi-function chunks (if available).
        // Multi-function chunks have a "verify_qK" function alongside the
        // default "decode_q1". We detect this by checking if the chunk has
        // the verify function and load it with a separate configuration.
        //
        // Opt-in: verify chunks add ~600 MB ANE-resident. Only load if
        // the caller has enabled a speculative path (LLM_EAGLE3_ENABLE=1
        // for the trained EAGLE-3 draft, or SPECULATIVE_PROFILE=1 for
        // the cross-vocab Qwen drafter). Default mode (no env set) skips
        // verify entirely so baseline memory footprint matches pre-spec.
        let specRequested =
            ProcessInfo.processInfo.environment["LLM_EAGLE3_ENABLE"] == "1"
            || ProcessInfo.processInfo.environment["SPECULATIVE_PROFILE"] != nil
        var v1: MLModel?, v2: MLModel?, v3: MLModel?, v4: MLModel?
        var detectedK = 0
        if specRequested {
        do {
            let verifyConfig = MLModelConfiguration()
            verifyConfig.computeUnits = effectiveUnits
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

        // EAGLE-3 fallback: if multi-function verify isn't present, try loading
        // standalone verify_chunk{1..4}.mlmodelc files (produced by
        // build_eagle3_verify.py). These have the same schema as the
        // verify_qK function and work with the existing verifyCandidates()
        // path. All four must be present to enable speculative.
        if v1 == nil || v2 == nil || v3 == nil || v4 == nil {
            if findModel("verify_chunk1") != nil {
                do {
                    let verifyConfig = MLModelConfiguration()
                    verifyConfig.computeUnits = effectiveUnits
                    applyHints(verifyConfig)
                    let verifyT0 = CFAbsoluteTimeGetCurrent()
                    try await withThrowingTaskGroup(of: (String, MLModel).self) { group in
                        for name in ["verify_chunk1", "verify_chunk2", "verify_chunk3", "verify_chunk4"] {
                            guard let url = findModel(name) else { continue }
                            group.addTask { (name, try MLModel(contentsOf: url, configuration: verifyConfig)) }
                        }
                        for try await (name, model) in group {
                            switch name {
                            case "verify_chunk1": v1 = model
                            case "verify_chunk2": v2 = model
                            case "verify_chunk3": v3 = model
                            case "verify_chunk4": v4 = model
                            default: break
                            }
                        }
                    }
                    if v1 != nil && v2 != nil && v3 != nil && v4 != nil {
                        // Standalone eagle3 verify chunks emit token_ids as (T,) 1D;
                        // multi-function verify_qK emits (1, T). Handle both.
                        if let desc = v4!.modelDescription.outputDescriptionsByName["token_ids"],
                           let c = desc.multiArrayConstraint {
                            if c.shape.count >= 2 {
                                detectedK = c.shape[1].intValue
                            } else if c.shape.count == 1 {
                                detectedK = c.shape[0].intValue
                            }
                        }
                        if detectedK == 0 {
                            // Fallback: assume K=3 for EAGLE-3 build convention.
                            detectedK = 3
                            print("[Load] Could not infer K from verify_chunk4 token_ids shape; defaulting to 3")
                        }
                        let vDt = CFAbsoluteTimeGetCurrent() - verifyT0
                        print("[Load] Standalone verify chunks loaded (K=\(detectedK)) in \(String(format: "%.1f", vDt))s")
                    } else {
                        print("[Load] Partial verify_chunk*.mlmodelc set — speculative disabled")
                        v1 = nil; v2 = nil; v3 = nil; v4 = nil
                    }
                } catch {
                    print("[Load] verify_chunk*.mlmodelc load failed (\(error)) — speculative disabled")
                    v1 = nil; v2 = nil; v3 = nil; v4 = nil
                }
            }
        }
        }  // end if specRequested
        if !specRequested {
            print("[Load] speculative disabled by default — set LLM_EAGLE3_ENABLE=1 or SPECULATIVE_PROFILE=1 to load verify chunks")
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
        // Fallback default matches v1.2.0+ ship (PREFILL_N=1024). Older
        // bundles export at the shape their graph was traced at; the probe
        // below overrides whenever p1 is available. When LLM_DEFER_PREFILL=1
        // is active and p1 is still nil at init, this fallback is the value
        // held until `attachPrefill` re-reads the real shape.
        var prefillN = 1024
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

        // ANE pipeline prewarm (Phase 0b): dummy decode steps at load time
        // force the ANE compiler to finalize dispatch schedules and resident
        // weight layouts before the first user token arrives. KV cache is
        // reset afterwards so the dummy tokens leave no state behind.
        //
        // NOTE: this is the EARLY prewarm. After this returns, CoreMLLLM.load
        // may load additional models (EAGLE-3 draft/fusion, cross-vocab Qwen,
        // etc.) which can evict this ANE cache — call engine.finalPrewarm()
        // at end of full load to re-warm decode + verify paths.
        let warmT0 = CFAbsoluteTimeGetCurrent()
        for i in 0..<4 {
            _ = try engine.predictStep(tokenID: 0, position: i)
        }
        engine.reset()
        print("[Load] ANE prewarm (4 steps) done in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - warmT0))s")

        // Deferred prefill: load the 4 prefill chunks in the background and
        // attach them once done. Callers can `await engine.prefillLoadTask?.value`
        // to block on completion. Most code paths instead gate on `hasPrefill`,
        // which stays false during the load window and silently falls back to
        // per-token decode.
        if deferPrefill && hasPrefillFiles {
            let directoryCopy = directory
            let prefillConfigCopy = prefillConfig
            let task = Task.detached(priority: .utility) { [weak engine] () throws -> Void in
                let bgT0 = CFAbsoluteTimeGetCurrent()
                func bgFind(_ name: String) -> URL? {
                    let c = directoryCopy.appendingPathComponent("\(name).mlmodelc")
                    if FileManager.default.fileExists(atPath:
                        c.appendingPathComponent("coremldata.bin").path) { return c }
                    let p = directoryCopy.appendingPathComponent("\(name).mlpackage")
                    if FileManager.default.fileExists(atPath: p.path) { return p }
                    return nil
                }
                func bgLoad(_ name: String) throws -> MLModel {
                    guard let url = bgFind(name) else {
                        throw CoreMLLLMError.modelNotFound(name)
                    }
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let m = try MLModel(contentsOf: url, configuration: prefillConfigCopy)
                    print("[Load/bg] \(name) done in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")
                    return m
                }
                let bp1 = try bgLoad("prefill_chunk1")
                let bp2 = try bgLoad("prefill_chunk2")
                let bp3 = try bgLoad("prefill_chunk3")
                let bp4 = try bgLoad("prefill_chunk4")
                guard let engine else { return }
                engine.attachPrefill(bp1, bp2, bp3, bp4)
                let bgDt = CFAbsoluteTimeGetCurrent() - bgT0
                print("[Load/bg] all 4 prefill chunks ready in \(String(format: "%.1f", bgDt))s — warming")
                try? engine.warmPrefillOnly()
            }
            engine.setPrefillLoadTask(task)
        }

        return engine
    }

    /// Final prewarm after ALL auxiliary models (EAGLE-3 draft/fusion,
    /// cross-vocab Qwen, etc.) have loaded — those loads can evict the
    /// decode chunks' ANE cache, causing the first user prompt to see
    /// ~50ms extra per decode step (observed on iPhone 17 Pro).
    /// Call once from CoreMLLLM.load after everything is loaded.
    public func finalPrewarm() throws {
        let t0 = CFAbsoluteTimeGetCurrent()
        // Warm prefill path first — the N=prefillN batched prefill graph is
        // a separate ANE specialization from decode. Without this, the first
        // user prompt pays a cold-compile hit (~100-300ms) on top of normal
        // prefill latency. 1-token input is enough: the buffer width is
        // still prefillN, so ANE compiles the full-shape path.
        if hasPrefill {
            _ = try? runPrefill(tokenIDs: [0])
        }
        // 8 T=1 decode steps (double the early prewarm) so ANE has more
        // dispatch samples to stabilize.
        for i in 0..<8 {
            _ = try predictStep(tokenID: 0, position: i)
        }
        // Warm verify path too — verify_qK is a separate compiled graph
        // that predictStep never touches, so first verifyCandidates is
        // otherwise a cold ANE hit (~100ms extra on first spec burst).
        if hasVerify {
            let dummy: [Int32] = Array(repeating: 0, count: verifyK)
            _ = try? verifyCandidates(tokens: dummy, startPosition: 0)
        }
        // Transition warmup: the first decode after a populated-KV prefill
        // is still cold even with prefill+decode already warmed above
        // (~63ms vs 36ms steady on iPhone 17 Pro, v0.8.0). Simulate the
        // runtime prefill→decode handoff one extra time to pre-pay that
        // per-configuration setup cost.
        if hasPrefill {
            _ = try? runPrefill(tokenIDs: [0])
            for i in 1...2 {
                _ = try? predictStep(tokenID: 0, position: i)
            }
        }
        reset()
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print("[Load] Final prewarm (prefill + decode + verify + transition) done in \(String(format: "%.2f", dt))s")
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
        self._prefillChunk1 = prefillChunk1; self._prefillChunk2 = prefillChunk2
        self._prefillChunk3 = prefillChunk3; self._prefillChunk4 = prefillChunk4
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
        lastVerifyInputTokens = []
        lastNewKSliding1 = nil; lastNewVSliding1 = nil
        lastNewKFull1 = nil;    lastNewVFull1 = nil
        lastNewKSliding2 = nil; lastNewVSliding2 = nil
        lastNewKFull2 = nil;    lastNewVFull2 = nil
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
        // EAGLE-3 hidden tap (present only in EAGLE-3 decode chunks).
        lastHiddenAtL8 = out2.featureValue(for: "hidden_at_L8")?.multiArrayValue
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
        let out3 = try chunk3.prediction(from: in3)
        let h3 = out3.featureValue(for: "hidden_states_out")!.multiArrayValue!
        // EAGLE-3 hidden tap (present only in EAGLE-3 decode chunks).
        lastHiddenAtL17 = out3.featureValue(for: "hidden_at_L17")?.multiArrayValue
        let tC3End = CFAbsoluteTimeGetCurrent()
        profileANEWait += (tC3End - tC3Wait0)
        profileC3 += (tC3End - tC3Start)

        // Chunk 4
        let tC4Start = CFAbsoluteTimeGetCurrent()
        var d4 = shared; d4["hidden_states"] = MLFeatureValue(multiArray: h3)
        let in4 = try MLDictionaryFeatureProvider(dictionary: d4)
        let tC4Wait0 = CFAbsoluteTimeGetCurrent()
        let out4 = try chunk4.prediction(from: in4)
        // EAGLE-3 hidden tap (present only in EAGLE-3 decode chunks).
        lastHiddenAtL34 = out4.featureValue(for: "hidden_at_L34")?.multiArrayValue
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

        let next = out4.featureValue(for: "token_id")!.multiArrayValue![0].intValue
        self.lastArgmaxAfterDecode = next
        return next
    }

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
        // 11c: capture per-T K/V slices for selective commit; do NOT write to persistent cache yet.
        lastNewKSliding1 = out1.featureValue(for: "new_K_sliding")?.multiArrayValue
        lastNewVSliding1 = out1.featureValue(for: "new_V_sliding")?.multiArrayValue
        lastNewKFull1    = out1.featureValue(for: "new_K_full")?.multiArrayValue
        lastNewVFull1    = out1.featureValue(for: "new_V_full")?.multiArrayValue

        // Verify chunk 2: emits per-T slices + within-verify extended kv13/kv14 for chunks 3/4.
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
        lastNewKSliding2 = out2.featureValue(for: "new_K_sliding")?.multiArrayValue
        lastNewVSliding2 = out2.featureValue(for: "new_V_sliding")?.multiArrayValue
        lastNewKFull2    = out2.featureValue(for: "new_K_full")?.multiArrayValue
        lastNewVFull2    = out2.featureValue(for: "new_V_full")?.multiArrayValue
        let kv13k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        lastKV13K = kv13k; lastKV13V = kv13v
        lastKV14K = kv14k; lastKV14V = kv14v
        // Remember verify inputs for commitAccepted's match check.
        lastVerifyInputTokens = tokens

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
        // 11c: capture per-T K/V slices; persistent cache untouched until commitAccepted.
        lastNewKSliding1 = out1.featureValue(for: "new_K_sliding")?.multiArrayValue
        lastNewVSliding1 = out1.featureValue(for: "new_V_sliding")?.multiArrayValue
        lastNewKFull1    = out1.featureValue(for: "new_K_full")?.multiArrayValue
        lastNewVFull1    = out1.featureValue(for: "new_V_full")?.multiArrayValue

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
        // EAGLE-3 hidden tap (present only in EAGLE-3 decode chunks; nil for
        // verify_chunk2 that predates the EAGLE-3 fusion build).
        lastHiddenAtL8 = out2.featureValue(for: "hidden_at_L8")?.multiArrayValue
        // 11c write-after-accept: capture per-T raw K/V slices; commitAccepted
        // selectively writes only the accepted prefix into persistent storage.
        // No copyBack(...) calls here — verify MUST NOT pollute persistent KV.
        lastNewKSliding2 = out2.featureValue(for: "new_K_sliding")?.multiArrayValue
        lastNewVSliding2 = out2.featureValue(for: "new_V_sliding")?.multiArrayValue
        lastNewKFull2    = out2.featureValue(for: "new_K_full")?.multiArrayValue
        lastNewVFull2    = out2.featureValue(for: "new_V_full")?.multiArrayValue
        let kv13k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        lastKV13K = kv13k; lastKV13V = kv13v
        lastKV14K = kv14k; lastKV14V = kv14v
        lastVerifyInputTokens = tokens

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

    /// Embed a single token id into a (1, 1, hidden) fp16 MLMultiArray, with
    /// the hidden-scale factor already applied (matches draft training
    /// convention). Exposed for `SpeculativeLoop`'s `tokenEmbed` closure.
    func embedToken(_ tokenID: Int32) throws -> MLMultiArray {
        try embedTokens.lookup(Int(tokenID),
                               shape: [1, 1, NSNumber(value: config.hiddenSize)])
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
        // cache shape: (slots, nkv, W, maxHd). Prefill output: (1, nkv, N, hd).
        let shape = cache.shape.map { $0.intValue }
        let nkv = shape[1]; let W = shape[2]; let maxHd = shape[3]
        let slotStride = nkv * W * maxHd
        let nkvStrideDst = W * maxHd
        let srcShape = srcArr.shape.map { $0.intValue }
        let N = srcShape[2]
        let nkvStrideSrc = N * hd
        let dst = cache.dataPointer.bindMemory(to: UInt16.self, capacity: cache.count)
        let s = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
        // SWA cache is right-aligned (see makeSlidingCausalMask): valid tokens
        // live in cache[W-valid..W-1]. When realLen > W, only the last W
        // source positions fit in the window.
        let writeCount = min(realLen, W)
        let sourceStart = realLen - writeCount
        let startCachePos = W - writeCount
        for head in 0..<nkv {
            for p in 0..<writeCount {
                let srcOff = head * nkvStrideSrc + (sourceStart + p) * hd
                let dstOff = slot * slotStride + head * nkvStrideDst + (startCachePos + p) * maxHd
                for j in 0..<hd { dst[dstOff + j] = s[srcOff + j] }
            }
        }
    }

    private func writeFullFromPrefill(src: MLFeatureProvider, name: String,
                                       cache: MLMultiArray, slot: Int,
                                       realLen: Int, hd: Int) throws {
        guard let srcArr = src.featureValue(for: name)?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        let shape = cache.shape.map { $0.intValue }
        let nkv = shape[1]; let ctx = shape[2]; let maxHd = shape[3]
        let slotStride = nkv * ctx * maxHd
        let nkvStrideDst = ctx * maxHd
        let srcShape = srcArr.shape.map { $0.intValue }
        let N = srcShape[2]
        let nkvStrideSrc = N * hd
        let dst = cache.dataPointer.bindMemory(to: UInt16.self, capacity: cache.count)
        let s = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
        for head in 0..<nkv {
            for p in 0..<realLen {
                let srcOff = head * nkvStrideSrc + p * hd
                let dstOff = slot * slotStride + head * nkvStrideDst + p * maxHd
                for j in 0..<hd { dst[dstOff + j] = s[srcOff + j] }
            }
        }
    }

    // KV slot mapping — config-driven for E2B (35 layers) and E4B (42 layers).
    //
    // Each prefill chunk emits one K{local}/V{local} pair per own-KV
    // non-producer layer in its [start, end) range (local index = position
    // within the chunk's non-producer list, matches Python's
    // `chunk_output_names` ordering). Producer layers emit kv13_* (sliding
    // producer) and kv14_* (full producer) aliases instead.
    //
    // Decode-side slot assignment follows `_layer_kv_map` in
    // gemma4_swa_chunks.py: within a chunk, sliding slots count up
    // per-sliding-layer, full slots count up per-full-layer. The Swift map
    // below must agree, so slots are computed by the same iteration order.
    private enum KVKind { case sliding, full }

    private struct KVSlotMap {
        let sliding: [(String, Int, MLMultiArray, Int)]
        let full: [(String, Int, MLMultiArray, Int)]
    }

    private func buildKVSlotMap(chunkIdx: Int) -> KVSlotMap {
        let hd = 256   // sliding head_dim (Gemma 4 both variants)
        let ghd = 512  // full head_dim
        let boundaries = config.chunkBoundaries
        let (start, end) = boundaries[chunkIdx - 1]
        let kS: MLMultiArray = (chunkIdx == 1) ? kSliding1 : kSliding2
        let vS: MLMultiArray = (chunkIdx == 1) ? vSliding1 : vSliding2
        let kF: MLMultiArray = (chunkIdx == 1) ? kFull1    : kFull2
        let vF: MLMultiArray = (chunkIdx == 1) ? vFull1    : vFull2

        var sliding: [(String, Int, MLMultiArray, Int)] = []
        var full: [(String, Int, MLMultiArray, Int)] = []
        var localNonProd = 0
        var slidingSlot = 0
        var fullSlot = 0
        let slidingProducer = config.kvSlidingProducer
        let fullProducer = config.kvFullProducer

        for layerIdx in start..<end {
            if config.isKvShared(layerIdx) { continue }
            let isFull = config.isFullAttention(layerIdx)
            // Resolve the output-name pair for this layer.
            let name: (String, String)
            if layerIdx == slidingProducer {
                name = ("kv13_k", "kv13_v")
            } else if layerIdx == fullProducer {
                name = ("kv14_k", "kv14_v")
            } else {
                name = ("K\(localNonProd)", "V\(localNonProd)")
                localNonProd += 1
            }
            if isFull {
                full.append((name.0, fullSlot, kF, ghd))
                full.append((name.1, fullSlot, vF, ghd))
                fullSlot += 1
            } else {
                sliding.append((name.0, slidingSlot, kS, hd))
                sliding.append((name.1, slidingSlot, vS, hd))
                slidingSlot += 1
            }
        }
        return KVSlotMap(sliding: sliding, full: full)
    }

    private func kvMapChunk1Sliding() -> [(String, Int, MLMultiArray, Int)] {
        buildKVSlotMap(chunkIdx: 1).sliding
    }
    private func kvMapChunk1Full() -> [(String, Int, MLMultiArray, Int)] {
        buildKVSlotMap(chunkIdx: 1).full
    }
    private func kvMapChunk2Sliding() -> [(String, Int, MLMultiArray, Int)] {
        buildKVSlotMap(chunkIdx: 2).sliding
    }
    private func kvMapChunk2Full() -> [(String, Int, MLMultiArray, Int)] {
        buildKVSlotMap(chunkIdx: 2).full
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

// MARK: - EAGLE-3 speculative decoding (Phase 2B)

extension ChunkedEngine {
    /// True when all four verify chunks are loaded and the most recent decode
    /// step captured EAGLE-3 hidden taps (L8/L17/L34).
    var canSpeculate: Bool {
        verifyChunk1 != nil && verifyChunk2 != nil
            && verifyChunk3 != nil && verifyChunk4 != nil
            && lastHiddenAtL8 != nil && lastHiddenAtL17 != nil && lastHiddenAtL34 != nil
    }

    // MARK: - Verify-side mask / RoPE / input builders

    /// Full-attention causal mask for T batched queries at positions
    /// [position, position + T - 1], last dim ctx + T.
    ///   mask[t, i] for i in 0..<ctx = 0 if i < position else -inf
    ///   mask[t, ctx+j] = 0 if j <= t else -inf
    func makeVerifyCausalMaskFull(position: Int, T: Int, ctx: Int) throws -> MLMultiArray {
        let lastDim = ctx + T
        let arr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: T), NSNumber(value: lastDim)], dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: T * lastDim)
        // Default: -inf
        for i in 0..<(T * lastDim) { p[i] = 0xFC00 }
        for t in 0..<T {
            let base = t * lastDim
            // Cache portion: 0..<position allowed.
            for i in 0..<min(position, ctx) { p[base + i] = 0 }
            // New-K portion: 0..t allowed within the trailing T slots.
            for j in 0...t { p[base + ctx + j] = 0 }
        }
        return arr
    }

    /// Sliding causal mask for T batched queries. Last dim W + T.
    ///   mask[t, i] for i in 0..<W: 0 iff cache slot i is within the
    ///     position-indexed window AND < position. When cache is partially
    ///     filled (position < W), slots [0, W-position-1] are invalid.
    ///   mask[t, W+j]: 0 iff j <= t.
    func makeVerifyCausalMaskSliding(position: Int, T: Int, W: Int) throws -> MLMultiArray {
        let lastDim = W + T
        let arr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: T), NSNumber(value: lastDim)], dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: T * lastDim)
        for i in 0..<(T * lastDim) { p[i] = 0xFC00 }
        // Valid cache length = min(position, W). Cache slot i represents
        // abs position (position - validCache + i) for i >= W - validCache.
        let validCache = min(position, W)
        let cacheStart = W - validCache
        for t in 0..<T {
            let base = t * lastDim
            // For query t at abs pos (position + t), sliding window admits
            // abs range [position + t - W + 1, position + t]. Map to cache
            // slot index. Cache slots [cacheStart .. W-1] are valid.
            // All valid cache slots satisfy the window condition when the
            // window is at least the cache length (always true here since
            // cache length = validCache ≤ W ≤ W + t).
            for i in cacheStart..<W { p[base + i] = 0 }
            // New positions within the trailing T slots.
            for j in 0...t { p[base + W + j] = 0 }
        }
        return arr
    }

    /// Stack embed(token) for each token into a single (1, T, hidden) fp16 array.
    func buildVerifyHidden(tokenIDs: [Int32]) throws -> MLMultiArray {
        let T = tokenIDs.count
        let h = config.hiddenSize
        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: T), NSNumber(value: h)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: T * h)
        for (i, tid) in tokenIDs.enumerated() {
            let e = try embedTokens.lookup(Int(tid), shape: [1, 1, NSNumber(value: h)])
            let src = e.dataPointer.bindMemory(to: UInt16.self, capacity: h)
            memcpy(dst.advanced(by: i * h), src, h * MemoryLayout<UInt16>.stride)
        }
        return arr
    }

    /// Stack per-layer-raw embedding for each token into (1, T, numLayers*pld) fp16.
    func buildVerifyPLR(tokenIDs: [Int32]) throws -> MLMultiArray {
        let T = tokenIDs.count
        let totalDim = config.numLayers * config.perLayerDim
        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: T), NSNumber(value: totalDim)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: T * totalDim)
        memset(dst, 0, T * totalDim * MemoryLayout<UInt16>.stride)
        for (i, tid) in tokenIDs.enumerated() {
            let raw = embedPerLayer.lookupRaw(Int(tid))
            memcpy(dst.advanced(by: i * totalDim), raw, totalDim * MemoryLayout<UInt16>.stride)
        }
        return arr
    }

    /// Read T consecutive rows from a RoPE table starting at `position`, stack
    /// into shape (1, 1, T, dim).
    func buildVerifyRoPE(table: Data?, position: Int, T: Int, dim: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: T), NSNumber(value: dim)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: T * dim)
        guard let table else {
            memset(dst, 0, T * dim * MemoryLayout<UInt16>.stride); return arr
        }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        table.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for t in 0..<T {
                let off = headerSize + (position + t) * rowBytes
                if off + rowBytes <= table.count {
                    memcpy(dst.advanced(by: t * dim), base.advanced(by: off), rowBytes)
                } else {
                    memset(dst.advanced(by: t * dim), 0, rowBytes)
                }
            }
        }
        return arr
    }
}

// MARK: - SpeculativeTarget conformance

extension ChunkedEngine: SpeculativeTarget {
    public func lastHiddenMulti(at layerIndices: [Int]) throws -> [MLMultiArray] {
        try layerIndices.map { idx in
            switch idx {
            case 8:
                guard let h = lastHiddenAtL8 else {
                    throw SpeculativeError.missingModel("lastHiddenAtL8 not captured yet")
                }
                return h
            case 17:
                guard let h = lastHiddenAtL17 else {
                    throw SpeculativeError.missingModel("lastHiddenAtL17 not captured yet")
                }
                return h
            case 34:
                guard let h = lastHiddenAtL34 else {
                    throw SpeculativeError.missingModel("lastHiddenAtL34 not captured yet")
                }
                return h
            default:
                throw SpeculativeError.missingModel("no hidden tap for layer \(idx)")
            }
        }
    }

    public func commitAccepted(_ tokens: [Int32]) throws {
        // 11c write-after-accept:
        //   For each accepted token at position currentPosition + t:
        //     - if it equals the verify input at slot t (i.e. that verify K/V is
        //       valid for this token), commit verify slice t into persistent cache.
        //     - else (correction, or beyond the verify range), run a T=1 predictStep
        //       to compute and write its K/V.
        //   Then advance currentPosition by tokens.count.
        let N = tokens.count
        if N == 0 { return }
        let P = currentPosition

        let inputs = lastVerifyInputTokens
        let K = inputs.count
        var M = 0
        while M < N && M < K && tokens[M] == inputs[M] {
            M += 1
        }

        if M > 0 {
            try commitKVSlices(count: M, basePosition: P)
        }

        // Tail tokens (correction / bonus / beyond verify range): compute K/V via T=1.
        for t in M..<N {
            _ = try predictStep(tokenID: Int(tokens[t]), position: P + t)
        }

        currentPosition = P + N
    }

    /// Commits the first `count` per-T K/V slices captured during the most
    /// recent verify call into the persistent IOSurface-backed caches.
    /// Sliding caches are shifted left by `count` and the new slices appended.
    /// Full caches are scattered into ctx-positions [basePosition .. basePosition+count-1].
    /// `count` MUST be ≤ K (the verify batch size) and ≤ verify slice tensors' second dim.
    private func commitKVSlices(count M: Int, basePosition P: Int) throws {
        guard M > 0,
              let nKs1 = lastNewKSliding1, let nVs1 = lastNewVSliding1,
              let nKf1 = lastNewKFull1,    let nVf1 = lastNewVFull1,
              let nKs2 = lastNewKSliding2, let nVs2 = lastNewVSliding2,
              let nKf2 = lastNewKFull2,    let nVf2 = lastNewVFull2 else {
            // No staged slices — caller must have called verifyCandidates first.
            // (PromptLookupLoop's stub-zero path is no longer supported; callers must
            //  pass actual accepted tokens.)
            return
        }
        let W = config.slidingWindow
        let ctx = config.contextLength
        let maxHd = 512
        let slidingHd = 256

        // Sliding writes (shift left by M, append M new rows per slot).
        commitSlidingSlots(buf: kSliding1, slices: nKs1, slotCount: 7,
                           M: M, W: W, slidingHd: slidingHd, maxHd: maxHd)
        commitSlidingSlots(buf: vSliding1, slices: nVs1, slotCount: 7,
                           M: M, W: W, slidingHd: slidingHd, maxHd: maxHd)
        commitSlidingSlots(buf: kSliding2, slices: nKs2, slotCount: 5,
                           M: M, W: W, slidingHd: slidingHd, maxHd: maxHd)
        commitSlidingSlots(buf: vSliding2, slices: nVs2, slotCount: 5,
                           M: M, W: W, slidingHd: slidingHd, maxHd: maxHd)

        // Full writes (scatter at positions P..P+M-1; full hd == maxHd, no padding).
        commitFullSlots(buf: kFull1, slices: nKf1, slotCount: 1,
                        M: M, P: P, ctx: ctx, fullHd: maxHd)
        commitFullSlots(buf: vFull1, slices: nVf1, slotCount: 1,
                        M: M, P: P, ctx: ctx, fullHd: maxHd)
        commitFullSlots(buf: kFull2, slices: nKf2, slotCount: 2,
                        M: M, P: P, ctx: ctx, fullHd: maxHd)
        commitFullSlots(buf: vFull2, slices: nVf2, slotCount: 2,
                        M: M, P: P, ctx: ctx, fullHd: maxHd)
    }

    private func commitSlidingSlots(buf: MLMultiArray, slices: MLMultiArray,
                                    slotCount: Int,
                                    M: Int, W: Int, slidingHd: Int, maxHd: Int) {
        let bufBase = buf.dataPointer.bindMemory(to: UInt16.self, capacity: buf.count)
        let srcBase = slices.dataPointer.bindMemory(to: UInt16.self, capacity: slices.count)
        let K = slices.shape[2].intValue
        precondition(M <= K, "commitSlidingSlots: M=\(M) > K=\(K)")
        let bytesPerSlidingRow = slidingHd * MemoryLayout<UInt16>.stride
        let bytesPadHigh       = (maxHd - slidingHd) * MemoryLayout<UInt16>.stride
        for slot in 0..<slotCount {
            let slotPtr = bufBase.advanced(by: slot * W * maxHd)
            // Shift left by M positions: dst rows [0..W-M-1] = src rows [M..W-1].
            memmove(slotPtr,
                    slotPtr.advanced(by: M * maxHd),
                    (W - M) * maxHd * MemoryLayout<UInt16>.stride)
            // Append M new rows at end, zero-padded high half (maxHd-slidingHd).
            let srcSlot = srcBase.advanced(by: slot * K * slidingHd)
            for n in 0..<M {
                let dstRow = slotPtr.advanced(by: (W - M + n) * maxHd)
                let srcRow = srcSlot.advanced(by: n * slidingHd)
                memcpy(dstRow, srcRow, bytesPerSlidingRow)
                if bytesPadHigh > 0 {
                    memset(dstRow.advanced(by: slidingHd), 0, bytesPadHigh)
                }
            }
        }
    }

    private func commitFullSlots(buf: MLMultiArray, slices: MLMultiArray,
                                 slotCount: Int,
                                 M: Int, P: Int, ctx: Int, fullHd: Int) {
        let bufBase = buf.dataPointer.bindMemory(to: UInt16.self, capacity: buf.count)
        let srcBase = slices.dataPointer.bindMemory(to: UInt16.self, capacity: slices.count)
        let K = slices.shape[2].intValue
        precondition(M <= K, "commitFullSlots: M=\(M) > K=\(K)")
        let bytesPerRow = fullHd * MemoryLayout<UInt16>.stride
        for slot in 0..<slotCount {
            let slotPtr = bufBase.advanced(by: slot * ctx * fullHd)
            let srcSlot = srcBase.advanced(by: slot * K * fullHd)
            for n in 0..<M {
                let pos = P + n
                precondition(pos < ctx, "commitFullSlots: pos \(pos) >= ctx \(ctx)")
                memcpy(slotPtr.advanced(by: pos * fullHd),
                       srcSlot.advanced(by: n * fullHd),
                       bytesPerRow)
            }
        }
    }

    public func verifyCandidates(_ candidates: [Int32], K: Int) throws -> [Int32] {
        guard let v1 = verifyChunk1, let v2 = verifyChunk2,
              let v3 = verifyChunk3, let v4 = verifyChunk4 else {
            throw SpeculativeError.missingModel("verify_chunk{1..4} not loaded")
        }
        precondition(candidates.count == K, "candidates.count must equal K")
        let ctx = config.contextLength
        let W = config.slidingWindow
        let P = currentPosition

        let hiddenIn = try buildVerifyHidden(tokenIDs: candidates)
        let plRaw = try buildVerifyPLR(tokenIDs: candidates)
        // 11c verify chunks use (1,1,K,ctx) for full and (1,1,K,W) for sliding
        // (the cache IS W — new K rows are appended by torch.cat(slot[K:], k_new)
        // inside the Python verify graph, so the query-side mask is just the
        // cache mask). Old (W+K) / (ctx+K) shapes rejected on iPhone.
        let maskFull = try makeVerifyCausalMask(startPos: P, K: K, length: ctx)
        let maskSliding = try makeVerifySlidingMask(startPos: P, K: K, W: W)
        // update_indicator: (1, 1, ctx, K) — one-hot column k at abs pos P+k.
        // Required by the 11c verify chunks for in-graph full-attn K scatter.
        let updateIndicator = try MLMultiArray(
            shape: [1, 1, NSNumber(value: ctx), NSNumber(value: K)],
            dataType: .float16)
        let indPtr = updateIndicator.dataPointer.bindMemory(to: UInt16.self, capacity: ctx * K)
        memset(indPtr, 0, ctx * K * MemoryLayout<UInt16>.stride)
        for k in 0..<K {
            let pos = P + k
            if pos < ctx {
                indPtr[pos * K + k] = 0x3C00  // fp16 1.0
            }
        }
        let cosS = try buildVerifyRoPE(table: cosSlidingTable, position: P, T: K, dim: 256)
        let sinS = try buildVerifyRoPE(table: sinSlidingTable, position: P, T: K, dim: 256)
        let cosF = try buildVerifyRoPE(table: cosFullTable, position: P, T: K, dim: 512)
        let sinF = try buildVerifyRoPE(table: sinFullTable, position: P, T: K, dim: 512)

        // Verify chunk 1 — same KV cache inputs as decode (read-only here).
        let out1 = try v1.prediction(from: MLDictionaryFeatureProvider(dictionary: [
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
        // 11c write-after-accept: capture per-T K/V slices; verify does NOT
        // write to persistent cache. Names match the 11c verify chunks:
        // new_K_sliding / new_V_sliding / new_K_full / new_V_full.
        lastNewKSliding1 = out1.featureValue(for: "new_K_sliding")?.multiArrayValue
        lastNewVSliding1 = out1.featureValue(for: "new_V_sliding")?.multiArrayValue
        lastNewKFull1    = out1.featureValue(for: "new_K_full")?.multiArrayValue
        lastNewVFull1    = out1.featureValue(for: "new_V_full")?.multiArrayValue

        // Verify chunk 2 — emits within-verify kv13/kv14 used by chunks 3/4
        // in this verify call. Under 11c these are NOT persisted to the
        // device KV cache — commitAccepted writes per-T slices instead.
        let out2 = try v2.prediction(from: MLDictionaryFeatureProvider(dictionary: [
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
        // 11c verify chunks emit kv13_k / kv14_k (no `_out` suffix).
        let kv13k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14v = out2.featureValue(for: "kv14_v")!.multiArrayValue!
        // 11c per-T K/V slices for L8-14 (persistent cache update happens in
        // commitAccepted, not here).
        lastNewKSliding2 = out2.featureValue(for: "new_K_sliding")?.multiArrayValue
        lastNewVSliding2 = out2.featureValue(for: "new_V_sliding")?.multiArrayValue
        lastNewKFull2    = out2.featureValue(for: "new_K_full")?.multiArrayValue
        lastNewVFull2    = out2.featureValue(for: "new_V_full")?.multiArrayValue
        // Stamp verify input tokens so commitAccepted can decide which
        // accepted positions have a valid verified slice (commit directly)
        // vs require a T=1 recompute (correction / bonus past verify range).
        lastVerifyInputTokens = candidates

        let shared: [String: MLFeatureValue] = [
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS), "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF), "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13k), "kv13_v": MLFeatureValue(multiArray: kv13v),
            "kv14_k": MLFeatureValue(multiArray: kv14k), "kv14_v": MLFeatureValue(multiArray: kv14v),
        ]

        var d3 = shared; d3["hidden_states"] = MLFeatureValue(multiArray: h2)
        let h3 = try v3.prediction(from: MLDictionaryFeatureProvider(dictionary: d3))
            .featureValue(for: "hidden_states_out")!.multiArrayValue!

        var d4 = shared; d4["hidden_states"] = MLFeatureValue(multiArray: h3)
        let out4 = try v4.prediction(from: MLDictionaryFeatureProvider(dictionary: d4))
        guard let tokenIdsArr = out4.featureValue(for: "token_ids")?.multiArrayValue else {
            throw SpeculativeError.verifyFailed("verify_chunk4 missing token_ids")
        }
        // token_ids shape is (T,) int32.
        let n = tokenIdsArr.count
        let p = tokenIdsArr.dataPointer.bindMemory(to: Int32.self, capacity: n)
        return (0..<n).map { p[$0] }
    }

    // No `verifyCandidatesTopN` override — the standalone verify_chunk4
    // exposes only `token_ids` + per-position `token_logits` (argmax logit
    // value), not the full (T, vocab) logits needed to rank alternatives.
    // Rebuilding chunk4 with a top-N output is required before tolerance>1
    // can do anything useful. For now, fall through to the protocol default
    // extension, which wraps argmax as a single-entry top list — so
    // LLM_EAGLE3_TOLERANCE>1 degrades safely to strict argmax match.
}
