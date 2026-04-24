import CoreML
import Foundation
import Tokenizers

/// On-device LLM inference using CoreML with ANE optimization.
///
/// Supports both monolithic models (single .mlpackage) and chunked SWA models
/// (Gemma 4 E2B with 4 decode + 4 prefill chunks + external embeddings).
///
/// ```swift
/// let llm = try await CoreMLLLM.load(from: modelDirectory)
///
/// // Simple single-turn
/// let answer = try await llm.generate("What is the capital of France?")
///
/// // Streaming
/// for await token in try await llm.stream("Tell me a story") {
///     print(token, terminator: "")
/// }
///
/// // Multi-turn conversation
/// let messages: [CoreMLLLM.Message] = [
///     .init(role: .user, content: "Hi!"),
///     .init(role: .assistant, content: "Hello!"),
///     .init(role: .user, content: "What is 2+2?"),
/// ]
/// for await token in try await llm.stream(messages) {
///     print(token, terminator: "")
/// }
/// ```
public final class CoreMLLLM: @unchecked Sendable {
    private let tokenizer: any Tokenizer
    private var config: ModelConfig

    // Engine: exactly one of these is non-nil.
    private var chunkedEngine: ChunkedEngine?
    private var monolithicModel: MLModel?
    private var monolithicState: MLState?

    // EAGLE-3 speculative decoding (optional — nil if the fusion/draft/verify
    // assets aren't in the model directory).
    private var speculativeLoop: SpeculativeLoop?

    /// Test-only accessor for the chunked engine. `internal` so tests in the
    /// same module (via @testable import) can exercise low-level paths like
    /// commitAccepted / KV snapshot. Do NOT use from production code.
    internal var _testChunkedEngine: ChunkedEngine? { chunkedEngine }

    // Vision (lazy loaded to save memory)
    private var visionModel: MLModel?
    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?
    /// True when visionModelURL points to a `vision.ane.*` sibling — the
    /// square 48×48 fixed-grid encoder built by
    /// `convert_gemma4_multimodal.py --vision-ane`. Changes the
    /// preprocessing path (forced 768×768 resize, fp16 pixel values,
    /// no padding) and the ANE-friendly compute unit selection.
    private var visionUsesANEBuild: Bool = false

    // Optional video-grade vision encoder (max_soft_tokens=70 → 64
    // tokens/frame). When present, replaces the Phase 1 Swift 2×2 pool.
    private var videoVisionModel: MLModel?
    private var videoVisionModelURL: URL?
    private var videoVisionConfig: MLModelConfiguration?

    // Audio (lazy loaded to save memory)
    private var audioModel: MLModel?
    private var audioModelURL: URL?
    private var audioConfig: MLModelConfiguration?
    private var melFilterbank: [Float]?
    private var audioProjection: AudioProcessor.ProjectionWeights?
    private var audioMelFloor: Float = 0.001
    public private(set) var audioMelFrames: Int = 200
    public private(set) var audioNumTokens: Int = 50

    // Multi-turn: cache image features across turns
    private var cachedImageFeatures: MLMultiArray?

    // MTP speculative decoding
    private var mtpEngine: MtpSpeculativeEngine?
    /// Toggle MTP speculation on/off for benchmarking.
    public var mtpEnabled: Bool = true

    // Cross-vocabulary (Qwen -> Gemma) speculative decoding — Route B
    private var crossVocabEngine: CrossVocabSpeculativeEngine?
    /// Underlying Qwen drafter, also re-used by `drafterUnion`. Held
    /// separately so the union can drive it without going through the
    /// cross-vocab-only engine wrapper.
    private var crossVocabDrafter: CrossVocabDraft?
    /// Toggle cross-vocab speculation on/off. Defaults to OFF on 2026-04-15
    /// after on-device testing showed the Qwen drafter runs ~10× slower
    /// than Mac projection, producing 1.8 tok/s with degraded output on
    /// iPhone 17 Pro. Opt-in until drafter cost + bootstrap-TTFT + K=3↔K=1
    /// numerical alignment (roadmap 11c) are investigated. MTP preserves
    /// priority when loaded.
    public var crossVocabEnabled: Bool = false

    // Phase B Task 1 — union of cross-vocab + prompt-lookup{n=2, n=3}
    private var drafterUnion: DrafterUnion?
    /// Opt-in for Phase B union. Default off until iPhone baseline check
    /// confirms no regression (per merge discipline in docs/HANDOFF.md).
    /// Takes precedence over crossVocabEnabled when both are true.
    public var drafterUnionEnabled: Bool = false

    // Linear LookAhead / Jacobi (drafter-free, docs/LOOKAHEAD_PROBE_HANDOFF.md).
    // Drafts K-1 tokens per cycle via n-gram lookup + Jacobi warm-start,
    // verifies in one ANE dispatch. Opt-in; defaults off until iPhone
    // baseline check confirms no regression.
    private var lookaheadEngine: LookaheadEngine?
    public var lookaheadEnabled: Bool = false
    public var lookaheadAcceptanceRate: Double {
        lookaheadEngine?.acceptanceRate ?? 0
    }
    public var lookaheadTokensPerCycle: Double {
        lookaheadEngine?.tokensPerCycle ?? 0
    }

    // Generation metrics
    public private(set) var tokensPerSecond: Double = 0
    public var mtpAcceptanceRate: Double { mtpEngine?.acceptanceRate ?? 0 }
    public var mtpTokensPerRound: Double { mtpEngine?.tokensPerRound ?? 0 }
    public var crossVocabAcceptanceRate: Double { crossVocabEngine?.acceptanceRate ?? 0 }
    public var crossVocabTokensPerCycle: Double { crossVocabEngine?.tokensPerCycle ?? 0 }
    public var drafterUnionAcceptanceRate: Double { drafterUnion?.acceptanceRate ?? 0 }
    public var drafterUnionTokensPerCycle: Double { drafterUnion?.tokensPerCycle ?? 0 }
    public var drafterUnionPicks: [String: Int] {
        guard let u = drafterUnion else { return [:] }
        var out: [String: Int] = [:]
        for (k, v) in u.picks { out[k.rawValue] = v }
        return out
    }
    /// Hard-disable the cross-vocab source inside the union. Used by the
    /// Mac-side bit-exact verifier to keep CV out of the picture when the
    /// staging Qwen has the wrong context length (gotcha #2 in
    /// docs/SESSION_STATE.md). On iPhone leave this `false`.
    public func setDrafterUnionCrossVocabDisabled(_ disabled: Bool) {
        drafterUnion?.crossVocabDisabled = disabled
    }
    /// Override the union's PLD rolling-accept gate. Setting it above 1.0
    /// hard-disables PLD for the whole generation — useful for narrowing
    /// down whether divergence vs serial decode comes from PLD-induced
    /// verify-chunk drift or from union bookkeeping.
    public func setDrafterUnionPLDThreshold(_ value: Double) {
        drafterUnion?.pldThreshold = value
    }

    // Token-ID recording for offline accept-rate benches. These are populated
    // from the last `generate` / `stream` call and live until the next one.
    // See `docs/MAC_FIRST_EXECUTION_PLAN.md` §A1 for usage.
    public private(set) var lastPromptTokenIDs: [Int32] = []
    public private(set) var lastEmittedTokenIDs: [Int32] = []

    /// Expose the tokenizer so a harness can re-encode / decode arbitrary
    /// text without duplicating the swift-transformers wiring.
    public var tokenizerRef: any Tokenizer { tokenizer }

    // MARK: - Public Types

    /// A message in a multi-turn conversation.
    public struct Message: Sendable {
        public enum Role: String, Sendable {
            case user, assistant, system
        }
        public let role: Role
        public let content: String

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }
    }

    private init(config: ModelConfig, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    // MARK: - Public API

    /// Load a model from a local directory.
    ///
    /// Auto-detects layout:
    /// - If `chunk1.mlmodelc` exists → chunked SWA engine (Gemma 4 E2B)
    /// - Otherwise → monolithic model (`model.mlpackage` / `model.mlmodelc`)
    ///
    /// - Parameters:
    ///   - directory: Folder containing model files, embeddings, config
    ///   - computeUnits: CoreML compute units (default: `.cpuAndNeuralEngine`)
    ///   - onProgress: Optional callback for loading status updates
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        onProgress: ((String) -> Void)? = nil
    ) async throws -> CoreMLLLM {
        onProgress?("Reading config...")
        let config = try ModelConfig.load(from: directory)

        // Tokenizer
        onProgress?("Loading tokenizer...")
        let tokDir = directory.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)

        let llm = CoreMLLLM(config: config, tokenizer: tokenizer)

        // Opt-in gates for the speculative stack. Every component adds
        // resident ANE memory; the baseline (no env set) loads only the
        // T=1 decode chunks so footprint matches pre-EAGLE-3 builds.
        //
        //   LLM_EAGLE3_ENABLE=1     — load eagle3_draft + eagle3_fusion
        //                              (234 MB ANE-resident) AND verify
        //                              chunks (600 MB); speculative burst
        //                              path active.
        //   SPECULATIVE_PROFILE=1   — load cross-vocab Qwen drafter
        //                              (301 MB) + verify chunks (600 MB).
        //                              Validated 11c cv bench path.
        //   (neither)                 — pure T=1 decode, no aux models.
        //
        // EAGLE-3 measurement (Gemma 4 E2B, 500k training positions):
        // val accept 22%, iPhone throughput +5-8% over T=1. Default
        // opt-out keeps users who don't need speculation from paying
        // the memory cost.
        let eagle3Enabled =
            ProcessInfo.processInfo.environment["LLM_EAGLE3_ENABLE"] == "1"
        let specProfile =
            ProcessInfo.processInfo.environment["SPECULATIVE_PROFILE"] != nil
        let loadVerify = eagle3Enabled || specProfile
        let loadCV = specProfile
        if eagle3Enabled || specProfile {
            print("[Load] speculative gates: "
                  + "eagle3=\(eagle3Enabled) cv=\(loadCV) verify=\(loadVerify)")
        }

        // Auto-detect: chunked or monolithic
        let isChunked = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("chunk1.mlmodelc").path)
            || FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("chunk1.mlpackage").path)

        if isChunked {
            onProgress?("Loading chunks (first run = ANE compile, can take 1-2 min)...")
            llm.chunkedEngine = try await ChunkedEngine.load(
                from: directory, config: config, computeUnits: computeUnits)
            // MLComputePlan silent-fallback audit (§G2) — runs only when
            // COMPUTE_PLAN_AUDIT env var or UserDefaults key is set.
            await ComputePlanAudit.run(modelDirectory: directory,
                                       computeUnits: computeUnits)

            // Optional disk-backed prefix cache (LLM_PREFIX_CACHE=1).
            // Cache directory is namespaced by model directory's last
            // path component (e.g. "gemma4-e2b") so multiple models
            // don't collide. Capacity defaults to 256 MB which fits
            // 3-7 snapshots at 2K context (more at 8K = fewer entries).
            if let engine = llm.chunkedEngine,
               ProcessInfo.processInfo.environment["LLM_PREFIX_CACHE"] == "1" {
                do {
                    let cachesDir = try FileManager.default.url(
                        for: .cachesDirectory, in: .userDomainMask,
                        appropriateFor: nil, create: true)
                    let modelTag = directory.lastPathComponent.isEmpty
                        ? "default" : directory.lastPathComponent
                    let cacheDir = cachesDir
                        .appendingPathComponent("coreml-llm-prefix-cache")
                        .appendingPathComponent(modelTag)
                    let capStr = ProcessInfo.processInfo
                        .environment["LLM_PREFIX_CACHE_MB"]
                    let capMB = Int(capStr ?? "") ?? 256
                    engine.prefixCache = try PrefixCache(
                        directory: cacheDir,
                        capacityBytes: capMB * 1024 * 1024)
                    print("[PrefixCache] enabled at \(cacheDir.path) " +
                          "cap=\(capMB)MB existing=\(engine.prefixCache!.totalBytes()/(1024*1024))MB")
                } catch {
                    print("[PrefixCache] init failed: \(error)")
                }
            }

            // EAGLE-3 assets are optional. All three must be present or
            // speculative decoding stays off. Gemma 4 fusion layers are
            // [8, 17, 34] per eagle3_config.json.
            func findAsset(_ name: String) -> URL? {
                let compiled = directory.appendingPathComponent("\(name).mlmodelc")
                if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
                let pkg = directory.appendingPathComponent("\(name).mlpackage")
                if FileManager.default.fileExists(atPath: pkg.path) { return pkg }
                return nil
            }
            // EAGLE-3 speculative loop. Verify chunks are already loaded by
            // ChunkedEngine.load() (either as multi-function verify_qK or as
            // standalone verify_chunk*.mlmodelc). We just need draft + fusion
            // mlpackages here to wire up the SpeculativeLoop orchestrator.
            //
            // Opt-in: gated by LLM_EAGLE3_ENABLE=1. On iPhone 17 Pro with
            // Gemma 4 E2B the draft/fusion add ~234 MB ANE-resident and
            // yield only ~+5-8% throughput at current draft quality (val
            // 22.4% per-step, per-token compounded ~14%). Not worth the
            // memory by default.
            if eagle3Enabled,
               let fusionURL = findAsset("eagle3_fusion"),
               let draftURL = findAsset("eagle3_draft"),
               (llm.chunkedEngine?.hasVerify ?? false) {
                do {
                    onProgress?("Loading EAGLE-3 speculative (fusion + draft)...")
                    llm.speculativeLoop = try SpeculativeLoop(
                        fusionURL: fusionURL, draftURL: draftURL,
                        K: 3, fusionLayers: [8, 17, 34],
                        embedScale: config.embedScale)
                    print("[Load] EAGLE-3 speculative ready (K=3)")
                } catch {
                    print("[Load] EAGLE-3 unavailable: \(error). Falling back to T=1 decode.")
                    llm.speculativeLoop = nil
                }
            }
        } else {
            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = computeUnits
            let modelURL = directory.appendingPathComponent("model.mlmodelc")
            if FileManager.default.fileExists(atPath: modelURL.path) {
                onProgress?("Loading model...")
                llm.monolithicModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            } else {
                onProgress?("Compiling model...")
                let pkgURL = directory.appendingPathComponent("model.mlpackage")
                let compiled = try await MLModel.compileModel(at: pkgURL)
                llm.monolithicModel = try MLModel(contentsOf: compiled, configuration: mlConfig)
            }
            llm.monolithicState = llm.monolithicModel?.makeState()
        }

        // MTP drafter (optional — enables speculative decoding)
        let mtpCompiled = directory.appendingPathComponent("mtp_drafter.mlmodelc")
        let mtpPkg = directory.appendingPathComponent("mtp_drafter.mlpackage")
        let mtpURL: URL? = FileManager.default.fileExists(atPath: mtpCompiled.path) ? mtpCompiled
            : FileManager.default.fileExists(atPath: mtpPkg.path) ? mtpPkg : nil
        if let mtpURL, let engine = llm.chunkedEngine, engine.hasVerify {
            onProgress?("Loading MTP drafter...")
            do {
                let drafterSource = try MtpDraftSource(
                    modelURL: mtpURL, K: engine.verifyK)
                llm.mtpEngine = MtpSpeculativeEngine(
                    engine: engine, drafter: drafterSource)
                print("[MTP] Drafter loaded (K=\(engine.verifyK))")
            } catch {
                print("[MTP] Failed to load drafter: \(error)")
            }
        }

        // Cross-vocabulary drafter (Route B): Qwen 2.5 0.5B monolithic +
        // vocab map. Looks for `cross_vocab/qwen_drafter.mlmodelc` (or
        // `.mlpackage`) plus `cross_vocab/qwen_gemma_vocab.bin` under the
        // model directory. Skipped if absent OR if SPECULATIVE_PROFILE
        // is not set (gate prevents paying 301 MB ANE-resident cost
        // when speculation is off).
        let cvDir = directory.appendingPathComponent("cross_vocab")
        let cvCompiled = cvDir.appendingPathComponent("qwen_drafter.mlmodelc")
        let cvPkg = cvDir.appendingPathComponent("qwen_drafter.mlpackage")
        let cvMapURL = cvDir.appendingPathComponent("qwen_gemma_vocab.bin")
        let cvModelURL: URL? = FileManager.default.fileExists(atPath: cvCompiled.path) ? cvCompiled
            : FileManager.default.fileExists(atPath: cvPkg.path) ? cvPkg : nil
        if loadCV,
           let cvModelURL,
           FileManager.default.fileExists(atPath: cvMapURL.path),
           let engine = llm.chunkedEngine,
           engine.hasVerify {
            onProgress?("Loading cross-vocab drafter (Qwen 2.5 0.5B)...")
            do {
                let map = try CrossVocabMap(url: cvMapURL)
                // Qwen 2.5 0.5B supports 32K natively; cap at target's
                // context length so the two stay in lockstep.
                let drafter = try CrossVocabDraft(
                    modelURL: cvModelURL,
                    vocabMap: map,
                    K: engine.verifyK,
                    contextLength: config.contextLength,
                    computeUnits: .cpuAndGPU)
                llm.crossVocabEngine = CrossVocabSpeculativeEngine(
                    engine: engine, drafter: drafter)
                llm.crossVocabDrafter = drafter
                llm.drafterUnion = DrafterUnion(
                    engine: engine, crossVocab: drafter, K: engine.verifyK)
                print("[CrossVocab] Drafter loaded (K=\(engine.verifyK), "
                      + "coverage q->g=\(String(format: "%.1f", Double(map.qwenToGemma.filter { $0 >= 0 }.count) / Double(map.qwenVocabSize) * 100))%)")
            } catch {
                print("[CrossVocab] Failed to load drafter: \(error)")
            }
            // MLComputePlan audit on the drafter (Phase B Task 2). Runs
            // only when COMPUTE_PLAN_AUDIT is set, so production load
            // sees no extra cost. Tells us GPU placement vs CPU fallback
            // when investigating the iPhone perf regression.
            await ComputePlanAudit.runDrafter(modelDirectory: directory)
        }

        // PLD-only union: still useful when cross-vocab drafter assets are
        // absent (typical for stripped iPhone bundles). Phase B's union
        // collapses to prompt-lookup{n=2,n=3} which has near-zero cost.
        if llm.drafterUnion == nil,
           let engine = llm.chunkedEngine,
           engine.hasVerify {
            llm.drafterUnion = DrafterUnion(
                engine: engine, crossVocab: nil, K: engine.verifyK)
            print("[DrafterUnion] PLD-only mode (cross-vocab drafter not loaded)")
        }

        // LookaheadEngine: zero-weight, drafter-free speculation. Always
        // constructable when verify chunks are present — the actual
        // routing is gated by `lookaheadEnabled` or LLM_LOOKAHEAD_ENABLE.
        if let engine = llm.chunkedEngine, engine.hasVerify {
            llm.lookaheadEngine = LookaheadEngine(engine: engine)
            if ProcessInfo.processInfo.environment["LLM_LOOKAHEAD_ENABLE"] == "1" {
                llm.lookaheadEnabled = true
                print("[Lookahead] enabled via LLM_LOOKAHEAD_ENABLE=1 (K=\(engine.verifyK))")
            } else {
                print("[Lookahead] engine ready (K=\(engine.verifyK)); opt-in via lookaheadEnabled=true or LLM_LOOKAHEAD_ENABLE=1")
            }
        }

        // Vision model (optional, lazy loaded on first image).
        //
        // Default is the legacy variable-grid GPU build
        // (`vision.mlmodelc`). iPhone 17 Pro A19 A/B (2026-04-25)
        // measured predict 205 ms GPU vs 584 ms ANE (steady-state,
        // both prewarmed) — the Mac 8× ANE win does not reproduce on
        // A19, and the GPU path preserves aspect ratio while the ANE
        // build force-squashes to 48×48.
        //
        // LLM_VISION_FORCE_ANE=1 opts into the `vision.ane.*` build
        // for benchmarking / future A19 firmware retest. The `.v2.`
        // suffix still wins over the unsuffixed ANE file so a newer
        // converted copy can be dropped on-device without reinstall.
        let forceANE = ProcessInfo.processInfo.environment["LLM_VISION_FORCE_ANE"] == "1"
        let visionANEv2Compiled = directory.appendingPathComponent("vision.ane.v2.mlmodelc")
        let visionANECompiled = directory.appendingPathComponent("vision.ane.mlmodelc")
        let visionANEPkg = directory.appendingPathComponent("vision.ane.mlpackage")
        let visionCompiled = directory.appendingPathComponent("vision.mlmodelc")
        let visionPkg = directory.appendingPathComponent("vision.mlpackage")
        if forceANE, FileManager.default.fileExists(atPath: visionANEv2Compiled.path) {
            llm.visionModelURL = visionANEv2Compiled
            llm.visionUsesANEBuild = true
        } else if forceANE, FileManager.default.fileExists(atPath: visionANECompiled.path) {
            llm.visionModelURL = visionANECompiled
            llm.visionUsesANEBuild = true
        } else if forceANE, FileManager.default.fileExists(atPath: visionANEPkg.path) {
            llm.visionModelURL = visionANEPkg
            llm.visionUsesANEBuild = true
        } else if FileManager.default.fileExists(atPath: visionCompiled.path) {
            llm.visionModelURL = visionCompiled
        } else if FileManager.default.fileExists(atPath: visionPkg.path) {
            llm.visionModelURL = visionPkg
        } else if FileManager.default.fileExists(atPath: visionANEv2Compiled.path) {
            // No legacy file present — fall back to any ANE sibling so
            // vision still works on partially-deployed bundles.
            llm.visionModelURL = visionANEv2Compiled
            llm.visionUsesANEBuild = true
        } else if FileManager.default.fileExists(atPath: visionANECompiled.path) {
            llm.visionModelURL = visionANECompiled
            llm.visionUsesANEBuild = true
        } else if FileManager.default.fileExists(atPath: visionANEPkg.path) {
            llm.visionModelURL = visionANEPkg
            llm.visionUsesANEBuild = true
        }
        if llm.visionModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = llm.visionUsesANEBuild ? .cpuAndNeuralEngine : .cpuAndGPU
            llm.visionConfig = cfg
            let tag = llm.visionUsesANEBuild ? "ANE" : "GPU"
            let name = llm.visionModelURL!.lastPathComponent
            let forced = forceANE ? " (LLM_VISION_FORCE_ANE=1)" : ""
            print("[Vision] selected \(name) → \(tag)\(forced)")
            await ComputePlanAudit.runVision(modelURL: llm.visionModelURL!,
                                             computeUnits: cfg.computeUnits,
                                             backendTag: tag)
        }

        // Optional video-grade vision encoder. Ships alongside
        // vision.mlpackage when the HF release was built with
        // `convert_gemma4_multimodal.py --video-vision`.
        let videoVisionCompiled = directory.appendingPathComponent("vision_video.mlmodelc")
        let videoVisionPkg = directory.appendingPathComponent("vision_video.mlpackage")
        if FileManager.default.fileExists(atPath: videoVisionCompiled.path) {
            llm.videoVisionModelURL = videoVisionCompiled
        } else if FileManager.default.fileExists(atPath: videoVisionPkg.path) {
            llm.videoVisionModelURL = videoVisionPkg
        }
        if llm.videoVisionModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            llm.videoVisionConfig = cfg
        }

        // Audio model (optional, lazy loaded on first audio)
        let audioCompiled = directory.appendingPathComponent("audio.mlmodelc")
        let audioPkg = directory.appendingPathComponent("audio.mlpackage")
        if FileManager.default.fileExists(atPath: audioCompiled.path) {
            llm.audioModelURL = audioCompiled
        } else if FileManager.default.fileExists(atPath: audioPkg.path) {
            llm.audioModelURL = audioPkg
        }
        if llm.audioModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            llm.audioConfig = cfg

            let melURL = directory.appendingPathComponent("mel_filterbank.bin")
            if FileManager.default.fileExists(atPath: melURL.path) {
                llm.melFilterbank = try? AudioProcessor.loadMelFilterbank(from: melURL)
            }
            // Projection weights (Swift-side float32 computation)
            let projURL = directory.appendingPathComponent("output_proj_weight.npy")
            if FileManager.default.fileExists(atPath: projURL.path) {
                llm.audioProjection = try? AudioProcessor.ProjectionWeights.load(from: directory)
            }

            let audioConfURL = directory.appendingPathComponent("audio_config.json")
            if let data = try? Data(contentsOf: audioConfURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                llm.audioMelFrames = json["mel_frames"] as? Int ?? 200
                llm.audioNumTokens = json["num_tokens"] as? Int ?? 50
                // mel_floor / log_offset: HF stores it under either key; fall
                // back to the default (0.001) if absent. Match the value the
                // encoder was trained with or features drift.
                if let mf = json["log_offset"] as? Double {
                    llm.audioMelFloor = Float(mf)
                } else if let mf = json["mel_floor"] as? Double {
                    llm.audioMelFloor = Float(mf)
                }
            }
        }

        // Final prewarm — re-warm decode + verify paths after all auxiliary
        // models (EAGLE-3 draft/fusion, cross-vocab Qwen, etc.) have loaded.
        // Those loads can evict the early ANE cache set up during
        // ChunkedEngine.load, leaving the first user decode ~50ms slower
        // per step and the first spec burst ~100ms slower.
        if let engine = llm.chunkedEngine {
            do {
                try engine.finalPrewarm()
            } catch {
                print("[Load] final prewarm skipped: \(error)")
            }
        }

        // Warm the vision encoder in the background — first-call ANE
        // compile / weight paging costs ~550 ms on iPhone 17 Pro for
        // the still-image ANE build, which lands right on TTFT for the
        // first image prompt. A dry forward with zero pixel values and
        // a full 48×48 grid compiles the ANE graph and pages the
        // 326 MB of weights in so the first real image call drops to
        // steady-state (~200-300 ms).
        //
        // The legacy GPU build is now the default; prewarm it with a
        // dummy 48×48 (multiple-of-48) grid so first-call compile
        // doesn't land on TTFT for the first real image. GPU load is
        // cheap (~1 s on iPhone) but first predict compiles the graph
        // (~30 s observed) — running it on a utility queue hides the
        // cost behind user typing time.
        if let url = llm.visionModelURL,
           let cfg = llm.visionConfig,
           !llm.visionUsesANEBuild {
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                let m = try MLModel(contentsOf: url, configuration: cfg)
                llm.visionModel = m
                let dt = CFAbsoluteTimeGetCurrent() - t0
                print("[Load] vision GPU load done in \(String(format: "%.1f", dt))s")
                DispatchQueue.global(qos: .utility).async {
                    do {
                        let tw = CFAbsoluteTimeGetCurrent()
                        let pd = 16 * 16 * 3
                        let total = 2520
                        let pv = try MLMultiArray(
                            shape: [1, NSNumber(value: total), NSNumber(value: pd)],
                            dataType: .float32)
                        let pid = try MLMultiArray(
                            shape: [1, NSNumber(value: total), 2], dataType: .int32)
                        let pidp = pid.dataPointer.bindMemory(
                            to: Int32.self, capacity: total * 2)
                        var k = 0
                        for py in 0..<48 {
                            for px in 0..<48 {
                                pidp[k * 2] = Int32(px)
                                pidp[k * 2 + 1] = Int32(py)
                                k += 1
                            }
                        }
                        for i in (48 * 48)..<total {
                            pidp[i * 2] = -1
                            pidp[i * 2 + 1] = -1
                        }
                        let input = try MLDictionaryFeatureProvider(dictionary: [
                            "pixel_values": MLFeatureValue(multiArray: pv),
                            "pixel_position_ids": MLFeatureValue(multiArray: pid),
                        ])
                        _ = try m.prediction(from: input)
                        let dw = CFAbsoluteTimeGetCurrent() - tw
                        print("[Load] vision GPU prewarm predict in \(String(format: "%.1f", dw))s")
                    } catch {
                        print("[Load] vision GPU prewarm predict skipped: \(error)")
                    }
                }
            } catch {
                print("[Load] vision GPU load skipped: \(error)")
            }
        }
        if let url = llm.visionModelURL,
           let cfg = llm.visionConfig,
           llm.visionUsesANEBuild {
            // Load the vision MLModel synchronously *and* attach it
            // to `llm.visionModel` before the first user prompt. Then
            // kick off a dry forward on a utility queue so the ANE
            // compile + weight paging happen in the background. The
            // key property we need vs the previous DispatchQueue
            // approach: the FIRST user predict must hit the same
            // MLModel instance the prewarm warmed, otherwise we eat
            // the 500 ms compile cost again. Attaching synchronously
            // on the caller's thread guarantees that.
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                let m = try MLModel(contentsOf: url, configuration: cfg)
                llm.visionModel = m
                let dt = CFAbsoluteTimeGetCurrent() - t0
                print("[Load] vision ANE load done in \(String(format: "%.1f", dt))s")
                DispatchQueue.global(qos: .utility).async {
                    do {
                        let tw = CFAbsoluteTimeGetCurrent()
                        let pd = 16 * 16 * 3
                        let total = 48 * 48
                        let pv = try MLMultiArray(
                            shape: [1, NSNumber(value: total), NSNumber(value: pd)],
                            dataType: .float16)
                        let pid = try MLMultiArray(
                            shape: [1, NSNumber(value: total), 2], dataType: .int32)
                        let pidp = pid.dataPointer.bindMemory(
                            to: Int32.self, capacity: total * 2)
                        var k = 0
                        for py in 0..<48 {
                            for px in 0..<48 {
                                pidp[k * 2] = Int32(px)
                                pidp[k * 2 + 1] = Int32(py)
                                k += 1
                            }
                        }
                        let input = try MLDictionaryFeatureProvider(dictionary: [
                            "pixel_values": MLFeatureValue(multiArray: pv),
                            "pixel_position_ids": MLFeatureValue(multiArray: pid),
                        ])
                        _ = try m.prediction(from: input)
                        let dw = CFAbsoluteTimeGetCurrent() - tw
                        print("[Load] vision ANE prewarm predict in \(String(format: "%.1f", dw))s")
                    } catch {
                        print("[Load] vision ANE prewarm predict skipped: \(error)")
                    }
                }
            } catch {
                print("[Load] vision ANE load skipped: \(error)")
            }
        }

        // Also warm the EAGLE-3 draft + fusion mlpackages if loaded —
        // one dry forward through each so the first user burst doesn't
        // eat the ANE compile cost for these two graphs.
        if let sl = llm.speculativeLoop, let engine = llm.chunkedEngine {
            do {
                _ = try sl.drawBurst(
                    target: engine,
                    tTokNext: 0,
                    tokenEmbed: { try engine.embedToken($0) })
                engine.reset()
                print("[Load] EAGLE-3 draft+fusion prewarm done")
            } catch {
                print("[Load] EAGLE-3 prewarm skipped: \(error)")
            }
        }

        onProgress?("Ready")
        return llm
    }

    /// Download (if needed) and load a model in one call.
    ///
    /// ```swift
    /// let llm = try await CoreMLLLM.load(model: .gemma4e2b) { print($0) }
    /// ```
    ///
    /// If the model is already downloaded, skips straight to loading.
    public static func load(
        model: ModelDownloader.ModelInfo,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        onProgress: ((String) -> Void)? = nil
    ) async throws -> CoreMLLLM {
        let downloader = ModelDownloader.shared
        let modelURL: URL
        if let existing = downloader.localModelURL(for: model) {
            modelURL = existing
        } else {
            onProgress?("Downloading \(model.name)...")
            modelURL = try await downloader.download(model)
        }
        let directory = modelURL.deletingLastPathComponent()
        return try await load(from: directory, computeUnits: computeUnits,
                               onProgress: onProgress)
    }

    /// Whether EAGLE-3 speculative decoding is loaded and active for this model.
    public var supportsSpeculative: Bool { speculativeLoop != nil }

    /// Rolling acceptance rate observed on recent speculative bursts. 1.0 before
    /// any burst has run; decays toward 0 if the draft consistently mispredicts.
    public var speculativeAcceptance: Double {
        speculativeLoop?.rollingAcceptance ?? 0
    }

    /// Whether this model supports image input.
    public var supportsVision: Bool { visionModelURL != nil }

    /// Whether this model supports audio input.
    ///
    /// The projection (.npy files) is optional — newer audio.mlmodelc builds
    /// fuse the projection into the graph, so only the encoder + mel
    /// filterbank are required. Older 1024-dim-output encoders still need
    /// `audioProjection`; `AudioProcessor.process` decides at runtime.
    public var supportsAudio: Bool { audioModelURL != nil && melFilterbank != nil }

    /// Maximum audio duration in seconds that the model accepts.
    public var maxAudioDuration: TimeInterval {
        // mel_frames ≈ audio_samples / (hop_length * 4) → seconds = mel_frames * hop_length * 4 / sample_rate
        // Simplified: each mel frame ≈ 10ms, 4x subsample → each token ≈ 40ms
        Double(audioMelFrames) * 0.01
    }

    /// Model name from config.
    public var modelName: String { config.modelName }

    /// Context length from config.
    public var contextLength: Int { config.contextLength }

    // MARK: - Single-turn convenience

    /// Generate a complete response from a single prompt.
    public func generate(_ prompt: String, image: CGImage? = nil,
                         audio: [Float]? = nil,
                         maxTokens: Int = 2048) async throws -> String {
        let messages = [Message(role: .user, content: prompt)]
        return try await generate(messages, image: image, audio: audio, maxTokens: maxTokens)
    }

    /// Stream tokens from a single prompt.
    public func stream(_ prompt: String, image: CGImage? = nil,
                       audio: [Float]? = nil,
                       maxTokens: Int = 2048) async throws -> AsyncStream<String> {
        let messages = [Message(role: .user, content: prompt)]
        return try await stream(messages, image: image, audio: audio, maxTokens: maxTokens)
    }

    // MARK: - Multi-turn API

    /// Generate a complete response from a conversation.
    public func generate(_ messages: [Message], image: CGImage? = nil,
                         audio: [Float]? = nil,
                         maxTokens: Int = 2048) async throws -> String {
        var result = ""
        for await token in try await stream(messages, image: image, audio: audio,
                                             maxTokens: maxTokens) {
            result += token
        }
        return result
    }

    /// Stream tokens from a multi-turn conversation.
    ///
    /// If `image` is provided, it's processed and cached for the current turn.
    /// If `image` is nil but a previous image was cached, the cached features are reused.
    public func stream(_ messages: [Message], image: CGImage? = nil,
                       audio: [Float]? = nil,
                       maxTokens: Int = 2048) async throws -> AsyncStream<String> {
        // Process image (or reuse cached)
        var imageFeatures: MLMultiArray? = cachedImageFeatures
        if let image {
            let tVision = CFAbsoluteTimeGetCurrent()
            imageFeatures = try processImage(image)
            let dtVision = (CFAbsoluteTimeGetCurrent() - tVision) * 1000
            let which = visionUsesANEBuild ? "ANE" : "GPU"
            print("[Vision] encode (\(which)): \(String(format: "%.1f", dtVision)) ms")
            cachedImageFeatures = imageFeatures
        }

        // Process audio
        var audioFeatures: MLMultiArray?
        var actualAudioTokens = 0
        if let audio {
            let (features, tokenCount) = try processAudio(audio)
            audioFeatures = features
            actualAudioTokens = tokenCount
        }

        let hasImage = imageFeatures != nil
        let hasAudioInput = audioFeatures != nil
        let chatPrompt = buildPrompt(messages, hasImage: hasImage, hasAudio: hasAudioInput,
                                     audioTokenCount: actualAudioTokens)
        return try streamFromPrompt(
            chatPrompt,
            imageFeatures: imageFeatures,
            imageNumTokens: imageFeatures != nil ? 256 : 0,
            audioFeatures: audioFeatures,
            audioTokenCount: actualAudioTokens,
            maxTokens: maxTokens
        )
    }

    // MARK: - Video API

    /// Stream tokens with a video prompt. Frames are sampled at `options.fps`
    /// (capped to `options.maxFrames`); each frame contributes 256 vision
    /// soft tokens plus a `MM:SS` timestamp label, matching the Gemma 4
    /// video chat template. If `options.includeAudio` is set and the asset
    /// has an audio track, its PCM is fed through the Conformer encoder.
    ///
    /// Watch the prompt length: each frame costs ~261 tokens, so on a 2K
    /// chunk you want `maxFrames <= 7` and on 8K `maxFrames <= 30`.
    public func stream(_ messages: [Message], videoURL: URL,
                       videoOptions: VideoProcessor.Options = .init(),
                       maxTokens: Int = 2048) async throws -> AsyncStream<String> {
        let frames = try await VideoProcessor.extractFrames(
            from: videoURL, options: videoOptions)
        guard !frames.isEmpty else {
            throw CoreMLLLMError.videoDecodeFailed
        }
        let pcm: [Float]? = videoOptions.includeAudio
            ? try await VideoProcessor.extractAudioPCM16k(from: videoURL)
            : nil

        let tokensPerFrame = videoOptions.tokensPerFrame
        let combined = try concatFrameFeatures(frames.map { $0.image },
                                                tokensPerFrame: tokensPerFrame)

        var audioFeatures: MLMultiArray?
        var actualAudioTokens = 0
        if let pcm, !pcm.isEmpty, supportsAudio {
            let (features, tokenCount) = try processAudio(pcm)
            audioFeatures = features
            actualAudioTokens = tokenCount
        }

        let videoBlock = buildVideoBlock(timestamps: frames.map { $0.timestampSeconds },
                                          tokensPerFrame: tokensPerFrame)
        let audioBlock = actualAudioTokens > 0
            ? "<|audio>" + String(repeating: "<|audio|>", count: actualAudioTokens) + "<audio|>"
            : ""
        let chatPrompt = buildGemmaMediaPrompt(messages, mediaBlock: videoBlock,
                                                audioBlock: audioBlock)

        return try streamFromPrompt(
            chatPrompt,
            imageFeatures: combined,
            imageNumTokens: frames.count * tokensPerFrame,
            audioFeatures: audioFeatures,
            audioTokenCount: actualAudioTokens,
            maxTokens: maxTokens
        )
    }

    /// Generate a full response from a video prompt.
    public func generate(_ messages: [Message], videoURL: URL,
                         videoOptions: VideoProcessor.Options = .init(),
                         maxTokens: Int = 2048) async throws -> String {
        var result = ""
        for await token in try await stream(messages, videoURL: videoURL,
                                             videoOptions: videoOptions,
                                             maxTokens: maxTokens) {
            result += token
        }
        return result
    }

    /// Single-prompt convenience for video.
    public func generate(_ prompt: String, videoURL: URL,
                         videoOptions: VideoProcessor.Options = .init(),
                         maxTokens: Int = 2048) async throws -> String {
        try await generate([Message(role: .user, content: prompt)],
                            videoURL: videoURL, videoOptions: videoOptions,
                            maxTokens: maxTokens)
    }

    public func stream(_ prompt: String, videoURL: URL,
                       videoOptions: VideoProcessor.Options = .init(),
                       maxTokens: Int = 2048) async throws -> AsyncStream<String> {
        try await stream([Message(role: .user, content: prompt)],
                          videoURL: videoURL, videoOptions: videoOptions,
                          maxTokens: maxTokens)
    }

    // MARK: - Private core stream

    private func streamFromPrompt(
        _ chatPrompt: String,
        imageFeatures: MLMultiArray?,
        imageNumTokens: Int,
        audioFeatures: MLMultiArray?,
        audioTokenCount: Int,
        maxTokens: Int
    ) throws -> AsyncStream<String> {
        let tokenIDs = tokenizer.encode(text: chatPrompt)

        reset()

        // Clear recording buffers and record the prompt IDs for this turn.
        self.lastPromptTokenIDs = tokenIDs.map { Int32($0) }
        self.lastEmittedTokenIDs = []

        let mutableSelf = self
        let imgFeats = imageFeatures
        let imgTokenLimit = imageNumTokens
        let audFeats = audioFeatures
        let tokens = tokenIDs
        let audTokenCount = audioTokenCount
        let ctxLimit = config.contextLength

        // Decode-loop QoS: defaults to inherited (.userInitiated when called from
        // UI). Set LLM_DECODE_QOS=utility (or background) to bias toward
        // efficiency cores — trades a small tok/s loss for cooler sustained
        // operation. CPU memcpy/copyBack between ANE dispatches is the
        // dominant CPU draw at 31 tok/s; E-cores cut that ~4x.
        let qosEnv = ProcessInfo.processInfo.environment["LLM_DECODE_QOS"]?.lowercased()
        let decodePriority: TaskPriority?
        switch qosEnv {
        case "background": decodePriority = .background
        case "utility":    decodePriority = .utility
        case "userinitiated", "user": decodePriority = .userInitiated
        case "high":       decodePriority = .high
        default:           decodePriority = nil  // inherit
        }
        if decodePriority != nil {
            print("[QoS] LLM_DECODE_QOS=\(qosEnv!) — decode loop priority overridden")
        }

        return AsyncStream { continuation in
            Task(priority: decodePriority) {
                do {
                    let IMAGE_TOKEN_ID = 258880
                    let AUDIO_TOKEN_ID = 258881
                    let VIDEO_TOKEN_ID = 258884
                    var imageIdx = 0
                    var audioIdx = 0
                    var nextID = 0

                    func multimodalEmbedding(for tid: Int) -> MLMultiArray? {
                        // Image and video share the same `imgFeats` buffer:
                        // video frames are concatenated by `concatFrameFeatures`
                        // into the same (1, N, hidden) layout the image path
                        // uses, so a single counter walks both placeholder
                        // streams correctly.
                        if (tid == IMAGE_TOKEN_ID || tid == VIDEO_TOKEN_ID),
                           let f = imgFeats, imageIdx < imgTokenLimit {
                            let emb = engine?.sliceFeature(f, at: imageIdx)
                                ?? ImageProcessor.sliceFeature(f, at: imageIdx,
                                    hiddenSize: mutableSelf.config.hiddenSize)
                            imageIdx += 1
                            return emb
                        }
                        if tid == AUDIO_TOKEN_ID, let f = audFeats, audioIdx < audTokenCount {
                            let emb = engine?.sliceFeature(f, at: audioIdx)
                                ?? AudioProcessor.sliceFeature(f, at: audioIdx,
                                    hiddenSize: mutableSelf.config.hiddenSize)
                            audioIdx += 1
                            return emb
                        }
                        return nil
                    }

                    let engine = mutableSelf.chunkedEngine

                    if let engine {
                        // Prefill chunks load in the background (see
                        // ChunkedEngine `deferred prefill load`). If a long
                        // prompt lands while they're still loading, we used
                        // to block on the full load — but in practice that
                        // could take 60+ s during app warmup, dwarfing the
                        // ~8 s decode-loop alternative (272 tok × 30 ms).
                        // Decide at request time:
                        //   * prefill ready → use batched prefill
                        //   * not ready     → run decode-loop prefill now,
                        //                     don't wait
                        // For short prompts (< 64 tok) the decode loop is
                        // already comparable to batched prefill and we
                        // never waited.
                        let prefillReady = engine.hasPrefill
                        if !prefillReady, tokens.count >= 64 {
                            print("[Load] prefill chunks not ready — using decode-loop prefill for \(tokens.count) tok (avoids multi-s await)")
                        }
                        let prefillLen = prefillReady ? min(tokens.count, engine.prefillN) : 0
                        let useHybrid = prefillReady && prefillLen > 0

                        if useHybrid {
                            try autoreleasepool {
                                let batch = Array(tokens[0..<prefillLen])
                                nextID = try engine.runPrefill(
                                    tokenIDs: batch,
                                    imageFeatures: imgFeats,
                                    imageNumTokens: imgTokenLimit,
                                    audioFeatures: audFeats,
                                    audioNumTokens: audTokenCount
                                )
                            }
                            imageIdx = tokens[0..<prefillLen].filter {
                                $0 == IMAGE_TOKEN_ID || $0 == VIDEO_TOKEN_ID
                            }.count
                            audioIdx = tokens[0..<prefillLen].filter { $0 == AUDIO_TOKEN_ID }.count
                            engine.currentPosition = prefillLen

                            for step in prefillLen..<tokens.count {
                                let tid = tokens[step]
                                try autoreleasepool {
                                    if let emb = multimodalEmbedding(for: tid) {
                                        nextID = try engine.predictStep(tokenID: 0, position: step,
                                                                         imageEmbedding: emb)
                                    } else {
                                        nextID = try engine.predictStep(tokenID: tid, position: step)
                                    }
                                }
                                engine.currentPosition = step + 1
                            }
                        } else {
                            for (step, tid) in tokens.enumerated() {
                                try autoreleasepool {
                                    if let emb = multimodalEmbedding(for: tid) {
                                        nextID = try engine.predictStep(tokenID: 0, position: step,
                                                                         imageEmbedding: emb)
                                    } else {
                                        nextID = try engine.predictStep(tokenID: tid, position: step)
                                    }
                                }
                                engine.currentPosition = step + 1
                            }
                        }

                        // Decode loop with tok/s tracking
                        let eosIDs: Set<Int> = [1, 106, 151645]
                        let startTime = CFAbsoluteTimeGetCurrent()
                        var tokenCount = 0
                        let maxDecode = min(ctxLimit - engine.currentPosition, maxTokens)
                        // Drafter selection priority (highest first):
                        //   1. EAGLE-3 SpeculativeLoop (trained eagle drafter, best when present)
                        //   2. MTP (trained drafter, best when present)
                        //   3. DrafterUnion (Phase B; cv + pld-n2 + pld-n3)
                        //   4. CrossVocab alone (legacy, kept as opt-out fallback)
                        // Only the selected engine resets — the union and the
                        // CV-alone engine share an underlying CrossVocabDraft,
                        // so simultaneous use would corrupt Qwen state.
                        //
                        // LLM_EAGLE3_DISABLE=1 skips EAGLE-3 so DrafterUnion/CV
                        // can be benched standalone (e.g. while the EAGLE-3
                        // draft is a known-broken pre-retrain checkpoint).
                        let eagleDisabled = ProcessInfo.processInfo
                            .environment["LLM_EAGLE3_DISABLE"] == "1"
                        let eagleSpec = eagleDisabled ? nil : mutableSelf.speculativeLoop
                        // LookaheadEngine takes precedence over MTP/Union/CV
                        // when enabled — it is drafter-free and subsumes the
                        // union's PLD path while adding Jacobi warm-start.
                        let lookaheadSpec = (eagleSpec == nil && mutableSelf.lookaheadEnabled)
                            ? mutableSelf.lookaheadEngine : nil
                        let mtpSpec = (eagleSpec == nil && lookaheadSpec == nil
                                       && mutableSelf.mtpEnabled)
                            ? mutableSelf.mtpEngine : nil
                        let unionSpec = (eagleSpec == nil && lookaheadSpec == nil
                                         && mtpSpec == nil
                                         && mutableSelf.drafterUnionEnabled)
                            ? mutableSelf.drafterUnion : nil
                        let cvSpec = (eagleSpec == nil && lookaheadSpec == nil
                                      && mtpSpec == nil && unionSpec == nil
                                      && mutableSelf.crossVocabEnabled)
                            ? mutableSelf.crossVocabEngine : nil
                        lookaheadSpec?.reset()
                        lookaheadSpec?.setPrefillHistory(tokens.map { Int32($0) })
                        mtpSpec?.reset()
                        unionSpec?.reset()
                        unionSpec?.setPrefillHistory(tokens.map { Int32($0) })
                        cvSpec?.reset()
                        cvSpec?.setPrefillHistory(tokens.map { Int32($0) })

                        // First iteration is always plain T=1 decode so hidden_at_L*
                        // taps get populated before EAGLE-3 speculative can read them.
                        var didFirstDecode = false

                        var nid = Int32(nextID)
                        decodeLoop: while tokenCount < maxDecode {
                            if eosIDs.contains(Int(nid)) { break }
                            if engine.currentPosition >= ctxLimit { break }

                            let useEagle = (eagleSpec != nil) && didFirstDecode
                                && engine.canSpeculate
                                && (eagleSpec?.shouldSpeculate ?? false)
                            // One-shot debug of the 4 gating conditions when
                            // EAGLE-3 is loaded but not yet speculating.
                            if eagleSpec != nil && !useEagle && tokenCount < 3 {
                                print("[SpecDbg] useEagle=false  loaded=\(eagleSpec != nil)  firstDecode=\(didFirstDecode)  canSpec=\(engine.canSpeculate)  shouldSpec=\(eagleSpec?.shouldSpeculate ?? false)  hL8=\(engine.lastHiddenAtL8 != nil)  hL17=\(engine.lastHiddenAtL17 != nil)  hL34=\(engine.lastHiddenAtL34 != nil)  vchunk=\(engine.hasVerify)")
                            }

                            if useEagle, let sl = eagleSpec {
                                if tokenCount < 5 {
                                    print("[SpecDbg] ENTER EAGLE burst tokenCount=\(tokenCount) pos=\(engine.currentPosition) nid=\(nid)")
                                }
                                // EAGLE-3 speculative burst: yields 1..K+1 accepted tokens.
                                let accepted: [Int32]
                                do {
                                    accepted = try sl.drawBurst(
                                        target: engine,
                                        tTokNext: nid,
                                        tokenEmbed: { try engine.embedToken($0) })
                                } catch {
                                    // On any speculative failure, fall back to T=1 this step.
                                    print("[Spec] burst failed: \(error) — falling back to T=1")
                                    mutableSelf.lastEmittedTokenIDs.append(nid)
                                    let text = mutableSelf.tokenizer.decode(tokens: [Int(nid)])
                                    continuation.yield(text)
                                    tokenCount += 1
                                    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                                    if elapsed > 0 {
                                        mutableSelf.tokensPerSecond = Double(tokenCount) / elapsed
                                    }
                                    try autoreleasepool {
                                        let next = try engine.predictStep(
                                            tokenID: Int(nid), position: engine.currentPosition)
                                        nid = Int32(next)
                                    }
                                    engine.currentPosition += 1
                                    continue decodeLoop
                                }

                                // `accepted` always starts with the tTokNext we passed in.
                                // commitAccepted has already advanced currentPosition by
                                // accepted.count and refreshed hidden taps.
                                for tok in accepted {
                                    let t = Int(tok)
                                    if eosIDs.contains(t) {
                                        mutableSelf.lastEmittedTokenIDs.append(tok)
                                        let text = mutableSelf.tokenizer.decode(tokens: [t])
                                        continuation.yield(text)
                                        tokenCount += 1
                                        nid = tok
                                        break decodeLoop
                                    }
                                    mutableSelf.lastEmittedTokenIDs.append(tok)
                                    let text = mutableSelf.tokenizer.decode(tokens: [t])
                                    continuation.yield(text)
                                    tokenCount += 1
                                    if tokenCount >= maxDecode { break decodeLoop }
                                }
                                nid = Int32(engine.lastArgmaxAfterDecode)
                            } else if let se = lookaheadSpec, se.shouldSpeculate {
                                let emitted = try autoreleasepool {
                                    try se.speculateStep(nextID: &nid)
                                }
                                for tok in emitted {
                                    if eosIDs.contains(Int(tok)) {
                                        nid = tok
                                        break
                                    }
                                    mutableSelf.lastEmittedTokenIDs.append(tok)
                                    let text = mutableSelf.tokenizer.decode(tokens: [Int(tok)])
                                    continuation.yield(text)
                                    tokenCount += 1
                                }
                            } else if let se = mtpSpec, se.shouldSpeculate {
                                let emitted = try autoreleasepool {
                                    try se.speculateStep(nextID: &nid)
                                }
                                for tok in emitted {
                                    if eosIDs.contains(Int(tok)) {
                                        nid = tok
                                        break
                                    }
                                    let text = mutableSelf.tokenizer.decode(tokens: [Int(tok)])
                                    continuation.yield(text)
                                    tokenCount += 1
                                }
                            } else if let se = unionSpec, se.shouldSpeculate {
                                let emitted = try autoreleasepool {
                                    try se.speculateStep(nextID: &nid)
                                }
                                for tok in emitted {
                                    if eosIDs.contains(Int(tok)) {
                                        nid = tok
                                        break
                                    }
                                    mutableSelf.lastEmittedTokenIDs.append(tok)
                                    let text = mutableSelf.tokenizer.decode(tokens: [Int(tok)])
                                    continuation.yield(text)
                                    tokenCount += 1
                                }
                            } else if let se = cvSpec, se.shouldSpeculate {
                                let emitted = try autoreleasepool {
                                    try se.speculateStep(nextID: &nid)
                                }
                                for tok in emitted {
                                    if eosIDs.contains(Int(tok)) {
                                        nid = tok
                                        break
                                    }
                                    mutableSelf.lastEmittedTokenIDs.append(tok)
                                    let text = mutableSelf.tokenizer.decode(tokens: [Int(tok)])
                                    continuation.yield(text)
                                    tokenCount += 1
                                }
                            } else {
                                if tokenCount < 5 {
                                    print("[SpecDbg] ENTER T=1 tokenCount=\(tokenCount) pos=\(engine.currentPosition)")
                                }
                                mutableSelf.lastEmittedTokenIDs.append(nid)
                                let text = mutableSelf.tokenizer.decode(tokens: [Int(nid)])
                                continuation.yield(text)
                                tokenCount += 1
                                try autoreleasepool {
                                    let next = try engine.predictStep(
                                        tokenID: Int(nid), position: engine.currentPosition)
                                    nid = Int32(next)
                                }
                                engine.currentPosition += 1
                            }

                            // Any branch (T=1, MTP, Union, CrossVocab) counts
                            // as "first decode completed" — after which the
                            // EAGLE-3 speculative path is eligible to take
                            // over on subsequent steps. Without this, decode
                            // loops that kick in MTP/Union/CrossVocab first
                            // never let EAGLE-3 run (the flag stayed false).
                            if !didFirstDecode && tokenCount < 3 {
                                print("[SpecDbg] didFirstDecode set true (tokenCount=\(tokenCount))")
                            }
                            didFirstDecode = true

                            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                            if elapsed > 0 { mutableSelf.tokensPerSecond = Double(tokenCount) / elapsed }
                        }
                        nextID = Int(nid)
                    } else {
                        // Monolithic path
                        for (step, tid) in tokens.enumerated() {
                            try autoreleasepool {
                                if let emb = multimodalEmbedding(for: tid) {
                                    nextID = try mutableSelf.predictMonolithic(
                                        tokenID: 0, position: step, imageEmbedding: emb)
                                } else {
                                    nextID = try mutableSelf.predictMonolithic(
                                        tokenID: tid, position: step)
                                }
                            }
                        }
                        let eosIDs: Set<Int> = [1, 106, 151645]
                        let startTime = CFAbsoluteTimeGetCurrent()
                        var pos = tokens.count
                        var tokenCount = 0
                        for _ in 0..<maxTokens {
                            if eosIDs.contains(nextID) { break }
                            if pos >= ctxLimit { break }
                            mutableSelf.lastEmittedTokenIDs.append(Int32(nextID))
                            let text = mutableSelf.tokenizer.decode(tokens: [nextID])
                            continuation.yield(text)
                            tokenCount += 1
                            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                            if elapsed > 0 { mutableSelf.tokensPerSecond = Double(tokenCount) / elapsed }
                            try autoreleasepool {
                                nextID = try mutableSelf.predictMonolithic(
                                    tokenID: nextID, position: pos)
                            }
                            pos += 1
                        }
                    }
                } catch {
                    print("[CoreMLLLM] Error: \(error)")
                }
                continuation.finish()
            }
        }
    }

    /// Reset conversation state (clears KV cache and cached image features).
    public func reset() {
        if let engine = chunkedEngine {
            engine.reset()
        } else {
            monolithicState = monolithicModel?.makeState()
        }
        mtpEngine?.reset()
        crossVocabEngine?.reset()
        drafterUnion?.reset()
        lookaheadEngine?.reset()
        tokensPerSecond = 0
    }

    /// Clear cached image features (called between conversations).
    public func clearImageCache() {
        cachedImageFeatures = nil
    }

    // MARK: - Bench helpers (offline harness use only; not for production)

    /// Chunked-engine verify K (typically 3). Nil if the monolithic path is
    /// in use or verify chunks aren't loaded.
    public var benchVerifyK: Int? {
        guard let e = chunkedEngine, e.hasVerify else { return nil }
        return e.verifyK
    }

    /// Current decode position on the chunked engine, or nil on monolithic.
    public var benchCurrentPosition: Int? { chunkedEngine?.currentPosition }

    /// Run prefill + sequential `predictStep` for any tokens that didn't fit
    /// in a prefill chunk. After return, `benchCurrentPosition ==
    /// promptTokens.count` and `seed` is target's argmax (via `decode_q1`) for
    /// that position — the token to emit first. Resets engine + spec engines.
    ///
    /// Text prompts only; image/audio paths are not handled.
    public func benchPrefill(_ prompt: String) async throws -> (prompt: [Int32], seed: Int32) {
        guard let engine = chunkedEngine else {
            throw CoreMLLLMError.predictionFailed
        }
        let messages = [Message(role: .user, content: prompt)]
        let chatPrompt = buildPrompt(messages, hasImage: false, hasAudio: false,
                                     audioTokenCount: 0)
        let tokens = tokenizer.encode(text: chatPrompt)
        reset()
        self.lastPromptTokenIDs = tokens.map { Int32($0) }
        self.lastEmittedTokenIDs = []

        var nextID = 0
        let prefillLen = min(tokens.count, engine.prefillN)
        let useHybrid = engine.hasPrefill && prefillLen > 0
        if useHybrid {
            try autoreleasepool {
                let batch = Array(tokens[0..<prefillLen])
                nextID = try engine.runPrefill(tokenIDs: batch)
            }
            engine.currentPosition = prefillLen
            for step in prefillLen..<tokens.count {
                try autoreleasepool {
                    nextID = try engine.predictStep(tokenID: tokens[step], position: step)
                }
                engine.currentPosition = step + 1
            }
        } else {
            for (step, tid) in tokens.enumerated() {
                try autoreleasepool {
                    nextID = try engine.predictStep(tokenID: tid, position: step)
                }
                engine.currentPosition = step + 1
            }
        }
        return (tokens.map { Int32($0) }, Int32(nextID))
    }

    /// Run `verify_qK` at the current decode position. `tokens.count` must
    /// equal `benchVerifyK`. Returns target's argmax at each of K positions.
    /// Writes K KV slots starting at `benchCurrentPosition` but does NOT
    /// advance the position — use `benchAdvance(by:)` to commit.
    public func benchVerify(_ tokens: [Int32]) throws -> [Int32] {
        guard let engine = chunkedEngine, engine.hasVerify else {
            throw CoreMLLLMError.predictionFailed
        }
        return try engine.verifyCandidates(tokens: tokens,
                                           startPosition: engine.currentPosition)
    }

    /// Variant of `benchVerify` that also returns per-position top-`topK`
    /// `(token_id, logit_fp32)` pairs. Used by the tolerance-based accept
    /// variant of `accept-rate-bench`.
    ///
    /// Throws `CoreMLLLMError.verifyLogitsNotExposed` until the Track B
    /// (`feat/c0-verify-requant`) re-export of verify chunk 4 adds a
    /// `logits_fp16` output. Until that PR merges, callers must fall back to
    /// argmax-only acceptance via `benchVerify`.
    public func benchVerifyTopK(_ tokens: [Int32], topK: Int = 3) throws
        -> [[(Int32, Float)]]
    {
        guard let engine = chunkedEngine, engine.hasVerify else {
            throw CoreMLLLMError.predictionFailed
        }
        let (_, top) = try engine.verifyCandidatesWithLogits(
            tokens: tokens,
            startPosition: engine.currentPosition,
            topK: topK)
        return top
    }

    /// Advance `benchCurrentPosition` by `count`.
    public func benchAdvance(by count: Int) {
        chunkedEngine?.currentPosition += count
    }

    // MARK: - Private: monolithic prediction

    private func predictMonolithic(tokenID: Int, position: Int,
                                    imageEmbedding: MLMultiArray? = nil) throws -> Int {
        guard let model = monolithicModel, let state = monolithicState else {
            throw CoreMLLLMError.predictionFailed
        }
        let ctx = config.contextLength
        let hs = config.hiddenSize

        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(position))
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        for i in 0..<ctx { mp[i] = i <= position ? 0 : 0xFC00 }
        let umask = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16)
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(up, 0, ctx * MemoryLayout<UInt16>.stride)
        up[min(position, ctx - 1)] = 0x3C00

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ]

        let inputNames = model.modelDescription.inputDescriptionsByName
        if inputNames["per_layer_combined"] != nil, let engine = chunkedEngine {
            let emb = try engine.computePerLayerCombined(tokenID: tokenID,
                embedding: try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16))
            dict["per_layer_combined"] = MLFeatureValue(multiArray: emb)
        }
        if inputNames["image_embedding"] != nil {
            let imgEmb: MLMultiArray
            if let imageEmbedding { imgEmb = imageEmbedding }
            else {
                imgEmb = try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
                memset(imgEmb.dataPointer, 0, hs * MemoryLayout<UInt16>.stride)
            }
            dict["image_embedding"] = MLFeatureValue(multiArray: imgEmb)
        }

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict),
                                           using: state)
        return output.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    // MARK: - Private: vision

    private func processImage(_ image: CGImage) throws -> MLMultiArray {
        if visionModel == nil, let url = visionModelURL, let cfg = visionConfig {
            visionModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let vm = visionModel else { throw CoreMLLLMError.visionNotAvailable }
        if visionUsesANEBuild {
            return try ImageProcessor.processANE(image, with: vm)
        }
        return try ImageProcessor.process(image, with: vm)
    }

    private func processVideoFrame(_ image: CGImage) throws -> MLMultiArray {
        if videoVisionModel == nil, let url = videoVisionModelURL, let cfg = videoVisionConfig {
            videoVisionModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let vm = videoVisionModel else { throw CoreMLLLMError.visionNotAvailable }
        return try ImageProcessor.processVideoFrame(image, with: vm)
    }

    // MARK: - Private: audio

    /// Returns (features, actualTokenCount).
    /// actualTokenCount is based on real audio length, not the padded model input.
    private func processAudio(_ samples: [Float]) throws -> (MLMultiArray, Int) {
        if audioModel == nil, let url = audioModelURL, let cfg = audioConfig {
            audioModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let am = audioModel else { throw CoreMLLLMError.audioNotAvailable }
        guard let mel = melFilterbank else { throw CoreMLLLMError.audioNotAvailable }
        // projection is optional — AudioProcessor.process will fall back to
        // Swift-side projection only if the encoder outputs 1024-dim features.

        // Compute actual mel frames from audio length (matching HF Gemma4AudioFeatureExtractor)
        let padLeft = 160  // frameLength / 2, semicausal pad
        let paddedLen = padLeft + samples.count
        let unfoldSize = 321  // frameLength + 1
        let actualMelFrames = max(0, (paddedLen - unfoldSize) / 160 + 1)
        // After 2x Conv2d stride 2: tokens = ceil(ceil(melFrames / 2) / 2)
        let afterConv1 = (actualMelFrames + 1) / 2
        let actualTokens = min((afterConv1 + 1) / 2, audioNumTokens)

        let features = try AudioProcessor.process(samples, with: am,
                                                    melFilterbank: mel,
                                                    targetFrames: audioMelFrames,
                                                    projection: audioProjection,
                                                    melFloor: audioMelFloor)
        return (features, actualTokens)
    }

    // MARK: - Private: prompt building

    private func buildPrompt(_ messages: [Message], hasImage: Bool,
                              hasAudio: Bool = false,
                              audioTokenCount: Int = 0) -> String {
        if config.architecture.hasPrefix("qwen") {
            return buildQwenPrompt(messages)
        }
        return buildGemmaPrompt(messages, hasImage: hasImage, hasAudio: hasAudio,
                                audioTokenCount: audioTokenCount)
    }

    private func buildGemmaPrompt(_ messages: [Message], hasImage: Bool, hasAudio: Bool,
                                   audioTokenCount: Int = 0) -> String {
        let imageBlock = "<|image>" + String(repeating: "<|image|>", count: 256) + "<image|>"
        let audioBlock = "<|audio>" + String(repeating: "<|audio|>", count: audioTokenCount) + "<audio|>"
        let lastUserIdx = messages.lastIndex { $0.role == .user }

        var p = "<bos>"
        for (i, m) in messages.enumerated() {
            switch m.role {
            case .user:
                let isLast = i == lastUserIdx
                var mediaPrefix = ""
                if hasImage && isLast { mediaPrefix += imageBlock + "\n" }
                if hasAudio && isLast { mediaPrefix += audioBlock + "\n" }
                p += "<|turn>user\n\(mediaPrefix)\(m.content)<turn|>\n"
            case .assistant:
                p += "<|turn>model\n\(m.content)<turn|>\n"
            case .system:
                break
            }
        }
        return p + "<|turn>model\n"
    }

    /// Build the per-frame video block for the Gemma 4 chat template:
    ///   `MM:SS <|image><|video|>×K<image|>` joined by single spaces.
    ///
    /// The Gemma 4 tokenizer has separate placeholder ids for image
    /// (`<|image|>` 258880) and video (`<|video|>` 258884). HF's
    /// `Gemma4Processor` uses the video token inside each frame block so
    /// the model knows the sequence is a video, not a series of stills —
    /// using `<|image|>` here makes the model describe frames as
    /// independent images. BOI/EOI tags stay shared with the image path.
    private func buildVideoBlock(timestamps: [Double], tokensPerFrame: Int) -> String {
        let placeholder = String(repeating: "<|video|>", count: tokensPerFrame)
        return timestamps
            .map { "\(VideoProcessor.timestampLabel($0)) <|image>\(placeholder)<image|>" }
            .joined(separator: " ")
    }

    /// Gemma prompt variant that injects an arbitrary media block (already
    /// formatted) instead of the single-image block. Used by the video path.
    private func buildGemmaMediaPrompt(_ messages: [Message],
                                        mediaBlock: String,
                                        audioBlock: String) -> String {
        let lastUserIdx = messages.lastIndex { $0.role == .user }
        var p = "<bos>"
        for (i, m) in messages.enumerated() {
            switch m.role {
            case .user:
                let isLast = i == lastUserIdx
                var mediaPrefix = ""
                if isLast && !mediaBlock.isEmpty { mediaPrefix += mediaBlock + "\n" }
                if isLast && !audioBlock.isEmpty { mediaPrefix += audioBlock + "\n" }
                p += "<|turn>user\n\(mediaPrefix)\(m.content)<turn|>\n"
            case .assistant:
                p += "<|turn>model\n\(m.content)<turn|>\n"
            case .system:
                break
            }
        }
        return p + "<|turn>model\n"
    }

    // MARK: - Private: video feature concatenation

    /// Run each frame through the vision encoder and concatenate to a single
    /// (1, N*tokensPerFrame, H) MLMultiArray.
    ///
    /// The still-image encoder emits 280 tokens per frame (256 real + 24
    /// padding for a square input, laid out as a 16×16 grid). For video we
    /// want a lower token budget per frame (Gemma 4's `video_processor`
    /// uses `max_soft_tokens=70` ≈ 64 real). We cover three cases here:
    ///   - `tokensPerFrame = 64`, `vision_video.mlpackage` present:
    ///       use the purpose-built video encoder which already emits 64
    ///       tokens/frame — no Swift-side pooling needed.
    ///   - `tokensPerFrame = 256`: raw passthrough (first 256 of 280).
    ///   - `tokensPerFrame = 64`, no video encoder: 2×2 average-pool the
    ///       16×16 grid to 8×8 (Phase 1 fallback).
    ///   - other:                   first `tokensPerFrame` tokens of the 280
    ///                              (not semantically meaningful — debug only).
    private func concatFrameFeatures(_ frames: [CGImage],
                                      tokensPerFrame: Int) throws -> MLMultiArray {
        precondition(!frames.isEmpty)
        if tokensPerFrame == 64, videoVisionModelURL != nil {
            return try concatVideoFrameFeatures(frames)
        }
        let hidden = config.hiddenSize
        let total = frames.count * tokensPerFrame
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: total), NSNumber(value: hidden)],
            dataType: .float16)
        let dst = out.dataPointer.bindMemory(to: UInt16.self, capacity: total * hidden)
        memset(dst, 0, total * hidden * MemoryLayout<UInt16>.stride)
        for (i, frame) in frames.enumerated() {
            let feat = try processImage(frame)
            let src = feat.dataPointer.bindMemory(to: UInt16.self, capacity: feat.count)
            let dstFrame = dst.advanced(by: i * tokensPerFrame * hidden)
            if tokensPerFrame == 64 {
                pool16x16To8x8(src: src, dst: dstFrame, hidden: hidden)
            } else {
                memcpy(dstFrame, src,
                       tokensPerFrame * hidden * MemoryLayout<UInt16>.stride)
            }
        }
        return out
    }

    /// Video-encoder path for `concatFrameFeatures`. The encoder emits
    /// (1, 64, hidden) per frame; we memcpy each block into the combined
    /// (1, N·64, hidden) buffer. Kept separate from the still-image path
    /// so the pool fallback is easy to delete once every shipped model
    /// bundle includes `vision_video.mlpackage`.
    private func concatVideoFrameFeatures(_ frames: [CGImage]) throws -> MLMultiArray {
        precondition(!frames.isEmpty)
        let hidden = config.hiddenSize
        let perFrame = 64
        let total = frames.count * perFrame
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: total), NSNumber(value: hidden)],
            dataType: .float16)
        let dst = out.dataPointer.bindMemory(to: UInt16.self, capacity: total * hidden)
        memset(dst, 0, total * hidden * MemoryLayout<UInt16>.stride)
        for (i, frame) in frames.enumerated() {
            let feat = try processVideoFrame(frame)
            let src = feat.dataPointer.bindMemory(to: UInt16.self, capacity: feat.count)
            let dstFrame = dst.advanced(by: i * perFrame * hidden)
            memcpy(dstFrame, src,
                   perFrame * hidden * MemoryLayout<UInt16>.stride)
        }
        return out
    }

    /// Average-pool a 16×16 token grid (256 tokens, row-major) down to 8×8
    /// (64 tokens) by averaging each 2×2 block in fp32 and writing back
    /// fp16. `src` must point to 256 × hidden fp16 values; `dst` to 64 ×
    /// hidden fp16 values.
    private func pool16x16To8x8(src: UnsafeMutablePointer<UInt16>,
                                 dst: UnsafeMutablePointer<UInt16>,
                                 hidden: Int) {
        for by in 0..<8 {
            for bx in 0..<8 {
                let dstOff = (by * 8 + bx) * hidden
                for d in 0..<hidden {
                    var sum: Float = 0
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let r = by * 2 + dy
                            let c = bx * 2 + dx
                            let srcIdx = (r * 16 + c) * hidden + d
                            sum += Float(Float16(bitPattern: src[srcIdx]))
                        }
                    }
                    dst[dstOff + d] = (Float16(sum * 0.25)).bitPattern
                }
            }
        }
    }

    private func buildQwenPrompt(_ messages: [Message]) -> String {
        var p = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for m in messages {
            switch m.role {
            case .user:
                p += "<|im_start|>user\n\(m.content)<|im_end|>\n"
            case .assistant:
                p += "<|im_start|>assistant\n\(m.content)<|im_end|>\n"
            case .system:
                break
            }
        }
        return p + "<|im_start|>assistant\n"
    }
}

// MARK: - Error types

public enum CoreMLLLMError: LocalizedError {
    case configNotFound
    case predictionFailed
    case modelNotFound(String)
    case prefillNotAvailable
    case visionNotAvailable
    case audioNotAvailable
    case videoDecodeFailed
    /// The verify-chunk pipeline does not expose a `logits_fp16` output yet.
    /// Returned by `benchVerifyTopK` / `verifyCandidatesWithLogits` until the
    /// Track B re-export (`feat/c0-verify-requant`) lands on `main`.
    case verifyLogitsNotExposed

    public var errorDescription: String? {
        switch self {
        case .configNotFound: return "model_config.json not found"
        case .predictionFailed: return "Model prediction failed"
        case .modelNotFound(let name): return "Model file not found: \(name)"
        case .prefillNotAvailable: return "Prefill chunks not loaded"
        case .visionNotAvailable: return "Vision model not available"
        case .audioNotAvailable: return "Audio model not available"
        case .videoDecodeFailed: return "Could not decode any frames from video"
        case .verifyLogitsNotExposed:
            return "verify chunk 4 does not expose `logits_fp16` (Track B re-export pending)"
        }
    }
}
