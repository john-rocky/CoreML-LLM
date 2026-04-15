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

    // Vision (lazy loaded to save memory)
    private var visionModel: MLModel?
    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?

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

    /// The compute profile this instance was loaded with. Reported as
    /// `.custom(units)` when the caller used the legacy `computeUnits:`
    /// parameter. Inspected by the power-bench harness and the UI badge.
    public private(set) var computeProfile: ComputeProfile = .efficient

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
        try await load(from: directory,
                       profile: ComputeProfile.from(computeUnits),
                       onProgress: onProgress)
    }

    /// Load a model with a semantic `ComputeProfile`.
    ///
    /// - Parameters:
    ///   - directory: Folder containing model files, embeddings, config
    ///   - profile: `.efficient` (ANE, default), `.balanced` (`.all`),
    ///              `.performance` (GPU), or `.custom(MLComputeUnits)`.
    ///   - onProgress: Optional callback for loading status updates
    public static func load(
        from directory: URL,
        profile: ComputeProfile,
        onProgress: ((String) -> Void)? = nil
    ) async throws -> CoreMLLLM {
        let computeUnits = profile.mlComputeUnits
        onProgress?("Reading config (profile=\(profile.rawIdentifier))...")
        let config = try ModelConfig.load(from: directory)

        // Tokenizer
        onProgress?("Loading tokenizer...")
        let tokDir = directory.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)

        let llm = CoreMLLLM(config: config, tokenizer: tokenizer)
        llm.computeProfile = profile

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
        } else {
            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = computeUnits
            // V6-1: fixed-shape hint (iOS 18.2+). Skips per-call shape trace.
            if #available(iOS 18.2, macOS 15.2, *) {
                mlConfig.optimizationHints.reshapeFrequency = .infrequent
            }
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
        // model directory. Silently skipped if absent.
        let cvDir = directory.appendingPathComponent("cross_vocab")
        let cvCompiled = cvDir.appendingPathComponent("qwen_drafter.mlmodelc")
        let cvPkg = cvDir.appendingPathComponent("qwen_drafter.mlpackage")
        let cvMapURL = cvDir.appendingPathComponent("qwen_gemma_vocab.bin")
        let cvModelURL: URL? = FileManager.default.fileExists(atPath: cvCompiled.path) ? cvCompiled
            : FileManager.default.fileExists(atPath: cvPkg.path) ? cvPkg : nil
        if let cvModelURL,
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

        // Vision model (optional, lazy loaded on first image)
        let visionCompiled = directory.appendingPathComponent("vision.mlmodelc")
        let visionPkg = directory.appendingPathComponent("vision.mlpackage")
        if FileManager.default.fileExists(atPath: visionCompiled.path) {
            llm.visionModelURL = visionCompiled
        } else if FileManager.default.fileExists(atPath: visionPkg.path) {
            llm.visionModelURL = visionPkg
        }
        if llm.visionModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            llm.visionConfig = cfg
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
        try await load(model: model,
                       profile: ComputeProfile.from(computeUnits),
                       onProgress: onProgress)
    }

    /// Download (if needed) and load a model with a semantic `ComputeProfile`.
    public static func load(
        model: ModelDownloader.ModelInfo,
        profile: ComputeProfile,
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
        return try await load(from: directory, profile: profile,
                               onProgress: onProgress)
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
            imageFeatures = try processImage(image)
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
        let tokenIDs = tokenizer.encode(text: chatPrompt)

        reset()

        // Clear recording buffers and record the prompt IDs for this turn.
        self.lastPromptTokenIDs = tokenIDs.map { Int32($0) }
        self.lastEmittedTokenIDs = []

        let mutableSelf = self
        let imgFeats = imageFeatures
        let audFeats = audioFeatures
        let tokens = tokenIDs
        let audTokenCount = actualAudioTokens
        let ctxLimit = config.contextLength

        return AsyncStream { continuation in
            Task {
                do {
                    let IMAGE_TOKEN_ID = 258880
                    let AUDIO_TOKEN_ID = 258881
                    var imageIdx = 0
                    var audioIdx = 0
                    var nextID = 0

                    func multimodalEmbedding(for tid: Int) -> MLMultiArray? {
                        if tid == IMAGE_TOKEN_ID, let f = imgFeats, imageIdx < 256 {
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
                        let prefillLen = min(tokens.count, engine.prefillN)
                        let useHybrid = engine.hasPrefill && prefillLen > 0

                        if useHybrid {
                            try autoreleasepool {
                                let batch = Array(tokens[0..<prefillLen])
                                nextID = try engine.runPrefill(
                                    tokenIDs: batch,
                                    imageFeatures: imgFeats,
                                    audioFeatures: audFeats,
                                    audioNumTokens: audTokenCount
                                )
                            }
                            imageIdx = tokens[0..<prefillLen].filter { $0 == IMAGE_TOKEN_ID }.count
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
                        //   1. MTP (trained drafter, best when present)
                        //   2. DrafterUnion (Phase B; cv + pld-n2 + pld-n3)
                        //   3. CrossVocab alone (legacy, kept as opt-out fallback)
                        // Only the selected engine resets — the union and the
                        // CV-alone engine share an underlying CrossVocabDraft,
                        // so simultaneous use would corrupt Qwen state.
                        let mtpSpec = mutableSelf.mtpEnabled ? mutableSelf.mtpEngine : nil
                        let unionSpec = (mtpSpec == nil && mutableSelf.drafterUnionEnabled)
                            ? mutableSelf.drafterUnion : nil
                        let cvSpec = (mtpSpec == nil && unionSpec == nil
                                      && mutableSelf.crossVocabEnabled)
                            ? mutableSelf.crossVocabEngine : nil
                        mtpSpec?.reset()
                        unionSpec?.reset()
                        unionSpec?.setPrefillHistory(tokens.map { Int32($0) })
                        cvSpec?.reset()
                        cvSpec?.setPrefillHistory(tokens.map { Int32($0) })

                        var nid = Int32(nextID)
                        while tokenCount < maxDecode {
                            if eosIDs.contains(Int(nid)) { break }
                            if engine.currentPosition >= ctxLimit { break }

                            if let se = mtpSpec, se.shouldSpeculate {
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
        return try ImageProcessor.process(image, with: vm)
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
        case .verifyLogitsNotExposed:
            return "verify chunk 4 does not expose `logits_fp16` (Track B re-export pending)"
        }
    }
}
