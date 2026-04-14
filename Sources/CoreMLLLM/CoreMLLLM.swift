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

    // Generation metrics
    public private(set) var tokensPerSecond: Double = 0
    public var mtpAcceptanceRate: Double { mtpEngine?.acceptanceRate ?? 0 }
    public var mtpTokensPerRound: Double { mtpEngine?.tokensPerRound ?? 0 }

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

        let mutableSelf = self
        let imgFeats = imageFeatures
        let imgTokenLimit = imageNumTokens
        let audFeats = audioFeatures
        let tokens = tokenIDs
        let audTokenCount = audioTokenCount
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
                        if tid == IMAGE_TOKEN_ID, let f = imgFeats, imageIdx < imgTokenLimit {
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
                                    imageNumTokens: imgTokenLimit,
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
                        let specEngine = mutableSelf.mtpEnabled ? mutableSelf.mtpEngine : nil
                        specEngine?.reset()

                        var nid = Int32(nextID)
                        while tokenCount < maxDecode {
                            if eosIDs.contains(Int(nid)) { break }
                            if engine.currentPosition >= ctxLimit { break }

                            if let se = specEngine, se.shouldSpeculate {
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
                            } else {
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
        tokensPerSecond = 0
    }

    /// Clear cached image features (called between conversations).
    public func clearImageCache() {
        cachedImageFeatures = nil
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

    /// Build the per-frame video block for the Gemma 4 chat template:
    ///   `MM:SS <|image><|image|>×K<image|>` joined by single spaces.
    private func buildVideoBlock(timestamps: [Double], tokensPerFrame: Int) -> String {
        let placeholder = String(repeating: "<|image|>", count: tokensPerFrame)
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
    ///   - `tokensPerFrame = 256`: raw passthrough (first 256 of 280).
    ///   - `tokensPerFrame = 64`:  2×2 average-pool the 16×16 grid to 8×8.
    ///   - other:                   first `tokensPerFrame` tokens of the 280
    ///                              (not semantically meaningful — debug only).
    private func concatFrameFeatures(_ frames: [CGImage],
                                      tokensPerFrame: Int) throws -> MLMultiArray {
        precondition(!frames.isEmpty)
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

    public var errorDescription: String? {
        switch self {
        case .configNotFound: return "model_config.json not found"
        case .predictionFailed: return "Model prediction failed"
        case .modelNotFound(let name): return "Model file not found: \(name)"
        case .prefillNotAvailable: return "Prefill chunks not loaded"
        case .visionNotAvailable: return "Vision model not available"
        case .audioNotAvailable: return "Audio model not available"
        case .videoDecodeFailed: return "Could not decode any frames from video"
        }
    }
}
