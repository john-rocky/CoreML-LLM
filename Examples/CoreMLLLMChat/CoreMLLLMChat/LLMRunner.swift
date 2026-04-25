import CoreML
import CoreMLLLM
import Foundation
import Tokenizers
#if canImport(UIKit)
import UIKit
#endif

/// Thin @Observable wrapper around CoreMLLLM for the chat app.
///
/// Delegates all inference to the CoreMLLLM package. Adds app-specific
/// features: benchmark, ANE verification, memory report.
@Observable
final class LLMRunner {
    var isLoaded = false
    var isGenerating = false
    var loadingStatus = "Not loaded"
    var tokensPerSecond: Double = 0
    var modelName = ""
    var hasVision = false
    var hasAudio = false
    var maxAudioDuration: TimeInterval = 10.0

    // MTP speculation metrics
    var mtpAcceptanceRate: Double = 0
    var mtpTokensPerRound: Double = 0
    var hasMTP: Bool { llm?.mtpAcceptanceRate != nil }

    // Cross-vocabulary (Qwen) speculation metrics — Route B
    var crossVocabAcceptanceRate: Double = 0
    var crossVocabTokensPerCycle: Double = 0

    private var llm: CoreMLLLM?
    private var modelFolderURL: URL?

    // Qwen3.5 path: separate generator + tokenizer, selected when the
    // downloaded folder contains `qwen3_5_0_8b_decode_fp16_mseq128.mlpackage`.
    // Not integrated into CoreMLLLM because Qwen3.5 uses a completely
    // different architecture (hybrid Gated-DeltaNet SSM + attention).
    private var qwen35Generator: Qwen35Generator?
    private var qwen35Tokenizer: (any Tokenizer)?

    // Qwen3-VL 2B stateful path (Phase 1 ship): MLState + slice_update,
    // selected when `qwen3_vl_2b_stateful_chunks/` is present in the
    // model folder. Runs alongside the v1.4.0 recurrent path; vision
    // stays on the v1.4.0 generator, text goes through this one.
    private var qwen3vl2bStatefulGenerator: Qwen3VL2BStatefulGenerator?

    // Gemma 4 E2B stateful path: MLState + slice_update KV. Selected
    // when `gemma4_e2b_stateful_chunks/` is present. Independent of the
    // legacy ChunkedEngine path — runs through Gemma4StatefulEngine
    // (Sources/CoreMLLLM/Gemma4StatefulEngine.swift). Both the Conv2d
    // wrapper and Linear (cml9 PR #2577) variants land in the same
    // engine — only the chunk_*.mlpackage internals differ.
    private var gemma4StatefulEngine: Gemma4StatefulEngine?
    private var gemma4StatefulTokenizer: (any Tokenizer)?

    // Qwen3-VL 2B path: separate generator + tokenizer, selected when
    // the downloaded folder contains `qwen3_vl_2b_decode_chunks/`.
    // Plain GQA architecture (not the Qwen3.5 hybrid SSM), so it gets
    // its own runtime — Qwen3VL2BGenerator hardcodes head_dim=128,
    // num_kv_heads=8, 6 body chunks + head + mmap embed sidecar.
    private var qwen3vl2bGenerator: Qwen3VL2BGenerator?
    private var qwen3vl2bTokenizer: (any Tokenizer)?
    /// Optional vision encoder paired with the VL2B decode chunks.
    /// Allocated alongside the generator when `vision.mlmodelc` /
    /// `vision.mlpackage` is present in the model folder.
    private var qwen3vl2bVisionEncoder: Qwen3VL2BVisionEncoder?

    /// Cache the most recently encoded image's features so repeat-turn
    /// generates with the same image skip the encoder run AND reuse
    /// the same Qwen3VL2BVisionFeatures instance — the generator keys
    /// its persisted KV cache on `features.hidden`'s ObjectIdentifier
    /// so a stable instance is required to hit the fast path.
    private var cachedVisionImage: CGImage?
    private var cachedVisionFeatures: Qwen3VL2BVisionFeatures?

    // MARK: - Loading

    func loadModel(from url: URL) async throws {
        let folder = url.deletingLastPathComponent()

        // Release previous engines BEFORE allocating a new one — peak footprint
        // on model switch would otherwise hold both in memory simultaneously,
        // OOMing on 8 GB devices.
        if llm != nil || qwen35Generator != nil || qwen3vl2bGenerator != nil
            || qwen3vl2bStatefulGenerator != nil
            || gemma4StatefulEngine != nil
        {
            llm = nil
            qwen35Generator = nil
            qwen35Tokenizer = nil
            qwen3vl2bGenerator = nil
            qwen3vl2bStatefulGenerator = nil
            qwen3vl2bTokenizer = nil
            qwen3vl2bVisionEncoder = nil
            gemma4StatefulEngine = nil
            gemma4StatefulTokenizer = nil
            cachedVisionImage = nil
            cachedVisionFeatures = nil
            isLoaded = false
            modelName = ""
            hasVision = false
            hasAudio = false
            mtpAcceptanceRate = 0
            mtpTokensPerRound = 0
            crossVocabAcceptanceRate = 0
            crossVocabTokensPerCycle = 0
            loadingStatus = "Releasing previous model..."
            await Task.yield()
        }

        modelFolderURL = folder
        loadingStatus = "Loading..."

        // Qwen3.5 detection: the downloaded folder contains the decode
        // mlpackage directly — no `model_config.json` / `hf_model/` layout
        // that Gemma uses. Accept any of the shipping variants, including
        // the 4-chunk + embed.bin 2B layout under `qwen3_5_2b_decode_chunks/`.
        // Chunked detection honors BOTH `.mlpackage` (from the HF download
        // path) and `.mlmodelc` (from `devicectl copy to` sideload) so we
        // can iterate on device without re-uploading to HF between runs.
        let chunksDir = folder.appendingPathComponent("qwen3_5_2b_decode_chunks")
        let fm = FileManager.default
        func chunkPresent(_ base: String) -> Bool {
            fm.fileExists(atPath: chunksDir.appendingPathComponent("\(base).mlpackage").path)
                || fm.fileExists(atPath: chunksDir.appendingPathComponent("\(base).mlmodelc").path)
        }
        let embedPresent = fm.fileExists(atPath:
            chunksDir.appendingPathComponent("embed_weight.bin").path)
        if embedPresent && ["chunk_a", "chunk_b", "chunk_c", "chunk_d"].allSatisfy(chunkPresent) {
            try await loadQwen35(folder: folder)
            return
        }
        for mlpkgName in ["qwen3_5_0_8b_decode_int8_mseq128.mlpackage",
                          "qwen3_5_0_8b_decode_fp16_mseq128.mlpackage",
                          "qwen3_5_2b_decode_int8_mseq128.mlpackage"] {
            let path = folder.appendingPathComponent(mlpkgName)
            if FileManager.default.fileExists(atPath: path.path) {
                try await loadQwen35(folder: folder)
                return
            }
        }

        // Qwen3-VL 2B STATEFUL detection (Phase 1): chunk_0..N +
        // chunk_head + embed_weight.bin under qwen3_vl_2b_stateful_chunks/.
        // Tolerate both layouts depending on what `localModelURL`
        // returned (outer model folder OR inner chunks subdir):
        //   A. folder = .../qwen3-vl-2b-stateful/        → chunks at folder + subdir
        //   B. folder = .../qwen3_vl_2b_stateful_chunks/ → chunks live in folder itself
        func statefulCandidates(_ base: URL) -> URL? {
            for cand in [
                base.appendingPathComponent("qwen3_vl_2b_stateful_chunks"),
                base,
            ] {
                let embed = cand.appendingPathComponent("embed_weight.bin")
                let head = cand.appendingPathComponent("chunk_head.mlpackage")
                let headC = cand.appendingPathComponent("chunk_head.mlmodelc")
                let c0 = cand.appendingPathComponent("chunk_0.mlpackage")
                let c0C = cand.appendingPathComponent("chunk_0.mlmodelc")
                if fm.fileExists(atPath: embed.path)
                    && (fm.fileExists(atPath: head.path) || fm.fileExists(atPath: headC.path))
                    && (fm.fileExists(atPath: c0.path) || fm.fileExists(atPath: c0C.path))
                {
                    return cand
                }
            }
            return nil
        }
        if let chunksRoot = statefulCandidates(folder) {
            // chunksRoot is the inner chunks dir; the generator's
            // resolveURLs expects the OUTER folder so it can append
            // its own subdir. Pass chunksRoot.deletingLastPathComponent().
            try await loadQwen3VL2BStateful(folder: chunksRoot.deletingLastPathComponent())
            return
        }

        // Gemma 4 E2B STATEFUL detection: chunk_{1..4}.mlpackage/.mlmodelc
        // + embed_tokens_q8.bin under gemma4_e2b_stateful_chunks/. Both
        // the Conv2d wrapper variant (folder=gemma4-e2b-stateful) and the
        // Linear variant (folder=gemma4-e2b-stateful-linear, Plan 3 A/B)
        // share the same internal layout — Gemma4StatefulEngine handles
        // both transparently because the only difference is the MIL graph
        // inside each chunk_*.mlpackage.
        let gemma4StatefulDir = folder.appendingPathComponent("gemma4_e2b_stateful_chunks")
        let gemma4StatefulPresent = fm.fileExists(atPath:
            gemma4StatefulDir.appendingPathComponent("embed_tokens_q8.bin").path)
            && (1...4).allSatisfy { i in
                fm.fileExists(atPath:
                    gemma4StatefulDir.appendingPathComponent("chunk_\(i).mlpackage").path)
                || fm.fileExists(atPath:
                    gemma4StatefulDir.appendingPathComponent("chunk_\(i).mlmodelc").path)
            }
        if gemma4StatefulPresent {
            try await loadGemma4Stateful(folder: gemma4StatefulDir)
            return
        }

        // Qwen3-VL 2B detection: 6 body chunks + chunk_head + embed
        // sidecar, all under qwen3_vl_2b_decode_chunks/.
        let vl2bDir = folder.appendingPathComponent("qwen3_vl_2b_decode_chunks")
        func vl2bChunkPresent(_ base: String) -> Bool {
            fm.fileExists(atPath: vl2bDir.appendingPathComponent("\(base).mlpackage").path)
                || fm.fileExists(atPath: vl2bDir.appendingPathComponent("\(base).mlmodelc").path)
        }
        let vl2bEmbedPresent = fm.fileExists(atPath:
            vl2bDir.appendingPathComponent("embed_weight.bin").path)
        let vl2bAllChunks = (0..<4).allSatisfy { vl2bChunkPresent("chunk_\($0)") }
            && vl2bChunkPresent("chunk_head")
        if vl2bEmbedPresent && vl2bAllChunks {
            try await loadQwen3VL2B(folder: folder)
            return
        }

        // LookAhead K=8 probe bundles ship a `probe.marker` file so the runner
        // can enable verify-chunk loading AND auto-route decode through
        // LookaheadEngine without requiring the user to set env vars in the
        // Xcode scheme. Normal bundles don't have this file, so their load
        // path is unchanged.
        let probeMarker = folder.appendingPathComponent("probe.marker")
        let isProbeBundle = FileManager.default.fileExists(atPath: probeMarker.path)
        if isProbeBundle {
            setenv("SPECULATIVE_PROFILE", "1", 1)
            setenv("LLM_LOOKAHEAD_ENABLE", "1", 1)
            print("[LLMRunner] probe.marker detected — SPECULATIVE_PROFILE=1 + LLM_LOOKAHEAD_ENABLE=1 forced")
        }

        llm = try await CoreMLLLM.load(from: folder) { [weak self] status in
            Task { @MainActor in
                self?.loadingStatus = status
            }
        }

        modelName = llm!.modelName
        hasVision = llm!.supportsVision
        hasAudio = llm!.supportsAudio
        maxAudioDuration = llm!.maxAudioDuration
        isLoaded = true
        loadingStatus = "Ready"
        print("[LLMRunner] loaded: vision=\(hasVision) audio=\(hasAudio) model=\(modelName)")

        // 11c iPhone bench (Task #9): when SPECULATIVE_PROFILE is set, switch
        // off the (incompatible-ctx) MTP drafter and route through the cross-vocab /
        // PLD union instead so the verify path is actually exercised. Without
        // this, the default mtpEnabled=true silently falls through to no-spec
        // when the MTP drafter mlmodel is incompatible with the engine config.
        if ProcessInfo.processInfo.environment["SPECULATIVE_PROFILE"] != nil {
            llm!.mtpEnabled = false
            llm!.drafterUnionEnabled = true
            llm!.crossVocabEnabled = true
            print("[LLMRunner] SPECULATIVE_PROFILE=1 — mtp=off union=on cv=on")
        }

        // 11c iPhone diagnostic: SPEC_OFF=1 disables ALL speculative paths so
        // we can measure pure serial decode speed (isolates ANE compile / chunk
        // perf from spec-engine overhead). Overrides SPECULATIVE_PROFILE.
        if ProcessInfo.processInfo.environment["SPEC_OFF"] != nil {
            llm!.mtpEnabled = false
            llm!.drafterUnionEnabled = false
            llm!.crossVocabEnabled = false
            print("[LLMRunner] SPEC_OFF=1 — pure serial decode")
        }
    }

    // MARK: - Generation

    func generate(messages: [ChatMessage], image: CGImage? = nil,
                  audio: [Float]? = nil) async throws -> AsyncStream<String> {
        if qwen35Generator != nil {
            return try await generateQwen35(messages: messages)
        }
        if qwen3vl2bStatefulGenerator != nil {
            return try await generateQwen3VL2BStateful(
                messages: messages, image: image)
        }
        if gemma4StatefulEngine != nil {
            return try await generateGemma4Stateful(messages: messages)
        }
        if qwen3vl2bGenerator != nil {
            return try await generateQwen3VL2B(messages: messages, image: image)
        }
        guard let llm else {
            throw NSError(domain: "LLMRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        isGenerating = true
        tokensPerSecond = 0

        let coreMessages = toCoreMessages(messages)
        let innerStream = try await llm.stream(coreMessages, image: image, audio: audio)
        return wrapStream(innerStream, engine: llm)
    }

    /// Variant that routes through Gemma 4's video chat template:
    /// frames sampled at `videoOptions.fps` (capped by `maxFrames`),
    /// optional audio from the same clip if `includeAudio` is set.
    func generate(messages: [ChatMessage], videoURL: URL,
                  videoOptions: VideoProcessor.Options) async throws -> AsyncStream<String> {
        guard let llm else {
            throw NSError(domain: "LLMRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        isGenerating = true
        tokensPerSecond = 0

        let coreMessages = toCoreMessages(messages)
        let innerStream = try await llm.stream(
            coreMessages, videoURL: videoURL, videoOptions: videoOptions)
        return wrapStream(innerStream, engine: llm)
    }

    private func toCoreMessages(_ messages: [ChatMessage]) -> [CoreMLLLM.Message] {
        messages.compactMap { m -> CoreMLLLM.Message? in
            switch m.role {
            case .user: return .init(role: .user, content: m.content)
            case .assistant: return .init(role: .assistant, content: m.content)
            case .system: return nil
            }
        }
    }

    private func wrapStream(_ inner: AsyncStream<String>,
                             engine: CoreMLLLM) -> AsyncStream<String> {
        let runner = self
        return AsyncStream { continuation in
            // `@Observable` state mutations must land on the main actor,
            // otherwise SwiftUI observation notifications can be dropped
            // and `isGenerating` stays `true` from the view's perspective
            // even after EOS — the symptom that manifested as
            // "ChatView unresponsive after turn 3 EOS". The Qwen path
            // already hops to @MainActor on its `isGenerating = false`
            // defer; this hoists the entire loop onto @MainActor so every
            // Observable mutation (tokensPerSecond, loadingStatus, …)
            // gets the same treatment with a single actor hop per token.
            Task { @MainActor in
                defer { runner.isGenerating = false }
                for await token in inner {
                    continuation.yield(token)
                    runner.tokensPerSecond = engine.tokensPerSecond
                    runner.mtpAcceptanceRate = engine.mtpAcceptanceRate
                    runner.mtpTokensPerRound = engine.mtpTokensPerRound
                    runner.crossVocabAcceptanceRate = engine.crossVocabAcceptanceRate
                    runner.crossVocabTokensPerCycle = engine.crossVocabTokensPerCycle
                }
                runner.loadingStatus = "Ready"
                continuation.finish()
            }
        }
    }

    func resetConversation() {
        llm?.reset()
        llm?.clearImageCache()
        // Qwen3.5 is stateless per-call (state is built recurrently from
        // scratch each generate), so nothing to reset here.
        // Qwen3-VL 2B stateful: drop the persisted KV cache + vision
        // feature cache so the next turn rebuilds from scratch.
        qwen3vl2bStatefulGenerator?.resetPersistedState()
        cachedVisionImage = nil
        cachedVisionFeatures = nil
    }

    // MARK: - Qwen3.5 dispatch

    private func loadQwen35(folder: URL) async throws {
        loadingStatus = "Loading Qwen tokenizer..."
        let tok = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3.5-0.8B")
        loadingStatus = "Compiling decode model (first run only, can take a few minutes on ANE)..."
        let gen = Qwen35Generator()
        gen.modelFolderOverride = folder
        // Trigger compile by calling load — falls back to lazy on first generate.
        do {
            try gen.load()
        } catch {
            throw NSError(domain: "LLMRunner", code: 20,
                userInfo: [NSLocalizedDescriptionKey:
                    "Qwen3.5 load failed: \(error.localizedDescription)"])
        }
        qwen35Generator = gen
        qwen35Tokenizer = tok
        // Display variant reflects which decode bundle shipped in the folder.
        // 2B 5-chunk takes precedence over 2B monolithic (the monolithic
        // path stays for Mac compatibility but fails iPhone ANE budget).
        // Honors both `.mlpackage` (HF download) and `.mlmodelc`
        // (devicectl sideload) for the chunked layout. Check chunk_a as
        // a representative marker — full presence is validated in the
        // router block above.
        let chunksDir = folder.appendingPathComponent("qwen3_5_2b_decode_chunks")
        let chunkAPkg = chunksDir.appendingPathComponent("chunk_a.mlpackage")
        let chunkAMlc = chunksDir.appendingPathComponent("chunk_a.mlmodelc")
        let mono2B = folder.appendingPathComponent("qwen3_5_2b_decode_int8_mseq128.mlpackage")
        if FileManager.default.fileExists(atPath: chunkAPkg.path)
            || FileManager.default.fileExists(atPath: chunkAMlc.path) {
            modelName = "Qwen3.5 2B (mmap embed + 4-chunk)"
        } else if FileManager.default.fileExists(atPath: mono2B.path) {
            modelName = "Qwen3.5 2B"
        } else {
            modelName = "Qwen3.5 0.8B"
        }
        hasVision = false
        hasAudio = false
        isLoaded = true
        loadingStatus = "Ready"
        print("[LLMRunner] loaded Qwen3.5 (\(modelName)) from \(folder.lastPathComponent)")
    }

    private func generateQwen35(messages: [ChatMessage]) async throws -> AsyncStream<String> {
        guard let gen = qwen35Generator, let tok = qwen35Tokenizer else {
            throw NSError(domain: "LLMRunner", code: 21,
                userInfo: [NSLocalizedDescriptionKey: "Qwen3.5 not loaded"])
        }
        isGenerating = true
        tokensPerSecond = 0

        // Apply Qwen's chat template — user/assistant turns wrapped in
        // <|im_start|>/<|im_end|> delimiters. SYSTEM messages are filtered
        // out because the app uses them for UI status ("Loading...",
        // "Model loaded!") — not actual model instructions. Leaving them
        // in confuses Qwen's instruct alignment and produces degenerate
        // looping output.
        let chatMessages: [[String: Any]] = messages.compactMap { m in
            let role: String
            switch m.role {
            case .user: role = "user"
            case .assistant: role = "assistant"
            case .system: return nil  // skip UI-status system messages
            }
            return ["role": role, "content": m.content]
        }
        let inputIds: [Int] = (try? tok.applyChatTemplate(messages: chatMessages))
            ?? tok.encode(text: messages.last?.content ?? "")
        let inputIdsInt32 = inputIds.map { Int32($0) }

        // Qwen3.5 decode mlpackage is built with max_seq=2048, so total
        // tokens (prompt + generation) must fit in 2048. Compute
        // remaining budget from the prompt length; cap at a reasonable
        // chat ceiling. If prompt alone already exceeds the budget,
        // throw a clear error.
        let maxSeq = 2048
        let remaining = maxSeq - inputIds.count - 1
        if remaining < 1 {
            throw NSError(domain: "LLMRunner", code: 22,
                userInfo: [NSLocalizedDescriptionKey:
                    "Qwen3.5 prompt (\(inputIds.count) tokens) exceeds max_seq=\(maxSeq). Shorten the message or clear the chat history."])
        }
        let maxNew = min(remaining, 1024)  // soft cap to avoid long hangs

        // Qwen3.5 has multiple stop tokens that all legitimately end a
        // turn. Stopping on any of them prevents the model from leaking
        // the text of a special token (e.g. "<|endoftext|>") into the
        // visible stream and then fabricating a new "Human:" turn.
        //   248044 = <|endoftext|>
        //   248045 = <|im_start|>   (start of next turn — we should stop)
        //   248046 = <|im_end|>     (end of current turn)
        var eosSet: Set<Int32> = [248044, 248045, 248046]
        if let eid = tok.eosTokenId { eosSet.insert(Int32(eid)) }

        let genStart = Date()
        return AsyncStream { continuation in
            Task { [weak self] in
                defer { Task { @MainActor in self?.isGenerating = false } }
                // Accumulated-decode streaming. Qwen BPE often splits multi-
                // byte UTF-8 (emoji, CJK glyphs) across multiple tokens —
                // decoding each token individually yields broken UTF-8 that
                // renders as mojibake (U+FFFD replacement characters). Keep
                // a growing buffer of token IDs, decode the full sequence
                // each step, and emit only the delta string. Cost is O(N²)
                // in decode bytes but negligible at chat token rates.
                var accumIds: [Int] = []
                var emittedText = ""
                var tokenCount = 0
                do {
                    // Plain greedy — Mac ANE bench (INT8 / FP16) shows no
                    // loops once the full Qwen EOS set is honored
                    // (248044/248045/248046). rep_penalty previously
                    // compensated for an EOS miss, not a real loop. Keep
                    // the path available by upping this arg when
                    // investigating.
                    _ = try await gen.generate(
                        inputIds: inputIdsInt32, maxNewTokens: maxNew,
                        temperature: 0.0, topK: 40, repetitionPenalty: 1.0,
                        eosTokenIds: eosSet,
                        onToken: { [weak self] tokenId in
                            tokenCount += 1
                            if eosSet.contains(tokenId) { return }
                            accumIds.append(Int(tokenId))
                            let current = tok.decode(tokens: accumIds)
                            if current.count > emittedText.count,
                               current.hasPrefix(emittedText) {
                                let delta = String(current.dropFirst(emittedText.count))
                                continuation.yield(delta)
                                emittedText = current
                            }
                            let elapsed = Date().timeIntervalSince(genStart)
                            if elapsed > 0 {
                                Task { @MainActor in
                                    self?.tokensPerSecond = Double(tokenCount) / elapsed
                                }
                            }
                        })
                } catch {
                    continuation.yield("[Error: \(error.localizedDescription)]")
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Qwen3-VL 2B dispatch

    private func loadQwen3VL2B(folder: URL) async throws {
        loadingStatus = "Loading Qwen3-VL tokenizer..."
        let tok = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3-VL-2B-Instruct")
        loadingStatus = "Compiling Qwen3-VL 2B chunks (first run only, can take 15+ minutes on ANE)..."
        let gen = Qwen3VL2BGenerator()
        gen.modelFolderOverride = folder
        do {
            try gen.load()
        } catch {
            throw NSError(domain: "LLMRunner", code: 30,
                userInfo: [NSLocalizedDescriptionKey:
                    "Qwen3-VL 2B load failed: \(error.localizedDescription)"])
        }
        qwen3vl2bGenerator = gen
        qwen3vl2bTokenizer = tok

        // Optional vision encoder (Phase 2c/2d ship). chunk_0_vision must
        // also be present for the end-to-end vision path to work; the
        // generator throws at generate() time if only one half of the
        // vision bundle is present.
        var visionTag = ""
        if let visionURL = Qwen3VL2BVisionEncoder.resolveModel(folder: folder) {
            let enc = Qwen3VL2BVisionEncoder()
            do {
                try enc.load(modelURL: visionURL)
                qwen3vl2bVisionEncoder = enc
                visionTag = gen.hasVisionChunk0 ? " + vision" : " (vision encoder only, no chunk_0_vision)"
            } catch {
                print("[LLMRunner] Qwen3-VL 2B vision encoder load failed: \(error)")
            }
        }

        modelName = "Qwen3-VL 2B\(visionTag)"
        // The chat UI enables its image picker on hasVision — we only
        // flip it true when BOTH the encoder and chunk_0_vision are
        // loaded so the picker isn't a dead end.
        hasVision = qwen3vl2bVisionEncoder != nil && gen.hasVisionChunk0
        hasAudio = false
        isLoaded = true
        loadingStatus = "Ready"
        print("[LLMRunner] loaded Qwen3-VL 2B\(visionTag) from \(folder.lastPathComponent)")
    }

    private func generateQwen3VL2B(messages: [ChatMessage],
                                    image: CGImage? = nil) async throws -> AsyncStream<String> {
        guard let gen = qwen3vl2bGenerator, let tok = qwen3vl2bTokenizer else {
            throw NSError(domain: "LLMRunner", code: 31,
                userInfo: [NSLocalizedDescriptionKey: "Qwen3-VL 2B not loaded"])
        }
        isGenerating = true
        tokensPerSecond = 0

        // ---- Build input IDs ----
        // Text-only path uses the stock chat template. Vision path
        // requires a `<|vision_start|><|image_pad|>×196<|vision_end|>`
        // block spliced in before the latest user message; we do this
        // by stringly building the prompt with the Qwen3-VL special
        // tokens, which the tokenizer encodes to their reserved IDs
        // (151652 / 151655 / 151653).
        let inputIdsInt32: [Int32]
        var visionFeatures: Qwen3VL2BVisionFeatures?
        if let image, let encoder = qwen3vl2bVisionEncoder {
            visionFeatures = try await encoder.encode(image)
            inputIdsInt32 = try buildQwen3VLVisionPromptIds(
                tokenizer: tok, history: messages)
        } else {
            let chatMessages: [[String: Any]] = messages.compactMap { m in
                let role: String
                switch m.role {
                case .user: role = "user"
                case .assistant: role = "assistant"
                case .system: return nil  // skip UI status messages
                }
                return ["role": role, "content": m.content]
            }
            let ids: [Int] = (try? tok.applyChatTemplate(messages: chatMessages))
                ?? tok.encode(text: messages.last?.content ?? "")
            inputIdsInt32 = ids.map { Int32($0) }
        }
        let inputIds = inputIdsInt32.map { Int($0) }

        // Qwen3-VL 2B chunks are built with max_seq=1024 and 3 body
        // chunks (12 layers each). Trades some tok/s vs the 512-ctx
        // build for enough room that typical chat turns don't truncate
        // at ~10 lines of Japanese.
        let maxSeq = 1024
        let remaining = maxSeq - inputIds.count - 1
        if remaining < 1 {
            throw NSError(domain: "LLMRunner", code: 32,
                userInfo: [NSLocalizedDescriptionKey:
                    "Qwen3-VL 2B prompt (\(inputIds.count) tokens) exceeds max_seq=\(maxSeq). Shorten the message or clear the chat history."])
        }
        let maxNew = min(remaining, 960)

        // Qwen3-VL EOS set: <|endoftext|>=151643, <|im_end|>=151645,
        // <|im_start|>=151644 (next-turn marker → also a stop).
        var eosSet: Set<Int32> = [151643, 151644, 151645]
        if let eid = tok.eosTokenId { eosSet.insert(Int32(eid)) }

        let genStart = Date()
        return AsyncStream { continuation in
            Task { [weak self] in
                defer { Task { @MainActor in self?.isGenerating = false } }
                var accumIds: [Int] = []
                var emittedText = ""
                var tokenCount = 0
                do {
                    _ = try await gen.generate(
                        inputIds: inputIdsInt32, maxNewTokens: maxNew,
                        temperature: 0.0, eosTokenIds: eosSet,
                        visionFeatures: visionFeatures,
                        onToken: { [weak self] tokenId in
                            tokenCount += 1
                            if eosSet.contains(tokenId) { return }
                            accumIds.append(Int(tokenId))
                            // Same accumulate-decode pattern as Qwen3.5 to
                            // preserve multi-byte UTF-8 across BPE token
                            // splits. Qwen3-VL has vocab=151K (vs Qwen3.5's
                            // 248K) so emoji/CJK BPE splits are common —
                            // the trailing token in `current` may be a
                            // partial UTF-8 sequence that decodes to U+FFFD
                            // (replacement char `◇`). Drop trailing FFFDs
                            // before emitting; the next token will complete
                            // the sequence and we'll emit the full glyph.
                            var current = tok.decode(tokens: accumIds)
                            while current.last == "\u{FFFD}" {
                                current.removeLast()
                            }
                            if current.count > emittedText.count,
                               current.hasPrefix(emittedText) {
                                let delta = String(current.dropFirst(emittedText.count))
                                continuation.yield(delta)
                                emittedText = current
                            }
                            let elapsed = Date().timeIntervalSince(genStart)
                            if elapsed > 0 {
                                Task { @MainActor in
                                    self?.tokensPerSecond = Double(tokenCount) / elapsed
                                }
                            }
                        })
                } catch {
                    continuation.yield("[Error: \(error.localizedDescription)]")
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Qwen3-VL 2B stateful dispatch (Phase 1 ship)

    private func loadQwen3VL2BStateful(folder: URL) async throws {
        loadingStatus = "Loading Qwen3-VL tokenizer..."
        let tok = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3-VL-2B-Instruct")
        loadingStatus = "Compiling Qwen3-VL 2B stateful chunks (first run only)..."
        let gen = Qwen3VL2BStatefulGenerator(cfg: .defaultFourChunk)
        gen.modelFolderOverride = folder
        try gen.load()
        qwen3vl2bStatefulGenerator = gen
        qwen3vl2bTokenizer = tok

        // Optional vision: encoder + chunk_0_vision must both be present
        // to enable image input. If either is missing, fall back to
        // text-only.
        var visionTag = ""
        if gen.hasVisionChunk,
           let visionURL = Qwen3VL2BVisionEncoder.resolveModel(folder: folder) {
            let enc = Qwen3VL2BVisionEncoder()
            do {
                try enc.load(modelURL: visionURL)
                qwen3vl2bVisionEncoder = enc
                visionTag = " + vision"
            } catch {
                print("[LLMRunner] stateful vision encoder load failed: \(error)")
            }
        }
        modelName = "Qwen3-VL 2B (stateful)\(visionTag)"
        hasVision = qwen3vl2bVisionEncoder != nil && gen.hasVisionChunk
        hasAudio = false

        // ANE pre-warm: front-load each chunk's first-call compile
        // (multi-second per chunk on the iPhone Neural Engine) so the
        // first user send doesn't pay it. Surface as a distinct status
        // string so the user knows what the wait is.
        loadingStatus = "Warming ANE..."
        do {
            let warmStart = Date()
            try await gen.prewarm()
            let warmMs = Int(Date().timeIntervalSince(warmStart) * 1000)
            print("[LLMRunner] Qwen3-VL 2B stateful prewarm: \(warmMs) ms")
        } catch {
            // Non-fatal: the first generate will simply pay the
            // compile cost itself. Log and continue.
            print("[LLMRunner] Qwen3-VL 2B stateful prewarm failed: \(error)")
        }

        isLoaded = true
        loadingStatus = "Ready"
        print("[LLMRunner] Qwen3-VL 2B stateful — \(gen.status)")
    }

    private func generateQwen3VL2BStateful(messages: [ChatMessage],
                                            image: CGImage? = nil
    ) async throws -> AsyncStream<String> {
        guard let gen = qwen3vl2bStatefulGenerator,
              let tok = qwen3vl2bTokenizer
        else {
            throw NSError(domain: "LLMRunner", code: 33,
                userInfo: [NSLocalizedDescriptionKey:
                    "Qwen3-VL 2B stateful not loaded"])
        }
        isGenerating = true
        tokensPerSecond = 0

        // Build prompt + run vision encoder if image is present.
        var visionFeatures: Qwen3VL2BVisionFeatures? = nil
        let inputIdsInt32: [Int32]
        if let image, let encoder = qwen3vl2bVisionEncoder {
            // Reuse cached features when the same CGImage instance is
            // sent again; encoding is the dominant non-prefill cost on
            // a fresh image (~hundreds of ms on iPhone 17 Pro). The
            // generator additionally keys its persisted KV state on the
            // returned features.hidden ObjectIdentifier, so handing
            // back the same struct is what makes turn-2 KV reuse fire.
            if let cachedImage = cachedVisionImage, cachedImage === image,
               let cached = cachedVisionFeatures {
                visionFeatures = cached
            } else {
                let f = try await encoder.encode(image)
                cachedVisionImage = image
                cachedVisionFeatures = f
                visionFeatures = f
            }
            inputIdsInt32 = try buildQwen3VLVisionPromptIds(
                tokenizer: tok, history: messages)
        } else {
            let chatMessages: [[String: Any]] = messages.compactMap { m in
                let role: String
                switch m.role {
                case .user: role = "user"
                case .assistant: role = "assistant"
                case .system: return nil
                }
                return ["role": role, "content": m.content]
            }
            let ids: [Int] = (try? tok.applyChatTemplate(messages: chatMessages))
                ?? tok.encode(text: messages.last?.content ?? "")
            inputIdsInt32 = ids.map { Int32($0) }
        }

        let maxSeq = 2048
        let remaining = maxSeq - inputIdsInt32.count - 1
        if remaining < 1 {
            throw NSError(domain: "LLMRunner", code: 34,
                userInfo: [NSLocalizedDescriptionKey:
                    "prompt (\(inputIdsInt32.count) tokens) exceeds max_seq=\(maxSeq). "
                    + "Clear chat or shorten."])
        }
        let maxNew = min(remaining, 1024)

        var eosSet: Set<Int32> = [151643, 151644, 151645]
        if let eid = tok.eosTokenId { eosSet.insert(Int32(eid)) }

        return AsyncStream { continuation in
            Task { [weak self] in
                defer { Task { @MainActor in self?.isGenerating = false } }
                var accumIds: [Int] = []
                var emittedText = ""
                var tokenCount = 0
                // Measure pure decode tok/s: skip prefill + vision encode
                // by anchoring on the first decode token.
                var firstTokenAt: Date?
                do {
                    _ = try await gen.generate(
                        inputIds: inputIdsInt32,
                        maxNewTokens: maxNew,
                        eosTokenIds: eosSet,
                        visionFeatures: visionFeatures,
                        onToken: { [weak self] tokenId in
                            tokenCount += 1
                            if eosSet.contains(tokenId) { return }
                            accumIds.append(Int(tokenId))
                            var current = tok.decode(tokens: accumIds)
                            while current.last == "\u{FFFD}" {
                                current.removeLast()
                            }
                            if current.count > emittedText.count,
                               current.hasPrefix(emittedText) {
                                let delta = String(current.dropFirst(emittedText.count))
                                continuation.yield(delta)
                                emittedText = current
                            }
                            if firstTokenAt == nil { firstTokenAt = Date() }
                            if let start = firstTokenAt {
                                let elapsed = Date().timeIntervalSince(start)
                                // Subtract 1 because the first token cost is
                                // bundled into the prefill window we're
                                // excluding from the denominator.
                                let n = max(tokenCount - 1, 0)
                                if elapsed > 0 && n > 0 {
                                    Task { @MainActor in
                                        self?.tokensPerSecond = Double(n) / elapsed
                                    }
                                }
                            }
                        })
                } catch {
                    continuation.yield("[Error: \(error.localizedDescription)]")
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Gemma 4 stateful (text-only)

    private func loadGemma4Stateful(folder: URL) async throws {
        loadingStatus = "Loading Gemma 4 tokenizer..."
        let hfDir = folder.appendingPathComponent("hf_model")
        let tok = try await AutoTokenizer.from(modelFolder: hfDir)
        loadingStatus = "Compiling Gemma 4 E2B stateful chunks (first run only)..."
        let engine = Gemma4StatefulEngine()
        try await engine.load(modelDirectory: folder)
        gemma4StatefulEngine = engine
        gemma4StatefulTokenizer = tok

        // Surface the variant in the model name so the user can tell A/B
        // apart in the UI. folder.path looks like
        // ".../Documents/Models/gemma4-e2b-stateful{,-linear}/gemma4_e2b_stateful_chunks"
        let parent = folder.deletingLastPathComponent().lastPathComponent
        let variantTag = parent.contains("linear") ? " (Linear)" : ""
        modelName = "Gemma 4 E2B (stateful)\(variantTag)"
        hasVision = false
        hasAudio = false
        isLoaded = true
        loadingStatus = "Ready"
        print("[LLMRunner] Gemma 4 E2B stateful loaded — \(modelName) at \(folder.lastPathComponent)")
    }

    private func generateGemma4Stateful(messages: [ChatMessage])
        async throws -> AsyncStream<String>
    {
        guard let engine = gemma4StatefulEngine,
              let tok = gemma4StatefulTokenizer
        else {
            throw NSError(domain: "LLMRunner", code: 41,
                userInfo: [NSLocalizedDescriptionKey:
                    "Gemma 4 stateful not loaded"])
        }
        isGenerating = true
        tokensPerSecond = 0

        // Build the Gemma 4 prompt manually — the tokenizer config in the
        // bundle has no chat_template, and Gemma 4 uses different turn
        // markers than Gemma 2/3 (`<|turn>` / `<turn|>` vs the older
        // `<start_of_turn>` / `<end_of_turn>`). Mirrors CoreMLLLM.swift's
        // buildGemmaPrompt path so the model sees the same token sequence
        // it was trained on.
        var prompt = "<bos>"
        for m in messages {
            switch m.role {
            case .user:
                prompt += "<|turn>user\n\(m.content)<turn|>\n"
            case .assistant:
                prompt += "<|turn>model\n\(m.content)<turn|>\n"
            case .system:
                continue  // skip UI status messages ("Model loaded!" etc.)
            }
        }
        prompt += "<|turn>model\n"
        let inputIds = tok.encode(text: prompt)
        let inputIdsInt32 = inputIds.map { Int32($0) }

        // Gemma 4 stop tokens: EOS (1) + <end_of_turn> (106).
        var eosSet: Set<Int32> = [1, 106]
        if let eid = tok.eosTokenId { eosSet.insert(Int32(eid)) }

        // Special tokens we never want to render as user-visible text.
        // The Engine's generate emits tokens BEFORE the next-iteration EOS
        // check breaks the loop, so without filtering here `<end_of_turn>`
        // (106) and `<start_of_turn>` (105) leak into the chat bubble.
        // Limited to verified Gemma 4 control tokens — earlier "safety"
        // additions were removed to avoid masking real vocabulary tokens
        // (e.g. emoji that share low IDs in some tokenizer revisions).
        let skipSet: Set<Int32> = [1, 105, 106]  // <eos>, <start_of_turn>, <end_of_turn>

        let genStart = Date()
        return AsyncStream { continuation in
            Task { [weak self] in
                defer { Task { @MainActor in self?.isGenerating = false } }
                // Accumulated-decode streaming for clean multi-byte UTF-8
                // (Gemma SentencePiece often splits CJK / emoji across
                // tokens; decoding individually produces mojibake).
                var accum: [Int] = []
                var emittedString = ""
                var totalEmitted = 0
                do {
                    _ = try await engine.generate(
                        inputIds: inputIdsInt32,
                        maxNewTokens: 256,
                        eosTokenIds: eosSet,
                        onToken: { tokenId in
                            if skipSet.contains(tokenId) { return }
                            accum.append(Int(tokenId))
                            let current = tok.decode(tokens: accum)
                            if current.count > emittedString.count {
                                let delta = String(
                                    current.suffix(current.count - emittedString.count))
                                continuation.yield(delta)
                                emittedString = current
                            }
                            totalEmitted += 1
                        })
                    let dt = Date().timeIntervalSince(genStart)
                    if dt > 0 {
                        let tps = Double(totalEmitted) / dt
                        Task { @MainActor in
                            self?.tokensPerSecond = tps
                        }
                    }
                } catch {
                    continuation.yield("[Error: \(error.localizedDescription)]")
                }
                continuation.finish()
            }
        }
    }

    /// Build the token ID sequence for a vision-augmented Qwen3-VL 2B
    /// prompt. Emits the same prefix the HF processor would produce for
    /// `[{role:"user", content:[{type:"image"},{type:"text", text:...}]}]`
    /// by wrapping 196 `<|image_pad|>` markers between
    /// `<|vision_start|>` / `<|vision_end|>` before the user's text.
    ///
    /// The vision block is injected into the FIRST user message so the
    /// image-pad span sits at a fixed sequence offset across turns —
    /// this lets Qwen3VL2BStatefulGenerator's per-session KV cache
    /// match by longest-common-prefix (turn 1's prompt becomes a strict
    /// prefix of turn 2's prompt). If the latest user turn carries a
    /// distinct image, the persisted state is invalidated upstream
    /// (vision fingerprint mismatch in the generator), so always
    /// pinning the block to the first user turn is safe.
    ///
    /// Special tokens (151644/151645/151652/151653/151655) are spliced
    /// in as raw IDs so we don't depend on the tokenizer recognizing
    /// them from a literal "<|…|>" string.
    private func buildQwen3VLVisionPromptIds(
        tokenizer tok: any Tokenizer,
        history: [ChatMessage]
    ) throws -> [Int32] {
        let imStart: Int32 = 151644
        let imEnd:   Int32 = 151645
        let visionStart: Int32 = 151652
        let visionEnd:   Int32 = 151653
        let imagePad:    Int32 = 151655
        let newline: [Int32] = tok.encode(text: "\n").map { Int32($0) }
        let userRole:      [Int32] = tok.encode(text: "user").map { Int32($0) }
        let assistantRole: [Int32] = tok.encode(text: "assistant").map { Int32($0) }

        let firstUserIdx = history.firstIndex(where: { $0.role == .user })

        var ids: [Int32] = []
        for (idx, m) in history.enumerated() {
            let role: [Int32]
            switch m.role {
            case .user:       role = userRole
            case .assistant:  role = assistantRole
            case .system:     continue
            }
            ids.append(imStart)
            ids += role
            ids += newline
            if idx == firstUserIdx {
                ids.append(visionStart)
                ids.append(contentsOf: Array(repeating: imagePad, count: 196))
                ids.append(visionEnd)
            }
            ids += tok.encode(text: m.content).map { Int32($0) }
            ids.append(imEnd)
            ids += newline
        }
        // Assistant preamble (no content yet — the model fills it in).
        ids.append(imStart)
        ids += assistantRole
        ids += newline
        return ids
    }

    // MARK: - Battery / sustained-throughput benchmark

    struct BenchmarkProgress {
        var elapsed: TimeInterval
        var totalTokens: Int
        var round: Int
        var avgTokPerSec: Double
        var batteryStart: Float
        var batteryNow: Float
        var thermal: ProcessInfo.ThermalState
    }

    struct BenchmarkResult {
        var duration: TimeInterval
        var totalTokens: Int
        var rounds: Int
        var avgTokPerSec: Double
        var batteryStart: Float
        var batteryEnd: Float
        var thermalStart: ProcessInfo.ThermalState
        var thermalEnd: ProcessInfo.ThermalState
        var abortedThermal: Bool = false
        var batteryLog: [(TimeInterval, Float)] = []

        var batteryDelta: Float { batteryStart - batteryEnd }
        var drainedPercent: Double { Double(batteryDelta) * 100.0 }
        var drainedPerMinute: Double { duration > 0 ? drainedPercent / (duration / 60.0) : 0 }
        var tokensPerPercent: Double { drainedPercent > 0 ? Double(totalTokens) / drainedPercent : 0 }
    }

    private static let benchmarkPrompt =
        "Write a very long, detailed article about the history of artificial intelligence from the 1950s through today. Cover: early symbolic AI and Alan Turing, the first and second AI winters, the rise of neural networks, deep learning breakthroughs like AlexNet and ResNet, the attention mechanism and transformers, the scaling era with GPT-2/3/4, reinforcement learning milestones, and the current era of multimodal foundation models running on-device. Be verbose and thorough."

    #if os(iOS)
    @MainActor
    func runBenchmark(
        duration: TimeInterval,
        onProgress: @escaping (BenchmarkProgress) -> Void
    ) async throws -> BenchmarkResult {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let startBat = UIDevice.current.batteryLevel
        let startThermal = ProcessInfo.processInfo.thermalState
        let startTime = Date()

        var totalTokens = 0
        var round = 0
        var abortedThermal = false
        var batteryLog: [(TimeInterval, Float)] = [(0, startBat)]
        var lastLoggedLevel = startBat
        let prompt = ChatMessage(role: .user, content: Self.benchmarkPrompt)

        func isThermalUnsafe() -> Bool {
            let s = ProcessInfo.processInfo.thermalState
            return s == .serious || s == .critical
        }

        while Date().timeIntervalSince(startTime) < duration {
            if isThermalUnsafe() { abortedThermal = true; break }
            round += 1
            let stream = try await generate(messages: [prompt], image: nil)
            for await _ in stream {
                totalTokens += 1
                let elapsed = Date().timeIntervalSince(startTime)
                let currentLevel = UIDevice.current.batteryLevel
                if currentLevel >= 0 && currentLevel != lastLoggedLevel {
                    batteryLog.append((elapsed, currentLevel))
                    lastLoggedLevel = currentLevel
                }
                if totalTokens % 20 == 0 {
                    onProgress(BenchmarkProgress(
                        elapsed: elapsed, totalTokens: totalTokens, round: round,
                        avgTokPerSec: elapsed > 0 ? Double(totalTokens) / elapsed : 0,
                        batteryStart: startBat, batteryNow: currentLevel,
                        thermal: ProcessInfo.processInfo.thermalState))
                }
                if elapsed >= duration { break }
                if isThermalUnsafe() { abortedThermal = true; break }
            }
            if abortedThermal { break }
            if Date().timeIntervalSince(startTime) >= duration { break }
        }

        let endTime = Date()
        let endBat = UIDevice.current.batteryLevel
        let dur = endTime.timeIntervalSince(startTime)
        batteryLog.append((dur, endBat))
        return BenchmarkResult(
            duration: dur, totalTokens: totalTokens, rounds: round,
            avgTokPerSec: dur > 0 ? Double(totalTokens) / dur : 0,
            batteryStart: startBat, batteryEnd: endBat,
            thermalStart: startThermal, thermalEnd: ProcessInfo.processInfo.thermalState,
            abortedThermal: abortedThermal, batteryLog: batteryLog)
    }
    #endif

    static func thermalString(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "?"
        }
    }

    // MARK: - ANE placement verification

    @available(iOS 17.0, macOS 14.0, *)
    func verifyANEPlacement() async -> String {
        guard let folder = modelFolderURL else {
            return "No model folder (load a model first)."
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        let visionCfg = MLModelConfiguration()
        visionCfg.computeUnits = .cpuAndGPU

        struct Entry { let label: String; let url: URL; let cfg: MLModelConfiguration }
        var entries: [Entry] = []
        let names = ["chunk1", "chunk2", "chunk3", "chunk4",
                     "prefill_chunk1", "prefill_chunk2", "prefill_chunk3", "prefill_chunk4"]
        for name in names {
            if let u = findModel(in: folder, name: name) {
                entries.append(Entry(label: name, url: u, cfg: cfg))
            }
        }
        if let u = findModel(in: folder, name: "vision") {
            entries.append(Entry(label: "vision", url: u, cfg: visionCfg))
        }
        if entries.isEmpty { return "No chunks found." }

        var lines: [String] = ["MLComputePlan placement:"]
        var tAll = 0, aAll = 0, gAll = 0, cAll = 0
        for e in entries {
            do {
                let plan = try await MLComputePlan.load(contentsOf: e.url, configuration: e.cfg)
                let (total, ane, gpu, cpu) = countOps(plan: plan)
                tAll += total; aAll += ane; gAll += gpu; cAll += cpu
                let dispatched = ane + gpu + cpu
                let pct = dispatched > 0 ? Int((Double(ane) / Double(dispatched) * 100).rounded()) : 0
                let label = e.label.padding(toLength: 16, withPad: " ", startingAt: 0)
                lines.append("  \(label) \(ane)/\(dispatched) ANE (\(pct)%)  GPU=\(gpu) CPU=\(cpu)")
            } catch {
                lines.append("  \(e.label): failed — \(error.localizedDescription)")
            }
        }
        let dAll = aAll + gAll + cAll
        let pAll = dAll > 0 ? Int((Double(aAll) / Double(dAll) * 100).rounded()) : 0
        lines.append("  TOTAL            \(aAll)/\(dAll) ANE (\(pAll)%)  GPU=\(gAll) CPU=\(cAll)")

        #if os(iOS)
        lines.append("")
        lines.append(memoryReport())
        #endif
        return lines.joined(separator: "\n")
    }

    func memoryReport() -> String {
        var lines = [String]()
        #if os(iOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        let state = UIDevice.current.batteryState
        let stateStr = state == .charging ? "charging" : state == .full ? "full" : "unplugged"
        lines.append("Battery: \(level >= 0 ? "\(Int(level * 100))%" : "?") (\(stateStr)), thermal: \(Self.thermalString(ProcessInfo.processInfo.thermalState))")
        #endif
        lines.append("Memory (task_vm_info):")
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let phys = Double(info.phys_footprint) / 1024 / 1024
            let resident = Double(info.resident_size) / 1024 / 1024
            let compressed = Double(info.compressed) / 1024 / 1024
            lines.append("  phys_footprint: \(String(format: "%.1f", phys)) MB  resident: \(String(format: "%.1f", resident)) MB  compressed: \(String(format: "%.1f", compressed)) MB")
        }
        let available = os_proc_available_memory()
        lines.append("  os_proc_available: \(available / 1024 / 1024) MB")
        return lines.joined(separator: "\n")
    }

    // MARK: - Private helpers

    private func findModel(in folder: URL, name: String) -> URL? {
        let compiled = folder.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
        let pkg = folder.appendingPathComponent("\(name).mlpackage")
        if FileManager.default.fileExists(atPath: pkg.path) { return pkg }
        return nil
    }

    @available(iOS 17.0, macOS 14.0, *)
    private func countOps(plan: MLComputePlan) -> (total: Int, ane: Int, gpu: Int, cpu: Int) {
        // Multi-function chunks (build_verify_chunks.py output) name their
        // entry points "decode_q1" / "verify_qK", not "main". Fall through to
        // any available function so audit works across both layouts.
        guard case let .program(program) = plan.modelStructure else {
            return (0, 0, 0, 0)
        }
        let fn = program.functions["decode_q1"]
            ?? program.functions["main"]
            ?? program.functions.values.first
        guard let function = fn else { return (0, 0, 0, 0) }
        var total = 0, ane = 0, gpu = 0, cpu = 0
        var stack: [MLModelStructure.Program.Block] = [function.block]
        while let block = stack.popLast() {
            for op in block.operations {
                total += 1
                if let usage = plan.deviceUsage(for: op) {
                    switch usage.preferred {
                    case .neuralEngine: ane += 1
                    case .gpu:          gpu += 1
                    case .cpu:          cpu += 1
                    @unknown default:   break
                    }
                }
                for inner in op.blocks { stack.append(inner) }
            }
        }
        return (total, ane, gpu, cpu)
    }
}
