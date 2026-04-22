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

    // MARK: - Loading

    func loadModel(from url: URL) async throws {
        let folder = url.deletingLastPathComponent()

        // Release previous engines BEFORE allocating a new one — peak footprint
        // on model switch would otherwise hold both in memory simultaneously,
        // OOMing on 8 GB devices.
        if llm != nil || qwen35Generator != nil {
            llm = nil
            qwen35Generator = nil
            qwen35Tokenizer = nil
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
        // that Gemma uses. Accept either the INT8 (default shipping) or
        // fp16 (legacy / high-precision) variant.
        let qwen35Int8 = folder.appendingPathComponent(
            "qwen3_5_0_8b_decode_int8_mseq128.mlpackage")
        let qwen35Fp16 = folder.appendingPathComponent(
            "qwen3_5_0_8b_decode_fp16_mseq128.mlpackage")
        if FileManager.default.fileExists(atPath: qwen35Int8.path) ||
           FileManager.default.fileExists(atPath: qwen35Fp16.path) {
            try await loadQwen35(folder: folder)
            return
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
            Task {
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
        modelName = "Qwen3.5 0.8B"
        hasVision = false
        hasAudio = false
        isLoaded = true
        loadingStatus = "Ready"
        print("[LLMRunner] loaded Qwen3.5 from \(folder.lastPathComponent)")
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

        // Qwen3.5 decode mlpackage is built with max_seq=128, so total
        // tokens (prompt + generation) must fit in 128. Compute remaining
        // budget from the prompt length; cap at a reasonable chat ceiling.
        // If prompt alone already exceeds the budget, throw a clear error.
        let maxSeq = 128
        let remaining = maxSeq - inputIds.count - 1
        if remaining < 1 {
            throw NSError(domain: "LLMRunner", code: 22,
                userInfo: [NSLocalizedDescriptionKey:
                    "Qwen3.5 prompt (\(inputIds.count) tokens) exceeds max_seq=\(maxSeq). Shorten the message or clear the chat history."])
        }
        let maxNew = min(remaining, 120)  // soft cap to avoid long hangs

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
