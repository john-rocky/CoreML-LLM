import CoreML
import CoreMLLLM
import Foundation
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

    // MARK: - Loading

    func loadModel(from url: URL) async throws {
        let folder = url.deletingLastPathComponent()
        modelFolderURL = folder
        loadingStatus = "Loading..."

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
    }

    // MARK: - Generation

    func generate(messages: [ChatMessage], image: CGImage? = nil,
                  audio: [Float]? = nil) async throws -> AsyncStream<String> {
        guard let llm else {
            throw NSError(domain: "LLMRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        isGenerating = true
        tokensPerSecond = 0

        // Convert app ChatMessage → CoreMLLLM.Message
        let coreMessages = messages.compactMap { m -> CoreMLLLM.Message? in
            switch m.role {
            case .user: return .init(role: .user, content: m.content)
            case .assistant: return .init(role: .assistant, content: m.content)
            case .system: return nil
            }
        }

        let innerStream = try await llm.stream(coreMessages, image: image, audio: audio)
        let runner = self
        let engine = llm

        return AsyncStream { continuation in
            Task {
                defer { runner.isGenerating = false }
                for await token in innerStream {
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

    struct ThermalSample {
        var t: TimeInterval
        var state: ProcessInfo.ThermalState
        var batteryLevel: Float
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
        var thermalTrajectory: [LLMRunner.ThermalSample] = []

        // iPhone 17 Pro nominal battery capacity. Override for other devices.
        // Source: Apple spec sheet (14.03 Wh = 50508 J).
        var batteryCapacityWh: Double = 14.03

        var batteryDelta: Float { batteryStart - batteryEnd }
        var drainedPercent: Double { Double(batteryDelta) * 100.0 }
        var drainedPerMinute: Double { duration > 0 ? drainedPercent / (duration / 60.0) : 0 }
        var drainedPerHour: Double { drainedPerMinute * 60.0 }
        var tokensPerPercent: Double { drainedPercent > 0 ? Double(totalTokens) / drainedPercent : 0 }

        /// Energy per decoded token in millijoules, derived from battery-gauge delta.
        /// Coarse (1% gauge resolution); trust only for runs ≥ 10 min.
        var mJPerToken: Double {
            guard totalTokens > 0, drainedPercent > 0 else { return 0 }
            let joules = drainedPercent / 100.0 * batteryCapacityWh * 3600.0
            return joules * 1000.0 / Double(totalTokens)
        }

        var timeToFair: TimeInterval? {
            thermalTrajectory.first { $0.state == .fair || $0.state == .serious || $0.state == .critical }?.t
        }
        var timeToSerious: TimeInterval? {
            thermalTrajectory.first { $0.state == .serious || $0.state == .critical }?.t
        }

        func csv() -> String {
            var lines = ["t_seconds,battery_pct,thermal_state,source"]
            for s in thermalTrajectory {
                let pct = s.batteryLevel >= 0 ? Int(s.batteryLevel * 100) : -1
                lines.append("\(Int(s.t)),\(pct),\(LLMRunner.thermalString(s.state)),thermal")
            }
            for (t, lvl) in batteryLog {
                let pct = lvl >= 0 ? Int(lvl * 100) : -1
                lines.append("\(Int(t)),\(pct),,battery")
            }
            lines.append("")
            lines.append("# summary")
            lines.append("# duration_s=\(Int(duration))")
            lines.append("# total_tokens=\(totalTokens)")
            lines.append("# avg_tok_per_sec=\(String(format: "%.2f", avgTokPerSec))")
            lines.append("# drained_percent=\(String(format: "%.2f", drainedPercent))")
            lines.append("# drained_per_hour=\(String(format: "%.2f", drainedPerHour))")
            lines.append("# mJ_per_token=\(String(format: "%.2f", mJPerToken))")
            lines.append("# time_to_fair_s=\(timeToFair.map { String(Int($0)) } ?? "never")")
            lines.append("# time_to_serious_s=\(timeToSerious.map { String(Int($0)) } ?? "never")")
            lines.append("# thermal_start=\(LLMRunner.thermalString(thermalStart))")
            lines.append("# thermal_end=\(LLMRunner.thermalString(thermalEnd))")
            lines.append("# aborted_thermal=\(abortedThermal)")
            lines.append("# battery_capacity_wh=\(batteryCapacityWh)")
            return lines.joined(separator: "\n")
        }
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
        var thermalTrajectory: [ThermalSample] = [
            ThermalSample(t: 0, state: startThermal, batteryLevel: startBat)
        ]
        var nextThermalSampleAt: TimeInterval = 30
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
                if elapsed >= nextThermalSampleAt {
                    thermalTrajectory.append(ThermalSample(
                        t: elapsed,
                        state: ProcessInfo.processInfo.thermalState,
                        batteryLevel: currentLevel))
                    nextThermalSampleAt += 30
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
        let endThermal = ProcessInfo.processInfo.thermalState
        let dur = endTime.timeIntervalSince(startTime)
        batteryLog.append((dur, endBat))
        thermalTrajectory.append(ThermalSample(t: dur, state: endThermal, batteryLevel: endBat))
        return BenchmarkResult(
            duration: dur, totalTokens: totalTokens, rounds: round,
            avgTokPerSec: dur > 0 ? Double(totalTokens) / dur : 0,
            batteryStart: startBat, batteryEnd: endBat,
            thermalStart: startThermal, thermalEnd: endThermal,
            abortedThermal: abortedThermal, batteryLog: batteryLog,
            thermalTrajectory: thermalTrajectory)
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
        guard case let .program(program) = plan.modelStructure,
              let main = program.functions["main"] else {
            return (0, 0, 0, 0)
        }
        var total = 0, ane = 0, gpu = 0, cpu = 0
        var stack: [MLModelStructure.Program.Block] = [main.block]
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
