// Minimal CLI smoke test for CoreMLLLM on Mac.
//
// Usage:
//   swift run -c release coreml-llm-smoke <model-dir> [prompt] [maxTokens]
//
// Goals:
//   * Prove the library loads a chunked Gemma model end-to-end on Mac.
//   * Verify no regression from the Route B drafter refactor — baseline
//     decode must still work when no cross-vocab / MTP drafter is present.
//   * Emit tok/s so we can see whether the decode path is healthy.

import CoreML
import CoreMLLLM
import Foundation

@main
struct Smoke {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            fputs("usage: \(args[0]) <model-dir-or-repo> [prompt] [maxTokens]\n", stderr)
            fputs("       repo form: org/name  (e.g. mlboydaisuke/lfm2.5-350m-coreml)\n", stderr)
            fputs("       or model id           (e.g. lfm2.5-350m)\n", stderr)
            exit(2)
        }
        let target = args[1]
        let prompt = args.count >= 3
            ? args[2]
            : "Write three short sentences about the ocean."
        let maxTokens = args.count >= 4 ? (Int(args[3]) ?? 64) : 64

        // "repo or id" vs "path": an HF repo path / registered model id
        // doesn't begin with `/`, `./`, `../`, and isn't an existing entry
        // on disk.
        let looksLikeRepo =
            !target.hasPrefix("/") && !target.hasPrefix("./") && !target.hasPrefix("../")
            && !FileManager.default.fileExists(atPath: target)

        // Allow forcing target compute unit via SMOKE_TARGET_DEVICE env.
        let smokeUnits: MLComputeUnits
        switch ProcessInfo.processInfo.environment["SMOKE_TARGET_DEVICE"] {
        case "cpu":      smokeUnits = .cpuOnly
        case "gpu":      smokeUnits = .cpuAndGPU
        case "ane":      smokeUnits = .cpuAndNeuralEngine
        case "all":      smokeUnits = .all
        default:         smokeUnits = .cpuAndNeuralEngine
        }

        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            let llm: CoreMLLLM
            let onProgress: (String) -> Void = { print("[load] \($0)") }
            if looksLikeRepo {
                print("[smoke] loading via repo: \(target)  device=\(smokeUnits)")
                llm = try await CoreMLLLM.load(
                    repo: target, computeUnits: smokeUnits, onProgress: onProgress)
            } else {
                let modelDir = URL(fileURLWithPath: target)
                print("[smoke] loading model from: \(modelDir.path)  device=\(smokeUnits)")
                llm = try await CoreMLLLM.load(
                    from: modelDir, computeUnits: smokeUnits, onProgress: onProgress)
            }
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print("[smoke] loaded in \(String(format: "%.1f", dt))s — model=\(llm.modelName)")

            // TEMP Phase B trip: mirror LLMRunner iPhone-trip overrides so
            // the Mac CLI exercises the same DrafterUnion + CV path.
            if ProcessInfo.processInfo.environment["UNION_TRIP"] != nil {
                llm.mtpEnabled = false
                llm.drafterUnionEnabled = true
                llm.crossVocabEnabled = true
                print("[smoke] UNION_TRIP=1 — mtp=off union=on cv=on")
            }

            // MTP sampling temperature overrides for free-form sampling tests.
            // MTP_TEMPERATURE = target T (rejection acceptance scale).
            // MTP_DRAFTER_TEMP = drafter T (overrides MTP_TEMPERATURE if set).
            if let s = ProcessInfo.processInfo.environment["MTP_TEMPERATURE"],
               let v = Float(s) {
                llm.mtpSamplingTemperature = v
                print("[smoke] MTP_TEMPERATURE=\(v) — sampling on")
            }
            if let s = ProcessInfo.processInfo.environment["MTP_DRAFTER_TEMP"],
               let v = Float(s) {
                llm.mtpDrafterTemperature = v
                print("[smoke] MTP_DRAFTER_TEMP=\(v)")
            }

            print("[smoke] prompt: \(prompt)")
            print("[smoke] max_tokens=\(maxTokens)")

            var collected = ""
            let stream = try await llm.stream(prompt, maxTokens: maxTokens)
            for await tok in stream {
                collected += tok
                FileHandle.standardOutput.write(Data(tok.utf8))
            }
            print("\n---")
            print("[smoke] tok/s = \(String(format: "%.2f", llm.tokensPerSecond))")
            print("[smoke] output length = \(collected.count) chars")
            print("[smoke] mtp accept = \(String(format: "%.2f", llm.mtpAcceptanceRate))")
            print("[smoke] cross-vocab accept = \(String(format: "%.2f", llm.crossVocabAcceptanceRate))")
            exit(0)
        } catch {
            fputs("[smoke] error: \(error)\n", stderr)
            exit(1)
        }
    }
}
