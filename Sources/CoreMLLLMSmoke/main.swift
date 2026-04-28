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

        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            let llm: CoreMLLLM
            if looksLikeRepo {
                print("[smoke] loading via repo: \(target)")
                llm = try await CoreMLLLM.load(repo: target) { msg in
                    print("[load] \(msg)")
                }
            } else {
                let modelDir = URL(fileURLWithPath: target)
                print("[smoke] loading model from: \(modelDir.path)")
                llm = try await CoreMLLLM.load(from: modelDir) { msg in
                    print("[load] \(msg)")
                }
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
