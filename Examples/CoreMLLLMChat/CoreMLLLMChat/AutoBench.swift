// AutoBench — single-shot tok/s bench for headless iPhone runs.
//
// Activated by env var LLM_AUTOBENCH=1 (pass via xcrun devicectl
// `--environment-variables '{"LLM_AUTOBENCH": "1"}'`). After the first
// successful model load, iterates over a fixed prompt set, prints
// `[AutoBench] <label>: tok/s=<value>` lines for each, then exits the
// process. Captures via `xcrun devicectl device process launch --console`.
//
// Optional env knobs:
//   LLM_AUTOBENCH_MAX_TOKENS — token budget per prompt (default 256)
//   LLM_AUTOBENCH_MODEL — model folder name under Documents/Models
//                         (default "gemma4-e2b")
//   LLM_AUTOBENCH_PROMPTS — comma-separated label list to restrict the
//                           prompt set ("narrative,code,list,yes")
import Foundation
import CoreMLLLM

#if os(iOS)
enum AutoBench {
    private static let prompts: [(String, String)] = [
        ("narrative",
         "Write a detailed essay about the ocean and marine life. Include facts about coral reefs, deep sea ecosystems, and the impact of climate change."),
        ("code",
         "Write a Python class implementing a binary search tree with insert, delete, and find methods."),
        ("list",
         "List 30 facts about ancient Roman emperors with their reign dates."),
        ("yes",
         "Say yes 30 times."),
    ]

    /// Call this in ChatView's body `.task`. Returns immediately if
    /// `LLM_AUTOBENCH` is not set. Otherwise loads the model, runs all
    /// prompts, prints results, and exits.
    @MainActor
    static func runIfRequested(runner: LLMRunner) async {
        let env = ProcessInfo.processInfo.environment
        guard env["LLM_AUTOBENCH"] == "1" else { return }

        let maxTokens = Int(env["LLM_AUTOBENCH_MAX_TOKENS"] ?? "256") ?? 256
        let modelFolder = env["LLM_AUTOBENCH_MODEL"] ?? "gemma4-e2b"
        let selected: Set<String>? = (env["LLM_AUTOBENCH_PROMPTS"]).map {
            Set($0.split(separator: ",").map(String.init))
        }

        print("[AutoBench] starting model=\(modelFolder) maxTokens=\(maxTokens)")

        let docs = FileManager.default.urls(for: .documentDirectory,
                                             in: .userDomainMask).first!
        let bundleDir = docs.appendingPathComponent("Models")
            .appendingPathComponent(modelFolder)
        let modelURL = bundleDir.appendingPathComponent("model.mlpackage")

        do {
            let loadStart = CFAbsoluteTimeGetCurrent()
            try await runner.loadModel(from: modelURL)
            let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
            print(String(format: "[AutoBench] model loaded in %.0f ms", loadMs))
        } catch {
            print("[AutoBench] LOAD_ERROR: \(error.localizedDescription)")
            exit(1)
        }

        // Warmup: short prompt to warm the drafter + first-cycle ANE
        // residency before the timed prompts. Otherwise the cold first
        // MTP cycle hits 0/K acceptance, rolling EMA decays below
        // fallback threshold, and MTP auto-bails for the rest of the
        // first timed prompt. Skip via LLM_AUTOBENCH_NO_WARMUP=1.
        if env["LLM_AUTOBENCH_NO_WARMUP"] != "1" {
            print("[AutoBench] warmup (24 tokens, discarded)")
            let warmStart = CFAbsoluteTimeGetCurrent()
            let warmMsg = ChatMessage(role: .user, content: "Hello.")
            do {
                let stream = try await runner.generate(messages: [warmMsg])
                var n = 0
                for await _ in stream {
                    n += 1
                    if n >= 24 { break }
                }
            } catch {
                print("[AutoBench] warmup ERROR: \(error.localizedDescription)")
            }
            let warmMs = (CFAbsoluteTimeGetCurrent() - warmStart) * 1000
            print(String(format: "[AutoBench] warmup done in %.0f ms", warmMs))
        }

        for (label, prompt) in prompts {
            if let selected, !selected.contains(label) { continue }
            print("[AutoBench] === \(label) ===")
            let userMsg = ChatMessage(role: .user, content: prompt)
            do {
                let runStart = CFAbsoluteTimeGetCurrent()
                let stream = try await runner.generate(messages: [userMsg])
                var tokenCount = 0
                var output = ""
                for await tok in stream {
                    output += tok
                    tokenCount += 1
                    if tokenCount >= maxTokens { break }
                }
                let wallSec = CFAbsoluteTimeGetCurrent() - runStart
                let tps = runner.tokensPerSecond
                print(String(format:
                    "[AutoBench] %@: tokens=%d wall=%.2fs tok/s=%.2f",
                    label as NSString, tokenCount, wallSec, tps))
                let preview = output.prefix(120)
                    .replacingOccurrences(of: "\n", with: " ")
                print("[AutoBench] \(label)_output: \(preview)")
            } catch {
                print("[AutoBench] \(label) ERROR: \(error.localizedDescription)")
            }
        }
        print("[AutoBench] done")
        // Flush stdio before exit so devicectl --console captures everything.
        fflush(stdout)
        fflush(stderr)
        exit(0)
    }
}
#endif
