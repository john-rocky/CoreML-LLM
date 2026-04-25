#if os(macOS)
// Mac-side bit-exact verifier for DrafterUnion (Phase B Task 1).
//
// Runs each prompt twice on the same loaded model:
//   1. Serial decode (mtp=off, union=off, cv=off)
//   2. DrafterUnion (mtp=off, union=on, cv=off)
//
// At temperature = 0 the two emitted token streams MUST match. Exits
// non-zero on any divergence so this can gate the iPhone trip.
//
// Usage:
//   swift run -c release union-bitexact \
//       [--model <dir>] [--max-tokens 64] [--prompts code,chat,...]
// Defaults to staging-2k-fast-prefill, 64 tokens, all prompt categories.

import CoreMLLLM
import Foundation

struct Args {
    var modelDir: URL
    var maxTokens: Int = 64
    var categories: Set<String> = []
    /// "union" (default), "pld-only", or "fallback-only". The latter two
    /// help isolate whether divergence vs serial comes from union
    /// bookkeeping (fallback-only) vs PLD-induced verify drift (pld-only).
    var mode: String = "union"
}

func parseArgs() -> Args {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let defaultModel = home
        .appendingPathComponent("Downloads")
        .appendingPathComponent("coreml-llm-artifacts")
        .appendingPathComponent("staging-2k-fast-prefill")
        .appendingPathComponent("gemma4-e2b")
    var a = Args(modelDir: defaultModel)
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        let s = argv[i]
        switch s {
        case "--model":      i += 1; a.modelDir = URL(fileURLWithPath: argv[i])
        case "--max-tokens": i += 1; a.maxTokens = Int(argv[i]) ?? 64
        case "--prompts":    i += 1; a.categories = Set(argv[i].split(separator: ",").map(String.init))
        case "--mode":       i += 1; a.mode = argv[i]
        case "-h", "--help":
            print("""
                union-bitexact: verify DrafterUnion emits the same token
                stream as serial decode at temperature = 0.

                Options:
                  --model <dir>        Model directory (default staging-2k-fast-prefill)
                  --max-tokens <N>     Tokens per prompt (default 64)
                  --prompts <list>     Comma-separated category filter (chat,code,qa,summary)
                """)
            exit(0)
        default:
            FileHandle.standardError.write(Data("warning: unknown arg \(s)\n".utf8))
        }
        i += 1
    }
    return a
}

@main
struct Verifier {
    // One representative prompt per A5 category, kept short so the bench
    // finishes in a few minutes on Mac Studio.
    static let corpus: [(String, String, String)] = [
        ("chat-1",    "chat",
         "Explain what a transformer is in two short sentences."),
        ("code-1",    "code",
         "Write a Python function that returns the n-th Fibonacci number using memoization."),
        ("qa-1",     "qa",
         "What is the capital of France, and which river runs through it?"),
        ("summary-1", "summary",
         "Summarize in one sentence: The mitochondrion is a double-membrane-bound organelle "
         + "found in most eukaryotic cells, often called the powerhouse of the cell because "
         + "it generates most of the cell's supply of adenosine triphosphate."),
    ]

    static func main() async {
        let args = parseArgs()
        do {
            let llm = try await CoreMLLLM.load(from: args.modelDir)
            print("[verify] model loaded")

            var failures = 0
            var checks = 0
            for (id, cat, prompt) in corpus {
                if !args.categories.isEmpty && !args.categories.contains(cat) { continue }
                checks += 1

                // Serial baseline.
                llm.mtpEnabled = false
                llm.crossVocabEnabled = false
                llm.drafterUnionEnabled = false
                llm.reset()
                _ = try await llm.generate(prompt, maxTokens: args.maxTokens)
                let serial = llm.lastEmittedTokenIDs

                // Union under test. Cross-vocab is gated out via a
                // very high rolling-accept threshold because the Mac
                // staging Qwen is compiled at ctx=512 but CoreMLLLM.load
                // hands it ctx=2048, which crashes at first prediction
                // (gotcha #2 in docs/SESSION_STATE.md). The CV verify
                // walk was already proven correct by PR #45's
                // CrossVocabSpeculativeEngine; the new code under test
                // here is the PLD path inside DrafterUnion. The iPhone
                // trip will exercise the CV-as-source path with a
                // properly-sized Qwen.
                llm.mtpEnabled = false
                llm.crossVocabEnabled = false
                llm.drafterUnionEnabled = true
                llm.setDrafterUnionCrossVocabDisabled(true)
                if args.mode == "fallback-only" {
                    // Disable PLD too; everything routes through fallbackSingleStep.
                    llm.setDrafterUnionPLDThreshold(99.0)
                }
                llm.reset()
                _ = try await llm.generate(prompt, maxTokens: args.maxTokens)
                let union = llm.lastEmittedTokenIDs

                let firstDivergence = zip(serial, union).enumerated()
                    .first { _, pair in pair.0 != pair.1 }?.offset
                let ok = serial == union
                let mark = ok ? "PASS" : "FAIL"
                print("[\(mark)] \(id) (\(cat)) — serial=\(serial.count) union=\(union.count) "
                    + (firstDivergence.map { "diverge@\($0)" } ?? "match"))
                if !ok {
                    failures += 1
                    let cmp = min(serial.count, union.count)
                    let from = max(0, (firstDivergence ?? cmp) - 3)
                    let to = min(cmp, (firstDivergence ?? cmp) + 5)
                    print("   serial[\(from)..<\(to)]: \(Array(serial[from..<to]))")
                    print("   union [\(from)..<\(to)]: \(Array(union[from..<to]))")
                }
            }

            print("[verify] \(checks - failures)/\(checks) prompts bit-exact")
            exit(failures == 0 ? 0 : 1)
        } catch {
            fputs("[verify] error: \(error)\n", stderr)
            exit(2)
        }
    }
}
#endif
