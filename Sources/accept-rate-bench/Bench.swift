//
//  main.swift — offline accept-rate bench (Phase A1–A3).
//
//  Runs the shipping `CoreMLLLM` pipeline on each prompt in
//  `Prompts.swift` at `temperature = 0`, records the emitted token IDs,
//  then replays each `Drafter` against the emitted sequence to measure
//  chain-accept rate per position.
//
//  Uses oracle replay rather than calling `verify_qK`, which is
//  mathematically identical at temp=0 and avoids mutating the KV cache.
//
//  Usage:
//      swift build -c release --target accept-rate-bench
//      .build/release/accept-rate-bench [--model <dir>] [--max-tokens 128] [--K 3]
//
//  Defaults to `~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b`.
//

import Foundation
import CoreMLLLM

// MARK: - CLI parsing

struct BenchArgs {
    var modelDir: URL
    var maxTokens: Int = 128
    var K: Int = 3
    var outJSON: URL?
    var promptFilter: String?  // run only prompts whose category or id contains this
}

func parseArgs() -> BenchArgs {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let defaultModel = home
        .appendingPathComponent("Downloads")
        .appendingPathComponent("coreml-llm-artifacts")
        .appendingPathComponent("staging-2k-fast-prefill")
        .appendingPathComponent("gemma4-e2b")

    var args = BenchArgs(modelDir: defaultModel)
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--model":
            i += 1
            args.modelDir = URL(fileURLWithPath: argv[i])
        case "--max-tokens":
            i += 1
            args.maxTokens = Int(argv[i]) ?? 128
        case "--K":
            i += 1
            args.K = Int(argv[i]) ?? 3
        case "--out":
            i += 1
            args.outJSON = URL(fileURLWithPath: argv[i])
        case "--filter":
            i += 1
            args.promptFilter = argv[i]
        case "-h", "--help":
            print("""
                Usage: accept-rate-bench [options]
                  --model <dir>        Model directory (default: staging-2k-fast-prefill)
                  --max-tokens <N>     Tokens to decode per prompt (default: 128)
                  --K <K>              Draft burst size (default: 3)
                  --out <path>         Write JSON results here in addition to stdout
                  --filter <substr>    Only run prompts whose category or id contains <substr>
                """)
            exit(0)
        default:
            FileHandle.standardError.write(Data("warning: unknown arg \(a)\n".utf8))
        }
        i += 1
    }
    return args
}

// MARK: - Output record

struct PromptResult: Codable {
    let id: String
    let category: String
    let promptLen: Int
    let emittedLen: Int
    let drafters: [String: AcceptStats]
}

struct BenchReport: Codable {
    let modelDir: String
    let K: Int
    let maxTokens: Int
    let generatedAt: Date
    let prompts: [PromptResult]

    /// Aggregate chain-accept per drafter across all prompts in a category.
    func aggregate(by category: String, drafterName: String) -> AcceptStats {
        var agg = AcceptStats(K: K)
        for p in prompts where p.category == category {
            if let st = p.drafters[drafterName] {
                for (k, v) in st.histogram.enumerated() { agg.histogram[k] += v }
                agg.totalBursts += st.totalBursts
            }
        }
        return agg
    }
}

// MARK: - Entry

@main
struct Main {
    static func main() async {
        let args = parseArgs()
        print("[bench] model:      \(args.modelDir.path)")
        print("[bench] max-tokens: \(args.maxTokens)")
        print("[bench] K:          \(args.K)")
        print("[bench] prompts:    \(benchPrompts.count) (filter=\(args.promptFilter ?? "-"))")
        print("")

        do {
            print("[bench] loading model…")
            let loadStart = Date()
            let llm = try await CoreMLLLM.load(from: args.modelDir)
            print("[bench] loaded in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")
            print("")

            var drafters: [Drafter] = [
                PromptLookupDrafter(ngramSize: 2),
                PromptLookupDrafter(ngramSize: 3),
                SuffixTreeDrafter(),
            ]

            // Disable both in-engine spec paths so generate() records the
            // pure target-decode sequence. Cross-vocab is measured below via
            // the direct CrossVocabDraft API wrapped in an oracle-replay
            // adapter.
            llm.mtpEnabled = false
            llm.crossVocabEnabled = false

            let cvDir = args.modelDir.appendingPathComponent("cross_vocab")
            let cvCompiled = cvDir.appendingPathComponent("qwen_drafter.mlmodelc")
            let cvPkg = cvDir.appendingPathComponent("qwen_drafter.mlpackage")
            let cvMapURL = cvDir.appendingPathComponent("qwen_gemma_vocab.bin")
            let cvModelURL: URL? = FileManager.default.fileExists(atPath: cvCompiled.path) ? cvCompiled
                : FileManager.default.fileExists(atPath: cvPkg.path) ? cvPkg : nil
            if let cvModelURL, FileManager.default.fileExists(atPath: cvMapURL.path) {
                do {
                    let map = try CrossVocabMap(url: cvMapURL)
                    // Qwen 2.5 0.5B mlpackage in sibling repo was compiled
                    // with ctx=512. Using 2048 triggers a MultiArray shape
                    // mismatch on causal_mask inputs.
                    let cvDraft = try CrossVocabDraft(
                        modelURL: cvModelURL,
                        vocabMap: map,
                        K: args.K,
                        contextLength: 512,
                        computeUnits: .cpuAndGPU)
                    drafters.append(CrossVocabOracleDrafter(drafter: cvDraft, K: args.K))
                    let cov = Double(map.qwenToGemma.filter { $0 >= 0 }.count) / Double(map.qwenVocabSize) * 100
                    print("[bench] cross-vocab drafter loaded (qwen→gemma coverage=\(String(format: "%.1f%%", cov)))")
                } catch {
                    print("[bench] cross-vocab drafter unavailable: \(error)")
                }
            } else {
                print("[bench] cross-vocab artifacts not found under \(cvDir.path) — skipping")
            }

            var results: [PromptResult] = []

            for p in benchPrompts {
                if let f = args.promptFilter,
                   !(p.id.contains(f) || p.category.contains(f)) { continue }

                print("[\(p.category)/\(p.id)] generating…")
                let genStart = Date()
                // Discard the text; we only care about the token IDs.
                _ = try await llm.generate(p.text, maxTokens: args.maxTokens)
                let genSec = Date().timeIntervalSince(genStart)
                let prompt = llm.lastPromptTokenIDs
                let emitted = llm.lastEmittedTokenIDs
                print("    promptLen=\(prompt.count) emittedLen=\(emitted.count) in \(String(format: "%.1f", genSec))s (\(String(format: "%.1f", Double(emitted.count) / max(genSec, 0.001))) tok/s)")

                var stats: [String: AcceptStats] = [:]
                for d in drafters {
                    let s = replayDrafter(d, prompt: prompt, emitted: emitted, K: args.K)
                    stats[d.name] = s
                    let chain = s.chainAccept.map { String(format: "%.2f", $0) }.joined(separator: " ")
                    print("    \(d.name): chainAccept=[\(chain)] E[tok/burst]=\(String(format: "%.2f", s.expectedTokensPerBurst))")
                }

                results.append(PromptResult(
                    id: p.id, category: p.category,
                    promptLen: prompt.count, emittedLen: emitted.count,
                    drafters: stats))
            }

            let report = BenchReport(
                modelDir: args.modelDir.path,
                K: args.K,
                maxTokens: args.maxTokens,
                generatedAt: Date(),
                prompts: results)

            // Summary by category
            print("")
            print("=== summary ===")
            let cats = Set(benchPrompts.map { $0.category }).sorted()
            let dnames = drafters.map { $0.name }
            for cat in cats {
                print("[\(cat)]")
                for d in dnames {
                    let a = report.aggregate(by: cat, drafterName: d)
                    let chain = a.chainAccept.map { String(format: "%.2f", $0) }.joined(separator: " ")
                    print("  \(d): N=\(a.totalBursts) chain=[\(chain)] E[tok/burst]=\(String(format: "%.2f", a.expectedTokensPerBurst))")
                }
            }

            if let out = args.outJSON {
                let enc = JSONEncoder()
                enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                enc.dateEncodingStrategy = .iso8601
                try enc.encode(report).write(to: out)
                print("[bench] wrote \(out.path)")
            }
        } catch {
            FileHandle.standardError.write(Data("[bench] error: \(error)\n".utf8))
            exit(1)
        }
    }
}
