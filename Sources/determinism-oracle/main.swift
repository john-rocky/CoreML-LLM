// Mac-side CoreML determinism oracle (quality-gate PoC).
//
// Runs a small fixed corpus through CoreMLLLM at argmax (all drafters off)
// and either:
//   * --record   writes the emitted token IDs + sha256 to an oracle JSON
//   * (default)  verifies the current run matches the committed oracle
//
// This is a CoreML→CoreML regression test: it catches silent changes in
// the shipped decode path (e.g. a chunk re-conversion, a runtime flag
// flip, a tokenizer-config drift). It does NOT compare against PyTorch
// goldens — that's a future extension which only adds another reference
// format; the harness stays the same.
//
// Usage:
//   swift run -c release determinism-oracle --record \
//       --model ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b \
//       --oracle Tests/oracles/gemma4-e2b-argmax.json
//
//   swift run -c release determinism-oracle \
//       --model ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b \
//       --oracle Tests/oracles/gemma4-e2b-argmax.json
//
// Exits 0 on all-match / successful record, 1 on any mismatch, 2 on error.

#if os(macOS)
import CoreMLLLM
import CryptoKit
import Foundation

struct Args {
    var modelDir: URL
    var oraclePath: URL
    var maxTokens: Int = 16
    var record: Bool = false
    var label: String?
}

func defaultModelDir() -> URL {
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Downloads")
        .appendingPathComponent("coreml-llm-artifacts")
        .appendingPathComponent("staging-2k-fast-prefill")
        .appendingPathComponent("gemma4-e2b")
}

func parseArgs() -> Args {
    var a = Args(
        modelDir: defaultModelDir(),
        oraclePath: URL(fileURLWithPath: "Tests/oracles/gemma4-e2b-argmax.json")
    )
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        let s = argv[i]
        switch s {
        case "--model":      i += 1; a.modelDir = URL(fileURLWithPath: argv[i])
        case "--oracle":     i += 1; a.oraclePath = URL(fileURLWithPath: argv[i])
        case "--max-tokens": i += 1; a.maxTokens = Int(argv[i]) ?? 16
        case "--label":      i += 1; a.label = argv[i]
        case "--record":     a.record = true
        case "-h", "--help":
            print("""
                determinism-oracle: CoreML argmax regression gate.

                Options:
                  --model <dir>        Model directory (default staging-2k-fast-prefill/gemma4-e2b)
                  --oracle <path>      Oracle JSON path (default Tests/oracles/gemma4-e2b-argmax.json)
                  --max-tokens <N>     Tokens per prompt (default 16)
                  --label <s>          Human label written into the oracle (e.g. git-sha)
                  --record             Rewrite the oracle from the current run (use when expected)
                """)
            exit(0)
        default:
            FileHandle.standardError.write(Data("warning: unknown arg \(s)\n".utf8))
        }
        i += 1
    }
    return a
}

struct OracleEntry: Codable {
    let id: String
    let prompt: String
    let maxTokens: Int
    let tokens: [Int32]
    let sha256: String
}

struct OracleFile: Codable {
    let schema: Int
    let model: String
    let label: String?
    let entries: [OracleEntry]
}

func sha256Hex(_ tokens: [Int32]) -> String {
    var hasher = SHA256()
    tokens.withUnsafeBufferPointer { buf in
        let bytes = UnsafeRawBufferPointer(buf)
        hasher.update(bufferPointer: bytes)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

func runCorpus(on llm: CoreMLLLM, maxTokens: Int) async throws -> [OracleEntry] {
    var out: [OracleEntry] = []
    out.reserveCapacity(Oracle.corpus.count)
    for (id, prompt) in Oracle.corpus {
        // Turn every drafter off so only the baseline serial decode runs.
        llm.mtpEnabled = false
        llm.drafterUnionEnabled = false
        llm.crossVocabEnabled = false
        llm.lookaheadEnabled = false
        llm.reset()
        _ = try await llm.generate(prompt, maxTokens: maxTokens)
        let tokens = llm.lastEmittedTokenIDs
        out.append(OracleEntry(
            id: id, prompt: prompt, maxTokens: maxTokens,
            tokens: tokens, sha256: sha256Hex(tokens)
        ))
    }
    return out
}

func record(to path: URL, model: String, label: String?, entries: [OracleEntry]) throws {
    let file = OracleFile(schema: 1, model: model, label: label, entries: entries)
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try enc.encode(file)
    try FileManager.default.createDirectory(
        at: path.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try data.write(to: path)
    print("[oracle] wrote \(entries.count) entries to \(path.path)")
}

func verify(against path: URL, entries: [OracleEntry]) throws -> Int {
    guard FileManager.default.fileExists(atPath: path.path) else {
        fputs("[oracle] error: no oracle at \(path.path) — run with --record first\n", stderr)
        return 2
    }
    let data = try Data(contentsOf: path)
    let dec = JSONDecoder()
    let ref = try dec.decode(OracleFile.self, from: data)
    let byID = Dictionary(uniqueKeysWithValues: ref.entries.map { ($0.id, $0) })

    var failures = 0
    for entry in entries {
        guard let want = byID[entry.id] else {
            print("[FAIL] \(entry.id) — missing from oracle (rerecord if expected)")
            failures += 1
            continue
        }
        if want.sha256 == entry.sha256 && want.tokens == entry.tokens {
            print("[PASS] \(entry.id) — \(entry.tokens.count) tokens, sha=\(entry.sha256.prefix(8))")
        } else {
            failures += 1
            let firstDiff = zip(want.tokens, entry.tokens).enumerated()
                .first { _, pair in pair.0 != pair.1 }?.offset
            print("[FAIL] \(entry.id) — diverged"
                + (firstDiff.map { " @\($0)" } ?? "")
                + " (want=\(want.tokens.count) tok, got=\(entry.tokens.count) tok)")
            let cmp = min(want.tokens.count, entry.tokens.count)
            let from = max(0, (firstDiff ?? cmp) - 2)
            let to = min(cmp, (firstDiff ?? cmp) + 4)
            if from < to {
                print("   want[\(from)..<\(to)]: \(Array(want.tokens[from..<to]))")
                print("   got [\(from)..<\(to)]: \(Array(entry.tokens[from..<to]))")
            }
        }
    }
    return failures
}

@main
struct Oracle {
    // Small fixed corpus. Keep short so Mac Studio finishes in a few seconds.
    // IDs are stable across runs — renaming is a breaking change for
    // oracles that already pin to them.
    static let corpus: [(id: String, prompt: String)] = [
        ("qa-capital",
         "What is the capital of France?"),
        ("code-fib",
         "Write a Python function that returns the n-th Fibonacci number."),
        ("summary-mito",
         "Summarize in one short sentence: the mitochondrion generates ATP in eukaryotic cells."),
    ]

    static func main() async {
        let args = parseArgs()
        do {
            print("[oracle] loading \(args.modelDir.path)")
            let llm = try await CoreMLLLM.load(from: args.modelDir)
            print("[oracle] loaded model=\(llm.modelName), corpus=\(Oracle.corpus.count) prompts, maxTokens=\(args.maxTokens)")

            let entries = try await runCorpus(on: llm, maxTokens: args.maxTokens)

            if args.record {
                try record(
                    to: args.oraclePath,
                    model: llm.modelName,
                    label: args.label,
                    entries: entries
                )
                exit(0)
            } else {
                let failures = try verify(against: args.oraclePath, entries: entries)
                print("[oracle] \(entries.count - failures)/\(entries.count) prompts match")
                exit(failures == 0 ? 0 : 1)
            }
        } catch {
            fputs("[oracle] error: \(error)\n", stderr)
            exit(2)
        }
    }
}
#else
@main
struct Oracle {
    static func main() {
        fputs("determinism-oracle is Mac-only\n", FileHandle.standardError)
    }
}
#endif
