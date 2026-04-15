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

enum BenchMode: String {
    /// Compare drafter proposals against `llm.generate(...)`'s emitted token
    /// stream (target's `decode_q1` argmax). Pre-PR #62 default; oracle replay.
    case oracle
    /// Compare drafter proposals against target's argmax from `verify_qK`
    /// run with zero-padded slots 1..K-1. Target's prediction at P+1 depends
    /// only on slot 0; slots 1..K-1 are irrelevant for argmax[0]. See v3
    /// findings — this mode ≈ oracle on Mac.
    case argmax
    /// Live-equivalent: drafter proposals feed verify slots 1..K-1, and
    /// chain-match is counted against `verify_qK`'s returned argmax[0..K-2]
    /// (K-1 comparisons). Matches what `DrafterUnion.speculateStep` measures.
    /// Histogram uses K+1 slots but only slots 0..K-1 are filled (max
    /// matchCount = K-1 since we cap proposals at K-1); E[tok/burst] max = K.
    case chain
    /// Run oracle + argmax side-by-side (pre-chain-mode behaviour).
    case both
    /// Run all three modes side-by-side.
    case all
}

struct BenchArgs {
    var modelDir: URL
    var maxTokens: Int = 128
    var K: Int = 3
    var outJSON: URL?
    var promptFilter: String?  // run only prompts whose category or id contains this
    var mode: BenchMode = .oracle
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
        case "--mode":
            i += 1
            guard let m = BenchMode(rawValue: argv[i]) else {
                FileHandle.standardError.write(Data("error: --mode must be oracle | argmax | chain | both | all\n".utf8))
                exit(2)
            }
            args.mode = m
        case "-h", "--help":
            print("""
                Usage: accept-rate-bench [options]
                  --model <dir>        Model directory (default: staging-2k-fast-prefill)
                  --max-tokens <N>     Tokens to decode per prompt (default: 128)
                  --K <K>              Draft burst size (default: 3)
                  --out <path>         Write JSON results here in addition to stdout
                  --filter <substr>    Only run prompts whose category or id contains <substr>
                  --mode <m>           oracle (default) | argmax | chain | both | all. See PHASE_B_V3_ARGMAX_FINDINGS.md
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
    /// Tokens emitted via `decode_q1` (oracle mode). Zero if mode didn't run.
    let emittedLen: Int
    /// Tokens emitted via the zero-padded `verify_qK` argmax chain (argmax
    /// mode) OR the drafter-fed verify chain (chain mode, taking drafter #0's
    /// argmax[0] as the committed next token). Zero if neither mode ran.
    let emittedArgmaxLen: Int
    /// How many leading tokens in the `decode_q1` and `verify_qK` chains
    /// agreed (if both oracle and argmax ran). Nil otherwise.
    let decodeVerifyAgreePrefix: Int?
    /// Oracle-mode drafter stats (drafter proposals vs `decode_q1` emitted).
    let drafters: [String: AcceptStats]
    /// Argmax-mode drafter stats (drafter proposals vs zero-padded `verify_qK`).
    let draftersArgmax: [String: AcceptStats]
    /// Chain-mode drafter stats — drafter proposals feed verify slots 1..K-1
    /// and chain-match runs against `verify_qK`'s returned argmax[0..K-2].
    /// Histogram indices K and above remain zero by construction (max match
    /// count = K-1 since proposals are capped at K-1).
    let draftersChain: [String: AcceptStats]
}

struct BenchReport: Codable {
    let modelDir: String
    let mode: String
    let K: Int
    let maxTokens: Int
    let generatedAt: Date
    let prompts: [PromptResult]

    enum Source { case oracle, argmax, chain }

    /// Aggregate chain-accept per drafter across all prompts in a category.
    func aggregate(by category: String, drafterName: String, source: Source) -> AcceptStats {
        var agg = AcceptStats(K: K)
        for p in prompts where p.category == category {
            let src: [String: AcceptStats]
            switch source {
            case .oracle: src = p.drafters
            case .argmax: src = p.draftersArgmax
            case .chain:  src = p.draftersChain
            }
            if let st = src[drafterName] {
                for (k, v) in st.histogram.enumerated() where k < agg.histogram.count {
                    agg.histogram[k] += v
                }
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

            let runOracle = (args.mode == .oracle || args.mode == .both || args.mode == .all)
            let runArgmax = (args.mode == .argmax || args.mode == .both || args.mode == .all)
            let runChain  = (args.mode == .chain  || args.mode == .all)
            if (runArgmax || runChain), llm.benchVerifyK == nil {
                FileHandle.standardError.write(Data(
                    "error: argmax / chain modes require a chunked engine with verify_qK chunks loaded\n".utf8))
                exit(3)
            }
            print("[bench] mode:       \(args.mode.rawValue)")

            var results: [PromptResult] = []

            for p in benchPrompts {
                if let f = args.promptFilter,
                   !(p.id.contains(f) || p.category.contains(f)) { continue }

                var prompt: [Int32] = []
                var emittedDecode: [Int32] = []
                var emittedVerify: [Int32] = []
                var oracleStats: [String: AcceptStats] = [:]
                var argmaxStats: [String: AcceptStats] = [:]
                var chainStats: [String: AcceptStats] = [:]

                if runOracle {
                    print("[\(p.category)/\(p.id)] oracle generating…")
                    let genStart = Date()
                    _ = try await llm.generate(p.text, maxTokens: args.maxTokens)
                    let genSec = Date().timeIntervalSince(genStart)
                    prompt = llm.lastPromptTokenIDs
                    emittedDecode = llm.lastEmittedTokenIDs
                    print("    promptLen=\(prompt.count) emittedLen=\(emittedDecode.count) in \(String(format: "%.1f", genSec))s (\(String(format: "%.1f", Double(emittedDecode.count) / max(genSec, 0.001))) tok/s)")

                    for d in drafters {
                        let s = replayDrafter(d, prompt: prompt, emitted: emittedDecode, K: args.K)
                        oracleStats[d.name] = s
                        let chain = s.chainAccept.map { String(format: "%.2f", $0) }.joined(separator: " ")
                        print("    [oracle] \(d.name): chainAccept=[\(chain)] E[tok/burst]=\(String(format: "%.2f", s.expectedTokensPerBurst))")
                    }
                }

                if runArgmax {
                    print("[\(p.category)/\(p.id)] argmax generating via verify_qK…")
                    let genStart = Date()
                    let (pTok, emittedV) = try await generateArgmaxChain(
                        llm: llm, prompt: p.text, maxTokens: args.maxTokens)
                    let genSec = Date().timeIntervalSince(genStart)
                    emittedVerify = emittedV
                    if prompt.isEmpty { prompt = pTok }
                    print("    promptLen=\(pTok.count) emittedLen=\(emittedVerify.count) in \(String(format: "%.1f", genSec))s")

                    for d in drafters {
                        let s = replayDrafter(d, prompt: pTok, emitted: emittedVerify, K: args.K)
                        argmaxStats[d.name] = s
                        let chain = s.chainAccept.map { String(format: "%.2f", $0) }.joined(separator: " ")
                        print("    [argmax] \(d.name): chainAccept=[\(chain)] E[tok/burst]=\(String(format: "%.2f", s.expectedTokensPerBurst))")
                    }
                }

                if runChain {
                    print("[\(p.category)/\(p.id)] chain-following via verify_qK…")
                    let genStart = Date()
                    let (pTok, chainEmitted, stats) = try await runChainMode(
                        llm: llm, prompt: p.text, maxTokens: args.maxTokens,
                        K: args.K, drafters: drafters)
                    let genSec = Date().timeIntervalSince(genStart)
                    if prompt.isEmpty { prompt = pTok }
                    if emittedVerify.isEmpty { emittedVerify = chainEmitted }
                    print("    promptLen=\(pTok.count) emittedLen=\(chainEmitted.count) in \(String(format: "%.1f", genSec))s")
                    for d in drafters {
                        let s = stats[d.name] ?? AcceptStats(K: args.K)
                        chainStats[d.name] = s
                        let chain = s.chainAccept.map { String(format: "%.2f", $0) }.joined(separator: " ")
                        print("    [chain] \(d.name): chainAccept=[\(chain)] E[tok/burst]=\(String(format: "%.2f", s.expectedTokensPerBurst))")
                    }
                }

                var agreePrefix: Int? = nil
                if runOracle && runArgmax {
                    var m = 0
                    let limit = min(emittedDecode.count, emittedVerify.count)
                    while m < limit && emittedDecode[m] == emittedVerify[m] { m += 1 }
                    agreePrefix = m
                    print("    decode_q1 / verify_qK agree for first \(m)/\(limit) tokens")
                }

                results.append(PromptResult(
                    id: p.id, category: p.category,
                    promptLen: prompt.count,
                    emittedLen: emittedDecode.count,
                    emittedArgmaxLen: emittedVerify.count,
                    decodeVerifyAgreePrefix: agreePrefix,
                    drafters: oracleStats,
                    draftersArgmax: argmaxStats,
                    draftersChain: chainStats))
            }

            let report = BenchReport(
                modelDir: args.modelDir.path,
                mode: args.mode.rawValue,
                K: args.K,
                maxTokens: args.maxTokens,
                generatedAt: Date(),
                prompts: results)

            // Summary by category
            print("")
            print("=== summary ===")
            let cats = Set(benchPrompts.map { $0.category }).sorted()
            let dnames = drafters.map { $0.name }
            let modes: [(String, BenchReport.Source)] = {
                switch args.mode {
                case .oracle: return [("oracle", .oracle)]
                case .argmax: return [("argmax", .argmax)]
                case .chain:  return [("chain",  .chain)]
                case .both:   return [("oracle", .oracle), ("argmax", .argmax)]
                case .all:    return [("oracle", .oracle), ("argmax", .argmax), ("chain", .chain)]
                }
            }()
            for cat in cats {
                print("[\(cat)]")
                for (label, src) in modes {
                    for d in dnames {
                        let a = report.aggregate(by: cat, drafterName: d, source: src)
                        let chain = a.chainAccept.map { String(format: "%.2f", $0) }.joined(separator: " ")
                        print("  [\(label)] \(d): N=\(a.totalBursts) chain=[\(chain)] E[tok/burst]=\(String(format: "%.2f", a.expectedTokensPerBurst))")
                    }
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

// MARK: - Argmax chain generation

/// Open-generate from the target using `verify_qK` to produce the argmax
/// chain. Each step calls `benchVerify` with the current seed in slot 0 and
/// zeros in slots 1..K-1; only `targetArgmax[0]` is consumed, so the slot 1+
/// inputs don't matter (target's prediction at P+1 depends only on slot 0's
/// token at position P). One verify call per committed token.
///
/// Slots 1..K-1 of the KV cache written by each call are overwritten by the
/// next call (which starts at P+1), so no downstream corruption.
func generateArgmaxChain(llm: CoreMLLLM, prompt: String, maxTokens: Int) async throws
    -> (prompt: [Int32], emitted: [Int32])
{
    guard let K = llm.benchVerifyK else {
        throw BenchError.verifyUnavailable
    }
    let (promptTokens, seed) = try await llm.benchPrefill(prompt)

    // Gemma EOS set matches CoreMLLLM's production loop.
    let eosIDs: Set<Int32> = [1, 106, 151645]

    var emitted: [Int32] = []
    emitted.reserveCapacity(maxTokens)
    var nextID = seed

    var verifyTokens = [Int32](repeating: 0, count: K)

    for _ in 0..<maxTokens {
        if eosIDs.contains(nextID) { break }
        verifyTokens[0] = nextID
        // Slots 1..K-1 left as 0 — intentional; see function doc.
        for i in 1..<K { verifyTokens[i] = 0 }
        let argmax = try llm.benchVerify(verifyTokens)
        emitted.append(nextID)
        llm.benchAdvance(by: 1)
        nextID = argmax[0]
    }
    return (promptTokens, emitted)
}

enum BenchError: Error {
    case verifyUnavailable
}

// MARK: - Chain-following (live-equivalent) mode
//
// For each committed position P:
//   1. For each drafter in `drafters`:
//        - Ask for K-1 proposals given history up through P.
//        - Call verify with [nextID, p0, p1, ..., p_{K-2}] (padded with 0 if
//          the drafter returned fewer than K-1 proposals).
//        - Chain-match proposals against returned argmax[0..compareLen-1]
//          (where compareLen = min(K-1, proposals.count)).
//        - Record histogram[matchCount]++. Max matchCount = K-1, so
//          histogram slot K is never filled.
//   2. Use the FIRST drafter's `argmax[0]` as the chain's next token
//      (argmax[0] only depends on slot 0, so all drafters produce the same
//      value modulo fp16 jitter). Emit current nextID. Advance by 1.
//
// Semantics match `DrafterUnion.speculateStep` per-source measurement: each
// drafter sees drafter's own proposals in verify slots 1..K-1, so its
// argmax[1..K-1] is computed under the same conditions as live Union.
func runChainMode(
    llm: CoreMLLLM, prompt: String, maxTokens: Int, K: Int, drafters: [Drafter]
) async throws -> (prompt: [Int32], emitted: [Int32], stats: [String: AcceptStats]) {
    guard let verifyK = llm.benchVerifyK else { throw BenchError.verifyUnavailable }
    precondition(verifyK == K,
                 "--K \(K) but engine verify_qK expects K=\(verifyK)")
    let (promptTokens, seed) = try await llm.benchPrefill(prompt)

    for d in drafters { d.reset() }
    // Ingest all but the last prompt token (same convention as oracle replay
    // so stateful CrossVocab's `committedPosition` sits at P where P is the
    // last committed token's position).
    for tok in promptTokens.dropLast() { for d in drafters { d.ingest(tok) } }

    let eosIDs: Set<Int32> = [1, 106, 151645]
    var emitted: [Int32] = []
    emitted.reserveCapacity(maxTokens)
    var stats: [String: AcceptStats] = [:]
    for d in drafters { stats[d.name] = AcceptStats(K: K) }

    var history = promptTokens
    var nextID = seed
    var verifyTokens = [Int32](repeating: 0, count: K)

    for _ in 0..<maxTokens {
        if eosIDs.contains(nextID) { break }
        verifyTokens[0] = nextID

        var chainNext: Int32? = nil

        for d in drafters {
            let props = d.propose(history: history, K: K - 1)
            let useCount = min(props.count, K - 1)
            for i in 1..<K {
                verifyTokens[i] = (i - 1 < useCount) ? props[i - 1] : 0
            }
            let argmax = try llm.benchVerify(verifyTokens)

            var match = 0
            for k in 0..<useCount {
                if props[k] == argmax[k] { match += 1 } else { break }
            }
            stats[d.name]!.histogram[match] += 1
            stats[d.name]!.totalBursts += 1

            // First drafter's argmax[0] seeds the chain; subsequent drafters'
            // argmax[0] values should match modulo fp16 jitter (depends only
            // on slot 0, same across all calls).
            if chainNext == nil { chainNext = argmax[0] }
        }

        emitted.append(nextID)
        history.append(nextID)
        for d in drafters { d.ingest(nextID) }
        llm.benchAdvance(by: 1)
        nextID = chainNext ?? nextID
    }
    return (promptTokens, emitted, stats)
}
