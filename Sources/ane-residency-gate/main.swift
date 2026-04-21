// T2: ANE residency CI gate.
//
// Loads each chunk{1..4}.{mlmodelc,mlpackage} in a model directory, runs
// MLComputePlan to read per-op device placement, and exits non-zero if
// any chunk's ANE op fraction drops below the configured threshold.
//
// Also emits a JSON line per chunk so PRs can store a baseline and diff
// it on the next run. The JSON is human-readable on stdout; redirect to
// a file when archiving as a baseline.
//
// Usage:
//   swift run ane-residency-gate \
//       --model-dir /path/to/gemma4-e2b \
//       --threshold 0.995 \
//       [--baseline previous_baseline.json] \
//       [--write-baseline current.json]
//
// Exit codes:
//   0  all chunks at or above threshold
//   1  at least one chunk below threshold
//   2  required chunk missing (load failure or absent file)
//   3  argument error
//
// See docs/LITERT_PERF_ADOPTIONS.md §T2.

import CoreML
import CoreMLLLM
import Foundation

struct Args {
    var modelDir: URL
    var threshold: Double = 0.995
    var baselinePath: URL? = nil
    var writeBaselinePath: URL? = nil
    var chunkNames: [String] = ["chunk1", "chunk2", "chunk3", "chunk4"]
}

func parseArgs() -> Args {
    var modelDir: URL? = nil
    var threshold: Double = 0.995
    var baseline: URL? = nil
    var writeBaseline: URL? = nil
    var chunkNames: [String] = ["chunk1", "chunk2", "chunk3", "chunk4"]

    var i = 1
    let argv = CommandLine.arguments
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--model-dir":
            i += 1
            modelDir = URL(fileURLWithPath: argv[i])
        case "--threshold":
            i += 1
            guard let v = Double(argv[i]), v >= 0, v <= 1 else {
                FileHandle.standardError.write("threshold must be in [0,1]\n".data(using: .utf8)!)
                exit(3)
            }
            threshold = v
        case "--baseline":
            i += 1
            baseline = URL(fileURLWithPath: argv[i])
        case "--write-baseline":
            i += 1
            writeBaseline = URL(fileURLWithPath: argv[i])
        case "--chunks":
            i += 1
            chunkNames = argv[i].split(separator: ",").map(String.init)
        case "--help", "-h":
            print("""
            Usage: ane-residency-gate --model-dir <path> [options]
              --threshold F          minimum acceptable ANE fraction per chunk (default 0.995)
              --baseline FILE        prior JSON to diff against (no failure on diff alone)
              --write-baseline FILE  write current results to FILE
              --chunks A,B,C         override chunk names (default: chunk1,chunk2,chunk3,chunk4)
            """)
            exit(0)
        default:
            FileHandle.standardError.write("unknown arg: \(a)\n".data(using: .utf8)!)
            exit(3)
        }
        i += 1
    }
    guard let m = modelDir else {
        FileHandle.standardError.write("--model-dir required\n".data(using: .utf8)!)
        exit(3)
    }
    return Args(modelDir: m, threshold: threshold,
                baselinePath: baseline, writeBaselinePath: writeBaseline,
                chunkNames: chunkNames)
}

func loadBaseline(_ url: URL) -> [String: ComputePlanAudit.ChunkResult] {
    guard let data = try? Data(contentsOf: url),
          let arr = try? JSONDecoder().decode([ComputePlanAudit.ChunkResult].self, from: data) else {
        FileHandle.standardError.write(
            "warning: baseline at \(url.path) unreadable or invalid; skipping diff\n"
                .data(using: .utf8)!)
        return [:]
    }
    return Dictionary(uniqueKeysWithValues: arr.map { ($0.chunkName, $0) })
}

let args = parseArgs()

let results = await ComputePlanAudit.audit(
    modelDirectory: args.modelDir,
    chunkNames: args.chunkNames,
    computeUnits: .cpuAndNeuralEngine)

if results.isEmpty {
    FileHandle.standardError.write("no chunks loaded from \(args.modelDir.path)\n"
        .data(using: .utf8)!)
    exit(2)
}

// Pretty table to stdout.
func col(_ s: String, _ width: Int) -> String {
    s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}
func rcol(_ s: String, _ width: Int) -> String {
    s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
}
print(col("chunk", 12) + rcol("total", 9) + rcol("ANE", 9) + rcol("non-ANE", 9) + rcol("fraction", 11))
for r in results {
    let frac = String(format: "%.4f", r.aneFraction)
    print(col(r.chunkName, 12)
          + rcol(String(r.totalOps), 9)
          + rcol(String(r.aneOps), 9)
          + rcol(String(r.nonANE), 9)
          + rcol(frac, 11))
}

// Optional diff vs baseline.
if let bp = args.baselinePath {
    let prior = loadBaseline(bp)
    print("\nDiff vs \(bp.lastPathComponent):")
    for r in results {
        guard let p = prior[r.chunkName] else {
            print("  \(r.chunkName): NEW (no baseline)")
            continue
        }
        let drift = r.aneFraction - p.aneFraction
        let driftStr = String(format: "%+.4f", drift)
        let opDrift = (r.totalOps - p.totalOps)
        if abs(drift) < 1e-6 && opDrift == 0 {
            print("  \(r.chunkName): unchanged")
        } else {
            print("  \(r.chunkName): fraction \(driftStr) (ops \(opDrift))")
        }
    }
}

// Write current as new baseline if asked.
if let wb = args.writeBaselinePath {
    do {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(results)
        try data.write(to: wb, options: .atomic)
        print("\nBaseline written to \(wb.path)")
    } catch {
        FileHandle.standardError.write("failed to write baseline: \(error)\n"
            .data(using: .utf8)!)
    }
}

// Gate: any chunk below threshold = failure.
let failures = results.filter { $0.aneFraction < args.threshold || $0.totalOps == 0 }
if !failures.isEmpty {
    FileHandle.standardError.write("\nFAIL: \(failures.count) chunk(s) below threshold \(args.threshold):\n"
        .data(using: .utf8)!)
    for r in failures {
        let why = r.totalOps == 0 ? "no ops (load failure?)" :
            String(format: "%.4f < %.4f", r.aneFraction, args.threshold)
        FileHandle.standardError.write("  \(r.chunkName): \(why)\n".data(using: .utf8)!)
    }
    exit(1)
}
print("\nPASS: all chunks at or above threshold \(args.threshold).")
