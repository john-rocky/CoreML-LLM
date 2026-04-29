// LookAhead K=8 probe harness.
//
// Goal: measure pure verify_qK=8 wall-clock on ANE so we can decide whether
// LookAhead / Jacobi decoding is worth full implementation. See
// docs/experiments/LOOKAHEAD_PROBE_HANDOFF.md for the go / no-go gate.
//
// Usage:
//   swift run -c release verify-k8-probe <model-dir> [iterations] [prompt]
//
// The bundle at <model-dir> must contain chunk{1-4}.mlmodelc built with
// --K 8 (conversion/build_verify_chunks.py). `benchVerifyK` must report 8
// or the probe aborts — we don't want to silently measure K=3.

import CoreMLLLM
import Darwin
import Foundation

@main
struct Probe {
    static func main() async {
        // Force verify chunks to load. ChunkedEngine gates verify loading on
        // LLM_EAGLE3_ENABLE=1 or SPECULATIVE_PROFILE != nil; we don't need
        // EAGLE-3 weights so SPECULATIVE_PROFILE is the cleaner switch.
        setenv("SPECULATIVE_PROFILE", "1", 1)

        let args = CommandLine.arguments
        guard args.count >= 2 else {
            fputs("usage: \(args[0]) <model-dir> [iterations=25] [prompt]\n", stderr)
            exit(2)
        }
        let modelDir = URL(fileURLWithPath: args[1])
        let iterations = args.count >= 3 ? (Int(args[2]) ?? 25) : 25
        let prompt = args.count >= 4
            ? args[3]
            : "The quick brown fox jumps over the lazy dog."

        do {
            print("[probe] model=\(modelDir.path)")
            print("[probe] iterations=\(iterations)")
            let t0 = CFAbsoluteTimeGetCurrent()
            let llm = try await CoreMLLLM.load(from: modelDir) { msg in
                print("[load] \(msg)")
            }
            let loadDt = CFAbsoluteTimeGetCurrent() - t0
            print("[probe] loaded in \(String(format: "%.1f", loadDt))s — model=\(llm.modelName)")

            guard let k = llm.benchVerifyK else {
                fputs("[probe] ERROR: verify chunks not loaded. Rebuild with --K 8 or check bundle.\n", stderr)
                exit(1)
            }
            guard k == 8 else {
                fputs("[probe] ERROR: benchVerifyK=\(k), expected 8. Rebuild conversion with --K 8.\n", stderr)
                exit(1)
            }
            print("[probe] benchVerifyK=\(k) ✓")

            // Prefill. The prefill result seeds the position counter; its
            // magnitude doesn't matter for verify cost (ANE dispatch is
            // position-independent once the mask is built).
            let (promptTokens, seed) = try await llm.benchPrefill(prompt)
            let prefillLen = promptTokens.count
            print("[probe] prefill tokens=\(prefillLen) seed=\(seed) pos=\(llm.benchCurrentPosition ?? -1)")

            // Fixed candidate pattern. Actual token values don't change
            // verify runtime cost — ANE runs the same graph regardless of
            // input integer values. Use a repeated pattern to keep the
            // probe deterministic.
            let candidates: [Int32] = [
                Int32(seed), 100, 200, 500, 1000, 2000, 5000, 10000,
            ]
            precondition(candidates.count == k)

            // Warmup: first call pays ANE compile-schedule cost. Discard 3.
            for _ in 0..<3 {
                _ = try llm.benchVerify(candidates)
            }

            // Timed loop. CFAbsoluteTimeGetCurrent resolution is ~µs on
            // darwin which is fine for 20-60 ms measurements.
            var times: [Double] = []
            times.reserveCapacity(iterations)
            for _ in 0..<iterations {
                let ts = CFAbsoluteTimeGetCurrent()
                _ = try llm.benchVerify(candidates)
                let dt = CFAbsoluteTimeGetCurrent() - ts
                times.append(dt * 1000.0)  // ms
            }

            let sorted = times.sorted()
            let minMs = sorted.first!
            let maxMs = sorted.last!
            let medianMs = sorted[sorted.count / 2]
            let p90Idx = min(sorted.count - 1, Int(Double(sorted.count) * 0.90))
            let p99Idx = min(sorted.count - 1, Int(Double(sorted.count) * 0.99))
            let p90Ms = sorted[p90Idx]
            let p99Ms = sorted[p99Idx]
            let meanMs = times.reduce(0, +) / Double(times.count)
            let varMs = times.map { pow($0 - meanMs, 2) }.reduce(0, +) / Double(times.count)
            let stdMs = sqrt(varMs)

            print("")
            print("[probe] verify_qK=\(k) timing over \(iterations) iters (ms)")
            print("[probe]   min    = \(String(format: "%6.2f", minMs))")
            print("[probe]   median = \(String(format: "%6.2f", medianMs))")
            print("[probe]   mean   = \(String(format: "%6.2f", meanMs))")
            print("[probe]   p90    = \(String(format: "%6.2f", p90Ms))")
            print("[probe]   p99    = \(String(format: "%6.2f", p99Ms))")
            print("[probe]   max    = \(String(format: "%6.2f", maxMs))")
            print("[probe]   std    = \(String(format: "%6.2f", stdMs))")

            // Decision gate — matches docs/experiments/LOOKAHEAD_PROBE_HANDOFF.md §2.4.
            let verdict: String
            if medianMs < 50 {
                verdict = "GO — LookAhead implementation is worth committing to"
            } else if medianMs <= 70 {
                verdict = "MARGINAL — defer unless Metal Phase 3 slips"
            } else {
                verdict = "KILL — K-linear scaling confirmed, LookAhead dead on ANE"
            }
            print("")
            print("[probe] verdict: \(verdict)")
            exit(0)
        } catch {
            fputs("[probe] error: \(error)\n", stderr)
            exit(1)
        }
    }
}
