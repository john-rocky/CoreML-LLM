// MoE dispatch-latency probe.
//
// Phase B's Python ct.models.MLModel.predict() numbers were
// overhead-contaminated. This harness measures the REAL Swift-side
// dispatch latency the production path would see: warm model,
// pre-allocated reused input buffer (IOSurface-backed when possible),
// tight prediction() loop.
//
// Designs measured (each in fp16 AND INT4 — production uses INT4):
//   single_expert      — one SwiGLU expert
//   layer_gather       — 60 experts as in-graph constants + runtime
//                        gather, one layer's routed experts per call
//   dense_backbone_NL  — N fused STATIC layers (attn + router + shared
//                        expert + norms) — the part that CAN chunk
//   multifunction_N    — N expert functions, selected by functionName
//
// Build artifacts first:
//   pyenv shell lama-cml
//   python conversion/phase_b_redux_build.py --out-dir /tmp/moe_probe
//
// Then:
//   swift run -c release moe-dispatch-probe /tmp/moe_probe [iterations]

import CoreML
import CoreVideo
import Darwin
import Foundation

let HIDDEN = 2048
let TOP_K = 4
let N_LAYERS = 24  // Qwen1.5-MoE-A2.7B

// ---- input buffers --------------------------------------------------

func makeInput() -> (MLMultiArray, Bool) {
    let shape: [NSNumber] = [1, NSNumber(value: HIDDEN), 1, 1]
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        kCVPixelBufferMetalCompatibilityKey as String: true,
    ]
    let st = CVPixelBufferCreate(
        kCFAllocatorDefault, 1, HIDDEN,
        kCVPixelFormatType_OneComponent16Half,
        attrs as CFDictionary, &pb)
    if st == kCVReturnSuccess, let pb = pb {
        CVPixelBufferLockBaseAddress(pb, [])
        memset(CVPixelBufferGetBaseAddress(pb)!, 0,
               CVPixelBufferGetDataSize(pb))
        CVPixelBufferUnlockBaseAddress(pb, [])
        if let arr = try? MLMultiArray(pixelBuffer: pb, shape: shape) {
            return (arr, true)
        }
    }
    let arr = try! MLMultiArray(shape: shape, dataType: .float16)
    memset(arr.dataPointer, 0, HIDDEN * MemoryLayout<UInt16>.stride)
    return (arr, false)
}

// ---- timing ---------------------------------------------------------

struct Stats {
    let min, median, mean, p90, p99, max: Double
    static func from(_ raw: [Double]) -> Stats {
        let s = raw.sorted()
        let mean = raw.reduce(0, +) / Double(raw.count)
        return Stats(
            min: s.first!, median: s[s.count / 2], mean: mean,
            p90: s[Swift.min(s.count - 1, Int(Double(s.count) * 0.90))],
            p99: s[Swift.min(s.count - 1, Int(Double(s.count) * 0.99))],
            max: s.last!)
    }
    func line(_ label: String) -> String {
        String(format: "%@  min %7.3f  med %7.3f  mean %7.3f  p90 %7.3f  p99 %7.3f",
               label.padding(toLength: 34, withPad: " ", startingAt: 0),
               min, median, mean, p90, p99)
    }
}

func unitName(_ u: MLComputeUnits) -> String {
    switch u {
    case .cpuOnly: return "cpu"
    case .cpuAndGPU: return "gpu"
    case .cpuAndNeuralEngine: return "ane"
    case .all: return "all"
    @unknown default: return "?"
    }
}

func resolveCompiled(_ dir: URL, _ name: String) async -> URL? {
    let mlmodelc = dir.appendingPathComponent("\(name).mlmodelc")
    if FileManager.default.fileExists(atPath: mlmodelc.path) { return mlmodelc }
    let mlpackage = dir.appendingPathComponent("\(name).mlpackage")
    guard FileManager.default.fileExists(atPath: mlpackage.path) else { return nil }
    do {
        let compiled = try await MLModel.compileModel(at: mlpackage)
        try? FileManager.default.removeItem(at: mlmodelc)
        try FileManager.default.copyItem(at: compiled, to: mlmodelc)
        return mlmodelc
    } catch {
        fputs("  [\(name)] compile failed: \(error)\n", stderr)
        return nil
    }
}

// ---- probes ---------------------------------------------------------

func timeLoop(_ iterations: Int, _ body: () -> Void) -> Stats {
    for _ in 0..<10 { body() }  // warm-up
    var times: [Double] = []
    times.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let t = CFAbsoluteTimeGetCurrent()
        body()
        times.append((CFAbsoluteTimeGetCurrent() - t) * 1000.0)
    }
    return Stats.from(times)
}

func probeSingle(_ url: URL, _ units: MLComputeUnits, _ label: String,
                 _ iterations: Int) -> Stats? {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units
    guard let model = try? MLModel(contentsOf: url, configuration: cfg) else {
        fputs("  [\(label)] load failed\n", stderr); return nil
    }
    let (input, _) = makeInput()
    guard let provider = try? MLDictionaryFeatureProvider(
        dictionary: ["x_bc1t": MLFeatureValue(multiArray: input)]) else { return nil }
    let stats = timeLoop(iterations) { _ = try? model.prediction(from: provider) }
    print(stats.line(label)); return stats
}

func probeLayerGather(_ url: URL, _ units: MLComputeUnits, _ label: String,
                      _ iterations: Int) -> Stats? {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units
    guard let model = try? MLModel(contentsOf: url, configuration: cfg) else {
        fputs("  [\(label)] load failed\n", stderr); return nil
    }
    let (x, _) = makeInput()
    guard let idx = try? MLMultiArray(shape: [4], dataType: .int32),
          let w = try? MLMultiArray(shape: [4], dataType: .float16) else { return nil }
    for i in 0..<4 { idx[i] = NSNumber(value: Int32(i)); w[i] = NSNumber(value: Float(0.25)) }
    guard let provider = try? MLDictionaryFeatureProvider(dictionary: [
        "x_bc1t": MLFeatureValue(multiArray: x),
        "topk_idx": MLFeatureValue(multiArray: idx),
        "topk_weights": MLFeatureValue(multiArray: w),
    ]) else { return nil }
    var it = 0
    let stats = timeLoop(iterations) {
        for k in 0..<4 { idx[k] = NSNumber(value: Int32((it * 4 + k) % 60)) }
        it += 1
        _ = try? model.prediction(from: provider)
    }
    print(stats.line(label)); return stats
}

func probeDenseBackbone(_ url: URL, _ units: MLComputeUnits, _ label: String,
                        _ iterations: Int, kvWindow: Int) -> Stats? {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units
    guard let model = try? MLModel(contentsOf: url, configuration: cfg) else {
        fputs("  [\(label)] load failed\n", stderr); return nil
    }
    let (x, _) = makeInput()
    guard let kv = try? MLMultiArray(
        shape: [1, NSNumber(value: HIDDEN), 1, NSNumber(value: kvWindow)],
        dataType: .float16) else { return nil }
    memset(kv.dataPointer, 0, HIDDEN * kvWindow * MemoryLayout<UInt16>.stride)
    guard let provider = try? MLDictionaryFeatureProvider(dictionary: [
        "x_bc1t": MLFeatureValue(multiArray: x),
        "kv_window": MLFeatureValue(multiArray: kv),
    ]) else { return nil }
    let stats = timeLoop(iterations) { _ = try? model.prediction(from: provider) }
    print(stats.line(label)); return stats
}

/// Load N expert functions as N handles via functionName, time bursts
/// of TOP_K calls — one decode layer's routed-expert dispatch with no
/// gather (ANE-viable, unlike layer_gather).
func probeMultifunction(_ url: URL, _ nFuncs: Int, _ units: MLComputeUnits,
                        _ label: String, _ iterations: Int) -> Stats? {
    var handles: [MLModel] = []
    for i in 0..<nFuncs {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        cfg.functionName = "expert_\(i)"
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else {
            fputs("  [\(label)] load expert_\(i) failed\n", stderr); return nil
        }
        handles.append(m)
    }
    let (input, _) = makeInput()
    guard let provider = try? MLDictionaryFeatureProvider(
        dictionary: ["x_bc1t": MLFeatureValue(multiArray: input)]) else { return nil }
    for h in handles { for _ in 0..<3 { _ = try? h.prediction(from: provider) } }
    var it = 0
    let stats = timeLoop(iterations) {
        let base = (it * TOP_K) % nFuncs
        it += 1
        for k in 0..<TOP_K {
            _ = try? handles[(base + k) % nFuncs].prediction(from: provider)
        }
    }
    print(stats.line(label)); return stats
}

// ---- main -----------------------------------------------------------

let args = CommandLine.arguments
guard args.count >= 2 else {
    fputs("usage: \(args[0]) <probe-dir> [iterations=200]\n", stderr); exit(2)
}
let probeDir = URL(fileURLWithPath: args[1])
let iterations = args.count >= 3 ? (Int(args[2]) ?? 200) : 200
let units: [MLComputeUnits] = [.cpuAndNeuralEngine, .cpuAndGPU, .all]

print("=== MoE dispatch-latency probe (probe-dir: \(probeDir.path), iter: \(iterations)) ===\n")

// stats keyed by "design/variant/unit"
var S: [String: Stats] = [:]

print("--- single_expert ---")
for variant in ["single_expert", "single_expert_int4"] {
    if let url = await resolveCompiled(probeDir, variant) {
        for u in units {
            let label = "\(variant)/\(unitName(u))"
            if let s = probeSingle(url, u, label, iterations) { S[label] = s }
        }
    }
}

print("\n--- layer_gather (60 experts constants + gather, 1 layer/call) ---")
for variant in ["layer_gather", "layer_gather_int4"] {
    if let url = await resolveCompiled(probeDir, variant) {
        for u in units {
            let label = "\(variant)/\(unitName(u))"
            if let s = probeLayerGather(url, u, label, iterations) { S[label] = s }
        }
    }
}

// multifunction: fp16 + INT4. ANE can't gather but CAN run
// function-selected INT4 experts — this is the all-ANE routed path.
var nFuncs = 16
if let entries = try? FileManager.default.contentsOfDirectory(atPath: probeDir.path) {
    for e in entries where e.hasPrefix("multifunction_") && !e.contains("_int4") {
        let d = e.replacingOccurrences(of: "multifunction_", with: "")
            .replacingOccurrences(of: ".mlpackage", with: "")
            .replacingOccurrences(of: ".mlmodelc", with: "")
        if let n = Int(d) { nFuncs = n }
    }
}
print("\n--- multifunction expert dispatch (TOP_K=\(TOP_K) burst, N=\(nFuncs)) ---")
for variant in ["multifunction_\(nFuncs)", "multifunction_\(nFuncs)_int4"] {
    if let url = await resolveCompiled(probeDir, variant) {
        for u in units {
            let label = "\(variant)/\(unitName(u))"
            if let s = probeMultifunction(url, nFuncs, u, label, iterations) {
                S[label] = s
            }
        }
    }
}

// dense backbone: 1L (correct per-layer cost) AND 6L (chunked-static analysis)
print("\n--- dense_backbone (1L = correct per-layer; 6L = chunked-static) ---")
for variant in ["dense_backbone_1L", "dense_backbone_1L_int4",
                "dense_backbone_6L", "dense_backbone_6L_int4"] {
    if let url = await resolveCompiled(probeDir, variant) {
        for u in units {
            let label = "\(variant)/\(unitName(u))"
            if let s = probeDenseBackbone(url, u, label, iterations, kvWindow: 256) {
                S[label] = s
            }
        }
    }
}

// ---- extrapolation --------------------------------------------------
// Correct structure: per layer = dense_1L + routed-expert dispatch, in
// strict sequence (routed output feeds next layer's dense). The dense
// part can't fuse 6 layers because routed experts interleave.
// Routed-expert options measured: layer_gather (gather, GPU-only) and
// multifunction (TOP_K separate function calls, ANE-viable).

print("\n=== Full-decode extrapolation (Qwen1.5-MoE-A2.7B, \(N_LAYERS) layers) ===")
print("Per layer = dense_backbone_1L + routed-expert dispatch (serial). + ~2ms head.\n")

struct DesignResult { let label: String; let msPerToken: Double; let tps: Double }
var designs: [DesignResult] = []

func addDesign(_ name: String, dbKey: String, routedKey: String, routedIsBurst: Bool) {
    guard let db = S[dbKey], let routed = S[routedKey] else { return }
    let densePart = db.median * Double(N_LAYERS)
    // routed: layer_gather/multifunction-burst are already one-layer cost.
    let routedPart = routed.median * Double(N_LAYERS)
    let total = densePart + routedPart + 2.0
    designs.append(DesignResult(label: name, msPerToken: total, tps: 1000.0 / total))
    print(String(format: "  %@: dense %.1f + routed %.1f + head 2.0 = %.1f ms → %.1f tok/s",
                 name.padding(toLength: 38, withPad: " ", startingAt: 0) as NSString,
                 densePart, routedPart, total, 1000.0 / total))
}

// Design 1: layer_gather routed (GPU-only — ANE gather is dead)
for v in ["", "_int4"] {
    for u in ["gpu"] {
        addDesign("gather \(v.isEmpty ? "fp16" : "int4")/dense+routed \(u)",
                  dbKey: "dense_backbone_1L\(v)/\(u)",
                  routedKey: "layer_gather\(v)/\(u)", routedIsBurst: false)
    }
}
// Design 2: multifunction routed (ANE-viable)
for v in ["", "_int4"] {
    for u in ["ane", "gpu", "all"] {
        addDesign("multifn \(v.isEmpty ? "fp16" : "int4")/dense+routed \(u)",
                  dbKey: "dense_backbone_1L\(v)/\(u)",
                  routedKey: "multifunction_\(nFuncs)\(v)/\(u)", routedIsBurst: true)
    }
}
// Design 3: mixed — dense on ANE-INT4 (its best), routed via multifunction
// ANE-INT4 (its best). Same unit, so no cross-unit handoff penalty.
if let db = S["dense_backbone_1L_int4/ane"],
   let mf = S["multifunction_\(nFuncs)_int4/ane"] {
    let total = db.median * Double(N_LAYERS) + mf.median * Double(N_LAYERS) + 2.0
    designs.append(DesignResult(label: "BEST all-ANE-INT4 (dense+multifn)",
                                msPerToken: total, tps: 1000.0 / total))
}

print("\nReference: Gemma 4 E2B ~35 tok/s baseline; 1.5× target ~52 tok/s")

// ---- gate -----------------------------------------------------------

print("\n=== Phase B redux gate ===")
let best = designs.max(by: { $0.tps < $1.tps })
if let b = best {
    if b.tps >= 52.0 {
        print(String(format: "PASS — best design '%@' → %.1f tok/s ≥ 52 (1.5×). Build the full prototype.", b.label as NSString, b.tps))
    } else if b.tps >= 42.0 {
        print(String(format: "PROMISING — best '%@' → %.1f tok/s. Above baseline; needs per-op tuning or another lever to hit 1.5×.", b.label as NSString, b.tps))
    } else if b.tps >= 35.0 {
        print(String(format: "PARTIAL — best '%@' → %.1f tok/s, ~parity with Gemma 4.", b.label as NSString, b.tps))
    } else {
        print(String(format: "WEAK — best '%@' → %.1f tok/s.", b.label as NSString, b.tps))
    }
}
exit(0)
