// pplx-embed-bench — Swift fidelity + latency harness for the pplx-embed CoreML model.
//
// Native int8 model output is not readable from the Python CoreML bridge on macOS26,
// so this validates the int8 deliverable in Swift: it loads the model, runs the
// pre-tokenized fixtures from export_swift_fixtures.py, reads the int8 output via the
// dtype-agnostic NSNumber subscript, computes cosine vs the fp32-reference int8, and
// times warm latency.
//
// Usage:
//   swift run -c release pplx-embed-bench \
//       --model <bundleDir with encoder.mlpackage|.mlmodelc> \
//       --fixtures /tmp/pplx_fixtures.json \
//       --compute-units cpuAndNE --iters 20

import CoreML
import Foundation

struct Fixture: Decodable { let text: String; let input_ids: [Int]; let n: Int; let ref_int8: [Int] }
struct Fixtures: Decodable { let L: Int; let hf_repo: String; let items: [Fixture] }

func arg(_ name: String, _ def: String) -> String {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return def
}

func computeUnits(_ s: String) -> MLComputeUnits {
    switch s {
    case "all": return .all
    case "cpuOnly": return .cpuOnly
    case "cpuAndGPU": return .cpuAndGPU
    default: return .cpuAndNeuralEngine
    }
}

func cosine(_ a: [Double], _ b: [Double]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    if na < 1e-12 || nb < 1e-12 { return Double.nan }
    return dot / (sqrt(na) * sqrt(nb))
}

let modelDir = arg("--model", "")
let fixturesPath = arg("--fixtures", "/tmp/pplx_fixtures.json")
let cuName = arg("--compute-units", "cpuAndNE")
let iters = Int(arg("--iters", "20")) ?? 20

guard !modelDir.isEmpty else { FileHandle.standardError.write("--model required\n".data(using: .utf8)!); exit(2) }

let fx = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: URL(fileURLWithPath: fixturesPath)))
let L = fx.L

// Load (compile .mlpackage if no .mlmodelc).
let cfg = MLModelConfiguration()
cfg.computeUnits = computeUnits(cuName)
let bundle = URL(fileURLWithPath: modelDir)
let mlmodelc = bundle.appendingPathComponent("encoder.mlmodelc")
let mlpackage = bundle.appendingPathComponent("encoder.mlpackage")
let modelURL: URL
if FileManager.default.fileExists(atPath: mlmodelc.path) {
    modelURL = mlmodelc
} else if FileManager.default.fileExists(atPath: mlpackage.path) {
    print("compiling \(mlpackage.lastPathComponent) …")
    modelURL = try await MLModel.compileModel(at: mlpackage)
} else {
    // allow passing the .mlpackage / .mlmodelc directly
    modelURL = bundle.pathExtension == "mlpackage"
        ? try await MLModel.compileModel(at: bundle) : bundle
}
let model = try MLModel(contentsOf: modelURL, configuration: cfg)
print("loaded \(modelURL.lastPathComponent)  compute-units=\(cuName)  L=\(L)  fixtures=\(fx.items.count)")

func makeInputs(_ f: Fixture) throws -> MLDictionaryFeatureProvider {
    let ids = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
    let attn = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float16)
    for i in 0..<L {
        ids[i] = NSNumber(value: Int32(i < f.input_ids.count ? f.input_ids[i] : 0))
        attn[i] = NSNumber(value: Float(i < f.n ? 1.0 : 0.0))
    }
    return try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids, "attention_mask": attn])
}

func readEmbedding(_ out: MLFeatureProvider) -> ([Double], String)? {
    guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else { return nil }
    let d = arr.count
    var v = [Double](repeating: 0, count: d)
    for i in 0..<d { v[i] = arr[i].doubleValue }   // dtype-agnostic read
    return (v, "\(arr.dataType.rawValue)")
}

// Fidelity pass.
var cosines: [Double] = []
var dtypeSeen = ""
for f in fx.items {
    let prov = try makeInputs(f)
    let out = try model.prediction(from: prov)
    guard let (vec, dt) = readEmbedding(out) else { print("  no embedding output"); continue }
    dtypeSeen = dt
    let ref = f.ref_int8.map { Double($0) }
    let c = cosine(vec, ref)
    cosines.append(c)
    print(String(format: "  n=%4d  cos=%.6f  %@", f.n, c, String(f.text.prefix(28))))
}
let valid = cosines.filter { !$0.isNaN }
let minc = valid.min() ?? Double.nan
let meanc = valid.isEmpty ? Double.nan : valid.reduce(0, +) / Double(valid.count)
print(String(format: "[FIDELITY] mean=%.6f min=%.6f  (output MLMultiArray dataType.rawValue=%@)", meanc, minc, dtypeSeen))

// Latency pass (warm) on the longest fixture.
let longest = fx.items.max(by: { $0.n < $1.n })!
let provL = try makeInputs(longest)
_ = try model.prediction(from: provL)   // warm
var times: [Double] = []
for _ in 0..<iters {
    let t0 = DispatchTime.now().uptimeNanoseconds
    _ = try model.prediction(from: provL)
    let t1 = DispatchTime.now().uptimeNanoseconds
    times.append(Double(t1 - t0) / 1_000_000.0)
}
times.sort()
let med = times[times.count / 2]
let mean = times.reduce(0, +) / Double(times.count)
print(String(format: "[LATENCY] median=%.1fms mean=%.1fms min=%.1fms max=%.1fms  (n=%d, L=%d, tokens=%d, units=%@)",
             med, mean, times.first!, times.last!, iters, L, longest.n, cuName))

let gate = 0.997
print((minc >= gate) ? "PASS vs 0.997 gate" : "FAIL vs 0.997 gate")
