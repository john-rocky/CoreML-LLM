// Qwen3.5-0.8B decode-step throughput + drift benchmark.
//
// Loads `qwen3_5_0_8b_decode_fp16_mseq128.mlmodelc` (auto-compiled from the
// mlpackage built by conversion/test_qwen3_5_full_decode_trace.py). Runs
// tokens from each oracle prompt through the decode step starting from
// all-zero states (so output doesn't match a prefill-seeded context, but
// the numerics and throughput on the decode path are exactly what actual
// generation will use).
//
// Key Mac reference numbers (M4 Studio):
//   CPU fp16:   cos 0.99992  top-1 100%  ~50 tok/s
//   CPU+ANE:    cos 0.99     top-1 40%   ~40 tok/s  (same ANE fp16 drift)
//
// LiteRT-LM baseline on iPhone 17 Pro: 56.5 tok/s. Mac CPU path is already
// ~50 tok/s; iPhone CPU typically beats Mac laptop/studio CPU on such
// per-token workloads, so this is the path most likely to cross the
// LiteRT bar without needing ANE drift resolved.

import CoreML
import Foundation

@Observable
final class Qwen35DecodeBenchmark {
    struct PromptResult: Identifiable {
        let id = UUID()
        let prompt: String
        let S: Int
        let lastCos: Double
        let top1Match: Bool
        let top1InTop3: Bool
        let top1InTop5: Bool
        let top1InTop10: Bool
        let top5Overlap: Int   // ref top-5 ∩ ANE top-5 (0..5)
        let totalMs: Double
        var tokPerSec: Double { Double(S) / (totalMs / 1000.0) }
    }

    var status = "Idle"
    var running = false
    var results: [PromptResult] = []
    var meanCos: Double = 0
    var worstCos: Double = 1.0
    var top1Rate: Double = 0
    var top1InTop3Rate: Double = 0
    var top1InTop5Rate: Double = 0
    var top1InTop10Rate: Double = 0
    var meanTop5Overlap: Double = 0
    var meanTokPerSec: Double = 0
    var units: Qwen35Benchmark.UnitsChoice = .cpuOnly   // CPU is fastest + accurate

    private let vocab = 248320
    private let maxSeq = 128
    private let numLayers = 24
    private let rotaryDim = 64   // head_dim * 0.25 = 256 * 0.25

    // Linear-attention state shapes (per 18 layers, i % 4 != 3)
    private let linearConvShape: [Int] = [1, 6144, 4]
    private let linearRecShape:  [Int] = [1, 16, 128, 128]
    // Full-attention state shapes (per 6 layers, i % 4 == 3)
    private let fullKvShape:     [Int] = [1, 2, 128, 256]

    private func isLinearAttn(layer i: Int) -> Bool { i % 4 != 3 }

    // RoPE cos/sin precomputed once at init (max_seq, rotary_dim) as fp16.
    // @ObservationIgnored avoids the macro-vs-lazy conflict; filled in init().
    @ObservationIgnored private var cosTable: [Float] = []
    @ObservationIgnored private var sinTable: [Float] = []

    init() {
        cosTable = buildRopeTable(cos: true)
        sinTable = buildRopeTable(cos: false)
    }

    private func buildRopeTable(cos: Bool) -> [Float] {
        // rotary_dim = 64 (head_dim * 0.25 for partial RoPE)
        // theta[i] = 10_000_000 ** (-2i / rotary_dim), i in 0..<rotary_dim/2
        // angle[p, i] = p * theta[i]
        // cos/sin: duplicated in half-half layout: [cos(a), cos(a), sin(a), sin(a)]-ish
        // HF Qwen3_5 uses interleaved "rotate_half" which means the layout is
        // [a0, a1, ..., a_{rd/2-1}, a0, a1, ..., a_{rd/2-1}] - i.e. the table
        // of dim rd has first half duplicated onto second half.
        let rd = rotaryDim
        let half = rd / 2
        let base: Float = 10_000_000.0
        var out = [Float](repeating: 0, count: maxSeq * rd)
        for p in 0..<maxSeq {
            for i in 0..<half {
                let theta = powf(base, Float(-2 * i) / Float(rd))
                let angle = Float(p) * theta
                let v: Float = cos ? cosf(angle) : sinf(angle)
                out[p * rd + i]        = v
                out[p * rd + i + half] = v
            }
        }
        return out
    }

    private var decode: MLModel?
    private var oracle: Qwen35Benchmark.Oracle?

    // MARK: - Load

    /// Prefer Documents/<name>.mlmodelc (hot-swap via devicectl, avoids
    /// Xcode bundle caching and stale ANE E5 cache). Falls back to
    /// Documents/<name>.mlpackage (compile on-device) then app bundle.
    private func resolveModelURL(_ base: String) throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let docMLC = docs.appendingPathComponent("\(base).mlmodelc")
        if FileManager.default.fileExists(atPath: docMLC.path) {
            print("[Qwen35Decode] loading from Documents (mlmodelc): \(docMLC.path)")
            return docMLC
        }
        let docPkg = docs.appendingPathComponent("\(base).mlpackage")
        if FileManager.default.fileExists(atPath: docPkg.path) {
            print("[Qwen35Decode] compiling from Documents (mlpackage): \(docPkg.path)")
            return try MLModel.compileModel(at: docPkg)
        }
        if let bundled = Bundle.main.url(forResource: base, withExtension: "mlmodelc") {
            print("[Qwen35Decode] loading from app bundle: \(bundled.path)")
            return bundled
        }
        throw NSError(domain: "Qwen35DecodeBenchmark", code: 1,
                      userInfo: [NSLocalizedDescriptionKey:
                          "\(base) not found in Documents or app bundle"])
    }

    func loadArtifacts() throws {
        let mlcURL = try resolveModelURL("qwen3_5_0_8b_decode_fp16_mseq128")
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units.mlComputeUnits
        decode = try MLModel(contentsOf: mlcURL, configuration: cfg)

        guard let jsonURL = Bundle.main.url(
            forResource: "qwen3_5_oracle_ios", withExtension: "json"
        ) else {
            throw NSError(domain: "Qwen35DecodeBenchmark", code: 2,
                          userInfo: [NSLocalizedDescriptionKey:
                              "qwen3_5_oracle_ios.json not found in bundle"])
        }
        let data = try Data(contentsOf: jsonURL)
        oracle = try JSONDecoder().decode(Qwen35Benchmark.Oracle.self, from: data)
        status = "Loaded: units=\(units.rawValue)"
    }

    // MARK: - State tensors

    private func makeZeroStates() throws -> [String: MLMultiArray] {
        var dict: [String: MLMultiArray] = [:]
        for i in 0..<numLayers {
            let shapeA = isLinearAttn(layer: i) ? linearConvShape : fullKvShape
            let shapeB = isLinearAttn(layer: i) ? linearRecShape  : fullKvShape
            let a = try MLMultiArray(shape: shapeA.map { NSNumber(value: $0) },
                                      dataType: .float16)
            let b = try MLMultiArray(shape: shapeB.map { NSNumber(value: $0) },
                                      dataType: .float16)
            // MLMultiArray is zero-initialized; no need to memset
            dict["state_\(i)_a"] = a
            dict["state_\(i)_b"] = b
        }
        return dict
    }

    private func copyState(from dict: [String: MLMultiArray], into feat: inout [String: MLFeatureValue]) {
        for (k, v) in dict {
            feat[k] = MLFeatureValue(multiArray: v)
        }
    }

    // MARK: - Inference

    func run() async {
        running = true
        defer { running = false }
        results.removeAll()
        meanCos = 0; worstCos = 1.0; top1Rate = 0; meanTokPerSec = 0
        do {
            try loadArtifacts()
        } catch {
            status = "Load failed: \(error.localizedDescription)"
            return
        }
        guard let decode, let oracle else { return }

        var collected: [PromptResult] = []
        for (pi, rec) in oracle.records.enumerated() {
            status = "Prompt \(pi+1)/\(oracle.records.count): \(rec.prompt.prefix(30))..."

            var states: [String: MLMultiArray]
            do { states = try makeZeroStates() }
            catch { status = "State alloc failed"; return }

            var lastLogits: [Float] = []
            let t0 = Date()
            for t in 0..<rec.S_real {
                // input_token (1, 1) int32
                let inpTok = try? MLMultiArray(shape: [1, 1], dataType: .int32)
                guard let inpTok else { status = "tok alloc fail"; return }
                inpTok.dataPointer.assumingMemoryBound(to: Int32.self)[0] = rec.input_ids[t]

                // position (1,) float32
                let pos = try? MLMultiArray(shape: [1], dataType: .float32)
                guard let pos else { status = "pos alloc fail"; return }
                pos.dataPointer.assumingMemoryBound(to: Float.self)[0] = Float(t)

                // cos / sin (1, 1, rd) fp16 from table
                let cosArr = try? MLMultiArray(shape: [1, 1, NSNumber(value: rotaryDim)], dataType: .float16)
                let sinArr = try? MLMultiArray(shape: [1, 1, NSNumber(value: rotaryDim)], dataType: .float16)
                guard let cosArr, let sinArr else { status = "cos/sin alloc"; return }
                let cosP = cosArr.dataPointer.assumingMemoryBound(to: UInt16.self)
                let sinP = sinArr.dataPointer.assumingMemoryBound(to: UInt16.self)
                for i in 0..<rotaryDim {
                    cosP[i] = floatToFp16Bits(cosTable[t * rotaryDim + i])
                    sinP[i] = floatToFp16Bits(sinTable[t * rotaryDim + i])
                }

                var feat: [String: MLFeatureValue] = [
                    "input_token": MLFeatureValue(multiArray: inpTok),
                    "position":    MLFeatureValue(multiArray: pos),
                    "cos":         MLFeatureValue(multiArray: cosArr),
                    "sin":         MLFeatureValue(multiArray: sinArr),
                ]
                copyState(from: states, into: &feat)

                let provider: MLDictionaryFeatureProvider
                do {
                    provider = try MLDictionaryFeatureProvider(dictionary: feat)
                } catch {
                    status = "provider build: \(error.localizedDescription)"
                    return
                }

                let out: MLFeatureProvider
                do {
                    out = try await decode.prediction(from: provider)
                } catch {
                    status = "predict fail at t=\(t): \(error.localizedDescription)"
                    return
                }

                // Update states
                for i in 0..<numLayers {
                    if let a = out.featureValue(for: "new_state_\(i)_a")?.multiArrayValue,
                       let b = out.featureValue(for: "new_state_\(i)_b")?.multiArrayValue {
                        states["state_\(i)_a"] = a
                        states["state_\(i)_b"] = b
                    }
                }

                if t == rec.S_real - 1 {
                    guard let logitsArr = out.featureValue(for: "logits")?.multiArrayValue else {
                        status = "missing logits"; return
                    }
                    lastLogits = extractLogits(logitsArr, vocab: vocab)
                }
            }
            let totalMs = Date().timeIntervalSince(t0) * 1000

            let refBytes = Data(base64Encoded: rec.last_logits_fp16_b64)!
            let ref = fp16BytesToFloat32(refBytes, count: vocab)
            let c = cosine(lastLogits, ref)
            let aneTop10 = topKIndices(lastLogits, k: 10)
            let refTop10 = topKIndices(ref, k: 10)
            let aneTop5Set = Set(aneTop10.prefix(5))
            let refTop5Set = Set(refTop10.prefix(5))
            let match = (aneTop10[0] == rec.top1_id)
            let inTop3 = aneTop10.prefix(3).contains(rec.top1_id)
            let inTop5 = aneTop10.prefix(5).contains(rec.top1_id)
            let inTop10 = aneTop10.contains(rec.top1_id)
            let overlap5 = aneTop5Set.intersection(refTop5Set).count
            collected.append(PromptResult(prompt: rec.prompt, S: rec.S_real,
                                           lastCos: c, top1Match: match,
                                           top1InTop3: inTop3, top1InTop5: inTop5,
                                           top1InTop10: inTop10, top5Overlap: overlap5,
                                           totalMs: totalMs))
        }

        results = collected
        meanCos = collected.map { $0.lastCos }.reduce(0, +) / Double(collected.count)
        worstCos = collected.map { $0.lastCos }.min() ?? 0
        let N = Double(collected.count)
        top1Rate = Double(collected.filter { $0.top1Match }.count) / N
        top1InTop3Rate = Double(collected.filter { $0.top1InTop3 }.count) / N
        top1InTop5Rate = Double(collected.filter { $0.top1InTop5 }.count) / N
        top1InTop10Rate = Double(collected.filter { $0.top1InTop10 }.count) / N
        meanTop5Overlap = Double(collected.map { $0.top5Overlap }.reduce(0, +)) / N
        // Exclude first prompt (warmup)
        let throughputs = collected.dropFirst().map { $0.tokPerSec }
        meanTokPerSec = throughputs.isEmpty ? 0 : throughputs.reduce(0, +) / Double(throughputs.count)

        status = String(format:
            "Done — top1=%.0f%% top3=%.0f%% top5=%.0f%% ovl5=%.1f/5 cos=%.4f tok/s=%.1f (%@)",
            top1Rate * 100, top1InTop3Rate * 100, top1InTop5Rate * 100,
            meanTop5Overlap, meanCos, meanTokPerSec, units.rawValue as CVarArg)
    }

    private func topKIndices(_ v: [Float], k: Int) -> [Int] {
        // Partial top-K: single pass maintaining a sorted array of size k.
        // Sufficient for k=10 and v.count=248320.
        var topIdx = [Int](); topIdx.reserveCapacity(k)
        var topVal = [Float](); topVal.reserveCapacity(k)
        for i in 0..<v.count {
            let x = v[i]
            if topIdx.count < k {
                // Insert sorted
                var pos = 0
                while pos < topIdx.count && topVal[pos] > x { pos += 1 }
                topIdx.insert(i, at: pos)
                topVal.insert(x, at: pos)
            } else if x > topVal[k - 1] {
                var pos = 0
                while pos < k && topVal[pos] > x { pos += 1 }
                topIdx.insert(i, at: pos)
                topVal.insert(x, at: pos)
                topIdx.removeLast()
                topVal.removeLast()
            }
        }
        return topIdx
    }

    // MARK: - Numerics helpers (duplicated from Qwen35Benchmark for independence)

    private func extractLogits(_ arr: MLMultiArray, vocab: Int) -> [Float] {
        var out = [Float](repeating: 0, count: vocab)
        if arr.dataType == .float32 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for v in 0..<vocab { out[v] = p[v] }
        } else if arr.dataType == .float16 {
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            for v in 0..<vocab { out[v] = fp16BitsToFloat(p[v]) }
        } else {
            for v in 0..<vocab {
                out[v] = arr[[0, 0, NSNumber(value: v)] as [NSNumber]].floatValue
            }
        }
        return out
    }

    private func fp16BytesToFloat32(_ data: Data, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<count { out[i] = fp16BitsToFloat(p[i]) }
        }
        return out
    }

    private func fp16BitsToFloat(_ b: UInt16) -> Float { Float(Float16(bitPattern: b)) }
    private func floatToFp16Bits(_ f: Float) -> UInt16 { Float16(f).bitPattern }

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot: Double = 0, na: Double = 0, nb: Double = 0
        let n = min(a.count, b.count)
        for i in 0..<n { let x = Double(a[i]), y = Double(b[i]); dot += x*y; na += x*x; nb += y*y }
        return dot / ((na.squareRoot() * nb.squareRoot()) + 1e-12)
    }

    private func argmax(_ v: [Float]) -> Int {
        var best = 0; var bv: Float = -.infinity
        for i in 0..<v.count { if v[i] > bv { bv = v[i]; best = i } }
        return best
    }
}
