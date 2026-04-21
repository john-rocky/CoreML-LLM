// Qwen3.5-0.8B prefill ANE-drift + throughput benchmark for iPhone 17 Pro.
//
// This is a research harness, not a production feature. It loads a bundled
// `qwen3_5_0_8b_fp16_seq64.mlpackage` (auto-compiled by Xcode into the app
// bundle as `qwen3_5_0_8b_fp16_seq64.mlmodelc`), replays the 10 Phase-1
// oracle prompts, and reports:
//   - cosine similarity of the last-position logits vs the fp32 HF reference
//   - top-1 next-token match
//   - per-prompt prefill latency → tok/s
//
// The fp16 mlpackage was produced by
// conversion/test_qwen3_5_full_model_trace.py on Mac. Drift vs oracle on Mac
// ANE (M4) is worst-pos cos ≈ 0.843 / top-1 80%. Question this harness
// answers: does A18 ANE behave the same, better, or worse?

import CoreML
import Foundation

@Observable
final class Qwen35Benchmark {
    struct PromptResult: Identifiable {
        let id = UUID()
        let prompt: String
        let S: Int
        let lastCos: Double
        let top1Match: Bool
        let prefillMs: Double
    }

    struct OracleRecord: Decodable {
        let prompt: String
        let input_ids: [Int32]
        let S_real: Int
        let top1_id: Int
        let top1_text: String
        let last_logits_fp16_b64: String
    }

    struct Oracle: Decodable {
        let model_id: String
        let vocab_size: Int
        let seq_len_bundle: Int
        let records: [OracleRecord]
    }

    enum UnitsChoice: String, CaseIterable {
        case cpuAndNE = "CPU+ANE"
        case cpuOnly  = "CPU only"
        case all      = "All"

        var mlComputeUnits: MLComputeUnits {
            switch self {
            case .cpuAndNE: return .cpuAndNeuralEngine
            case .cpuOnly:  return .cpuOnly
            case .all:      return .all
            }
        }
    }

    var status = "Idle"
    var running = false
    var results: [PromptResult] = []
    var meanCos: Double = 0
    var worstCos: Double = 1.0
    var top1Rate: Double = 0
    var meanPrefillMs: Double = 0
    var tokensPerSecond: Double = 0
    var units: UnitsChoice = .cpuAndNE

    private var chunkA: MLModel?
    private var chunkB: MLModel?
    private var oracle: Oracle?
    private var vocabSize: Int = 248320
    private var seqLen: Int = 64
    private var hiddenSize: Int = 1024

    // MARK: - Bundle loading

    func loadArtifacts() throws {
        // Load chunked prefill: chunk_a (embed + layers 0-11) -> hidden
        //                      chunk_b (layers 12-23 + lm_head) -> logits
        // Chunking is Gemma 4's workaround for iOS 26.1 Core ML compiler
        // limits (can't fit a 24-layer single-graph mlpackage on ANE).
        guard let aURL = Bundle.main.url(
            forResource: "qwen3_5_chunk_a", withExtension: "mlmodelc"
        ) else {
            throw NSError(domain: "Qwen35Benchmark", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                              "qwen3_5_chunk_a.mlmodelc not found in app bundle"])
        }
        guard let bURL = Bundle.main.url(
            forResource: "qwen3_5_chunk_b", withExtension: "mlmodelc"
        ) else {
            throw NSError(domain: "Qwen35Benchmark", code: 2,
                          userInfo: [NSLocalizedDescriptionKey:
                              "qwen3_5_chunk_b.mlmodelc not found in app bundle"])
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units.mlComputeUnits
        chunkA = try MLModel(contentsOf: aURL, configuration: cfg)
        chunkB = try MLModel(contentsOf: bURL, configuration: cfg)

        // Oracle JSON
        guard let jsonURL = Bundle.main.url(
            forResource: "qwen3_5_oracle_ios", withExtension: "json"
        ) else {
            throw NSError(domain: "Qwen35Benchmark", code: 3,
                          userInfo: [NSLocalizedDescriptionKey:
                              "qwen3_5_oracle_ios.json not found in app bundle"])
        }
        let data = try Data(contentsOf: jsonURL)
        let decoded = try JSONDecoder().decode(Oracle.self, from: data)
        oracle = decoded
        vocabSize = decoded.vocab_size
        seqLen = decoded.seq_len_bundle
        status = "Loaded: \(decoded.records.count) prompts, vocab=\(vocabSize), seq=\(seqLen), units=\(units.rawValue)"
    }

    // MARK: - Inference

    func run() async {
        running = true
        defer { running = false }

        results.removeAll()
        meanCos = 0; worstCos = 1.0; top1Rate = 0
        meanPrefillMs = 0; tokensPerSecond = 0

        status = units == .cpuAndNE
            ? "Loading & compiling ANE graph (first run can take 30-90s for 1.5GB model)..."
            : "Loading model..."
        let loadStart = Date()
        do {
            try loadArtifacts()
        } catch {
            status = "Load failed: \(error.localizedDescription)"
            return
        }
        let loadMs = Date().timeIntervalSince(loadStart) * 1000
        print("[Qwen35Bench] model loaded in \(String(format: "%.0f", loadMs))ms")

        guard let chunkA, let chunkB, let oracle else { return }

        // Warm-up: run both chunks once so first-call overhead (ANE compile
        // + weight upload) doesn't skew timing. With chunked prefill, both
        // graphs need to be warmed up.
        status = units == .cpuAndNE
            ? "Warming up (ANE compile for both chunks, may take 30-90s)..."
            : "Warming up..."
        let warmStart = Date()
        do {
            let warm = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
            for i in 0..<seqLen { warm[[0, NSNumber(value: i)] as [NSNumber]] = 0 }
            let feat = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: warm)])
            let aOut = try await chunkA.prediction(from: feat)
            guard let hidden = aOut.featureValue(for: "hidden")?.multiArrayValue else {
                throw NSError(domain: "Qwen35Benchmark", code: 4,
                              userInfo: [NSLocalizedDescriptionKey: "chunk_a output 'hidden' missing"])
            }
            let bFeat = try MLDictionaryFeatureProvider(dictionary: ["hidden_in": MLFeatureValue(multiArray: hidden)])
            _ = try await chunkB.prediction(from: bFeat)
        } catch {
            status = "Warmup failed: \(error.localizedDescription)"
            return
        }
        let warmMs = Date().timeIntervalSince(warmStart) * 1000
        print("[Qwen35Bench] warmup took \(String(format: "%.0f", warmMs))ms")
        status = "Warmup done (\(Int(warmMs))ms). Starting benchmark..."

        // Diagnostic: feed the first oracle prompt's real input_ids (not zeros)
        // so we exercise the exact path the main loop uses, then dump:
        //   - logits tensor shape / strides / dtype
        //   - first 5 output logits at last real position
        //   - first 5 reference logits from oracle
        //   - argmax of output  vs  oracle top-1
        // This pinpoints whether the bug is in model output layout, reference
        // decode, or math.
        if let rec = oracle.records.first {
            do {
                let ids = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
                let p = ids.dataPointer.assumingMemoryBound(to: Int32.self)
                for i in 0..<seqLen { p[i] = i < rec.S_real ? rec.input_ids[i] : 0 }
                let feat = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: ids)])
                let aOut = try await chunkA.prediction(from: feat)
                guard let hidden = aOut.featureValue(for: "hidden")?.multiArrayValue else {
                    print("[Qwen35Bench] diagnostic: chunk_a output 'hidden' missing")
                    throw NSError(domain: "Qwen35Benchmark", code: 5,
                                  userInfo: [NSLocalizedDescriptionKey: "missing hidden"])
                }
                let bFeat = try MLDictionaryFeatureProvider(dictionary: ["hidden_in": MLFeatureValue(multiArray: hidden)])
                let w = try await chunkB.prediction(from: bFeat)
                if let arr = w.featureValue(for: "logits")?.multiArrayValue {
                    let dtypeStr = arr.dataType == .float32 ? "f32"
                                 : arr.dataType == .float16 ? "f16"
                                 : arr.dataType == .int32   ? "i32" : "?\(arr.dataType.rawValue)"
                    print("[Qwen35Bench] logits shape=\(arr.shape) strides=\(arr.strides) dtype=\(dtypeStr) count=\(arr.count) S_real=\(rec.S_real)")
                    let lastPos = rec.S_real - 1
                    let out = sliceLastPositionLogits(arr, seqLen: seqLen, position: lastPos, vocab: vocabSize)
                    let refBytes = Data(base64Encoded: rec.last_logits_fp16_b64)!
                    let ref = fp16ToFloat32(refBytes, count: vocabSize)
                    let outFirst = (0..<5).map { String(format: "%.4f", out[$0]) }.joined(separator: ",")
                    let refFirst = (0..<5).map { String(format: "%.4f", ref[$0]) }.joined(separator: ",")
                    let outMax = out.indices.max(by: { out[$0] < out[$1] }) ?? 0
                    let refMax = ref.indices.max(by: { ref[$0] < ref[$1] }) ?? 0
                    print("[Qwen35Bench] out[0..5]=\(outFirst)  argmax=\(outMax) val=\(out[outMax])")
                    print("[Qwen35Bench] ref[0..5]=\(refFirst)  argmax=\(refMax) val=\(ref[refMax])  oracle_top1=\(rec.top1_id)")
                    let outMin = out.min() ?? 0, outMaxV = out.max() ?? 0
                    let refMin = ref.min() ?? 0, refMaxV = ref.max() ?? 0
                    print("[Qwen35Bench] out range=[\(outMin),\(outMaxV)]  ref range=[\(refMin),\(refMaxV)]")
                }
            } catch {
                print("[Qwen35Bench] diagnostic predict failed: \(error)")
            }
        }

        var collected: [PromptResult] = []
        var totalMs = 0.0
        var totalTokens = 0

        for (pi, rec) in oracle.records.enumerated() {
            status = "Prompt \(pi+1)/\(oracle.records.count): \(rec.prompt.prefix(30))..."

            // Build input_ids (1, seq_len), pad with 0s
            guard let ids = try? MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32) else {
                status = "MLMultiArray alloc failed"; return
            }
            let ptr = ids.dataPointer.assumingMemoryBound(to: Int32.self)
            for i in 0..<seqLen {
                ptr[i] = i < rec.S_real ? rec.input_ids[i] : 0
            }

            let feat: MLDictionaryFeatureProvider
            do {
                feat = try MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": MLFeatureValue(multiArray: ids)
                ])
            } catch {
                status = "Feature build failed: \(error.localizedDescription)"
                return
            }

            // Time a chunked prediction: chunk_a -> hidden -> chunk_b -> logits.
            let t0 = Date()
            let out: MLFeatureProvider
            do {
                let aOut = try await chunkA.prediction(from: feat)
                guard let hidden = aOut.featureValue(for: "hidden")?.multiArrayValue else {
                    status = "chunk_a missing 'hidden'"; return
                }
                let bFeat = try MLDictionaryFeatureProvider(dictionary: [
                    "hidden_in": MLFeatureValue(multiArray: hidden)
                ])
                out = try await chunkB.prediction(from: bFeat)
            } catch {
                status = "Predict failed: \(error.localizedDescription)"
                return
            }
            let ms = Date().timeIntervalSince(t0) * 1000
            totalMs += ms
            totalTokens += rec.S_real

            // Extract last-position logits (position = S_real - 1)
            guard let logitsVal = out.featureValue(for: "logits")?.multiArrayValue else {
                status = "Missing 'logits' output"
                return
            }
            // Shape: (1, seq_len, vocab_size), float32
            let lastPos = rec.S_real - 1
            let lastLogits = sliceLastPositionLogits(logitsVal, seqLen: seqLen,
                                                     position: lastPos, vocab: vocabSize)

            // Reference: decode base64 fp16 → Float array
            let refBytes = Data(base64Encoded: rec.last_logits_fp16_b64)!
            let refFloats = fp16ToFloat32(refBytes, count: vocabSize)

            let cos = cosine(lastLogits, refFloats)
            let pred1 = argmax(lastLogits)
            let match = (pred1 == rec.top1_id)

            let result = PromptResult(prompt: rec.prompt, S: rec.S_real,
                                       lastCos: cos, top1Match: match, prefillMs: ms)
            collected.append(result)
        }

        results = collected
        meanCos = collected.map { $0.lastCos }.reduce(0, +) / Double(collected.count)
        worstCos = collected.map { $0.lastCos }.min() ?? 0
        top1Rate = Double(collected.filter { $0.top1Match }.count) / Double(collected.count)
        meanPrefillMs = totalMs / Double(collected.count)
        tokensPerSecond = Double(totalTokens) / (totalMs / 1000.0)

        status = String(format:
            "Done — mean cos=%.4f  worst=%.4f  top1=%.0f%%  prefill=%.1f ms  tok/s=%.1f",
            meanCos, worstCos, top1Rate * 100, meanPrefillMs, tokensPerSecond)
    }

    // MARK: - Numerics

    private func sliceLastPositionLogits(_ arr: MLMultiArray, seqLen: Int,
                                          position: Int, vocab: Int) -> [Float] {
        var out = [Float](repeating: 0, count: vocab)
        // Use the stride-aware byte offset per element so non-contiguous
        // MLMultiArray layouts (which CoreML produces after fp16->fp32 casts
        // on ANE outputs) work correctly. Raw dataPointer indexing assumes
        // default row-major stride and silently reads garbage otherwise.
        let strides = arr.strides.map(\.intValue)
        guard strides.count >= 3 else {
            for v in 0..<vocab {
                out[v] = arr[[0, NSNumber(value: position), NSNumber(value: v)] as [NSNumber]].floatValue
            }
            return out
        }
        let s0 = strides[0], s1 = strides[1], s2 = strides[2]
        let baseElems = 0 * s0 + position * s1
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for v in 0..<vocab { out[v] = p[baseElems + v * s2] }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            for v in 0..<vocab { out[v] = fp16BitsToFloat(p[baseElems + v * s2]) }
        default:
            for v in 0..<vocab {
                out[v] = arr[[0, NSNumber(value: position), NSNumber(value: v)] as [NSNumber]].floatValue
            }
        }
        return out
    }

    private func fp16ToFloat32(_ data: Data, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<count { out[i] = fp16BitsToFloat(p[i]) }
        }
        return out
    }

    private func fp16BitsToFloat(_ bits: UInt16) -> Float {
        // IEEE 754 half → float using the hardware _Float16 if available.
        #if arch(arm64)
        let h = Float16(bitPattern: bits)
        return Float(h)
        #else
        // Portable bit-level conversion
        let s = UInt32(bits >> 15) & 0x1
        let e = UInt32(bits >> 10) & 0x1F
        let m = UInt32(bits) & 0x3FF
        var f: UInt32 = 0
        if e == 0 {
            if m == 0 { f = s << 31 }
            else {
                var mm = m; var ee: UInt32 = 0
                while (mm & 0x400) == 0 { mm <<= 1; ee += 1 }
                f = (s << 31) | ((127 - 15 - ee + 1) << 23) | ((mm & 0x3FF) << 13)
            }
        } else if e == 31 {
            f = (s << 31) | (0xFF << 23) | (m << 13)
        } else {
            f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
        }
        return Float(bitPattern: f)
        #endif
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot: Double = 0, na: Double = 0, nb: Double = 0
        let n = min(a.count, b.count)
        for i in 0..<n {
            let av = Double(a[i]); let bv = Double(b[i])
            dot += av * bv; na += av * av; nb += bv * bv
        }
        let denom = (na.squareRoot() * nb.squareRoot()) + 1e-12
        return dot / denom
    }

    private func argmax(_ v: [Float]) -> Int {
        var best = 0; var bv: Float = -.infinity
        for i in 0..<v.count { if v[i] > bv { bv = v[i]; best = i } }
        return best
    }
}
