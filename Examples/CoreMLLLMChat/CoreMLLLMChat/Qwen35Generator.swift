// Phase 4e-4: end-to-end Qwen3.5-0.8B generation on device.
//
// Loads BOTH the stateful prefill mlpackage and the stateful decode mlpackage,
// chains them (prefill emits initial states -> decode consumes them), and
// runs a greedy generation loop with argmax sampling. Produces output token
// IDs; detokenization is out of scope here (the user can bundle or fetch a
// Qwen tokenizer separately).
//
// Required bundle resources:
//   qwen3_5_0_8b_prefill_stateful_fp16_seq64.mlmodelc
//   qwen3_5_0_8b_decode_fp16_mseq128.mlmodelc
//
// Dimensions (Qwen3.5-0.8B):
//   hidden=1024  num_layers=24 (18 linear + 6 full)  vocab=248320
//   linear state per layer: conv (1,6144,4) + rec (1,16,128,128)
//   full   state per layer: k_cache + v_cache (1,2,128,256)

import CoreML
import Foundation

@Observable
final class Qwen35Generator {
    struct Config {
        let seqLen: Int           // prefill fixed seq length (64)
        let maxSeq: Int           // decode + prefill max length (128)
        let vocab: Int            // 248320
        let numLayers: Int        // 24
        let rotaryDim: Int        // head_dim * 0.25 = 64
        let units: MLComputeUnits
        static let `default` = Config(seqLen: 64, maxSeq: 128, vocab: 248320,
                                      numLayers: 24, rotaryDim: 64, units: .cpuOnly)
    }

    var status = "Idle"
    var running = false
    var generatedIds: [Int32] = []
    var prefillMs: Double = 0
    var decodeMsAvg: Double = 0
    var tokensPerSecond: Double = 0

    private var prefill: MLModel?
    private var decode: MLModel?
    private let cfg: Config

    // RoPE cos/sin tables (max_seq, rotary_dim) — Qwen3.5 text: theta=1e7, partial=0.25
    private lazy var cosTable: [Float] = buildRope(isCos: true)
    private lazy var sinTable: [Float] = buildRope(isCos: false)

    init(cfg: Config = .default) {
        self.cfg = cfg
    }

    private func buildRope(isCos: Bool) -> [Float] {
        let rd = cfg.rotaryDim
        let half = rd / 2
        let base: Float = 10_000_000.0
        var out = [Float](repeating: 0, count: cfg.maxSeq * rd)
        for p in 0..<cfg.maxSeq {
            for i in 0..<half {
                let theta = powf(base, Float(-2 * i) / Float(rd))
                let a = Float(p) * theta
                let v = isCos ? cosf(a) : sinf(a)
                out[p * rd + i]        = v
                out[p * rd + i + half] = v
            }
        }
        return out
    }

    private func isLinearAttn(_ i: Int) -> Bool { i % 4 != 3 }

    // MARK: - Loading

    func load() throws {
        guard let pURL = Bundle.main.url(
            forResource: "qwen3_5_0_8b_prefill_stateful_fp16_seq64", withExtension: "mlmodelc"
        ) else {
            throw NSError(domain: "Qwen35Generator", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "prefill mlmodelc not found"])
        }
        guard let dURL = Bundle.main.url(
            forResource: "qwen3_5_0_8b_decode_fp16_mseq128", withExtension: "mlmodelc"
        ) else {
            throw NSError(domain: "Qwen35Generator", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "decode mlmodelc not found"])
        }
        let c = MLModelConfiguration()
        c.computeUnits = cfg.units
        prefill = try MLModel(contentsOf: pURL, configuration: c)
        decode = try MLModel(contentsOf: dURL, configuration: c)
        status = "Loaded (\(cfg.units == .cpuOnly ? "CPU" : cfg.units == .cpuAndNeuralEngine ? "CPU+ANE" : "All"))"
    }

    // MARK: - Generation entry

    /// Greedy-generate up to `maxNewTokens` tokens starting from `inputIds`.
    /// Returns the new tokens only (not echo-ed inputs).
    @discardableResult
    func generate(inputIds: [Int32], maxNewTokens: Int = 32) async throws -> [Int32] {
        running = true
        defer { running = false }
        generatedIds.removeAll()

        if prefill == nil || decode == nil { try load() }
        guard let prefill, let decode else { return [] }

        let S = inputIds.count
        guard S > 0, S <= cfg.seqLen else {
            throw NSError(domain: "Qwen35Generator", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "input length \(S) must be in (0, \(cfg.seqLen)]"])
        }

        // --- 1. Prefill ---
        status = "Prefill..."
        let prefillInputs = try makePrefillInputs(inputIds: inputIds)
        let t0 = Date()
        let pOut = try await prefill.prediction(from: prefillInputs)
        prefillMs = Date().timeIntervalSince(t0) * 1000

        // Extract logits[S-1] → first generated token
        guard let pLogits = pOut.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "Qwen35Generator", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "prefill: no logits"])
        }
        var nextToken = argmaxAtPosition(pLogits, position: S - 1, vocab: cfg.vocab)
        generatedIds.append(nextToken)

        // Extract per-layer states from prefill outputs
        var states: [String: MLMultiArray] = [:]
        for i in 0..<cfg.numLayers {
            guard let sa = pOut.featureValue(for: "state_\(i)_a")?.multiArrayValue,
                  let sb = pOut.featureValue(for: "state_\(i)_b")?.multiArrayValue else {
                throw NSError(domain: "Qwen35Generator", code: 5,
                    userInfo: [NSLocalizedDescriptionKey: "prefill: missing state_\(i)"])
            }
            states["state_\(i)_a"] = sa
            states["state_\(i)_b"] = sb
        }

        // --- 2. Decode loop ---
        status = "Decoding..."
        var decodeTotal = 0.0
        let decodeStart = Date()
        var position = S
        for step in 0..<(maxNewTokens - 1) {
            if position >= cfg.maxSeq { break }  // ran out of KV cache room
            let stepStart = Date()
            let dInputs = try makeDecodeInputs(token: nextToken, position: position, states: states)
            let dOut = try await decode.prediction(from: dInputs)
            decodeTotal += Date().timeIntervalSince(stepStart) * 1000

            // Update states
            for i in 0..<cfg.numLayers {
                if let a = dOut.featureValue(for: "new_state_\(i)_a")?.multiArrayValue,
                   let b = dOut.featureValue(for: "new_state_\(i)_b")?.multiArrayValue {
                    states["state_\(i)_a"] = a
                    states["state_\(i)_b"] = b
                }
            }

            guard let dLogits = dOut.featureValue(for: "logits")?.multiArrayValue else { break }
            nextToken = argmaxAtPosition(dLogits, position: 0, vocab: cfg.vocab)  // (1,1,V): pos=0
            generatedIds.append(nextToken)
            position += 1
            status = "Decoding... \(step + 2)/\(maxNewTokens)"
        }

        let totalDecodeMs = Date().timeIntervalSince(decodeStart) * 1000
        let decodedCount = max(generatedIds.count - 1, 1)
        decodeMsAvg = totalDecodeMs / Double(decodedCount)
        tokensPerSecond = Double(generatedIds.count) / ((prefillMs + totalDecodeMs) / 1000.0)
        status = String(format: "Done: %d tokens, prefill=%.0fms, decode avg=%.1fms/tok, %.1f tok/s",
                         generatedIds.count, prefillMs, decodeMsAvg, tokensPerSecond)
        return generatedIds
    }

    // MARK: - Input builders

    private func makePrefillInputs(inputIds: [Int32]) throws -> MLDictionaryFeatureProvider {
        let ids = try MLMultiArray(shape: [1, NSNumber(value: cfg.seqLen)], dataType: .int32)
        let p = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<cfg.seqLen { p[i] = i < inputIds.count ? inputIds[i] : 0 }
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids)
        ])
    }

    private func makeDecodeInputs(token: Int32, position: Int,
                                   states: [String: MLMultiArray]
                                   ) throws -> MLDictionaryFeatureProvider {
        let tok = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tok.dataPointer.assumingMemoryBound(to: Int32.self)[0] = token

        let pos = try MLMultiArray(shape: [1], dataType: .float32)
        pos.dataPointer.assumingMemoryBound(to: Float.self)[0] = Float(position)

        let rd = cfg.rotaryDim
        let cosArr = try MLMultiArray(shape: [1, 1, NSNumber(value: rd)], dataType: .float16)
        let sinArr = try MLMultiArray(shape: [1, 1, NSNumber(value: rd)], dataType: .float16)
        let cp = cosArr.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sp = sinArr.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<rd {
            cp[i] = Float16(cosTable[position * rd + i]).bitPattern
            sp[i] = Float16(sinTable[position * rd + i]).bitPattern
        }

        var feat: [String: MLFeatureValue] = [
            "input_token": MLFeatureValue(multiArray: tok),
            "position":    MLFeatureValue(multiArray: pos),
            "cos":         MLFeatureValue(multiArray: cosArr),
            "sin":         MLFeatureValue(multiArray: sinArr),
        ]
        for (k, v) in states {
            feat[k] = MLFeatureValue(multiArray: v)
        }
        return try MLDictionaryFeatureProvider(dictionary: feat)
    }

    // MARK: - Argmax over (1, S, V) or (1, 1, V)

    private func argmaxAtPosition(_ arr: MLMultiArray, position: Int, vocab: Int) -> Int32 {
        let base = position * vocab
        var best: Int32 = 0
        if arr.dataType == .float32 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            var bv: Float = -.infinity
            for v in 0..<vocab {
                let x = p[base + v]
                if x > bv { bv = x; best = Int32(v) }
            }
        } else if arr.dataType == .float16 {
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            var bv: Float = -.infinity
            for v in 0..<vocab {
                let x = Float(Float16(bitPattern: p[base + v]))
                if x > bv { bv = x; best = Int32(v) }
            }
        }
        return best
    }
}
