// Gate Zero: 2-layer MLState stub predict on iPhone ANE.
//
// Loads `gate_zero_stub.mlmodelc` (or `gate_zero_stub.mlpackage`,
// compiled on-device) from the app's Documents directory. Calls
// `makeState()`, runs one predict on `.cpuAndNeuralEngine`, reports
// pass/fail.
//
// We only care about one outcome: did ANE refuse the state + slice_update
// recipe? Green light = proceed with Phase 1 converter. Red light
// (error -14 / MILCompilerForANE Error=(11)) = stop and diagnose.

import CoreML
import Foundation

@Observable
final class GateZeroBenchmark {
    var status = "Idle"
    var running = false
    var passed: Bool? = nil
    var predictMs: Double = 0
    var outputNorm: Double = 0
    var errorText: String? = nil

    private let hiddenSize = 2048

    private func resolveModelURL(_ base: String) throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let docMLC = docs.appendingPathComponent("\(base).mlmodelc")
        if FileManager.default.fileExists(atPath: docMLC.path) {
            return docMLC
        }
        let docPkg = docs.appendingPathComponent("\(base).mlpackage")
        if FileManager.default.fileExists(atPath: docPkg.path) {
            return try MLModel.compileModel(at: docPkg)
        }
        throw NSError(
            domain: "GateZero", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                       "\(base).mlmodelc / .mlpackage not found in Documents — "
                       + "sideload via scripts/gate_zero_push.sh first"])
    }

    func run() async {
        running = true
        status = "Loading..."
        passed = nil
        errorText = nil
        defer { running = false }

        do {
            let url = try resolveModelURL("gate_zero_stub")
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: url, configuration: cfg)

            let state = model.makeState()

            // hidden_states: (1, HIDDEN, 1, 1) fp16 with a single 1.0
            let hidden = try MLMultiArray(
                shape: [1, NSNumber(value: hiddenSize), 1, 1],
                dataType: .float16)
            let ptr = hidden.dataPointer.bindMemory(
                to: UInt16.self, capacity: hiddenSize)
            for i in 0..<hiddenSize { ptr[i] = 0 }
            // 1.0 as fp16 = 0x3C00
            ptr[0] = 0x3C00

            // current_pos: (1,) int32 = 0
            let pos = try MLMultiArray(
                shape: [1], dataType: .int32)
            pos.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = 0

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLFeatureValue(multiArray: hidden),
                "current_pos": MLFeatureValue(multiArray: pos),
            ])

            status = "Predicting on ANE..."
            let t0 = CFAbsoluteTimeGetCurrent()
            let out = try await model.prediction(
                from: provider, using: state,
                options: MLPredictionOptions())
            predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

            guard
                let fv = out.featureValue(for: "output_hidden_states"),
                let arr = fv.multiArrayValue
            else {
                throw NSError(
                    domain: "GateZero", code: 2,
                    userInfo: [NSLocalizedDescriptionKey:
                               "output_hidden_states missing from prediction"])
            }

            let n = arr.count
            let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            var sqsum = 0.0
            var finite = true
            for i in 0..<n {
                let bits = p[i]
                let f = Float(fp16ToFloat(bits))
                if !f.isFinite { finite = false }
                sqsum += Double(f) * Double(f)
            }
            outputNorm = sqsum.squareRoot()

            if !finite {
                passed = false
                status = "FAIL — output contains NaN/Inf (ANE dispatch likely rejected)"
                return
            }

            passed = true
            status = String(format:
                "PASS — predict=%.1fms, ||out||=%.3f. ANE accepted MLState + slice_update.",
                predictMs, outputNorm)
        } catch {
            passed = false
            errorText = "\(error)"
            status = "FAIL — \(error.localizedDescription)"
        }
    }

    private func fp16ToFloat(_ bits: UInt16) -> Float {
        let sign = (bits >> 15) & 0x1
        let exponent = (bits >> 10) & 0x1f
        let mantissa = bits & 0x3ff
        if exponent == 0 && mantissa == 0 { return sign == 1 ? -0 : 0 }
        if exponent == 0x1f {
            return mantissa == 0
                ? (sign == 1 ? -Float.infinity : Float.infinity)
                : Float.nan
        }
        var bits32: UInt32
        if exponent == 0 {
            // denormal
            var m = UInt32(mantissa)
            var e: UInt32 = 0
            while (m & 0x400) == 0 { m <<= 1; e &+= 1 }
            m &= 0x3ff
            bits32 = (UInt32(sign) << 31) | ((127 &- 15 &- e) << 23) | (m << 13)
        } else {
            bits32 = (UInt32(sign) << 31)
                | ((UInt32(exponent) &+ (127 &- 15)) << 23)
                | (UInt32(mantissa) << 13)
        }
        return Float(bitPattern: bits32)
    }
}
