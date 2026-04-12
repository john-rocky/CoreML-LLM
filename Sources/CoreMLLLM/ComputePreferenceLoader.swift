//
//  ComputePreferenceLoader.swift
//  CoreMLLLM
//
//  Reads a `compute_preference.json` sidecar from an mlpackage directory
//  and returns a pre-configured `MLModelConfiguration`. Intended for the
//  GPU-prefill path (Approach A of docs/UNEXPLORED_APPROACHES.md): the
//  Python-side builder `conversion/build_prefill_gpu.py` drops a sidecar
//  into each mlpackage to signal the intended compute unit without baking
//  it into the graph.
//
//  Sidecar format (written by build_prefill_gpu.py):
//  {
//    "preferred_compute_units": "cpuAndGPU" | "cpuAndNeuralEngine" | "cpuOnly" | "all",
//    "notes": "...",
//    "xcode_min": "26.1",
//    "ios_min": "18",
//    "sliced_q": true
//  }
//
//  Usage:
//    let cfg = try ComputePreferenceLoader.configuration(for: prefillChunk1URL)
//    let model = try MLModel(contentsOf: prefillChunk1URL, configuration: cfg)
//

import CoreML
import Foundation

public enum ComputePreferenceError: Error, LocalizedError {
    case readFailed(String)
    case invalidSidecar(String)

    public var errorDescription: String? {
        switch self {
        case .readFailed(let s):     return "failed to read compute_preference.json: \(s)"
        case .invalidSidecar(let s): return "invalid compute_preference.json: \(s)"
        }
    }
}

public enum ComputePreferenceLoader {

    /// Load or infer an MLModelConfiguration for the mlpackage at `url`.
    ///
    /// If a `compute_preference.json` sidecar exists inside the mlpackage
    /// directory, it controls `computeUnits`. Otherwise returns a default
    /// configuration with `.cpuAndNeuralEngine` (the library's default).
    public static func configuration(for url: URL) throws -> MLModelConfiguration {
        let sidecar = url.appendingPathComponent("compute_preference.json")
        guard FileManager.default.fileExists(atPath: sidecar.path) else {
            // No sidecar → default ANE
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            return cfg
        }

        let data: Data
        do { data = try Data(contentsOf: sidecar) }
        catch { throw ComputePreferenceError.readFailed(error.localizedDescription) }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ComputePreferenceError.invalidSidecar("not a JSON object")
        }

        let pref = (json["preferred_compute_units"] as? String) ?? "cpuAndNeuralEngine"
        let cu: MLComputeUnits
        switch pref {
        case "cpuAndGPU":            cu = .cpuAndGPU
        case "cpuAndNeuralEngine":   cu = .cpuAndNeuralEngine
        case "cpuOnly":              cu = .cpuOnly
        case "all":                  cu = .all
        default:
            throw ComputePreferenceError.invalidSidecar("unknown preferred_compute_units: \(pref)")
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = cu
        return cfg
    }

    /// Convenience: load an MLModel with sidecar-driven compute units.
    public static func loadModel(contentsOf url: URL) throws -> MLModel {
        let cfg = try configuration(for: url)
        return try MLModel(contentsOf: url, configuration: cfg)
    }
}
