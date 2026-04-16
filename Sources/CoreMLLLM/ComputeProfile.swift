//
//  ComputeProfile.swift
//  CoreMLLLM
//
//  First-class, user-facing compute-unit selector. Maps a semantic profile
//  (efficient / balanced / performance / custom) onto `MLComputeUnits`.
//
//  Rationale: the Gemma 4 E2B INT4 chunked layout was optimised for the
//  Neural Engine (low power, fixed throughput), but iPhone 17 Pro's A19 Pro
//  GPU — exposed via Metal Performance Primitives / Tensor cores — can
//  outperform ANE for *decode* when the per-dispatch latency dominates.
//  Running the 4-chunk graph on `.cpuAndGPU` bypasses the ANE scheduler and
//  typically halves the ~2.3 ms round-trip, at the cost of higher battery
//  drain and thermal load. `.all` lets Core ML partition between both.
//
//  See docs/POWER_BENCH.md for measured numbers and the tethered
//  `powermetrics` instructions.
//

import CoreML
import Foundation

/// User-facing compute-profile selector. Maps onto `MLComputeUnits` but adds
/// semantic intent (power vs speed) so the UI can expose a stable toggle
/// even if Apple adds new compute units in the future.
public enum ComputeProfile: Sendable, Equatable, CustomStringConvertible {
    /// Low power, default behaviour. Uses `.cpuAndNeuralEngine` — the ANE
    /// draws roughly a third of the GPU's power for the same matmul, at the
    /// cost of a fixed ~2.3 ms per-chunk dispatch overhead on A19 Pro.
    case efficient

    /// Let Core ML partition between ANE and GPU automatically. Uses
    /// `.all`. Good default when you don't know the workload shape.
    case balanced

    /// Max throughput. Uses `.cpuAndGPU` — bypasses ANE entirely and runs
    /// the decode matmul on the A19 Pro GPU's tensor cores. Expected to
    /// beat ANE on decode-dominated workloads at the cost of ~2-3× battery
    /// drain and faster thermal throttling. The "gachi" / full-send mode.
    case performance

    /// Caller-supplied override. Useful for A/B testing or for new compute
    /// units the enum hasn't been updated to name yet.
    case custom(MLComputeUnits)

    /// Construct a profile from raw `MLComputeUnits`, collapsing onto a
    /// named case when the units match a named profile. Lets callers using
    /// the legacy `computeUnits:` parameter still get a recognisable
    /// `displayName` instead of "Custom".
    public static func from(_ units: MLComputeUnits) -> ComputeProfile {
        switch units {
        case .cpuAndNeuralEngine: return .efficient
        case .all:                return .balanced
        case .cpuAndGPU:          return .performance
        default:                  return .custom(units)
        }
    }

    /// Resolved CoreML compute units for this profile.
    public var mlComputeUnits: MLComputeUnits {
        switch self {
        case .efficient:     return .cpuAndNeuralEngine
        case .balanced:      return .all
        case .performance:   return .cpuAndGPU
        case .custom(let u): return u
        }
    }

    /// Stable identifier for persistence (UserDefaults, sidecar JSON).
    public var rawIdentifier: String {
        switch self {
        case .efficient:     return "efficient"
        case .balanced:      return "balanced"
        case .performance:   return "performance"
        case .custom(let u):
            switch u {
            case .cpuOnly:            return "custom.cpuOnly"
            case .cpuAndGPU:          return "custom.cpuAndGPU"
            case .cpuAndNeuralEngine: return "custom.cpuAndNeuralEngine"
            case .all:                return "custom.all"
            @unknown default:         return "custom.unknown"
            }
        }
    }

    /// Inverse of `rawIdentifier`. Returns `.efficient` for unknown strings.
    public static func fromRawIdentifier(_ s: String) -> ComputeProfile {
        switch s {
        case "efficient":                 return .efficient
        case "balanced":                  return .balanced
        case "performance":               return .performance
        case "custom.cpuOnly":            return .custom(.cpuOnly)
        case "custom.cpuAndGPU":          return .custom(.cpuAndGPU)
        case "custom.cpuAndNeuralEngine": return .custom(.cpuAndNeuralEngine)
        case "custom.all":                return .custom(.all)
        default:                          return .efficient
        }
    }

    /// Short human label for UI. English only (CLAUDE.md constraint).
    public var displayName: String {
        switch self {
        case .efficient:     return "Efficient (ANE)"
        case .balanced:      return "Balanced (ANE+GPU)"
        case .performance:   return "Performance (GPU)"
        case .custom:        return "Custom"
        }
    }

    /// One-line explanation for UI tooltips / pickers. English only.
    public var tagline: String {
        switch self {
        case .efficient:     return "Low power, coolest, ~baseline speed"
        case .balanced:      return "Core ML decides — often fastest overall"
        case .performance:   return "Max speed, hot, high battery drain"
        case .custom:        return "Caller-specified compute units"
        }
    }

    public var description: String { "ComputeProfile.\(rawIdentifier)" }

    /// All enum cases the UI should display as top-level options. `.custom`
    /// is intentionally excluded — it is only reachable programmatically.
    public static var uiSelectable: [ComputeProfile] {
        [.efficient, .balanced, .performance]
    }
}

// MARK: - UserDefaults persistence

public extension ComputeProfile {
    /// UserDefaults key for persisting the user's chosen profile.
    static let userDefaultsKey = "CoreMLLLM.ComputeProfile"

    /// Load the persisted profile, or `.efficient` if unset.
    static func loadPersisted(defaults: UserDefaults = .standard) -> ComputeProfile {
        guard let raw = defaults.string(forKey: userDefaultsKey) else {
            return .efficient
        }
        return fromRawIdentifier(raw)
    }

    /// Persist this profile so the next app launch picks it up.
    func persist(to defaults: UserDefaults = .standard) {
        defaults.set(rawIdentifier, forKey: Self.userDefaultsKey)
    }
}
