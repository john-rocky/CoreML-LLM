import Foundation

/// Heat-management helpers used by the model loader.
///
/// During cold load the ANE compile daemon, P-cores, and weight paging all
/// run at peak with no breathing room between phases. Inserting short
/// `coolDown` awaits between heavy phases lets iOS clock the chip clusters
/// down briefly. The amount scales with `ProcessInfo.thermalState` so that
/// already-hot devices wait longer.
///
/// Tunables:
///   LLM_LOAD_COOL_MS = base cool-down ms between phases (default 300, 0 disables)
///   LLM_LOAD_LITE    = "1" skips optional auxiliary prewarms (vision, EAGLE-3,
///                       finalPrewarm prefill/verify/transition tail) — for
///                       text-only sessions where heat matters more than
///                       first-call latency on those paths.
enum ThermalThrottle {
    static let baseCoolMs: Int = {
        if let s = ProcessInfo.processInfo.environment["LLM_LOAD_COOL_MS"],
           let v = Int(s) { return max(0, v) }
        return 300
    }()

    static let lite: Bool =
        ProcessInfo.processInfo.environment["LLM_LOAD_LITE"] == "1"

    /// Awaitable cool-down between heavy load phases. Multiplier comes from
    /// the current thermal state so a hot device pauses much longer:
    /// nominal 1×, fair 2×, serious 6×, critical 15×.
    static func coolDown(_ baseMsOverride: Int? = nil,
                         label: String = "phase") async {
        let base = baseMsOverride ?? baseCoolMs
        if base <= 0 { return }
        let state = ProcessInfo.processInfo.thermalState
        let mult: Int
        let stateName: String
        switch state {
        case .nominal:  mult = 1;  stateName = "nominal"
        case .fair:     mult = 2;  stateName = "fair"
        case .serious:  mult = 6;  stateName = "serious"
        case .critical: mult = 15; stateName = "critical"
        @unknown default: mult = 1; stateName = "unknown"
        }
        let ms = base * mult
        if mult > 1 {
            print("[Thermal] \(label) cool-down \(ms) ms (state=\(stateName))")
        }
        try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
    }
}
