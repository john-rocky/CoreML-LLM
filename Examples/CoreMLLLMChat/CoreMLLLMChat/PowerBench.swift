//
//  PowerBench.swift
//  CoreMLLLMChat
//
//  Cross-profile power / throughput harness. Runs a fixed-length generation
//  (default 200 tokens × 5 trials) under each `ComputeProfile`, records
//  tok/s + thermal state + battery delta + task CPU usage, and writes a CSV
//  to the Documents directory.
//
//  For precise joule measurements, tether the iPhone to a Mac via USB and
//  run `powermetrics` in parallel — see docs/POWER_BENCH.md.
//

#if os(iOS)
import CoreML
import CoreMLLLM
import Foundation
import UIKit
import Darwin.Mach

/// Self-contained power / throughput benchmark for the chat example.
/// Invoke via the "Power" menu in the chat UI, or programmatically from a
/// unit-test host. Emits `power_bench.csv` into Documents.
@MainActor
final class PowerBench {

    struct Config {
        /// Tokens to generate per trial. 200 is enough to paper over warmup
        /// (~20 tokens) while keeping each profile under ~10s on A19 Pro.
        var tokensPerTrial: Int = 200
        /// Number of trials per profile. 5 is a compromise between noise
        /// and thermal drift — thermal state is sampled at the *end* of
        /// each trial.
        var trialsPerProfile: Int = 5
        /// Prompt used for every trial. Intentionally short so prefill does
        /// not dominate the decode-dominated measurement.
        var prompt: String = "Explain the theory of relativity in simple terms."
        /// Profiles to run. Defaults to the 3 user-facing profiles.
        var profiles: [ComputeProfile] = ComputeProfile.uiSelectable
        /// When true, wait ~30 s between profiles to let thermal state
        /// recover. Cuts noise, at the cost of a longer total run.
        var coolDownBetweenProfiles: Bool = true
        /// Cool-down wait, seconds.
        var coolDownSeconds: TimeInterval = 30
    }

    struct TrialResult {
        let profile: ComputeProfile
        let trial: Int
        let tokens: Int
        let seconds: Double
        let tokPerSec: Double
        let thermalStart: ProcessInfo.ThermalState
        let thermalEnd: ProcessInfo.ThermalState
        let batteryStart: Float
        let batteryEnd: Float
        /// Aggregate task CPU usage in % (0-100*nCores) averaged over the trial.
        let avgCPUPercent: Double
        /// Rough joule estimate = (Δbattery% / 100) × nominal_battery_Wh × 3600.
        /// Only meaningful when the device was unplugged and the trial was
        /// long enough to see a battery tick (~30 s+). Tethered powermetrics
        /// is the authoritative measurement — this is a fallback.
        let joulesEstimate: Double?
    }

    struct Summary {
        let profile: ComputeProfile
        let trials: [TrialResult]
        var meanTokPerSec: Double {
            guard !trials.isEmpty else { return 0 }
            return trials.map(\.tokPerSec).reduce(0, +) / Double(trials.count)
        }
        var peakTokPerSec: Double { trials.map(\.tokPerSec).max() ?? 0 }
        var meanCPUPercent: Double {
            guard !trials.isEmpty else { return 0 }
            return trials.map(\.avgCPUPercent).reduce(0, +) / Double(trials.count)
        }
        var thermalExit: ProcessInfo.ThermalState {
            trials.last?.thermalEnd ?? .nominal
        }
        var totalJoulesEstimate: Double {
            trials.compactMap(\.joulesEstimate).reduce(0, +)
        }
    }

    /// Nominal battery capacity for iPhone 17 Pro in watt-hours. Used only
    /// for the coarse joule estimate; actual joules come from powermetrics.
    /// Apple publishes 13.97 Wh for the 17 Pro (4685 mAh @ 3.83 V nominal).
    /// Adjust per device — this is only a sanity-check scale.
    static let nominalBatteryWh: Double = 13.97

    let llmFolder: URL
    let config: Config

    init(llmFolder: URL, config: Config = Config()) {
        self.llmFolder = llmFolder
        self.config = config
    }

    /// Run the full matrix and write `power_bench.csv`. Returns one
    /// `Summary` per profile in the order they were executed.
    ///
    /// - Parameter onProgress: called on the main actor with a human-readable
    ///   status line (e.g. "performance trial 3/5: 47.8 tok/s").
    func run(onProgress: @escaping (String) -> Void) async throws -> [Summary] {
        UIDevice.current.isBatteryMonitoringEnabled = true
        var summaries: [Summary] = []

        for profile in config.profiles {
            onProgress("Loading model for \(profile.displayName)…")
            // Load a fresh LLM per profile so the ANE/GPU compilation is
            // honored. CoreML caches compiled assets, so this is fast on
            // subsequent loads of the same profile.
            let llm = try await CoreMLLLM.load(from: llmFolder, profile: profile) { s in
                Task { @MainActor in onProgress("[\(profile.rawIdentifier)] \(s)") }
            }
            // Speculative paths add variance that obscures the compute-unit
            // effect we're trying to isolate. Turn them off for the bench.
            llm.mtpEnabled = false
            llm.crossVocabEnabled = false
            llm.drafterUnionEnabled = false

            var trials: [TrialResult] = []
            for t in 1...config.trialsPerProfile {
                let trial = try await runTrial(llm: llm, profile: profile, trial: t)
                trials.append(trial)
                onProgress(String(
                    format: "[%@] trial %d/%d: %.1f tok/s  thermal=%@  cpu=%.0f%%",
                    profile.rawIdentifier,
                    t, config.trialsPerProfile,
                    trial.tokPerSec,
                    Self.thermalName(trial.thermalEnd),
                    trial.avgCPUPercent))
            }
            summaries.append(Summary(profile: profile, trials: trials))

            if config.coolDownBetweenProfiles && profile != config.profiles.last {
                onProgress("Cooling down \(Int(config.coolDownSeconds))s before next profile…")
                try? await Task.sleep(nanoseconds: UInt64(config.coolDownSeconds * 1e9))
            }
        }

        try writeCSV(summaries: summaries)
        return summaries
    }

    // MARK: - Trial

    private func runTrial(llm: CoreMLLLM,
                          profile: ComputeProfile,
                          trial: Int) async throws -> TrialResult {
        let thermalStart = ProcessInfo.processInfo.thermalState
        let batteryStart = UIDevice.current.batteryLevel

        // Warmup pass if this is the first trial of the profile — ensures
        // ANE/GPU caches are primed. Discarded from the metric.
        if trial == 1 {
            _ = try? await quickWarmup(llm: llm)
        }

        let cpuStart = Self.sampleTaskCPU()
        let t0 = CFAbsoluteTimeGetCurrent()

        var tokens = 0
        let stream = try await llm.stream(config.prompt, maxTokens: config.tokensPerTrial)
        for await _ in stream {
            tokens += 1
            if tokens >= config.tokensPerTrial { break }
        }

        let seconds = CFAbsoluteTimeGetCurrent() - t0
        let cpuEnd = Self.sampleTaskCPU()
        let thermalEnd = ProcessInfo.processInfo.thermalState
        let batteryEnd = UIDevice.current.batteryLevel

        let avgCPU = Self.cpuDeltaPercent(start: cpuStart, end: cpuEnd,
                                          wallSeconds: seconds)

        let joules: Double?
        if batteryStart > 0, batteryEnd > 0, batteryStart >= batteryEnd {
            let dropFrac = Double(batteryStart - batteryEnd)
            if dropFrac > 0 {
                joules = dropFrac * Self.nominalBatteryWh * 3600.0
            } else {
                joules = nil
            }
        } else {
            joules = nil
        }

        return TrialResult(
            profile: profile,
            trial: trial,
            tokens: tokens,
            seconds: seconds,
            tokPerSec: seconds > 0 ? Double(tokens) / seconds : 0,
            thermalStart: thermalStart,
            thermalEnd: thermalEnd,
            batteryStart: batteryStart,
            batteryEnd: batteryEnd,
            avgCPUPercent: avgCPU,
            joulesEstimate: joules)
    }

    private func quickWarmup(llm: CoreMLLLM) async throws -> Int {
        var n = 0
        let stream = try await llm.stream("Hi", maxTokens: 8)
        for await _ in stream { n += 1; if n >= 8 { break } }
        return n
    }

    // MARK: - CSV output

    private func writeCSV(summaries: [Summary]) throws {
        let docs = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask).first!
        let url = docs.appendingPathComponent("power_bench.csv")

        var rows: [String] = []
        rows.append("profile,trial,tokens,seconds,tok_per_sec,thermal_start,thermal_end,battery_start,battery_end,cpu_pct,joules_estimate")
        for s in summaries {
            for t in s.trials {
                rows.append([
                    s.profile.rawIdentifier,
                    "\(t.trial)",
                    "\(t.tokens)",
                    String(format: "%.3f", t.seconds),
                    String(format: "%.2f", t.tokPerSec),
                    Self.thermalName(t.thermalStart),
                    Self.thermalName(t.thermalEnd),
                    String(format: "%.3f", t.batteryStart),
                    String(format: "%.3f", t.batteryEnd),
                    String(format: "%.1f", t.avgCPUPercent),
                    t.joulesEstimate.map { String(format: "%.2f", $0) } ?? "",
                ].joined(separator: ","))
            }
        }
        rows.append("") // summary section
        rows.append("profile,mean_tok_per_sec,peak_tok_per_sec,mean_cpu_pct,thermal_exit,joules_total_estimate")
        for s in summaries {
            rows.append([
                s.profile.rawIdentifier,
                String(format: "%.2f", s.meanTokPerSec),
                String(format: "%.2f", s.peakTokPerSec),
                String(format: "%.1f", s.meanCPUPercent),
                Self.thermalName(s.thermalExit),
                String(format: "%.2f", s.totalJoulesEstimate),
            ].joined(separator: ","))
        }

        let csv = rows.joined(separator: "\n") + "\n"
        try csv.write(to: url, atomically: true, encoding: .utf8)
        print("[PowerBench] wrote \(url.path)")
    }

    // MARK: - Helpers

    static func thermalName(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Returns total task CPU time (user + system) in seconds.
    /// Uses `task_info(TASK_BASIC_INFO)` + per-thread info because
    /// `task_basic_info` alone only exposes cumulative resident usage.
    struct CPUSample {
        let userSec: Double
        let systemSec: Double
        var total: Double { userSec + systemSec }
    }

    static func sampleTaskCPU() -> CPUSample {
        var info = task_thread_times_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_thread_times_info>.size / MemoryLayout<natural_t>.size)
        let kr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_,
                          task_flavor_t(TASK_THREAD_TIMES_INFO),
                          rebound,
                          &count)
            }
        }
        guard kr == KERN_SUCCESS else { return CPUSample(userSec: 0, systemSec: 0) }
        let user = Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1e6
        let sys  = Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1e6
        return CPUSample(userSec: user, systemSec: sys)
    }

    static func cpuDeltaPercent(start: CPUSample, end: CPUSample,
                                wallSeconds: Double) -> Double {
        guard wallSeconds > 0 else { return 0 }
        let delta = max(0, end.total - start.total)
        // Percent = CPU-seconds / wall-seconds × 100. Can exceed 100% when
        // multiple threads are busy concurrently — that's expected and
        // informative.
        return (delta / wallSeconds) * 100.0
    }
}
#endif
