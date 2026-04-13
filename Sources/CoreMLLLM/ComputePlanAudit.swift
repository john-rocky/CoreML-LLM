import CoreML
import Foundation

/// MLComputePlan silent-fallback audit (UNEXPLORED_APPROACHES_V2 §G2).
///
/// Walks every MIL operation in chunk1–chunk4, queries the on-device compiler
/// for per-op device placement, and logs any operation whose preferred device
/// is NOT the Neural Engine. Gated behind the `COMPUTE_PLAN_AUDIT` environment
/// variable or UserDefaults key so it runs only during development.
///
/// Usage:
/// ```swift
/// // Set env var COMPUTE_PLAN_AUDIT=1 or UserDefaults bool "COMPUTE_PLAN_AUDIT"
/// await ComputePlanAudit.run(modelDirectory: url, computeUnits: .cpuAndNeuralEngine)
/// ```
enum ComputePlanAudit {

    /// Returns true when the audit should run.
    static var isEnabled: Bool {
        if ProcessInfo.processInfo.environment["COMPUTE_PLAN_AUDIT"] != nil { return true }
        return UserDefaults.standard.bool(forKey: "COMPUTE_PLAN_AUDIT")
    }

    /// Run the audit for all decode chunks found in `modelDirectory`.
    static func run(modelDirectory: URL,
                    computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async {
        guard isEnabled else { return }

        let chunkNames = ["chunk1", "chunk2", "chunk3", "chunk4"]
        var totalOps = 0
        var totalFallbacks = 0

        for name in chunkNames {
            let compiled = modelDirectory.appendingPathComponent("\(name).mlmodelc")
            let pkg = modelDirectory.appendingPathComponent("\(name).mlpackage")
            let url: URL
            if FileManager.default.fileExists(atPath: compiled.path) {
                url = compiled
            } else if FileManager.default.fileExists(atPath: pkg.path) {
                url = pkg
            } else {
                print("[ComputePlan] \(name): not found, skipping")
                continue
            }

            let config = MLModelConfiguration()
            config.computeUnits = computeUnits

            do {
                let plan = try await MLComputePlan.load(contentsOf: url,
                                                        configuration: config)
                let (ops, fallbacks) = auditPlan(plan, chunkName: name)
                totalOps += ops
                totalFallbacks += fallbacks
            } catch {
                print("[ComputePlan] \(name): failed to load plan — \(error)")
            }
        }

        print("[ComputePlan] ── summary: \(totalFallbacks) non-ANE op(s) out of \(totalOps) total across all chunks")
    }

    // MARK: - Private

    /// Walk a single compute plan, log non-ANE ops, return (totalOps, fallbackCount).
    private static func auditPlan(_ plan: MLComputePlan,
                                  chunkName: String) -> (Int, Int) {
        guard let program = plan.modelStructure.program else {
            print("[ComputePlan] \(chunkName): model structure is not a program")
            return (0, 0)
        }

        var totalOps = 0
        var fallbackOps = 0

        for (fnName, function) in program.functions {
            walkBlock(function.block, path: "\(chunkName)/\(fnName)",
                      plan: plan, totalOps: &totalOps, fallbackOps: &fallbackOps)
        }

        if fallbackOps == 0 {
            print("[ComputePlan] \(chunkName): all \(totalOps) ops on Neural Engine")
        } else {
            print("[ComputePlan] \(chunkName): \(fallbackOps)/\(totalOps) ops NOT on Neural Engine")
        }
        return (totalOps, fallbackOps)
    }

    private static func walkBlock(_ block: MLModelStructure.Program.Block,
                                  path: String,
                                  plan: MLComputePlan,
                                  totalOps: inout Int,
                                  fallbackOps: inout Int) {
        for op in block.operations {
            totalOps += 1

            let usage = plan.deviceUsage(for: op)
            let cost = plan.estimatedCost(of: op)

            let isANE: Bool
            if let preferred = usage?.preferred {
                isANE = preferred is MLNeuralEngineComputeDevice
            } else {
                isANE = false
            }

            if !isANE {
                fallbackOps += 1
                let device = deviceLabel(usage?.preferred)
                let costStr: String
                if let w = cost?.weight {
                    costStr = String(format: "cost=%.4f", w)
                } else {
                    costStr = "cost=n/a"
                }
                let outputNames = op.outputs.map(\.name).joined(separator: ",")
                print("[ComputePlan] \(path) | \(op.operatorName) → \(device) | \(costStr) | outputs=\(outputNames)")
            }

            // Recurse into nested blocks (e.g. cond, while_loop)
            for nested in op.blocks {
                walkBlock(nested, path: "\(path)/\(op.operatorName)",
                          plan: plan, totalOps: &totalOps, fallbackOps: &fallbackOps)
            }
        }
    }

    private static func deviceLabel(_ device: MLComputeDevice?) -> String {
        guard let device else { return "unknown" }
        if device is MLCPUComputeDevice { return "CPU" }
        if device is MLGPUComputeDevice { return "GPU" }
        if device is MLNeuralEngineComputeDevice { return "ANE" }
        return "other"
    }
}
