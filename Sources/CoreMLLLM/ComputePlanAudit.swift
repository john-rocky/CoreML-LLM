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

        // Include both the 4-chunk decode names and the 3-chunk rename
        // pair (chunk2_3way, chunk3_3way). Non-present chunks skip silently.
        let chunkNames = ["chunk1", "chunk2", "chunk3", "chunk4",
                          "chunk2_3way", "chunk3_3way"]
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

    /// Audit the cross-vocab Qwen drafter at `cross_vocab/qwen_drafter.mlmodelc`
    /// (or `.mlpackage`). Uses the same `.cpuAndGPU` units the runtime
    /// passes to `CrossVocabDraft.init` so the placement table reflects
    /// what actually runs on device. Phase B Task 2 — needed to localise
    /// the iPhone 1.8 tok/s regression: GPU placement = "drafter is GPU
    /// but slow" (model surgery), CPU fallback = "force compute units".
    static func runDrafter(modelDirectory: URL) async {
        guard isEnabled else { return }
        let cvDir = modelDirectory.appendingPathComponent("cross_vocab")
        let compiled = cvDir.appendingPathComponent("qwen_drafter.mlmodelc")
        let pkg = cvDir.appendingPathComponent("qwen_drafter.mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: pkg.path) {
            url = pkg
        } else {
            print("[ComputePlan] cross_vocab/qwen_drafter: not found, skipping")
            return
        }

        // Match `CrossVocabDraft.init` defaults: drafter loads with
        // .cpuAndGPU. If the runtime ever switches units, mirror that
        // change here so the audit reflects production placement.
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        do {
            let plan = try await MLComputePlan.load(contentsOf: url,
                                                    configuration: config)
            let (ops, fallbacks) = auditDrafterPlan(plan,
                                                    name: "qwen_drafter",
                                                    expectedDevice: "GPU")
            print("[ComputePlan] qwen_drafter: \(fallbacks) off-GPU op(s) out of \(ops) total")
        } catch {
            print("[ComputePlan] qwen_drafter: failed to load plan — \(error)")
        }
    }

    // MARK: - Private

    /// Walk a single compute plan, log non-ANE ops, return (totalOps, fallbackCount).
    private static func auditPlan(_ plan: MLComputePlan,
                                  chunkName: String) -> (Int, Int) {
        guard case .program(let program) = plan.modelStructure else {
            print("[ComputePlan] \(chunkName): model structure is not a program")
            return (0, 0)
        }

        var totalOps = 0
        var fallbackOps = 0
        // (opName, device) → count, for compact grouped summary when per-op
        // spam would bury the verdict in 100+ lines.
        var fallbackByKind: [String: Int] = [:]

        for (fnName, function) in program.functions {
            walkBlock(function.block, path: "\(chunkName)/\(fnName)",
                      plan: plan, totalOps: &totalOps,
                      fallbackOps: &fallbackOps,
                      fallbackByKind: &fallbackByKind)
        }

        if fallbackOps == 0 {
            print("[ComputePlan] \(chunkName): all \(totalOps) ops on Neural Engine")
        } else {
            print("[ComputePlan] \(chunkName): \(fallbackOps)/\(totalOps) ops NOT on Neural Engine")
            let sorted = fallbackByKind.sorted { $0.value > $1.value }
            for (kind, n) in sorted {
                print("[ComputePlan] \(chunkName) fallback | \(kind): \(n)")
            }
        }
        return (totalOps, fallbackOps)
    }

    private static func walkBlock(_ block: MLModelStructure.Program.Block,
                                  path: String,
                                  plan: MLComputePlan,
                                  totalOps: inout Int,
                                  fallbackOps: inout Int,
                                  fallbackByKind: inout [String: Int]) {
        // Skip constant/weight-loading ops — they run once at load, not per-step
        let constOpSuffixes: Set<String> = [
            "const", "constexpr_lut_to_dense", "constexpr_affine_dequantize",
            "constexpr_blockwise_shift_scale", "constexpr_sparse_to_dense",
            "constexpr_cast",
        ]
        func isConstOpName(_ name: String) -> Bool {
            // operatorName arrives as either "const" or "ios18.const" (the
            // MIL-version prefix is attached for ops defined in the iOSNN
            // dialect). Strip any leading "iosNN." so both forms match.
            if let dot = name.firstIndex(of: ".") {
                return constOpSuffixes.contains(String(name[name.index(after: dot)...]))
            }
            return constOpSuffixes.contains(name)
        }

        for op in block.operations {
            let isConstOp = isConstOpName(op.operatorName)
            if !isConstOp { totalOps += 1 }

            let usage = plan.deviceUsage(for: op)
            let cost = plan.estimatedCost(of: op)

            let isANE: Bool
            if let preferred = usage?.preferred {
                isANE = deviceLabel(preferred) == "ANE"
            } else {
                // const ops have no device — skip them silently
                isANE = isConstOp
            }

            if !isANE && !isConstOp {
                fallbackOps += 1
                let device = deviceLabel(usage?.preferred)
                let key = "\(op.operatorName) → \(device)"
                fallbackByKind[key, default: 0] += 1
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
                          plan: plan, totalOps: &totalOps, fallbackOps: &fallbackOps,
                          fallbackByKind: &fallbackByKind)
            }
        }
    }

    private static func deviceLabel(_ device: MLComputeDevice?) -> String {
        guard let device else { return "unknown" }
        // MLComputeDevice is an enum in iOS 18+
        switch device {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .neuralEngine: return "ANE"
        @unknown default: return "other"
        }
    }

    /// Walk a plan and report any op whose preferred device is not
    /// `expectedDevice` ("GPU" for the cross-vocab drafter). Mirrors
    /// `auditPlan` but with a different device baseline so the same
    /// audit infrastructure can cover both decode chunks (expect ANE)
    /// and the cross-vocab drafter (expect GPU).
    private static func auditDrafterPlan(_ plan: MLComputePlan,
                                         name: String,
                                         expectedDevice: String) -> (Int, Int) {
        guard case .program(let program) = plan.modelStructure else {
            print("[ComputePlan] \(name): model structure is not a program")
            return (0, 0)
        }
        let constOps: Set<String> = [
            "const", "constexpr_lut_to_dense", "constexpr_affine_dequantize",
            "constexpr_blockwise_shift_scale", "constexpr_sparse_to_dense",
            "constexpr_cast",
        ]
        var totalOps = 0
        var fallbackOps = 0
        for (fnName, function) in program.functions {
            walkDrafterBlock(function.block, path: "\(name)/\(fnName)",
                             plan: plan, expectedDevice: expectedDevice,
                             constOps: constOps,
                             totalOps: &totalOps, fallbackOps: &fallbackOps)
        }
        return (totalOps, fallbackOps)
    }

    private static func walkDrafterBlock(_ block: MLModelStructure.Program.Block,
                                         path: String,
                                         plan: MLComputePlan,
                                         expectedDevice: String,
                                         constOps: Set<String>,
                                         totalOps: inout Int,
                                         fallbackOps: inout Int) {
        for op in block.operations {
            let isConstOp = constOps.contains(op.operatorName)
            if !isConstOp { totalOps += 1 }
            let usage = plan.deviceUsage(for: op)
            let cost = plan.estimatedCost(of: op)
            let actualDevice = deviceLabel(usage?.preferred)
            let onExpected: Bool = isConstOp ? true : (actualDevice == expectedDevice)
            if !onExpected && !isConstOp {
                fallbackOps += 1
                let costStr: String
                if let w = cost?.weight {
                    costStr = String(format: "cost=%.4f", w)
                } else {
                    costStr = "cost=n/a"
                }
                let outputs = op.outputs.map(\.name).joined(separator: ",")
                print("[ComputePlan] \(path) | \(op.operatorName) → \(actualDevice) "
                    + "(expected \(expectedDevice)) | \(costStr) | outputs=\(outputs)")
            }
            for nested in op.blocks {
                walkDrafterBlock(nested, path: "\(path)/\(op.operatorName)",
                                 plan: plan, expectedDevice: expectedDevice,
                                 constOps: constOps,
                                 totalOps: &totalOps, fallbackOps: &fallbackOps)
            }
        }
    }
}
