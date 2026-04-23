#!/usr/bin/env swift
// Audit ANE placement for a single CoreML .mlpackage (or .mlmodelc).
//
// Usage:
//     swift scripts/audit_mlpackage_placement.swift <path to .mlpackage>
//
// Prints every op whose preferred device is not the Neural Engine, plus a
// summary line with (totalOps, onANE, onCPU, onGPU) counts and the ANE %.
// Const/constexpr/load-once ops are excluded from the dispatched total since
// they run at load time, not per step.

import CoreML
import Foundation

@available(macOS 14.4, iOS 17.4, *)
func deviceLabel(_ d: MLComputeDevice?) -> String {
    guard let d else { return "unknown" }
    switch d {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "other"
    }
}

@available(macOS 14.4, iOS 17.4, *)
func walk(_ block: MLModelStructure.Program.Block,
          path: String,
          plan: MLComputePlan,
          counts: inout [String: Int],
          fallbacks: inout [(String, String)]) {
    let constOps: Set<String> = [
        "const", "constexpr_lut_to_dense", "constexpr_affine_dequantize",
        "constexpr_blockwise_shift_scale", "constexpr_sparse_to_dense",
        "constexpr_cast",
    ]

    for op in block.operations {
        let isConst = constOps.contains(op.operatorName)
        if !isConst {
            let usage = plan.deviceUsage(for: op)
            let label = deviceLabel(usage?.preferred)
            counts[label, default: 0] += 1
            if label != "ANE" {
                fallbacks.append(("\(path)/\(op.operatorName)", label))
            }
        }
        for nested in op.blocks {
            walk(nested, path: "\(path)/\(op.operatorName)",
                 plan: plan, counts: &counts, fallbacks: &fallbacks)
        }
    }
}

@available(macOS 14.4, iOS 17.4, *)
func audit(_ url: URL) async throws {
    print("Auditing \(url.path)")
    let t0 = Date()
    let plan = try await MLComputePlan.load(
        contentsOf: url,
        configuration: {
            let c = MLModelConfiguration()
            c.computeUnits = .cpuAndNeuralEngine  // forbid GPU so we see real ANE cost
            return c
        }()
    )
    print("  loaded compute plan in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")

    guard case .program(let program) = plan.modelStructure else {
        print("  ERROR: model structure is not a program")
        return
    }

    var counts: [String: Int] = [:]
    var fallbacks: [(String, String)] = []

    for (fnName, function) in program.functions {
        walk(function.block, path: fnName, plan: plan,
             counts: &counts, fallbacks: &fallbacks)
    }

    let total = counts.values.reduce(0, +)
    let ane = counts["ANE", default: 0]
    let cpu = counts["CPU", default: 0]
    let gpu = counts["GPU", default: 0]
    let other = total - ane - cpu - gpu
    let pct = total > 0 ? Double(ane) / Double(total) * 100 : 0

    print("  dispatched ops: \(total)  (ANE=\(ane)  CPU=\(cpu)  GPU=\(gpu)  other=\(other))")
    print(String(format: "  ANE placement: %.2f%%", pct))

    if !fallbacks.isEmpty {
        // Group by operator name for a compact report
        var byOp: [String: (count: Int, device: String)] = [:]
        for (p, d) in fallbacks {
            let opName = String(p.split(separator: "/").last ?? Substring(p))
            if var entry = byOp[opName] {
                entry.count += 1
                byOp[opName] = entry
            } else {
                byOp[opName] = (1, d)
            }
        }
        print("  non-ANE ops by operator:")
        for (op, entry) in byOp.sorted(by: { $0.value.count > $1.value.count }) {
            print("    \(op)  ×\(entry.count)  → \(entry.device)")
        }
    }
}

guard CommandLine.arguments.count >= 2 else {
    print("usage: swift \(CommandLine.arguments[0]) <path-to-mlpackage-or-mlmodelc> [more paths...]")
    exit(2)
}

if #available(macOS 14.4, *) {
    let sema = DispatchSemaphore(value: 0)
    Task {
        for arg in CommandLine.arguments.dropFirst() {
            let url = URL(fileURLWithPath: arg)
            do {
                try await audit(url)
            } catch {
                print("  ERROR: \(error)")
            }
            print("")
        }
        sema.signal()
    }
    sema.wait()
} else {
    print("requires macOS 14.4+")
    exit(2)
}
