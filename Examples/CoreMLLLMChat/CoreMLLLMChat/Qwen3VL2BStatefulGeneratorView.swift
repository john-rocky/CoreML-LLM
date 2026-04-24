// SwiftUI shell for the stateful Qwen3-VL 2B generator (Phase 1 ship path).
// Runs a fixed smoke-test prompt (64 arbitrary tokens) through the chunks
// and reports decode tok/s plus phys_footprint so we can verify both
// targets: ≥25 tok/s decode AND < 500 MB phys_footprint during inference.

import CoreML
import SwiftUI
import Darwin.Mach

struct Qwen3VL2BStatefulGeneratorView: View {
    @State private var gen = Qwen3VL2BStatefulGenerator()
    @State private var tokensPerSec = ""
    @State private var decodedTokens = ""
    @State private var phys = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Stateful generate — Phase 1") {
                    Text("28-layer Qwen3-VL 2B text backbone via MLState + "
                         + "slice_update. Loads from "
                         + "Documents/Models/qwen3-vl-2b-stateful/.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button {
                        Task { await runSmoke() }
                    } label: {
                        HStack {
                            if gen.running { ProgressView() }
                            Text(gen.running ? "Generating..." : "Run 64-token smoke test")
                        }
                    }
                    .disabled(gen.running)
                    Text(gen.status).font(.caption).foregroundStyle(.secondary)
                }

                if !tokensPerSec.isEmpty {
                    Section("Throughput") {
                        Text(tokensPerSec).font(.callout)
                    }
                }
                if !phys.isEmpty {
                    Section("Memory") {
                        Text(phys).font(.callout)
                            .foregroundStyle(phys.contains("PASS") ? .green : .orange)
                    }
                }
                if !decodedTokens.isEmpty {
                    Section("Decoded token IDs") {
                        Text(decodedTokens)
                            .font(.caption2).textSelection(.enabled)
                    }
                }

                Section("Targets") {
                    Text("decode ≥ 25 tok/s (stretch ≥ 40)").font(.caption)
                    Text("phys_footprint < 500 MB during inference").font(.caption)
                }
            }
            .navigationTitle("VL 2B (stateful)")
        }
    }

    private func runSmoke() async {
        gen.running = true
        tokensPerSec = ""; decodedTokens = ""; phys = ""
        defer { gen.running = false }
        do {
            try gen.load()
            // 8-token fake prompt (token IDs don't need to be real for a
            // throughput smoke test — we just need a prefill then 64 decode).
            let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
            let out = try await gen.generate(inputIds: prompt, maxNewTokens: 64)
            tokensPerSec = gen.stats
            decodedTokens = out.map { String($0) }.joined(separator: ", ")
            let mb = physFootprintMB()
            phys = String(format:
                "phys_footprint: %.0f MB (%@)",
                mb, mb < 500 ? "PASS — ANE-resident" : "FAIL — likely GPU spill")
        } catch {
            gen.status = "FAIL — \(error.localizedDescription)"
        }
    }
}

private func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                        / MemoryLayout<natural_t>.size)
    let kr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(info.phys_footprint) / (1024.0 * 1024.0)
}
