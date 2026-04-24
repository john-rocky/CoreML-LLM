// SwiftUI shell for Gate Zero. Tap "Run" once per build.

import SwiftUI

struct GateZeroBenchmarkView: View {
    @State private var bench = GateZeroBenchmark()

    var body: some View {
        NavigationStack {
            Form {
                Section("Gate Zero") {
                    Text("2-layer MLState + slice_update stub, same KV shape "
                         + "(4, 8, 2048, 128) as Qwen3-VL 2B per chunk.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button {
                        Task { await bench.run() }
                    } label: {
                        HStack {
                            if bench.running { ProgressView() }
                            Text(bench.running ? "Running..." : "Predict on ANE")
                        }
                    }
                    .disabled(bench.running)
                }

                Section("Result") {
                    Text(bench.status)
                        .font(.callout)
                        .foregroundStyle(resultColor)
                    if let err = bench.errorText {
                        Text(err).font(.caption2).foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }

                Section("What this gates") {
                    Text("PASS → proceed with full Phase 1 converter "
                         + "(28-layer MLState + multifunction + 2-chunk + "
                         + "split LM head).")
                        .font(.caption)
                    Text("FAIL (-14 / Error=11) → stop, fall back to the "
                         + "recurrent pattern. Report the exact error.")
                        .font(.caption)
                }
            }
            .navigationTitle("Gate Zero (Phase 1)")
        }
    }

    private var resultColor: Color {
        switch bench.passed {
        case .some(true): return .green
        case .some(false): return .red
        case .none: return .primary
        }
    }
}
