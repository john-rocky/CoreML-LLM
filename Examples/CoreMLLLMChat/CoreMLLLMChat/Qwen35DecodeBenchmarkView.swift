// SwiftUI shell for the Qwen3.5 stateful-decode benchmark.
// See Qwen35DecodeBenchmark.swift for numerics.

import SwiftUI

struct Qwen35DecodeBenchmarkView: View {
    @State private var bench = Qwen35DecodeBenchmark()

    var body: some View {
        NavigationStack {
            Form {
                Section("Compute units") {
                    Picker("Units", selection: $bench.units) {
                        ForEach(Qwen35Benchmark.UnitsChoice.allCases, id: \.self) { u in
                            Text(u.rawValue).tag(u)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(bench.running)
                }

                Section("Run") {
                    Button {
                        Task { await bench.run() }
                    } label: {
                        HStack {
                            if bench.running { ProgressView() }
                            Text(bench.running ? "Running..." : "Run decode benchmark")
                        }
                    }
                    .disabled(bench.running)
                    Text(bench.status).font(.caption).foregroundStyle(.secondary)
                }

                if !bench.results.isEmpty {
                    Section("Summary") {
                        statRow("mean cos",      String(format: "%.4f", bench.meanCos))
                        statRow("worst-pos cos", String(format: "%.4f", bench.worstCos))
                        statRow("top-1 match",   String(format: "%.0f%%", bench.top1Rate * 100))
                        statRow("throughput",    String(format: "%.1f tok/s", bench.meanTokPerSec))
                    }
                    Section("Per prompt") {
                        ForEach(bench.results) { r in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(r.prompt).font(.caption).lineLimit(1)
                                HStack {
                                    Text("S=\(r.S)").font(.caption2).foregroundStyle(.secondary)
                                    Spacer()
                                    Text(String(format: "cos=%.4f", r.lastCos))
                                        .font(.caption2)
                                        .foregroundStyle(r.lastCos >= 0.99 ? .green : .orange)
                                    Image(systemName: r.top1Match ? "checkmark.circle.fill" : "xmark.circle.fill")
                                        .foregroundStyle(r.top1Match ? .green : .red)
                                    Text(String(format: "%.1f tok/s", r.tokPerSec))
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                Section("Notes") {
                    Text(
                        "Measures the stateful decode mlpackage. Mac M4 reference: "
                        + "CPU fp16 cos 0.99992 / top-1 100% / ~50 tok/s; "
                        + "CPU+ANE cos 0.99 / top-1 40% / ~40 tok/s. "
                        + "States start at zero (no prefill handoff yet); that only affects "
                        + "token-sequence alignment vs HF oracle, not latency or placement. "
                        + "LiteRT-LM baseline on iPhone 17 Pro is 56.5 tok/s."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Qwen3.5 Decode")
        }
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack { Text(label); Spacer(); Text(value).monospaced() }
    }
}
