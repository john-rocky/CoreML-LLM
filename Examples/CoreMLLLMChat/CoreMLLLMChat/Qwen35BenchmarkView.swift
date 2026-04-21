// SwiftUI shell for the Qwen3.5-0.8B prefill ANE benchmark.
// See Qwen35Benchmark.swift for the inference + numerics.

import SwiftUI

struct Qwen35BenchmarkView: View {
    @State private var bench = Qwen35Benchmark()

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
                            Text(bench.running ? "Running..." : "Run benchmark")
                        }
                    }
                    .disabled(bench.running)
                    Text(bench.status).font(.caption).foregroundStyle(.secondary)
                }

                if !bench.results.isEmpty {
                    Section("Summary") {
                        statRow("mean cos",     String(format: "%.4f", bench.meanCos))
                        statRow("worst-pos cos", String(format: "%.4f", bench.worstCos))
                        statRow("top-1 match",  String(format: "%.0f%%", bench.top1Rate * 100))
                        statRow("prefill",      String(format: "%.1f ms", bench.meanPrefillMs))
                        statRow("throughput",   String(format: "%.1f tok/s", bench.tokensPerSecond))
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
                                        .foregroundStyle(r.lastCos >= 0.95 ? .green : .orange)
                                    Image(systemName: r.top1Match ? "checkmark.circle.fill" : "xmark.circle.fill")
                                        .foregroundStyle(r.top1Match ? .green : .red)
                                    Text(String(format: "%.0fms", r.prefillMs))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                Section("Notes") {
                    Text(
                        "Mac M4 ANE baseline (reference): mean cos ≈ 0.98, worst-pos ≈ 0.84, top-1 ≈ 80%. "
                        + "CPU fp16 path on Mac: top-1 100%, worst-pos 0.998. "
                        + "This screen measures the A18 Pro ANE."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Qwen3.5 Benchmark")
        }
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack { Text(label); Spacer(); Text(value).monospaced() }
    }
}
