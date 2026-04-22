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
                        statRow("oracle top-1 in ANE top-3",
                                String(format: "%.0f%%", bench.top1InTop3Rate * 100))
                        statRow("oracle top-1 in ANE top-5",
                                String(format: "%.0f%%", bench.top1InTop5Rate * 100))
                        statRow("oracle top-1 in ANE top-10",
                                String(format: "%.0f%%", bench.top1InTop10Rate * 100))
                        statRow("mean top-5 overlap",
                                String(format: "%.1f / 5", bench.meanTop5Overlap))
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
                                    Text("t3:\(r.top1InTop3 ? "✓" : "✗")")
                                        .font(.caption2)
                                        .foregroundStyle(r.top1InTop3 ? .green : .red)
                                    Text("ovl:\(r.top5Overlap)/5")
                                        .font(.caption2).foregroundStyle(.secondary)
                                    Text(String(format: "%.0fms", r.prefillMs))
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                Section("Notes") {
                    Text(
                        "top-1 alone can be misleading due to argmax fragility on 248K vocab. "
                        + "top-3/top-5 containment and overlap show whether ANE's distribution "
                        + "matches the fp32 oracle for sampling-mode generation. "
                        + "Decode bench reference: iPhone ANE top-1=60% but top-3=100%. "
                        + "Prefill uses chunked SSM with Neumann iteration — may be more "
                        + "fp16-sensitive than decode."
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
