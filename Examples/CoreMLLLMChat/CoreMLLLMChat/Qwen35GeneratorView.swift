// SwiftUI demo screen for the Qwen3.5-0.8B end-to-end generator.
// Accepts a comma-separated list of token IDs (the user can paste these
// from a Qwen tokenizer). Produces a list of generated token IDs.
// Detokenization is deliberately out of scope here.

import SwiftUI

struct Qwen35GeneratorView: View {
    @State private var gen = Qwen35Generator()
    @State private var inputIdsText: String =
        "785,6722,315,9437,374"   // "The capital of France is"
    @State private var maxNewTokens: Int = 16
    @State private var errorMsg: String = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Input token IDs (comma-separated)") {
                    TextEditor(text: $inputIdsText)
                        .frame(minHeight: 60)
                        .font(.caption.monospaced())
                    Stepper("Max new tokens: \(maxNewTokens)",
                            value: $maxNewTokens, in: 1...96)
                }

                Section {
                    Button {
                        Task { await runGenerate() }
                    } label: {
                        HStack {
                            if gen.running { ProgressView() }
                            Text(gen.running ? "Generating..." : "Generate")
                        }
                    }
                    .disabled(gen.running)
                    Text(gen.status).font(.caption).foregroundStyle(.secondary)
                    if !errorMsg.isEmpty {
                        Text(errorMsg).font(.caption).foregroundStyle(.red)
                    }
                }

                if !gen.generatedIds.isEmpty {
                    Section("Results") {
                        statRow("prefill",     String(format: "%.0f ms", gen.prefillMs))
                        statRow("decode avg",  String(format: "%.1f ms/tok", gen.decodeMsAvg))
                        statRow("throughput",  String(format: "%.1f tok/s", gen.tokensPerSecond))
                        statRow("tokens",      "\(gen.generatedIds.count)")
                    }
                    Section("Generated token IDs") {
                        Text(gen.generatedIds.map(String.init).joined(separator: ","))
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                    }
                }

                Section("Notes") {
                    Text(
                        "Defaults to CPU compute units (winning path per Mac reference). "
                        + "Takes token IDs because this demo ships without a Qwen tokenizer; "
                        + "paste IDs you've encoded elsewhere (e.g. via AutoTokenizer in Python). "
                        + "Prefill seq=64 fixed; decode grows up to position 128."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Qwen3.5 Generate")
        }
    }

    private func runGenerate() async {
        errorMsg = ""
        let ids = inputIdsText
            .split(whereSeparator: { ",\n \t".contains($0) })
            .compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
        guard !ids.isEmpty else { errorMsg = "no valid token IDs"; return }
        do {
            _ = try await gen.generate(inputIds: ids, maxNewTokens: maxNewTokens)
        } catch {
            errorMsg = error.localizedDescription
        }
    }

    private func statRow(_ label: String, _ value: String) -> some View {
        HStack { Text(label); Spacer(); Text(value).monospaced() }
    }
}
