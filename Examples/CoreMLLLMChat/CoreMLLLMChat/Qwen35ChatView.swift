// Simple chat UI for Qwen3.5-0.8B running on CPU+ANE.
// Uses swift-transformers AutoTokenizer to go text <-> token IDs, and
// Qwen35Generator for the prefill+decode loop.
//
// Not wired into the main ChatView's multi-model infrastructure; this is
// a focused screen that demonstrates the end-to-end ANE shipping path.

import SwiftUI
import CoreML
import Tokenizers

enum ComputeChoice: String, CaseIterable, Identifiable {
    case cpu = "CPU"
    case gpu = "GPU"
    case ane = "ANE"

    var id: String { rawValue }
    var units: MLComputeUnits {
        switch self {
        case .cpu: return .cpuOnly
        case .gpu: return .cpuAndGPU
        case .ane: return .cpuAndNeuralEngine
        }
    }
    var note: String {
        switch self {
        case .cpu:
            return "Prefill + decode on CPU. ~1.6 GB RAM, ~20 tok/s."
        case .gpu:
            return "Prefill + decode on Metal GPU. ~3 GB Metal heap, ~28 tok/s. Bit-exact vs fp32."
        case .ane:
            return "Prefill + decode both on ANE. 0 GB Metal, ~22 tok/s. First run compiles ANE ~4 min — be patient, not a hang."
        }
    }
}

struct Qwen35ChatView: View {
    @State private var generator = Qwen35Generator()
    @State private var tokenizer: (any Tokenizer)?
    @State private var tokenizerStatus = "Loading tokenizer..."
    @State private var inputText = ""
    @State private var outputText = ""
    @State private var maxNewTokens: Int = 64
    @State private var errorMsg = ""
    @State private var compute: ComputeChoice = .gpu
    @State private var temperature: Double = 0.0  // argmax by default (fast on ANE)
    @State private var topK: Int = 40
    @State private var repetitionPenalty: Double = 1.0  // off by default

    var body: some View {
        NavigationStack {
            Form {
                Section("Compute") {
                    Picker("Units", selection: $compute) {
                        ForEach(ComputeChoice.allCases) { c in
                            Text(c.rawValue).tag(c)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(generator.running)
                    .onChange(of: compute) { _, newValue in
                        generator.setComputeUnits(prefill: newValue.units,
                                                   decode: newValue.units)
                    }
                    Text(compute.note).font(.caption).foregroundStyle(.secondary)
                }

                Section("Sampling") {
                    HStack {
                        Text("Temperature").font(.caption)
                        Slider(value: $temperature, in: 0...1.5, step: 0.1)
                        Text(String(format: "%.1f", temperature))
                            .font(.caption.monospacedDigit())
                    }
                    Stepper("Top-K: \(topK)", value: $topK, in: 1...200, step: 5)
                    HStack {
                        Text("Rep. penalty").font(.caption)
                        Slider(value: $repetitionPenalty, in: 1.0...1.5, step: 0.05)
                        Text(String(format: "%.2f", repetitionPenalty))
                            .font(.caption.monospacedDigit())
                    }
                }

                Section("Prompt") {
                    TextEditor(text: $inputText)
                        .frame(minHeight: 80)
                    Stepper("Max new tokens: \(maxNewTokens)",
                            value: $maxNewTokens, in: 8...96)
                }

                Section {
                    Button {
                        Task { await send() }
                    } label: {
                        HStack {
                            if generator.running { ProgressView() }
                            Text(generator.running ? "Generating..." : "Generate")
                        }
                    }
                    .disabled(generator.running || tokenizer == nil ||
                              inputText.trimmingCharacters(in: .whitespaces).isEmpty)
                    Text(tokenizer == nil ? tokenizerStatus : generator.status)
                        .font(.caption).foregroundStyle(.secondary)
                    if !errorMsg.isEmpty {
                        Text(errorMsg).font(.caption).foregroundStyle(.red)
                    }
                }

                if !outputText.isEmpty {
                    Section("Output") {
                        Text(outputText)
                            .font(.body)
                            .textSelection(.enabled)
                        HStack {
                            Text(String(format: "%.1f tok/s", generator.tokensPerSecond))
                                .font(.caption.monospaced())
                            Spacer()
                            Text(String(format: "%.0f ms prefill", generator.prefillMs))
                                .font(.caption.monospaced())
                            Spacer()
                            Text(String(format: "%.1f ms/tok decode", generator.decodeMsAvg))
                                .font(.caption.monospaced())
                        }
                        .foregroundStyle(.secondary)
                    }

                    if !generator.firstStepDebug.isEmpty {
                        Section("Debug: top-5 tokens at first step") {
                            ForEach(Array(generator.firstStepDebug.enumerated()), id: \.offset) { idx, pair in
                                let (id, logit) = pair
                                let text = tokenizer?.decode(tokens: [Int(id)]) ?? "?"
                                HStack {
                                    Text("#\(idx) id=\(id)").font(.caption.monospaced())
                                    Spacer()
                                    Text(text.prefix(20)).font(.caption.monospaced())
                                    Spacer()
                                    Text(String(format: "%.3f", logit))
                                        .font(.caption.monospaced())
                                }
                            }
                        }
                    }

                    if generator.decodeProfile.count > 0 {
                        Section("Profile: decode per-step breakdown (n=\(generator.decodeProfile.count))") {
                            let p = generator.decodeProfile
                            phaseRow("ANE predict", p.predict, p.total)
                            phaseRow("state copy (×48)", p.stateCopy, p.total)
                            phaseRow("inputs build", p.inputsBuild, p.total)
                            phaseRow("logit read / argmax", p.logitRead, p.total)
                            HStack {
                                Text("total measured").font(.caption.monospaced())
                                Spacer()
                                Text(String(format: "%.2f ms", p.total))
                                    .font(.caption.monospaced())
                                Spacer()
                                Text(String(format: "%.1f tok/s", 1000 / p.total))
                                    .font(.caption.monospaced())
                            }
                        }
                    }
                }

                Section("Notes") {
                    Text(
                        "Qwen3.5-0.8B instruct model with <|im_start|>/<|im_end|> "
                        + "chat template applied automatically.\n\n"
                        + "Temperature 0 (default) = greedy argmax, fastest on ANE "
                        + "(vDSP SIMD, ~1ms/step overhead). Temperature > 0 enables "
                        + "sampling (~20-50ms/step overhead, softer output).\n\n"
                        + "First ANE selection compiles ~4 min; subsequent runs "
                        + "are cached and fast."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Qwen3.5 Chat")
            .task { await loadTokenizer() }
        }
    }

    private func phaseRow(_ label: String, _ ms: Double, _ total: Double) -> some View {
        let pct = total > 0 ? (ms / total) * 100 : 0
        return HStack {
            Text(label).font(.caption.monospaced())
            Spacer()
            Text(String(format: "%.2f ms", ms)).font(.caption.monospaced())
            Spacer()
            Text(String(format: "%.0f%%", pct)).font(.caption.monospaced())
                .foregroundStyle(pct > 40 ? .red : .secondary)
        }
    }

    private func loadTokenizer() async {
        if tokenizer != nil { return }
        tokenizerStatus = "Loading Qwen tokenizer from HF..."
        do {
            // AutoTokenizer.from(pretrained:) downloads & caches the tokenizer
            // under .cachesDirectory on first use.
            let t = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3.5-0.8B")
            await MainActor.run {
                self.tokenizer = t
                tokenizerStatus = "Tokenizer ready"
            }
        } catch {
            await MainActor.run {
                tokenizerStatus = "Tokenizer failed: \(error.localizedDescription)"
            }
        }
    }

    private func send() async {
        errorMsg = ""
        outputText = ""
        guard let tok = tokenizer else { return }
        let prompt = inputText.trimmingCharacters(in: .whitespaces)
        // Apply Qwen3.5 chat template so the instruct-tuned model responds
        // in chat mode. Without this, raw `tok.encode(prompt)` routes
        // through the base-completion path and produces continuation-style
        // (or degenerate) output.
        let ids: [Int]
        do {
            let messages: [[String: Any]] = [["role": "user", "content": prompt]]
            ids = try tok.applyChatTemplate(messages: messages)
        } catch {
            await MainActor.run {
                errorMsg = "Chat template failed: \(error.localizedDescription); falling back to raw encode"
            }
            ids = tok.encode(text: prompt)
        }
        let idsInt32 = ids.map { Int32($0) }
        // Qwen3.5 emits <|im_end|> (248046) and often continues into a
        // new turn; also stop on <|endoftext|> if reported by the tokenizer.
        var eosSet: Set<Int32> = [248046]
        if let eosId = tok.eosTokenId { eosSet.insert(Int32(eosId)) }
        do {
            let newIds = try await generator.generate(
                inputIds: idsInt32,
                maxNewTokens: maxNewTokens,
                temperature: Float(temperature),
                topK: topK,
                repetitionPenalty: Float(repetitionPenalty),
                eosTokenIds: eosSet)
            // Strip trailing EOS before decoding
            let filtered = newIds.filter { !eosSet.contains($0) }
            let text = tok.decode(tokens: filtered.map(Int.init))
            await MainActor.run { outputText = text }
        } catch {
            await MainActor.run { errorMsg = error.localizedDescription }
        }
    }
}
