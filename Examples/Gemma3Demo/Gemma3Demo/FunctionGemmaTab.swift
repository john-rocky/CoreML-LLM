import SwiftUI
import CoreMLLLM

@MainActor
final class FunctionGemmaModel: ObservableObject {
    enum State {
        case idle
        case downloading(progress: Double, file: String)
        case loading
        case ready
        case generating
        case failed(String)
    }

    @Published var state: State = .idle
    @Published var output: String = ""
    @Published var stats: String = ""

    private var runner: FunctionGemma?

    func load(hfToken: String?) async {
        state = .downloading(progress: 0, file: "")
        do {
            let dir = FileManager.default
                .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Gemma3Demo")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)

            // Use the package's one-call API: download (cached after first run)
            // + load. The wrapper LocalAIKit will eventually wrap exactly this
            // shape — the example app is the proving ground.
            runner = try await FunctionGemma.downloadAndLoad(
                modelsDir: dir,
                hfToken: hfToken?.isEmpty == false ? hfToken : nil,
                onProgress: { [weak self] p in
                    Task { @MainActor in
                        let pct = p.bytesTotal > 0
                            ? Double(p.bytesReceived) / Double(p.bytesTotal)
                            : 0
                        self?.state = .downloading(progress: pct, file: p.currentFile)
                    }
                }
            )
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func generate(prompt: String, maxNew: Int) async {
        guard let runner else { return }
        state = .generating
        output = ""
        let t0 = Date()
        do {
            var count = 0
            for try await piece in runner.stream(prompt: prompt, maxNewTokens: maxNew) {
                output += piece
                count += 1
            }
            let dt = Date().timeIntervalSince(t0)
            stats = String(format: "%d tokens · %.1f tok/s", count, Double(count) / dt)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}

struct FunctionGemmaTab: View {
    @StateObject private var m = FunctionGemmaModel()
    @State private var prompt: String = "List three primary colors:"
    @State private var maxNew: Double = 80
    @State private var hfToken: String = ""

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                statusBanner

                if case .ready = m.state { generateUI }
                else if case .generating = m.state { generateUI }
                else { loadUI }

                Divider()
                ScrollView {
                    Text(m.output.isEmpty ? "(output will appear here)" : m.output)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(m.output.isEmpty ? .secondary : .primary)
                        .padding(8)
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(8)
                }
                if !m.stats.isEmpty {
                    Text(m.stats).font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding()
            .navigationTitle("FunctionGemma 270M")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusBanner: some View {
        Group {
            switch m.state {
            case .idle:
                Text("Tap Download to fetch the bundle (~840 MB) from HuggingFace.")
                    .font(.callout).foregroundStyle(.secondary)
            case .downloading(let p, let f):
                VStack(alignment: .leading, spacing: 4) {
                    Text("Downloading \(Int(p * 100))%").font(.callout)
                    Text(f).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    ProgressView(value: p).progressViewStyle(.linear)
                }
            case .loading:
                ProgressView("Loading model…")
            case .ready:
                Label("Ready (model loaded on Neural Engine)", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.callout)
            case .generating:
                Label("Generating…", systemImage: "ellipsis.bubble")
                    .foregroundStyle(.blue).font(.callout)
            case .failed(let msg):
                Label(msg, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red).font(.caption)
            }
        }
    }

    private var loadUI: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("HF token (only if model is gated)", text: $hfToken)
                .textFieldStyle(.roundedBorder)
                .autocapitalization(.none)
                .disableAutocorrection(true)
            Button("Download & Load") {
                Task { await m.load(hfToken: hfToken) }
            }
            .buttonStyle(.borderedProminent)
            .disabled({ if case .downloading = m.state { true } else { false } }())
        }
    }

    private var generateUI: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...4)
            HStack {
                Text("Max tokens: \(Int(maxNew))").font(.caption)
                Slider(value: $maxNew, in: 16...256, step: 16)
            }
            Button("Generate") {
                Task { await m.generate(prompt: prompt, maxNew: Int(maxNew)) }
            }
            .buttonStyle(.borderedProminent)
            .disabled({ if case .generating = m.state { true } else { prompt.isEmpty } }())
        }
    }
}

#Preview { FunctionGemmaTab() }
