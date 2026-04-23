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
    @Published var functionCall: String?
    @Published var stats: String = ""

    private var runner: FunctionGemma?

    // Canonical mobile-actions tool list from the FunctionGemma docs. The
    // demo doesn't actually execute anything — it just shows that the model
    // emits a parseable function-call payload for the right kind of prompt.
    static let demoTools: [[String: Any]] = [
        [
            "type": "function",
            "function": [
                "name": "toggle_flashlight",
                "description": "Turn the phone flashlight on or off.",
                "parameters": ["type": "object", "properties": [:], "required": []],
            ],
        ],
        [
            "type": "function",
            "function": [
                "name": "set_timer",
                "description": "Set a countdown timer.",
                "parameters": [
                    "type": "object",
                    "properties": ["minutes": ["type": "integer", "description": "Duration in minutes"]],
                    "required": ["minutes"],
                ],
            ],
        ],
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": [
                    "type": "object",
                    "properties": ["city": ["type": "string", "description": "City name"]],
                    "required": ["city"],
                ],
            ],
        ],
        [
            "type": "function",
            "function": [
                "name": "send_message",
                "description": "Send a text message to a contact.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "recipient": ["type": "string", "description": "Contact name"],
                        "body": ["type": "string", "description": "Message text"],
                    ],
                    "required": ["recipient", "body"],
                ],
            ],
        ],
    ]

    func load() async {
        do {
            let dir = FileManager.default
                .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Gemma3Demo")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)

            // If the bundle is already on disk, skip straight to the loading
            // phase — otherwise the UI would flash "Downloading 100%" for a
            // few seconds while the downloader just confirms files exist.
            let bundleURL: URL
            if let cached = Gemma3BundleDownloader.localBundle(.functionGemma270m, under: dir) {
                bundleURL = cached
            } else {
                state = .downloading(progress: 0, file: "")
                bundleURL = try await Gemma3BundleDownloader.download(
                    .functionGemma270m,
                    into: dir,
                    onProgress: { [weak self] p in
                        Task { @MainActor in
                            let pct = p.bytesTotal > 0
                                ? Double(p.bytesReceived) / Double(p.bytesTotal)
                                : 0
                            self?.state = .downloading(progress: pct, file: p.currentFile)
                        }
                    }
                )
            }

            state = .loading
            runner = try await FunctionGemma.load(bundleURL: bundleURL)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func generate(userPrompt: String, maxNew: Int) async {
        guard let runner else { return }
        state = .generating
        output = ""
        functionCall = nil
        let t0 = Date()
        do {
            var count = 0
            for try await piece in runner.stream(
                messages: [["role": "user", "content": userPrompt]],
                tools: Self.demoTools,
                maxNewTokens: maxNew)
            {
                output += piece
                count += 1
            }
            functionCall = runner.extractFunctionCall(from: output)
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
    @State private var prompt: String = "Turn on the flashlight"
    @State private var maxNew: Double = 80

    private let examples = [
        "Turn on the flashlight",
        "Set a timer for 5 minutes",
        "What's the weather in Tokyo?",
        "Text my mom I'll be late",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    statusBanner
                    if case .ready = m.state { generateUI }
                    else if case .generating = m.state { generateUI }
                    else { loadUI }

                    Divider()
                    if let call = m.functionCall {
                        functionCallCard(call)
                    }
                    rawOutputCard

                    if !m.stats.isEmpty {
                        Text(m.stats).font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding()
            }
            .navigationTitle("FunctionGemma 270M")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusBanner: some View {
        Group {
            switch m.state {
            case .idle:
                VStack(alignment: .leading, spacing: 4) {
                    Text("Tap Download to fetch the bundle (~840 MB) from HuggingFace.")
                        .font(.callout).foregroundStyle(.secondary)
                    Text("This model emits structured function calls for mobile actions.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
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
        Button("Download & Load") {
            Task { await m.load() }
        }
        .buttonStyle(.borderedProminent)
        .disabled({
            switch m.state {
            case .downloading, .loading: return true
            default: return false
            }
        }())
    }

    private var generateUI: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("User request", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...4)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(examples, id: \.self) { ex in
                        Button { prompt = ex } label: {
                            Text(ex).font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }
            HStack {
                Text("Max tokens: \(Int(maxNew))").font(.caption)
                Slider(value: $maxNew, in: 16...256, step: 16)
            }
            Button("Generate") {
                Task { await m.generate(userPrompt: prompt, maxNew: Int(maxNew)) }
            }
            .buttonStyle(.borderedProminent)
            .disabled({ if case .generating = m.state { true } else { prompt.isEmpty } }())
        }
    }

    private func functionCallCard(_ call: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Function call emitted", systemImage: "function")
                .font(.caption).foregroundStyle(.secondary)
            Text(call)
                .font(.system(.body, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color.blue.opacity(0.10))
                .cornerRadius(8)
        }
    }

    private var rawOutputCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("raw output").font(.caption).foregroundStyle(.secondary)
            Text(m.output.isEmpty ? "(output will appear here)" : m.output)
                .frame(maxWidth: .infinity, alignment: .leading)
                .font(.system(.footnote, design: .monospaced))
                .foregroundStyle(m.output.isEmpty ? .secondary : .primary)
                .padding(8)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(8)
        }
    }
}

#Preview { FunctionGemmaTab() }
