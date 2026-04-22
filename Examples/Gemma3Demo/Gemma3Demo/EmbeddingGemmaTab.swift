import SwiftUI
import CoreMLLLM

@MainActor
final class EmbeddingGemmaModel: ObservableObject {
    enum State {
        case idle
        case downloading(progress: Double, file: String)
        case loading
        case ready
        case computing
        case failed(String)
    }

    @Published var state: State = .idle
    @Published var cosine: Float?
    @Published var stats: String = ""

    private var runner: EmbeddingGemma?

    func load() async {
        do {
            let dir = FileManager.default
                .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Gemma3Demo")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)

            let bundleURL: URL
            if let cached = Gemma3BundleDownloader.localBundle(.embeddingGemma300m, under: dir) {
                bundleURL = cached
            } else {
                state = .downloading(progress: 0, file: "")
                bundleURL = try await Gemma3BundleDownloader.download(
                    .embeddingGemma300m,
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
            runner = try await EmbeddingGemma.load(bundleURL: bundleURL)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func compare(a: String, b: String, dim: Int?) async {
        guard let runner else { return }
        state = .computing
        let t0 = Date()
        do {
            let v1 = try runner.encode(text: a, task: .similarity, dim: dim)
            let v2 = try runner.encode(text: b, task: .similarity, dim: dim)
            cosine = Self.cosine(v1, v2)
            let dt = Date().timeIntervalSince(t0)
            stats = String(format: "2 encodes · %.2fs · dim=%d", dt, v1.count)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(max(na, 1e-12)) * sqrtf(max(nb, 1e-12)))
    }
}

struct EmbeddingGemmaTab: View {
    @StateObject private var m = EmbeddingGemmaModel()
    @State private var textA: String = "The cat sat on the mat."
    @State private var textB: String = "A feline rested on the rug."
    @State private var dim: Int = 768

    private let dimOptions = [768, 512, 256, 128]

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                statusBanner

                if case .ready = m.state { compareUI }
                else if case .computing = m.state { compareUI }
                else { loadUI }

                if let c = m.cosine {
                    cosineCard(c)
                }
                if !m.stats.isEmpty {
                    Text(m.stats).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding()
            .navigationTitle("EmbeddingGemma 300M")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusBanner: some View {
        Group {
            switch m.state {
            case .idle:
                Text("Tap Download to fetch the bundle (~588 MB) from HuggingFace.")
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
                Label("Ready (encoder loaded on Neural Engine)", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.callout)
            case .computing:
                ProgressView("Encoding…")
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

    private var compareUI: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Text A", text: $textA, axis: .vertical)
                .textFieldStyle(.roundedBorder).lineLimit(2)
            TextField("Text B", text: $textB, axis: .vertical)
                .textFieldStyle(.roundedBorder).lineLimit(2)
            HStack {
                Text("Dim:").font(.caption)
                Picker("Matryoshka dim", selection: $dim) {
                    ForEach(dimOptions, id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
            }
            Button("Compute cosine") {
                Task { await m.compare(a: textA, b: textB, dim: dim) }
            }
            .buttonStyle(.borderedProminent)
            .disabled({ if case .computing = m.state { true } else { textA.isEmpty || textB.isEmpty } }())
        }
    }

    private func cosineCard(_ c: Float) -> some View {
        VStack(spacing: 4) {
            Text("cosine similarity")
                .font(.caption).foregroundStyle(.secondary)
            Text(String(format: "%.4f", c))
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .foregroundStyle(c > 0.6 ? .green : (c > 0.3 ? .orange : .red))
            Text(c > 0.6 ? "very similar" : (c > 0.3 ? "somewhat similar" : "different"))
                .font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
}

#Preview { EmbeddingGemmaTab() }
