import CoreML
import CoreMLLLM
import Foundation

// Standalone EmbeddingGemma-300M demo.
//
// Usage:
//   swift run embeddinggemma-demo --bundle <path> [--text "..."] [--dim 768]
//   swift run embeddinggemma-demo --download --into <dir> [--hf-token hf_...]
//   swift run embeddinggemma-demo --bundle <path> --compare "text1" "text2"
//
// This sample is standalone; it does not combine EmbeddingGemma with
// Gemma 4. RAG / retrieval orchestration lives in LocalAIKit.

struct Args {
    var bundleURL: URL?
    var downloadInto: URL?
    var hfToken: String?
    var text: String = "The quick brown fox jumps over the lazy dog."
    var compare: [String] = []
    var task: EmbeddingGemma.Task?
    var dim: Int?
    var preview: Int = 8
}

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--bundle":
            if let v = it.next() { a.bundleURL = URL(fileURLWithPath: v) }
        case "--download":
            break
        case "--into":
            if let v = it.next() { a.downloadInto = URL(fileURLWithPath: v) }
        case "--hf-token":
            if let v = it.next() { a.hfToken = v }
        case "--text":
            if let v = it.next() { a.text = v }
        case "--compare":
            // Consume exactly two following args as the pair to compare.
            if let a1 = it.next(), let a2 = it.next() { a.compare = [a1, a2] }
        case "--task":
            if let v = it.next(), let t = EmbeddingGemma.Task(rawValue: v) { a.task = t }
        case "--dim":
            if let v = it.next(), let n = Int(v) { a.dim = n }
        case "--preview":
            if let v = it.next(), let n = Int(v) { a.preview = n }
        case "-h", "--help":
            print("""
            embeddinggemma-demo — standalone EmbeddingGemma-300M CLI

            Run with an existing bundle:
              --bundle <dir>         Path to output/embeddinggemma-300m/bundle
              --text "<text>"        Input text (default: "The quick brown fox …")
              --compare "A" "B"      Print cosine similarity of two strings
              --task <name>          Task prefix: retrieval_query | retrieval_document |
                                     classification | clustering | similarity |
                                     code_retrieval | question_answering | fact_verification
              --dim <n>              Matryoshka dim (768 | 512 | 256 | 128)
              --preview <n>          Print first N components of the vector (default 8)

            Download on first use:
              --download --into <dir> [--hf-token hf_...]
            """)
            exit(0)
        default:
            FileHandle.standardError.write("warning: unknown arg '\(arg)'\n".data(using: .utf8)!)
        }
    }
    return a
}

func cosine(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var dot: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    return dot / (sqrtf(max(na, 1e-12)) * sqrtf(max(nb, 1e-12)))
}

@main
struct Main {
    static func main() async throws {
        let args = parseArgs()

        let bundleURL: URL
        if let b = args.bundleURL {
            bundleURL = b
        } else if let dir = args.downloadInto {
            if let existing = Gemma3BundleDownloader.localBundle(.embeddingGemma300m, under: dir) {
                print("Using cached bundle at \(existing.path)")
                bundleURL = existing
            } else {
                print("Downloading EmbeddingGemma-300M into \(dir.path)...")
                var lastLine = ""
                bundleURL = try await Gemma3BundleDownloader.download(
                    .embeddingGemma300m,
                    into: dir,
                    hfToken: args.hfToken,
                    onProgress: { p in
                        let pct = p.bytesTotal > 0
                            ? Int(Double(p.bytesReceived) / Double(p.bytesTotal) * 100)
                            : 0
                        let line = "[\(pct)%] \(p.currentFile)"
                        if line != lastLine {
                            lastLine = line
                            FileHandle.standardError.write("  \(line)\n".data(using: .utf8)!)
                        }
                    })
                print("Downloaded to \(bundleURL.path)")
            }
        } else {
            FileHandle.standardError.write(
                "usage: embeddinggemma-demo --bundle <dir> | --download --into <dir>\n"
                    .data(using: .utf8)!)
            exit(2)
        }

        print("Loading model from \(bundleURL.path)...")
        let t0 = Date()
        let eg = try await EmbeddingGemma.load(bundleURL: bundleURL)
        print(String(format: "  loaded in %.1fs (max_seq_len=%d, embed_dim=%d)",
                     Date().timeIntervalSince(t0), eg.config.maxSeqLen, eg.config.embedDim))

        if args.compare.count == 2 {
            let t1 = Date()
            let v1 = try eg.encode(text: args.compare[0], task: args.task, dim: args.dim)
            let v2 = try eg.encode(text: args.compare[1], task: args.task, dim: args.dim)
            let c = cosine(v1, v2)
            print(String(format: "\ncosine(\"%@\", \"%@\")  @dim=%d = %.4f  (%.1fs)",
                         args.compare[0] as CVarArg,
                         args.compare[1] as CVarArg,
                         args.dim ?? eg.config.embedDim,
                         c,
                         Date().timeIntervalSince(t1)))
            return
        }

        print("\nInput: \(args.text)")
        print("Task prefix: \(args.task?.rawValue ?? "(none)")")
        print("Dim: \(args.dim ?? eg.config.embedDim)")
        let t1 = Date()
        let vec = try eg.encode(text: args.text, task: args.task, dim: args.dim)
        let dt = Date().timeIntervalSince(t1)
        print(String(format: "\nEncoded in %.2fs", dt))
        let shown = vec.prefix(args.preview).map { String(format: "% .4f", $0) }.joined(separator: " ")
        print("first \(args.preview): [\(shown)\(vec.count > args.preview ? " ..." : "")]")
        var norm: Float = 0
        for v in vec { norm += v * v }
        print(String(format: "norm: %.4f  (%d dims)", sqrtf(norm), vec.count))
    }
}
