import CoreML
import CoreMLLLM
import Foundation

// Standalone FunctionGemma-270M demo.
//
// Usage:
//   swift run functiongemma-demo --bundle <path> [--prompt "..."] [--max 128]
//   swift run functiongemma-demo --download --into <dir> [--hf-token hf_...]
//
// This sample intentionally does NOT combine FunctionGemma with Gemma 4 or
// any other model. Hybrid RAG / tool-calling pipelines belong in the
// LocalAIKit wrapper layer, not here.

struct Args {
    var bundleURL: URL?
    var downloadInto: URL?
    var hfToken: String?
    var prompt: String = "Turn on the flashlight"
    var maxTokens: Int = 128
    var raw: Bool = false
}

// Minimal tool list so FunctionGemma has somewhere to send its function calls.
let demoTools: [[String: Any]] = [
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
]

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--bundle":
            if let v = it.next() { a.bundleURL = URL(fileURLWithPath: v) }
        case "--download":
            break  // flag — paired with --into
        case "--into":
            if let v = it.next() { a.downloadInto = URL(fileURLWithPath: v) }
        case "--hf-token":
            if let v = it.next() { a.hfToken = v }
        case "--prompt":
            if let v = it.next() { a.prompt = v }
        case "--max":
            if let v = it.next(), let n = Int(v) { a.maxTokens = n }
        case "--raw":
            a.raw = true
        case "-h", "--help":
            print("""
            functiongemma-demo — standalone FunctionGemma-270M CLI

            Run with an existing bundle:
              --bundle <dir>         Path to output/functiongemma-270m/bundle
              --prompt "<text>"      User request (default: "Turn on the flashlight")
              --max <n>              Max new tokens (default: 128)
              --raw                  Skip chat template / tools (feeds prompt verbatim)

            Download on first use:
              --download --into <dir> [--hf-token hf_...]

            FunctionGemma-270M is fine-tuned for function calling. The default
            mode wraps --prompt in a chat template + a canonical tool list
            (flashlight / timer / weather), which is what the model expects.
            Without this, instruction-style prompts produce nonsense.
            """)
            exit(0)
        default:
            FileHandle.standardError.write("warning: unknown arg '\(arg)'\n".data(using: .utf8)!)
        }
    }
    return a
}

@main
struct Main {
    static func main() async throws {
        let args = parseArgs()

        let bundleURL: URL
        if let b = args.bundleURL {
            bundleURL = b
        } else if let dir = args.downloadInto {
            if let existing = Gemma3BundleDownloader.localBundle(.functionGemma270m, under: dir) {
                print("Using cached bundle at \(existing.path)")
                bundleURL = existing
            } else {
                print("Downloading FunctionGemma-270M into \(dir.path)...")
                var lastLine = ""
                bundleURL = try await Gemma3BundleDownloader.download(
                    .functionGemma270m,
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
                "usage: functiongemma-demo --bundle <dir> | --download --into <dir>\n"
                    .data(using: .utf8)!)
            exit(2)
        }

        print("Loading model from \(bundleURL.path)...")
        let t0 = Date()
        let fg = try await FunctionGemma.load(bundleURL: bundleURL)
        print(String(format: "  loaded in %.1fs", Date().timeIntervalSince(t0)))

        print("\nPrompt: \(args.prompt)")
        print("---")

        let t1 = Date()
        var count = 0
        let text: String
        if args.raw {
            text = try fg.generate(
                prompt: args.prompt,
                maxNewTokens: args.maxTokens,
                onToken: { piece in
                    FileHandle.standardOutput.write(piece.data(using: .utf8) ?? Data())
                    count += 1
                    return true
                })
        } else {
            text = try fg.generate(
                messages: [["role": "user", "content": args.prompt]],
                tools: demoTools,
                maxNewTokens: args.maxTokens,
                onToken: { piece in
                    FileHandle.standardOutput.write(piece.data(using: .utf8) ?? Data())
                    count += 1
                    return true
                })
        }
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        let dt = Date().timeIntervalSince(t1)
        FileHandle.standardError.write(
            String(format: "\n[%.1fs, %.1f tok/s, %d tokens]\n", dt, Double(count) / dt, count)
                .data(using: .utf8)!)

        if let call = fg.extractFunctionCall(from: text) {
            FileHandle.standardError.write(
                "function call: \(call)\n".data(using: .utf8)!)
        }
    }
}
