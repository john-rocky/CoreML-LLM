// pplx-embed-demo — minimal CLI around the PplxEmbed runtime.
//
// Plain:
//   swift run -c release pplx-embed-demo \
//       --bundle-dir output/pplx-embed \
//       --text "hello world" --text "bonjour le monde" --format int8
//
// Context (late chunking) — each --text is one chunk of a single document;
// use ';;' inside a --text to split a document into multiple chunks:
//   swift run -c release pplx-embed-demo \
//       --bundle-dir output/pplx-embed-context/L512-int8 \
//       --context --text "first chunk;;second chunk"

import CoreML
import CoreMLLLM
import Foundation

func args(_ name: String) -> [String] {
    let a = CommandLine.arguments
    var out: [String] = []
    var i = 0
    while i < a.count {
        if a[i] == name, i + 1 < a.count { out.append(a[i + 1]); i += 2 } else { i += 1 }
    }
    return out
}
func arg(_ name: String, _ def: String) -> String { args(name).first ?? def }
func flag(_ name: String) -> Bool { CommandLine.arguments.contains(name) }

func computeUnits(_ s: String) -> MLComputeUnits {
    switch s {
    case "all": return .all
    case "cpuOnly": return .cpuOnly
    case "cpuAndGPU": return .cpuAndGPU
    default: return .cpuAndNeuralEngine
    }
}

func l2norm(_ v: [Int8]) -> Double {
    var s = 0.0
    for x in v { s += Double(x) * Double(x) }
    return (s).squareRoot()
}

func summarize(_ v: [Int8]) -> String {
    let head = v.prefix(8).map { String($0) }.joined(separator: ", ")
    return "dim=\(v.count) first8=[\(head)] l2=\(String(format: "%.2f", l2norm(v)))"
}

// Either --bundle-dir (local) or --repo (download from HuggingFace) is required.
let bundleDir = arg("--bundle-dir", "")
let repo = arg("--repo", "")
guard !bundleDir.isEmpty || !repo.isEmpty else {
    FileHandle.standardError.write("--bundle-dir or --repo required\n".data(using: .utf8)!)
    exit(2)
}
let texts = args("--text")
guard !texts.isEmpty else {
    FileHandle.standardError.write("at least one --text required\n".data(using: .utf8)!)
    exit(2)
}
let isContext = flag("--context")
let format = PplxEmbed.Format(rawValue: arg("--format", "int8")) ?? .int8
let cu = computeUnits(arg("--compute-units", "cpuAndNE"))
let asJSON = flag("--json")   // emit raw int8 vectors as JSON (for parity checks)

let embedder: PplxEmbed
if !repo.isEmpty {
    // Download-then-run: pull only the requested buckets from HF (content-addressed
    // cache → the shared weight.bin is fetched once), then load.
    let buckets = args("--buckets").compactMap { Int($0) }
    let cacheDir = args("--cache-dir").first.map { URL(fileURLWithPath: $0) }
    let hfToken = args("--hf-token").first ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
    embedder = try await PplxEmbed.load(
        repo: repo,
        buckets: buckets.isEmpty ? [512, 1024, 2048] : buckets,
        into: cacheDir,
        computeUnits: cu,
        variant: isContext ? "context" : "plain",
        hfToken: hfToken,
        onProgress: { frac in
            FileHandle.standardError.write("\r[download] \(Int(frac * 100))%   "
                .data(using: .utf8)!)
        })
    FileHandle.standardError.write("\n".data(using: .utf8)!)
} else {
    embedder = try await PplxEmbed.load(
        bundleDir: URL(fileURLWithPath: bundleDir), computeUnits: cu)
}

// JSON mode: dump int8 vectors only (plain: [[Int8]]; context: [[[Int8]]]).
if asJSON {
    let enc = JSONEncoder()
    if isContext {
        let docs = texts.map { $0.components(separatedBy: ";;") }
        let int8 = try embedder.embedContext(docs)
        let data = try enc.encode(int8.map { $0.map { $0.map(Int.init) } })
        FileHandle.standardOutput.write(data)
    } else {
        let int8 = try embedder.embed(texts)
        let data = try enc.encode(int8.map { $0.map(Int.init) })
        FileHandle.standardOutput.write(data)
    }
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    exit(0)
}

if isContext {
    // Each --text is a document; ';;' splits it into chunks.
    let docs = texts.map { $0.components(separatedBy: ";;") }
    let int8 = try embedder.embedContext(docs)
    for (d, doc) in int8.enumerated() {
        print("doc[\(d)]: \(doc.count) chunk(s)")
        for (c, row) in doc.enumerated() {
            switch format {
            case .int8:
                print("  chunk[\(c)]: \(summarize(row))")
            case .binary:
                let b = PplxEmbed.binary(fromInt8: row)
                let head = b.prefix(8).map { String(Int($0)) }.joined(separator: ", ")
                print("  chunk[\(c)]: binary dim=\(b.count) first8=[\(head)]")
            case .ubinary:
                let u = PplxEmbed.ubinary(fromInt8: row)
                let head = u.prefix(8).map { String($0) }.joined(separator: ", ")
                print("  chunk[\(c)]: ubinary bytes=\(u.count) first8=[\(head)]")
            }
        }
    }
} else {
    let int8 = try embedder.embed(texts)
    for (i, row) in int8.enumerated() {
        let label = String(texts[i].prefix(40))
        switch format {
        case .int8:
            print("[\(i)] \"\(label)\" → \(summarize(row))")
        case .binary:
            let b = PplxEmbed.binary(fromInt8: row)
            let head = b.prefix(8).map { String(Int($0)) }.joined(separator: ", ")
            print("[\(i)] \"\(label)\" → binary dim=\(b.count) first8=[\(head)]")
        case .ubinary:
            let u = PplxEmbed.ubinary(fromInt8: row)
            let head = u.prefix(8).map { String($0) }.joined(separator: ", ")
            print("[\(i)] \"\(label)\" → ubinary bytes=\(u.count) first8=[\(head)]")
        }
    }
}
