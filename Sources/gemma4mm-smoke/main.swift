// Mac smoke test for Gemma4StatefulMultimodalEngine.
//
// Usage:
//   swift run -c release gemma4mm-smoke <bundle-dir> [prompt] [maxTokens]
//
// `bundle-dir` should be the directory containing chunk_{1,2,3}.mlmodelc
// and a `prefill_T288/` subdir — i.e. the inner
// `gemma4_e2b_stateful_chunks/` folder, NOT the outer parent.
//
// Produces text-only output (no image/audio attachment). Used to catch
// engine bugs without needing an iPhone roundtrip — Mac ANE compiler is
// more permissive than iPhone's, so chunk_2 is expected to load here
// even when iPhone fails MIL→EIR translation.

import CoreML
import CoreMLLLM
import Foundation
import Tokenizers

@main
struct Gemma4MMSmoke {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            fputs("usage: \(args[0]) <bundle-dir> [prompt] [maxTokens]\n", stderr)
            exit(2)
        }
        let bundleDir = URL(fileURLWithPath: args[1])
        let prompt = args.count >= 3
            ? args[2]
            : "Write three short sentences about the ocean."
        let maxTokens = args.count >= 4 ? (Int(args[3]) ?? 64) : 64

        do {
            print("[smoke] bundle: \(bundleDir.path)")
            // Tokenizer.
            let hfDir = bundleDir.appendingPathComponent("hf_model")
            let tok = try await AutoTokenizer.from(modelFolder: hfDir)
            print("[smoke] tokenizer loaded")

            // Engine.
            let engine = Gemma4StatefulMultimodalEngine()
            let t0 = CFAbsoluteTimeGetCurrent()
            try await engine.load(modelDirectory: bundleDir)
            let loadDt = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "[smoke] engine loaded in %.1fs", loadDt))
            print("[smoke] hasVision=\(engine.hasVision) hasAudio=\(engine.hasAudio)")

            // Build a Gemma 4 chat prompt (single user turn).
            let promptStr = "<bos><|turn>user\n\(prompt)<turn|>\n<|turn>model\n"
            let inputIds = tok.encode(text: promptStr).map { Int32($0) }
            print("[smoke] input_ids = \(inputIds.count) tokens")
            print("[smoke] prompt: \(prompt)")
            print("[smoke] max_tokens=\(maxTokens)")

            var eosSet: Set<Int32> = [1, 106]
            if let eid = tok.eosTokenId { eosSet.insert(Int32(eid)) }
            let skipSet: Set<Int32> = [1, 105, 106]

            var accum: [Int] = []
            var emittedString = ""
            var totalEmitted = 0
            let genStart = CFAbsoluteTimeGetCurrent()
            _ = try await engine.generate(
                inputIds: inputIds,
                maxNewTokens: maxTokens,
                eosTokenIds: eosSet,
                onToken: { tokenId in
                    if skipSet.contains(tokenId) { return }
                    accum.append(Int(tokenId))
                    let current = tok.decode(tokens: accum)
                    if current.count > emittedString.count {
                        let delta = String(
                            current.suffix(current.count - emittedString.count))
                        FileHandle.standardOutput.write(Data(delta.utf8))
                        emittedString = current
                    }
                    totalEmitted += 1
                })
            let dt = CFAbsoluteTimeGetCurrent() - genStart
            print("\n---")
            print(String(format: "[smoke] decode wall = %.2fs", dt))
            print(String(format: "[smoke] last decode tok/s = %.2f",
                         engine.lastDecodeTokensPerSecond))
            print("[smoke] tokens emitted (non-skipped): \(totalEmitted)")
            exit(0)
        } catch {
            fputs("[smoke] error: \(error)\n", stderr)
            exit(1)
        }
    }
}
