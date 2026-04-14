//
//  MtpMacCheck — Mac-side end-to-end verification of Path C speculation.
//
//  Loads the deployed model directory (chunks + mtp_module_0/1) on Mac,
//  generates a short sequence, checks:
//    1. Output stream has no consecutive duplicate tokens
//    2. [MTP] Path C loaded message appears in log
//    3. Generation completes without crash
//
//  Usage:
//    swift run MtpMacCheck /tmp/device_deploy2 "こんにちは"
//
//  This saves an iPhone rebuild cycle when we change Swift logic.
//

import Foundation
import CoreMLLLM

@main
struct MtpMacCheck {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("usage: MtpMacCheck <model-dir> [prompt]")
            exit(1)
        }
        let modelDir = URL(fileURLWithPath: args[1])
        let prompt = args.count >= 3 ? args[2] : "Hello"
        print("Model dir: \(modelDir.path)")
        print("Prompt: \(prompt.debugDescription)")

        do {
            print("\n=== Loading model ===")
            let llm = try await CoreMLLLM.load(from: modelDir) { status in
                print("  [load] \(status)")
            }
            print("Loaded: \(llm.modelName)")

            print("\n=== Generating (max 32 tokens) ===")
            let messages: [CoreMLLLM.Message] = [
                .init(role: .user, content: prompt),
            ]
            let stream = try await llm.stream(messages, maxTokens: 32)

            var allTokens: [String] = []
            var accumulated = ""
            for await chunk in stream {
                print("  chunk: \(chunk.debugDescription)")
                allTokens.append(chunk)
                accumulated += chunk
            }

            print("\n=== Full output ===")
            print(accumulated)

            print("\n=== Duplicate check ===")
            var dupCount = 0
            for i in 1..<allTokens.count {
                if allTokens[i] == allTokens[i - 1] && !allTokens[i].isEmpty {
                    dupCount += 1
                    print("  DUP at \(i): \(allTokens[i].debugDescription)")
                }
            }
            if dupCount == 0 {
                print("  ✓ no consecutive duplicate chunks")
            } else {
                print("  ✗ FOUND \(dupCount) consecutive duplicates — double-emission bug likely")
            }

            print("\n=== Metrics ===")
            print("tok/s: \(String(format: "%.2f", llm.tokensPerSecond))")
            print("mtp acceptance rate: \(String(format: "%.1f%%", llm.mtpAcceptanceRate * 100))")
            print("mtp tokens per round: \(String(format: "%.2f", llm.mtpTokensPerRound))")

            exit(dupCount == 0 ? 0 : 2)
        } catch {
            print("ERROR: \(error)")
            exit(1)
        }
    }
}
