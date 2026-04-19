// CLI smoke test for CoreMLLLM on Mac. Exercises the new LiteRT-LM-parity
// API (stop sequences, JSON mode, tool use, session clone, event
// streaming, metrics) when a model is available, plus the plain baseline
// decode path.
//
// Usage:
//   swift run -c release coreml-llm-smoke <model-dir> [prompt] [maxTokens]
//
// Env flags (choose one demo):
//   SMOKE_MODE=baseline  (default)  plain stream(prompt, maxTokens:)
//   SMOKE_MODE=events               streamEvents + return metrics
//   SMOKE_MODE=stop                 stop sequence demo ("3." cuts list)
//   SMOKE_MODE=json                 jsonMode=true + short JSON prompt
//   SMOKE_MODE=tool                 register a fake weather tool
//   SMOKE_MODE=session              save, continue, restore, continue
//   UNION_TRIP=1                    enable drafter union (legacy probe)

import CoreMLLLM
import Foundation

@main
struct Smoke {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            fputs("usage: \(args[0]) <model-dir> [prompt] [maxTokens]\n", stderr)
            exit(2)
        }
        let modelDir = URL(fileURLWithPath: args[1])
        let prompt = args.count >= 3
            ? args[2]
            : "Write three short sentences about the ocean."
        let maxTokens = args.count >= 4 ? (Int(args[3]) ?? 64) : 64
        let mode = ProcessInfo.processInfo.environment["SMOKE_MODE"]?.lowercased() ?? "baseline"

        do {
            print("[smoke] loading model from: \(modelDir.path)")
            let t0 = CFAbsoluteTimeGetCurrent()
            let llm = try await CoreMLLLM.load(from: modelDir) { msg in
                print("[load] \(msg)")
            }
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print("[smoke] loaded in \(String(format: "%.1f", dt))s — model=\(llm.modelName)")

            if ProcessInfo.processInfo.environment["UNION_TRIP"] != nil {
                llm.mtpEnabled = false
                llm.drafterUnionEnabled = true
                llm.crossVocabEnabled = true
                print("[smoke] UNION_TRIP=1 — mtp=off union=on cv=on")
            }

            print("[smoke] mode=\(mode) prompt=\"\(prompt)\" max_tokens=\(maxTokens)")

            switch mode {
            case "events":       try await demoEvents(llm: llm, prompt: prompt, maxTokens: maxTokens)
            case "stop":         try await demoStop(llm: llm, maxTokens: maxTokens)
            case "json":         try await demoJSON(llm: llm, maxTokens: maxTokens)
            case "tool":         try await demoTool(llm: llm, maxTokens: maxTokens)
            case "session":      try await demoSession(llm: llm, maxTokens: maxTokens)
            case "cancel":       try await demoCancel(llm: llm, prompt: prompt, maxTokens: maxTokens)
            default:             try await demoBaseline(llm: llm, prompt: prompt, maxTokens: maxTokens)
            }
            exit(0)
        } catch {
            fputs("[smoke] error: \(error)\n", stderr)
            exit(1)
        }
    }

    // MARK: - Demos

    static func demoBaseline(llm: CoreMLLLM, prompt: String, maxTokens: Int) async throws {
        var collected = ""
        let stream = try await llm.stream(prompt, maxTokens: maxTokens)
        for await tok in stream {
            collected += tok
            FileHandle.standardOutput.write(Data(tok.utf8))
        }
        print("\n---")
        print("[baseline] tok/s=\(fmt(llm.tokensPerSecond)) len=\(collected.count)")
    }

    static func demoEvents(llm: CoreMLLLM, prompt: String, maxTokens: Int) async throws {
        let options = GenerationOptions(maxTokens: maxTokens,
                                         returnTokenMetadata: true)
        let stream = try await llm.streamEvents(prompt, options: options)
        var ttft: TimeInterval = 0
        var decodeCount = 0
        var finalStats: GenerationStats?
        var reason: FinishReason?
        for await ev in stream {
            switch ev {
            case .firstToken(let t):
                ttft = t
                print("[events] firstToken after \(fmt(t))s")
            case .token(let txt, let meta):
                FileHandle.standardOutput.write(Data(txt.utf8))
                decodeCount += 1
                if let m = meta, decodeCount % 10 == 0 {
                    print("\n  [meta #\(decodeCount) pos=\(m.position) tid=\(m.tokenId) " +
                          "logit=\(m.maxLogit.map { fmt(Double($0)) } ?? "nil")]")
                }
            case .toolCall(let n, let a):
                print("\n[events] toolCall name=\(n) args=\(a)")
            case .finished(let r, let s):
                reason = r; finalStats = s
            }
        }
        print("\n---")
        print("[events] reason=\(reason?.rawValue ?? "?") " +
              "ttft=\(fmt(ttft))s decode=\(finalStats?.decodeTokens ?? 0) " +
              "tok/s=\(fmt(finalStats?.tokensPerSecond ?? 0))")
    }

    static func demoStop(llm: CoreMLLLM, maxTokens: Int) async throws {
        let options = GenerationOptions(
            maxTokens: maxTokens,
            stopSequences: ["3."],
            returnTokenMetadata: false)
        let p = "List three things: 1."
        print("[stop] prompt=\"\(p)\" stopSequences=[\"3.\"]")
        let stream = try await llm.streamEvents(p, options: options)
        for await ev in stream {
            switch ev {
            case .token(let t, _):
                FileHandle.standardOutput.write(Data(t.utf8))
            case .finished(let r, _):
                print("\n---")
                print("[stop] reason=\(r.rawValue) (expect stop_sequence)")
            default: break
            }
        }
    }

    static func demoJSON(llm: CoreMLLLM, maxTokens: Int) async throws {
        let options = GenerationOptions(maxTokens: maxTokens, jsonMode: true)
        let p = "Return a JSON object with one key \"hello\" set to \"world\". Output ONLY JSON."
        print("[json] jsonMode=on")
        let stream = try await llm.streamEvents(p, options: options)
        for await ev in stream {
            switch ev {
            case .token(let t, _):
                FileHandle.standardOutput.write(Data(t.utf8))
            case .finished(let r, _):
                print("\n---")
                print("[json] reason=\(r.rawValue) (expect json_complete)")
            default: break
            }
        }
    }

    static func demoTool(llm: CoreMLLLM, maxTokens: Int) async throws {
        let weather: @Sendable ([String: Any]) async throws -> String = { args in
            let city = (args["city"] as? String) ?? "unknown"
            return "It is 22°C and sunny in \(city)."
        }
        let tool = ToolSpec(
            name: "get_weather",
            description: "Get current weather for a city.",
            parameters: [
                "type": "object",
                "properties": ["city": ["type": "string"]],
                "required": ["city"],
            ],
            handler: weather)
        let p = "What's the weather in Tokyo? Use the get_weather tool."
        print("[tool] registered get_weather — prompt=\"\(p)\"")
        let result = try await llm.generate(p,
                                             options: GenerationOptions(maxTokens: maxTokens),
                                             tools: [tool])
        print("[tool] final text: \(result.text)")
        print("[tool] reason=\(result.reason.rawValue) decode=\(result.stats.decodeTokens)")
    }

    static func demoSession(llm: CoreMLLLM, maxTokens: Int) async throws {
        let firstPrompt = "My favourite colour is blue."
        print("[session] turn 1 prompt=\"\(firstPrompt)\"")
        _ = try await llm.generate(firstPrompt,
                                    options: GenerationOptions(maxTokens: maxTokens))
        guard let snap = llm.saveSession() else {
            print("[session] saveSession returned nil (monolithic model?)")
            return
        }
        print("[session] snapshot captured — tokens=\(snap.tokenCount)")
        // Continue in a fresh turn.
        let second = try await llm.generate("What did I just tell you?",
                                             options: GenerationOptions(maxTokens: maxTokens))
        print("[session] turn 2 said: \(second.text)")
        try llm.restoreSession(snap)
        let third = try await llm.generate("What did I just tell you?",
                                             options: GenerationOptions(maxTokens: maxTokens))
        print("[session] turn 3 (after restore) said: \(third.text)")
    }

    static func demoCancel(llm: CoreMLLLM, prompt: String, maxTokens: Int) async throws {
        let stream = try await llm.streamEvents(
            prompt, options: GenerationOptions(maxTokens: maxTokens))
        let task = Task {
            var tokens = 0
            for await ev in stream {
                switch ev {
                case .token(let t, _):
                    tokens += 1
                    FileHandle.standardOutput.write(Data(t.utf8))
                    if tokens >= 5 {
                        print("\n[cancel] cancelling after 5 tokens")
                        return tokens
                    }
                case .finished(let r, _):
                    print("\n[cancel] finished naturally reason=\(r.rawValue)")
                    return tokens
                default: break
                }
            }
            return tokens
        }
        try await Task.sleep(nanoseconds: 3_000_000_000)
        task.cancel()
        _ = await task.value
        print("[cancel] done")
    }

    static func fmt(_ d: Double) -> String { String(format: "%.2f", d) }
}

