import Foundation

/// Declaration of a tool the model can call. Mirrors the OpenAI /
/// Anthropic function-calling schema so callers can port JSON definitions
/// from those APIs directly.
public struct ToolSpec: @unchecked Sendable {
    public let name: String
    public let description: String
    /// Parameters as a JSON-schema dictionary. Pass the same object
    /// shape you'd send to OpenAI's `tools[].function.parameters`.
    public let parametersJSON: [String: Any]
    /// Handler invoked with the parsed JSON arguments. Return a string
    /// that will be fed back to the model as the tool's response.
    /// Must be `@Sendable` because the handler is invoked from the
    /// detached decode task.
    public let handler: @Sendable ([String: Any]) async throws -> String

    public init(name: String, description: String,
                parameters: [String: Any] = ["type": "object", "properties": [:]],
                handler: @Sendable @escaping ([String: Any]) async throws -> String) {
        self.name = name
        self.description = description
        self.parametersJSON = parameters
        self.handler = handler
    }
}

/// Parsed `<tool_call>` block.
struct ToolCallInvocation {
    let name: String
    let argumentsJSON: String
    /// Raw arguments parsed as a dictionary (best-effort; invalid JSON
    /// yields an empty dict and the handler must handle it).
    let argumentsDict: [String: Any]
}

/// Looks for a complete `<tool_call>...</tool_call>` block anywhere in
/// the running decode text. Returns the parsed invocation and the
/// text-range to strip from user-visible output.
struct ToolCallDetector {
    static let open = "<tool_call>"
    static let close = "</tool_call>"

    /// Returns `nil` while no complete tool call has been observed yet.
    static func tryParse(_ text: String) -> (invocation: ToolCallInvocation,
                                              prefix: String)? {
        guard let openRange = text.range(of: open),
              let closeRange = text.range(of: close,
                                          range: openRange.upperBound..<text.endIndex)
        else { return nil }
        let body = String(text[openRange.upperBound..<closeRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let prefix = String(text[..<openRange.lowerBound])

        guard let data = body.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let name = obj["name"] as? String
        else { return nil }
        let args = (obj["arguments"] as? [String: Any]) ?? [:]
        let argsJSON: String = {
            if let argsData = try? JSONSerialization.data(withJSONObject: args,
                                                           options: [.sortedKeys]),
               let s = String(data: argsData, encoding: .utf8) {
                return s
            }
            return "{}"
        }()
        let inv = ToolCallInvocation(name: name,
                                     argumentsJSON: argsJSON,
                                     argumentsDict: args)
        return (inv, prefix)
    }
}

// MARK: - JSON mode detector

/// Tracks the state of a top-level JSON value being emitted by the model,
/// character-by-character. Reports "complete" once the first object or
/// array closes with matched braces.
struct JSONCompletionDetector {
    private enum State {
        case searching           // pre-any-JSON — waiting for `{` or `[`
        case inValue(depth: Int, inString: Bool, escape: Bool, opener: Character)
        case done
    }
    private var state: State = .searching

    /// Advance with new text. Returns `true` when a complete top-level
    /// JSON value has been closed.
    mutating func feed(_ chunk: String) -> Bool {
        for ch in chunk {
            switch state {
            case .done: return true
            case .searching:
                if ch == "{" || ch == "[" {
                    state = .inValue(depth: 1, inString: false, escape: false, opener: ch)
                }
            case .inValue(var depth, var inString, var escape, let opener):
                if inString {
                    if escape { escape = false }
                    else if ch == "\\" { escape = true }
                    else if ch == "\"" { inString = false }
                } else {
                    if ch == "\"" { inString = true }
                    else if ch == opener { depth += 1 }
                    else if (opener == "{" && ch == "}") || (opener == "[" && ch == "]") {
                        depth -= 1
                        if depth == 0 { state = .done; return true }
                    }
                }
                state = .inValue(depth: depth, inString: inString, escape: escape, opener: opener)
            }
        }
        return false
    }
}

// MARK: - Stop-sequence matcher

/// Incremental substring matcher for user-supplied stop strings. Returns
/// the byte prefix of the stream that should be yielded before stopping,
/// plus a flag when any stop was hit. Operates over the whole cumulative
/// decoded buffer (cheap because the buffer is short, and stop strings
/// are short).
struct StopSequenceMatcher {
    let sequences: [String]

    init(_ sequences: [String]) {
        self.sequences = sequences.filter { !$0.isEmpty }
    }

    /// Returns `(safePrefix, matched)`: `safePrefix` is the part of
    /// `cumulative` that is safe to emit, and `matched` is non-nil when
    /// a stop sequence was found (value: the stop sequence that fired).
    func findStop(in cumulative: String) -> (safePrefix: String, matched: String?) {
        guard !sequences.isEmpty else { return (cumulative, nil) }
        var earliest: (range: Range<String.Index>, seq: String)?
        for seq in sequences {
            if let r = cumulative.range(of: seq) {
                if earliest == nil || r.lowerBound < earliest!.range.lowerBound {
                    earliest = (r, seq)
                }
            }
        }
        if let hit = earliest {
            return (String(cumulative[..<hit.range.lowerBound]), hit.seq)
        }
        return (cumulative, nil)
    }

    /// Returns `true` when the tail of `cumulative` is a *partial*
    /// prefix of any stop sequence — in that case the runtime should
    /// buffer emission until it either completes (stop) or breaks the
    /// match (emit the buffered chunk).
    func tailMightMatch(_ cumulative: String) -> Bool {
        guard !sequences.isEmpty else { return false }
        for seq in sequences {
            let maxN = min(seq.count, cumulative.count)
            for n in stride(from: maxN, through: 1, by: -1) {
                let tail = cumulative.suffix(n)
                if seq.hasPrefix(String(tail)) { return true }
            }
        }
        return false
    }
}
