import Foundation

/// Minimal, explicit chat template for the models this library ships with.
///
/// Every model's prompt format is encoded as a pure function of
/// `(messages, mediaBlocks)`. This deliberately avoids pulling in the
/// `jinja` dependency that HF `apply_chat_template` uses — the formats
/// we need (Gemma 4, Qwen) are simple string templates.
public struct ChatTemplate: Sendable {
    public enum Family: String, Sendable {
        case gemma4
        case qwen
    }

    public let family: Family

    public init(family: Family) { self.family = family }

    /// Render a conversation to the raw prompt string fed into the
    /// tokenizer. `imageBlock` / `audioBlock` are pre-formatted token
    /// strings (`<|image>...<image|>` and friends) and are inserted
    /// immediately before the last user turn, matching HF's
    /// `Gemma4Processor` behaviour.
    public func render(messages: [CoreMLLLM.Message],
                       imageBlock: String = "",
                       audioBlock: String = "",
                       addGenerationPrompt: Bool = true) -> String {
        switch family {
        case .gemma4:
            return renderGemma4(messages: messages,
                                imageBlock: imageBlock,
                                audioBlock: audioBlock,
                                addGenerationPrompt: addGenerationPrompt)
        case .qwen:
            return renderQwen(messages: messages,
                              addGenerationPrompt: addGenerationPrompt)
        }
    }

    // MARK: - Gemma 4

    private func renderGemma4(messages: [CoreMLLLM.Message],
                              imageBlock: String,
                              audioBlock: String,
                              addGenerationPrompt: Bool) -> String {
        let lastUserIdx = messages.lastIndex { $0.role == .user }

        // Collapse system prompts into the first user turn (Gemma 4 has no
        // explicit system role in its chat template, but users coming from
        // OpenAI/Anthropic-style APIs expect .system to be honoured).
        let systemText = messages
            .filter { $0.role == .system }
            .map { $0.content }
            .joined(separator: "\n")

        var p = "<bos>"
        var firstUserSeen = false
        for (i, m) in messages.enumerated() {
            switch m.role {
            case .user:
                let isLast = i == lastUserIdx
                var content = m.content
                if !firstUserSeen && !systemText.isEmpty {
                    content = systemText + "\n\n" + content
                    firstUserSeen = true
                }
                var mediaPrefix = ""
                if !imageBlock.isEmpty && isLast { mediaPrefix += imageBlock + "\n" }
                if !audioBlock.isEmpty && isLast { mediaPrefix += audioBlock + "\n" }
                p += "<|turn>user\n\(mediaPrefix)\(content)<turn|>\n"
            case .assistant:
                p += "<|turn>model\n\(m.content)<turn|>\n"
            case .system:
                // Handled by the collapse above. If no user turn follows,
                // synthesize a user-framed injection so the system prompt
                // isn't lost.
                break
            }
        }
        // If only a system message was present with no user turn.
        if !firstUserSeen && !systemText.isEmpty {
            p += "<|turn>user\n\(systemText)<turn|>\n"
        }
        if addGenerationPrompt {
            p += "<|turn>model\n"
        }
        return p
    }

    // MARK: - Qwen

    private func renderQwen(messages: [CoreMLLLM.Message],
                            addGenerationPrompt: Bool) -> String {
        var p = ""
        for m in messages {
            switch m.role {
            case .system:
                p += "<|im_start|>system\n\(m.content)<|im_end|>\n"
            case .user:
                p += "<|im_start|>user\n\(m.content)<|im_end|>\n"
            case .assistant:
                p += "<|im_start|>assistant\n\(m.content)<|im_end|>\n"
            }
        }
        if addGenerationPrompt {
            p += "<|im_start|>assistant\n"
        }
        return p
    }

    // MARK: - Tool-use prompt injection

    /// Prepend a tool-use system instruction to `messages`. Uses the
    /// Gemma 4 tool-call format: the model is asked to emit
    /// `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    /// on a line by itself when it wants to invoke a tool.
    ///
    /// The format intentionally matches Gemma 4's
    /// `tools.chat_template.jinja` so the model's training distribution
    /// covers it.
    public func injectTools(_ tools: [ToolSpec],
                            into messages: [CoreMLLLM.Message]) -> [CoreMLLLM.Message] {
        guard !tools.isEmpty else { return messages }
        let rendered = tools.map { tool -> String in
            let schema = (try? JSONSerialization.data(
                withJSONObject: tool.parametersJSON, options: [.sortedKeys])).flatMap {
                String(data: $0, encoding: .utf8)
            } ?? "{}"
            return "- name: \(tool.name)\n  description: \(tool.description)\n  parameters: \(schema)"
        }.joined(separator: "\n")
        let sys = """
        You have access to the following tools:
        \(rendered)

        When you want to use a tool, emit exactly one line of the form:
        <tool_call>{"name": "<tool-name>", "arguments": {...}}</tool_call>
        Then stop. The runtime will execute the tool and feed its result
        back as a <tool_response> message, after which you should
        continue the response using that result.
        """
        var out = messages
        out.insert(CoreMLLLM.Message(role: .system, content: sys), at: 0)
        return out
    }
}
