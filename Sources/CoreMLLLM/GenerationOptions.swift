import Foundation

/// User-facing generation knobs. Designed to match the common surface of
/// LiteRT-LM / llama.cpp / HuggingFace `GenerationConfig` so that porting
/// prompts from those stacks is a drop-in.
///
/// **Sampling caveat.** The shipped decode graph (`chunk4`) performs an
/// in-model argmax and exposes only `token_id` + `token_logit` (the max
/// logit value), not the full logit vector. This means `temperature`,
/// `topK`, `topP`, `minP`, and `repetitionPenalty` are **accepted by the
/// API but treated as greedy** unless:
///
/// 1. An MTP drafter is loaded (we reuse its top-K heads for per-step
///    multinomial sampling), OR
/// 2. `chunk4` was rebuilt with a top-K output (see
///    `conversion/build_chunk4_topk.py`).
///
/// When sampling is not available and `temperature > 0`, the runtime
/// logs a one-shot warning and falls back to greedy. Deterministic
/// behaviour at `temperature == 0` or `topK == 1` matches the previous
/// API exactly (bit-exact token stream).
public struct GenerationOptions: Sendable {
    /// Maximum number of decode tokens to emit (exclusive of prompt).
    public var maxTokens: Int

    /// Additional stop strings. Generation halts as soon as any of these
    /// appear in the cumulative decoded text. The matched suffix is
    /// stripped from the output. Empty strings are ignored.
    public var stopSequences: [String]

    /// Additional stop token ids. Generation halts when the decoder
    /// commits any of these, *in addition to* the model's built-in EOS
    /// set. Pass `[]` to accept only the built-in set.
    public var stopTokenIds: [Int]

    /// Sampling temperature (0 = greedy / argmax, bit-exact). Higher
    /// values widen the distribution. `Double.nan` ≡ 0.
    public var temperature: Double

    /// Keep only the top-K most probable tokens before sampling.
    /// `0` disables top-K. `1` is equivalent to greedy.
    public var topK: Int

    /// Keep the smallest set of tokens whose cumulative probability
    /// exceeds `topP`. `1.0` disables top-P. Typical: `0.9`–`0.95`.
    public var topP: Double

    /// Drop tokens whose probability is below `minP * max_prob`.
    /// `0.0` disables. Typical: `0.05`.
    public var minP: Double

    /// Logit multiplier for tokens that appeared in the last
    /// `repetitionWindow` decode tokens. `1.0` disables.
    public var repetitionPenalty: Double

    /// How many recent tokens to consider for `repetitionPenalty`.
    public var repetitionWindow: Int

    /// Random seed. When `nil`, uses `SystemRandomNumberGenerator`.
    /// Used only when temperature > 0.
    public var seed: UInt64?

    /// If true, stop generation when a complete, balanced JSON value is
    /// produced (top-level object or array). Useful for structured
    /// output without constrained decoding.
    public var jsonMode: Bool

    /// Return per-token metadata in `GenerationEvent.token` (position,
    /// token id, max-logit proxy). Has no cost — always cheap to enable.
    public var returnTokenMetadata: Bool

    public init(
        maxTokens: Int = 2048,
        stopSequences: [String] = [],
        stopTokenIds: [Int] = [],
        temperature: Double = 0,
        topK: Int = 0,
        topP: Double = 1.0,
        minP: Double = 0.0,
        repetitionPenalty: Double = 1.0,
        repetitionWindow: Int = 64,
        seed: UInt64? = nil,
        jsonMode: Bool = false,
        returnTokenMetadata: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.stopSequences = stopSequences
        self.stopTokenIds = stopTokenIds
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
        self.seed = seed
        self.jsonMode = jsonMode
        self.returnTokenMetadata = returnTokenMetadata
    }

    public static let `default` = GenerationOptions()

    /// True when any sampling setting deviates from greedy/argmax.
    var needsSampling: Bool {
        let t = temperature.isNaN ? 0 : temperature
        return t > 0 || (topK > 0 && topK != 1) || topP < 1.0 || minP > 0
            || repetitionPenalty != 1.0
    }
}

// MARK: - Events

/// Structured stream events emitted from `CoreMLLLM.streamEvents(...)`.
/// Use this in place of `AsyncStream<String>` when you need TTFT, stop
/// reasons, tool calls, or per-token metadata.
public enum GenerationEvent: Sendable {
    /// A committed decode token, as decoded text plus optional metadata.
    case token(text: String, meta: TokenMeta?)
    /// First decode token has left the model — `ttft` is wall-clock
    /// seconds from `stream` call to this moment.
    case firstToken(ttft: TimeInterval)
    /// The model requested a tool call (Gemma 4 `<tool_call>` block).
    /// Arguments are the raw JSON body; consumer is expected to call
    /// `llm.continueAfterTool(result:)` to resume.
    case toolCall(name: String, arguments: String)
    /// Terminal event. Exactly one is emitted per stream.
    case finished(reason: FinishReason, stats: GenerationStats)
}

public struct TokenMeta: Sendable {
    /// Absolute decode position at which this token was committed
    /// (0-based across the whole conversation — includes prompt).
    public let position: Int
    /// Model vocab id.
    public let tokenId: Int
    /// `chunk4.token_logit` value for this token. Bounded fp16, used as
    /// a cheap confidence proxy (not a proper logprob).
    public let maxLogit: Float?
}

public enum FinishReason: String, Sendable {
    case eos           = "eos"
    case maxTokens     = "max_tokens"
    case stopSequence  = "stop_sequence"
    case stopTokenId   = "stop_token_id"
    case cancelled     = "cancelled"
    case contextFull   = "context_full"
    case toolCall      = "tool_call"
    case jsonComplete  = "json_complete"
    case error         = "error"
}

public struct GenerationStats: Sendable {
    public let promptTokens: Int
    public let decodeTokens: Int
    public let ttft: TimeInterval
    public let prefillTime: TimeInterval
    public let decodeTime: TimeInterval
    public var tokensPerSecond: Double {
        decodeTime > 0 ? Double(decodeTokens) / decodeTime : 0
    }

    public init(promptTokens: Int, decodeTokens: Int,
                ttft: TimeInterval, prefillTime: TimeInterval,
                decodeTime: TimeInterval) {
        self.promptTokens = promptTokens
        self.decodeTokens = decodeTokens
        self.ttft = ttft
        self.prefillTime = prefillTime
        self.decodeTime = decodeTime
    }
}
