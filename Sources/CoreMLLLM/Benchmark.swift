import Foundation

/// Performance metrics from a generation run.
public struct Benchmark: Sendable {
    /// Tokens generated
    public let tokenCount: Int
    /// Time for prefill phase (seconds)
    public let prefillTime: TimeInterval
    /// Time for decode phase (seconds)
    public let decodeTime: TimeInterval
    /// Total generation time (seconds)
    public let totalTime: TimeInterval

    /// Prefill tokens per second
    public var prefillTokensPerSecond: Double {
        guard prefillTime > 0 else { return 0 }
        return Double(tokenCount) / prefillTime
    }

    /// Decode tokens per second
    public var decodeTokensPerSecond: Double {
        guard decodeTime > 0 else { return 0 }
        return Double(tokenCount) / decodeTime
    }

    /// Overall tokens per second
    public var tokensPerSecond: Double {
        guard totalTime > 0 else { return 0 }
        return Double(tokenCount) / totalTime
    }

    public var summary: String {
        String(
            format: "%d tokens in %.2fs (prefill: %.1f tok/s, decode: %.1f tok/s)",
            tokenCount, totalTime, prefillTokensPerSecond, decodeTokensPerSecond
        )
    }
}
