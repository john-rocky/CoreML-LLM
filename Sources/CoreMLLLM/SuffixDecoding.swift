//
//  SuffixDecoding.swift
//  CoreMLLLM
//
//  CPU-only speculative draft using suffix tree lookup.
//
//  At each decode step the last k output tokens are matched against a suffix
//  tree built from all prior model outputs. The most-frequent continuation
//  becomes the draft candidate, verified by the existing SpeculativeTarget
//  verifyCandidates path (Q=K verifier, when available) or by T=1 greedy
//  comparison for validation.
//
//  Draft cost: ~20 µs/token on CPU. ANE stays 100% available for verification.
//
//  References:
//    - SuffixDecoding arXiv 2411.04975
//    - CMU CSD blog (2025)
//    - Snowflake ArcticInference (vLLM)
//

import CoreML
import Foundation

public final class SuffixDecoding {

    /// Underlying suffix tree storing n-gram frequencies.
    public let tree: SuffixTree

    /// Maximum draft tokens per burst.
    public let K: Int

    /// Number of recent tokens used as lookup context.
    public let contextWindow: Int

    /// Minimum tree node frequency to consider a continuation.
    public let minCount: Int32

    /// Auto-prune when node count exceeds this limit.
    public let maxNodes: Int

    // Rolling token context for lookups
    private var recentTokens: [Int32] = []

    // Tokens generated in the current turn (ingested on endGeneration)
    private var currentGeneration: [Int32] = []

    // MARK: - Stats

    public private(set) var lookupHits: Int = 0
    public private(set) var lookupMisses: Int = 0
    public private(set) var tokensAccepted: Int = 0
    public private(set) var tokensDrafted: Int = 0

    // T=1 validation (single-token prediction accuracy)
    public private(set) var t1Attempts: Int = 0
    public private(set) var t1Correct: Int = 0

    // MARK: - Init

    public init(tree: SuffixTree = SuffixTree(),
                K: Int = 5,
                contextWindow: Int = 8,
                minCount: Int32 = 1,
                maxNodes: Int = 500_000) {
        self.tree = tree
        self.K = K
        self.contextWindow = contextWindow
        self.minCount = minCount
        self.maxNodes = maxNodes
    }

    // MARK: - Token Management

    /// Append a generated token to the rolling context.
    public func appendToken(_ token: Int32) {
        recentTokens.append(token)
        currentGeneration.append(token)
        if recentTokens.count > contextWindow * 2 {
            recentTokens.removeFirst(recentTokens.count - contextWindow)
        }
    }

    // MARK: - Draft Generation

    /// Generate draft candidates from the suffix tree.
    ///
    /// Returns up to K tokens that the tree predicts will follow the current
    /// context. Pure CPU, typically completes in ~20 µs.
    public func draft() -> [Int32] {
        let context = Array(recentTokens.suffix(contextWindow))
        guard !context.isEmpty else {
            lookupMisses += 1
            return []
        }
        let candidates = tree.lookup(context: context,
                                     maxLength: K,
                                     minCount: minCount)
        if candidates.isEmpty {
            lookupMisses += 1
        } else {
            lookupHits += 1
            tokensDrafted += candidates.count
        }
        return candidates
    }

    // MARK: - T=1 Validation

    /// Record a single-token prediction result for accuracy tracking.
    ///
    /// Call after each decode step with the draft's first prediction and the
    /// target model's actual output. This does NOT accelerate decoding — it
    /// validates suffix tree quality before Q=K batch verification is ready.
    public func recordT1(predicted: Int32?, actual: Int32) {
        guard let predicted else { return }
        t1Attempts += 1
        if predicted == actual { t1Correct += 1 }
    }

    /// T=1 accuracy as a percentage (0–100).
    public var t1Accuracy: Double {
        guard t1Attempts > 0 else { return 0 }
        return Double(t1Correct) / Double(t1Attempts) * 100
    }

    /// Fraction of lookups that returned at least one candidate.
    public var hitRate: Double {
        let total = lookupHits + lookupMisses
        guard total > 0 else { return 0 }
        return Double(lookupHits) / Double(total)
    }

    // MARK: - SpeculativeTarget Integration (Q=K Verifier)

    /// Execute one speculative burst using suffix tree drafts.
    ///
    /// Follows the same verify-then-accept pattern as SpeculativeLoop.drawBurst:
    /// draft K candidates on CPU, verify via SpeculativeTarget.verifyCandidates,
    /// accept the longest matching prefix, commit to KV cache.
    ///
    /// Requires ChunkedEngine to conform to SpeculativeTarget (Q=K verifier).
    /// Until then, use T=1 validation via appendToken + draft + recordT1.
    ///
    /// - Parameters:
    ///   - target: SpeculativeTarget-conforming engine.
    ///   - lastToken: Target's most recent argmax (will always be accepted).
    /// - Returns: Accepted tokens (1…K+1), or empty if no draft candidates.
    public func drawBurst(target: SpeculativeTarget,
                          lastToken: Int32) throws -> [Int32] {
        let drafts = draft()
        guard !drafts.isEmpty else { return [] }

        let draftK = drafts.count

        // Build verify input: [lastToken, drafts[0], …, drafts[K-2]]
        var verifyTokens = [lastToken]
        if draftK > 1 {
            verifyTokens.append(contentsOf: drafts.dropLast())
        }

        let targetArgmax = try target.verifyCandidates(verifyTokens, K: draftK)
        guard targetArgmax.count == draftK else { return [] }

        // Accept prefix up to first disagreement.
        // lastToken is always accepted (target's own pick from previous step).
        var accepted: [Int32] = [lastToken]
        var matched = 0
        for k in 0..<draftK {
            if drafts[k] == targetArgmax[k] {
                accepted.append(drafts[k])
                matched += 1
            } else {
                accepted.append(targetArgmax[k])
                break
            }
        }

        try target.commitAccepted(accepted)

        tokensAccepted += matched
        for token in accepted {
            appendToken(token)
        }

        return accepted
    }

    // MARK: - Generation Lifecycle

    /// Call when a generation completes. Ingests the full output into the tree
    /// for future draft lookups. Auto-prunes if maxNodes is exceeded.
    /// Insert runs on a background queue to avoid blocking the next generation.
    public func endGeneration() {
        let tokens = currentGeneration
        currentGeneration = []
        recentTokens = []

        if !tokens.isEmpty {
            let t = tree
            let maxN = maxNodes
            DispatchQueue.global(qos: .utility).async {
                t.insert(sequence: tokens)
                if t.nodeCount > maxN {
                    t.prune(minCount: 2)
                }
            }
        }
    }

    /// Reset context without ingesting (e.g., on cancellation or reset).
    public func resetContext() {
        recentTokens = []
        currentGeneration = []
    }

    /// Reset all accumulated stats.
    public func resetStats() {
        lookupHits = 0
        lookupMisses = 0
        tokensAccepted = 0
        tokensDrafted = 0
        t1Attempts = 0
        t1Correct = 0
    }

    // MARK: - Persistence

    /// Save the suffix tree to disk for cross-session persistence.
    public func save(to url: URL) throws {
        try tree.save(to: url)
    }

    /// Load a persisted suffix tree and create a SuffixDecoding instance.
    public static func load(from url: URL,
                            K: Int = 5,
                            contextWindow: Int = 8,
                            minCount: Int32 = 1,
                            maxNodes: Int = 500_000) throws -> SuffixDecoding {
        let tree = try SuffixTree.load(from: url)
        return SuffixDecoding(tree: tree, K: K, contextWindow: contextWindow,
                              minCount: minCount, maxNodes: maxNodes)
    }

    // MARK: - Diagnostics

    /// One-line stats summary for logging.
    public var statsSummary: String {
        let total = lookupHits + lookupMisses
        let hitPct = total > 0
            ? String(format: "%.1f", hitRate * 100) : "N/A"
        let t1Pct = t1Attempts > 0
            ? String(format: "%.1f", t1Accuracy) : "N/A"
        return "[SuffixDecoding] tree=\(tree.nodeCount) nodes, "
            + "\(tree.totalSequences) seqs | "
            + "hits=\(hitPct)% (\(lookupHits)/\(total)) | "
            + "T1=\(t1Pct)% (\(t1Correct)/\(t1Attempts))"
    }
}
