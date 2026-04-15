//
//  Drafters.swift — speculative draft sources for accept-rate bench.
//
//  Each drafter is stateless from the replay harness's perspective: it's
//  given the entire history up to a position and asked for up to K draft
//  tokens. The harness ingests the ACTUAL next token after every burst so
//  stateful drafters (suffix index) can update internally.
//
//  All drafters run purely on CPU. No ANE/GPU dispatch. Measurement is
//  numerics-free — just token ID comparisons against the temp=0 oracle.
//

import Foundation
import CoreMLLLM

/// Draft-source interface. The harness calls `reset()` before each prompt,
/// `ingest(_:)` after every actual-next-token is committed, and
/// `propose(...)` to sample drafts at the current position.
protocol Drafter: AnyObject {
    var name: String { get }
    func reset()
    func ingest(_ token: Int32)
    /// Propose up to K draft tokens given `history`. May return fewer than K
    /// on a miss; the harness treats an empty result as zero-match (baseline).
    func propose(history: [Int32], K: Int) -> [Int32]
}

/// Phase A2 candidate. Wraps the already-merged pure algorithm in
/// `Sources/CoreMLLLM/PromptLookupDraft.swift`. Stateless.
final class PromptLookupDrafter: Drafter {
    let name: String
    let ngramSize: Int

    init(ngramSize: Int = 3) {
        self.ngramSize = ngramSize
        self.name = "prompt-lookup-n\(ngramSize)"
    }

    func reset() {}
    func ingest(_ token: Int32) {}

    func propose(history: [Int32], K: Int) -> [Int32] {
        PromptLookupDraft.propose(history: history, ngramSize: ngramSize, maxDraftLen: K)
    }
}

/// Phase A3 candidate. Naive suffix scan over the running history. A proper
/// suffix tree is an O(N) vs O(N^2) optimisation; the naive scan is fine at
/// this corpus size (prompts + 128 emitted ≈ 500 tokens).
///
/// Scans ngramSize from 4 down to 2, returning the tokens following the
/// most-recent earlier match. Tracks the full session (prompt + emitted)
/// across bursts, so multi-turn self-quotation is captured.
final class SuffixTreeDrafter: Drafter {
    let name = "suffix-scan"

    func reset() {}
    func ingest(_ token: Int32) {}

    /// Stateless over the caller-supplied `history`. That's safe here
    /// because the replay harness passes the full committed prefix up to
    /// position P each iteration, so there's no delta to track.
    func propose(history: [Int32], K: Int) -> [Int32] {
        guard history.count >= 4 else { return [] }
        for ngram in stride(from: 4, through: 2, by: -1) {
            let tail = Array(history.suffix(ngram))
            let end = history.count - ngram
            guard end > 0 else { continue }
            var i = end - 1
            while i >= 0 {
                var match = true
                for j in 0..<ngram where history[i + j] != tail[j] {
                    match = false
                    break
                }
                if match {
                    let start = i + ngram
                    let sliceEnd = min(start + K, history.count)
                    guard start < sliceEnd else { return [] }
                    return Array(history[start..<sliceEnd])
                }
                i -= 1
            }
        }
        return []
    }
}

/// Phase A4 candidate. Wraps the shipping `CrossVocabDraft` (Qwen 2.5 0.5B
/// monolithic model + Qwen↔Gemma vocab map) in the oracle-replay protocol.
///
/// The drafter is stateful (owns Qwen MLState). We advance its
/// `committedPosition` exactly one step per oracle-replay iteration by
/// rewinding after each `draftBurst` and letting the harness's `ingest`
/// feed the actual-next token via `consume`.
final class CrossVocabOracleDrafter: Drafter {
    let name = "cross-vocab-qwen"
    private let drafter: CrossVocabDraft
    private let K: Int

    init(drafter: CrossVocabDraft, K: Int) {
        self.drafter = drafter
        self.K = K
    }

    func reset() {
        drafter.reset()
    }

    func ingest(_ token: Int32) {
        _ = try? drafter.consume(gemmaToken: token)
    }

    func propose(history: [Int32], K: Int) -> [Int32] {
        guard let seed = history.last else { return [] }
        let saved = drafter.committedPosition
        defer { drafter.committedPosition = saved }  // rewind — see class doc
        guard let burst = try? drafter.draftBurst(seed: seed) else { return [] }
        return Array(burst.drafts.prefix(K))
    }
}

/// Per-drafter accept statistics accumulated across prompts.
struct AcceptStats: Codable {
    /// Histogram: index k = count of bursts where exactly k tokens matched (0 ≤ k ≤ K).
    var histogram: [Int]
    var totalBursts: Int

    /// Probability that position k is accepted, conditional on positions
    /// 0..k-1 having been accepted. = chain accept rate at step k.
    var chainAccept: [Double] {
        guard totalBursts > 0 else { return Array(repeating: 0, count: histogram.count) }
        var out: [Double] = []
        var remaining = totalBursts
        for k in 0..<(histogram.count - 1) {
            let accepted = histogram[(k + 1)...].reduce(0, +)
            out.append(remaining > 0 ? Double(accepted) / Double(remaining) : 0)
            remaining = accepted
        }
        return out
    }

    /// Expected emitted tokens per burst = 1 + sum(chainAccept). The 1 is the
    /// always-committed "correction/bonus" token emitted after a miss or at
    /// the tail of a fully-accepted burst.
    var expectedTokensPerBurst: Double {
        1.0 + chainAccept.reduce(0, +)
    }

    init(K: Int) {
        self.histogram = Array(repeating: 0, count: K + 1)
        self.totalBursts = 0
    }
}

/// Oracle replay. Given the full token sequence the target produced at
/// temperature = 0, walk position by position: ask the drafter for K
/// proposals, count the matching prefix against the actual next K tokens.
///
/// Correctness argument: at temp=0 the target's argmax is deterministic,
/// so the verify chunk's K-way argmax at position P+1..P+K equals the
/// emitted[P+1..P+K] of a standard single-token decode. Chain accept is
/// therefore identical between this oracle replay and a real verify pass.
func replayDrafter(
    _ drafter: Drafter,
    prompt: [Int32],
    emitted: [Int32],
    K: Int
) -> AcceptStats {
    drafter.reset()
    // Ingest all prompt tokens except the LAST one. That matches the state
    // convention `CrossVocabDraft` expects at burst time: cp == P where P
    // is the position of the just-emitted / just-committed token whose
    // continuation we want to draft. Propose sees `history[0...P]` via the
    // explicit parameter, and the stateful drafter's cp stays at P until
    // ingest advances it.
    for tok in prompt.dropLast() { drafter.ingest(tok) }

    let all = prompt + emitted
    var stats = AcceptStats(K: K)
    let startPos = prompt.count - 1
    let lastStart = all.count - K - 1
    guard startPos <= lastStart else { return stats }

    for P in startPos...lastStart {
        let draft = drafter.propose(history: Array(all[0...P]), K: K)
        var match = 0
        for k in 0..<min(K, draft.count) {
            if draft[k] == all[P + 1 + k] {
                match += 1
            } else {
                break
            }
        }
        stats.histogram[match] += 1
        stats.totalBursts += 1
        // Advance drafter state past all[P] before next iteration.
        drafter.ingest(all[P])
    }
    return stats
}
