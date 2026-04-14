//
//  PromptLookupDraft.swift
//  CoreMLLLM
//
//  Prompt Lookup Decoding (PLD) — zero-training n-gram draft source.
//
//  Given the full token history (prompt + tokens generated so far), find
//  the most recent place where the last `ngramSize` tokens appeared
//  earlier in the sequence, and propose the tokens that followed as
//  speculative drafts. High acceptance on summaries / QA where the
//  answer quotes the prompt; near-zero on free-form chat.
//
//  Pure CPU, no ANE / MLModel dependency. Designed to feed into the
//  target's Q=K verifier (ChunkedEngine.verifyCandidates) alongside or
//  as a fallback to a neural drafter (EAGLE-3 / MTP).
//
//  Roadmap: Phase 2 Track C item 12.
//

import Foundation

public enum PromptLookupDraft {
    /// Find the most recent n-gram match in `history` whose suffix matches
    /// the last `ngramSize` tokens, and return up to `maxDraftLen` tokens
    /// that followed the match in the prior context.
    ///
    /// Returns `[]` when fewer than `2·ngramSize + 1` history tokens are
    /// available or no match exists.
    ///
    /// - Parameters:
    ///   - history: full token sequence (prompt + output so far), int32.
    ///   - ngramSize: suffix length to match. 2 or 3 works well in practice.
    ///   - maxDraftLen: upper bound on the proposed draft length.
    public static func propose(
        history: [Int32],
        ngramSize: Int = 3,
        maxDraftLen: Int = 8
    ) -> [Int32] {
        let n = history.count
        guard ngramSize > 0, maxDraftLen > 0, n >= 2 * ngramSize + 1 else { return [] }

        let tailStart = n - ngramSize
        // Valid match positions: i in 0..<(tailStart - ngramSize + 1) so the
        // match is entirely before the tail AND leaves at least 1 draft token
        // (draftStart = i + ngramSize must be strictly less than tailStart).
        let maxI = tailStart - ngramSize  // exclusive upper bound on i
        guard maxI > 0 else { return [] }

        return history.withUnsafeBufferPointer { buf in
            let p = buf.baseAddress!
            var i = maxI - 1
            while i >= 0 {
                var match = true
                for j in 0..<ngramSize where p[i + j] != p[tailStart + j] {
                    match = false
                    break
                }
                if match {
                    let draftStart = i + ngramSize
                    let draftEnd = min(draftStart + maxDraftLen, tailStart)
                    return Array(UnsafeBufferPointer(start: p.advanced(by: draftStart),
                                                     count: draftEnd - draftStart))
                }
                i -= 1
            }
            return []
        }
    }
}
