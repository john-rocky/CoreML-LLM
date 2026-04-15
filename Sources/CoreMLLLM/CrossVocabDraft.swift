//
//  CrossVocabDraft.swift
//  CoreMLLLM
//
//  Route B / Task 3 — cross-vocabulary speculative drafter.
//
//  Uses a pre-trained Qwen 2.5 0.5B monolithic model to propose K
//  continuation tokens for a Gemma 4 target. A pre-built vocabulary map
//  (`conversion/build_qwen_gemma_vocab_map.py`) translates ids in both
//  directions by matching decoded surface forms.
//
//  Design notes
//  ------------
//  Qwen's monolithic graph uses write-through KV via MLState: the causal
//  mask selects which positions attend, and the `update_mask` (one-hot at
//  the current position) selects which slot receives the new KV entry.
//  Writes at an already-used position overwrite what was there, so we can
//  draft K tokens optimistically (advancing state past committed) and
//  later re-align by overwriting positions the target corrected. Stale
//  draft KV at positions > committedPosition stays invisible because the
//  causal mask never attends beyond the current query position.
//
//  State is tracked by a single `committedPosition` counter. Each
//  speculation cycle:
//    1. `draftBurst(seed:)` writes K Qwen KV entries at positions
//       [P, P+K-1] and returns K Gemma draft ids plus the final Qwen id.
//    2. Target verify determines `matchCount` and (unless matchCount==K)
//       a correction token.
//    3. `applyCommit(matchCount:lastQwenProposal:)` re-anchors
//       committedPosition so the next cycle's seed write lands at the
//       right slot, and for the all-accepted case feeds the trailing
//       Qwen proposal so its KV joins the state.
//

import CoreML
import Foundation

// MARK: - Vocabulary map

/// Bidirectional Qwen <-> Gemma id map. Produced offline by
/// `conversion/build_qwen_gemma_vocab_map.py`. Negative value = miss.
public final class CrossVocabMap {
    public let qwenVocabSize: Int
    public let gemmaVocabSize: Int
    public let qwenToGemma: [Int32]
    public let gemmaToQwen: [Int32]

    private static let magic: [UInt8] = Array("QGVMAP01".utf8)

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 16 else {
            throw CoreMLLLMError.modelNotFound("vocab map: truncated")
        }
        let header = Array(data[0..<8])
        guard header == Self.magic else {
            throw CoreMLLLMError.modelNotFound("vocab map: bad magic")
        }
        let qvs = Int(data.withUnsafeBytes {
            $0.load(fromByteOffset: 8, as: UInt32.self) })
        let gvs = Int(data.withUnsafeBytes {
            $0.load(fromByteOffset: 12, as: UInt32.self) })
        let expected = 16 + 4 * (qvs + gvs)
        guard data.count == expected else {
            throw CoreMLLLMError.modelNotFound(
                "vocab map: size mismatch (have \(data.count), need \(expected))")
        }
        var q = [Int32](repeating: -1, count: qvs)
        var g = [Int32](repeating: -1, count: gvs)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
                .advanced(by: 16)
                .bindMemory(to: Int32.self, capacity: qvs + gvs)
            for i in 0..<qvs { q[i] = base[i] }
            for i in 0..<gvs { g[i] = base[qvs + i] }
        }
        self.qwenVocabSize = qvs
        self.gemmaVocabSize = gvs
        self.qwenToGemma = q
        self.gemmaToQwen = g
    }

    @inline(__always)
    public func qwen(_ qid: Int32) -> Int32 {
        (qid >= 0 && qid < Int32(qwenVocabSize)) ? qwenToGemma[Int(qid)] : -1
    }

    @inline(__always)
    public func gemma(_ gid: Int32) -> Int32 {
        (gid >= 0 && gid < Int32(gemmaVocabSize)) ? gemmaToQwen[Int(gid)] : -1
    }
}

// MARK: - Qwen drafter

/// Result of a single drafting burst.
public struct DraftBurst {
    /// Up to K Gemma draft token ids (empty if the seed was unmappable).
    public let drafts: [Int32]
    /// Qwen id of the LAST draft (`drafts.last`). Needed for the
    /// all-accepted case where the caller must feed this token to Qwen
    /// at the trailing position to keep KV in sync.
    public let lastQwenProposal: Int32
    /// First position written by this burst — = committedPosition at the
    /// start of the burst. For debugging / assertions.
    public let startPosition: Int
}

public final class CrossVocabDraft {

    /// Closure that runs one Qwen forward pass at `position` with input
    /// token `qwenToken`, returning Qwen's argmax. Returns -1 to signal a
    /// hard stop (e.g. exhausted context). The production constructor
    /// supplies an `MLModel.prediction`-backed implementation; tests can
    /// inject a scripted sequence.
    public typealias StepFn = (Int32, Int) throws -> Int32

    public let contextLength: Int
    public let vocabMap: CrossVocabMap
    public let K: Int

    /// Position where the next *committed* Qwen KV write will land. The
    /// drafter's state at positions [0, committedPosition) is guaranteed
    /// consistent with the Gemma side's accepted history. State at
    /// positions >= committedPosition may hold stale speculative writes.
    /// Setter is module-internal so the orchestrating speculative engine
    /// can re-anchor or fast-forward during bootstrap / miss cycles.
    /// Exposed settable so offline accept-rate benches (oracle replay) can
    /// rewind state between per-position measurements. Runtime callers
    /// should prefer `applyCommit(matchCount:burst:)` which does this
    /// correctly relative to the last burst.
    public var committedPosition: Int = 0

    private let step: StepFn

    // Production-only (kept alive so MLState persists). Tests use the
    // `stepFn:` initializer and leave these nil.
    private let model: MLModel?
    private let state: MLState?

    /// Production initializer — loads a Qwen monolithic model.
    public convenience init(modelURL: URL,
                            vocabMap: CrossVocabMap,
                            K: Int,
                            contextLength: Int = 2048,
                            computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw CoreMLLLMError.modelNotFound(modelURL.lastPathComponent)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let m = try MLModel(contentsOf: modelURL, configuration: cfg)
        let st = m.makeState()

        let idsBuf = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let posBuf = try MLMultiArray(shape: [1], dataType: .int32)
        let cmBuf = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: contextLength)], dataType: .float16)
        let umBuf = try MLMultiArray(
            shape: [1, 1, NSNumber(value: contextLength), 1], dataType: .float16)

        let ctx = contextLength
        let mlStep: StepFn = { [m, st, idsBuf, posBuf, cmBuf, umBuf] qwenToken, position in
            if position >= ctx { return -1 }

            idsBuf[[0, 0] as [NSNumber]] = NSNumber(value: qwenToken)
            posBuf[0] = NSNumber(value: Int32(position))
            let cmp = cmBuf.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
            for i in 0..<ctx { cmp[i] = (i <= position) ? 0 : 0xFC00 }
            let ump = umBuf.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
            memset(ump, 0, ctx * MemoryLayout<UInt16>.stride)
            ump[min(position, ctx - 1)] = 0x3C00

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids":    MLFeatureValue(multiArray: idsBuf),
                "position_ids": MLFeatureValue(multiArray: posBuf),
                "causal_mask":  MLFeatureValue(multiArray: cmBuf),
                "update_mask":  MLFeatureValue(multiArray: umBuf),
            ])
            let out = try m.prediction(from: provider, using: st)
            guard let feat = out.featureValue(for: "token_id")?.multiArrayValue else {
                throw CoreMLLLMError.predictionFailed
            }
            return Int32(truncatingIfNeeded: feat[0].int64Value)
        }

        self.init(stepFn: mlStep,
                  vocabMap: vocabMap,
                  K: K,
                  contextLength: contextLength,
                  model: m,
                  state: st)
    }

    /// Testable initializer. `stepFn` is called exactly once per Qwen
    /// forward pass — tests can script the returned argmax sequence to
    /// exercise the drafting state machine without loading CoreML.
    public init(stepFn: @escaping StepFn,
                vocabMap: CrossVocabMap,
                K: Int,
                contextLength: Int = 2048,
                model: MLModel? = nil,
                state: MLState? = nil) {
        self.step = stepFn
        self.vocabMap = vocabMap
        self.K = K
        self.contextLength = contextLength
        self.model = model
        self.state = state
    }

    /// Rewind the logical position counter. Write-through KV makes stale
    /// entries at rewound positions invisible because the causal mask at
    /// the fresh position never attends past itself.
    public func reset() {
        committedPosition = 0
    }

    /// Feed a Gemma token through Qwen at `committedPosition`, advancing
    /// by one. Returns Qwen's argmax (useful for prefill bootstrap to
    /// prime the first `nextID`). Returns -1 if the token is unmappable.
    @discardableResult
    public func consume(gemmaToken: Int32) throws -> Int32 {
        let qid = vocabMap.gemma(gemmaToken)
        guard qid >= 0 else { return -1 }
        let argmax = try stepQwen(qwenToken: qid, position: committedPosition)
        committedPosition += 1
        return argmax
    }

    /// Run K forward passes to produce K draft proposals starting from
    /// the Gemma `seed` token (the last token the target committed or
    /// the current `nextID`). Advances `committedPosition` by K.
    ///
    /// Returns at most K drafts — fewer if any Qwen argmax has no Gemma
    /// inverse (in which case drafting stops early and the caller should
    /// treat it as a miss cycle).
    public func draftBurst(seed: Int32) throws -> DraftBurst {
        let start = committedPosition
        guard let seedQwen = mapSeed(seed) else {
            return DraftBurst(drafts: [], lastQwenProposal: -1, startPosition: start)
        }

        // First prediction: feed seed, get the argmax for the first proposal.
        var nextQwen = try stepQwen(qwenToken: seedQwen, position: committedPosition)
        committedPosition += 1

        var drafts: [Int32] = []
        drafts.reserveCapacity(K)
        var lastQwen: Int32 = nextQwen

        for i in 0..<K {
            let gemmaId = vocabMap.qwen(nextQwen)
            if gemmaId < 0 {
                // Draft terminates short; do not advance further.
                break
            }
            drafts.append(gemmaId)
            lastQwen = nextQwen
            // After the K-th append we do NOT feed another Qwen step —
            // the verify path only needs K drafts, not K+1.
            if i == K - 1 { break }
            nextQwen = try stepQwen(qwenToken: nextQwen, position: committedPosition)
            committedPosition += 1
        }

        return DraftBurst(drafts: drafts,
                          lastQwenProposal: lastQwen,
                          startPosition: start)
    }

    /// Re-anchor Qwen state after the target's verify pass.
    ///
    /// - matchCount: how many drafts (0..=K) were accepted by target.
    /// - burst:      the burst result from `draftBurst`.
    ///
    /// Behaviour:
    ///   * matchCount < K  →  rewind committedPosition to
    ///     `startPosition + matchCount + 1`. Stale Qwen KV at positions
    ///     beyond will be overwritten by the next cycle's seed.
    ///   * matchCount == K →  feed `lastQwenProposal` to Qwen so its KV
    ///     joins the state at the trailing slot, then move forward one
    ///     so committedPosition = startPosition + K + 1.
    public func applyCommit(matchCount: Int, burst: DraftBurst) throws {
        let P = burst.startPosition
        if matchCount < burst.drafts.count || matchCount < K {
            // Rewind; on the next cycle the seed write at
            // committedPosition will overwrite any stale entry.
            committedPosition = P + matchCount + 1
            return
        }
        // matchCount == K: all drafts accepted. The last proposal's Qwen
        // KV was never written (we only feed K-1 proposals through after
        // the seed). Feed it now so the state contains KV for the full
        // committed prefix of length K+1.
        if burst.lastQwenProposal >= 0 {
            // draftBurst left committedPosition at P + K. We need a write
            // at position P + K, so use committedPosition directly.
            _ = try stepQwen(qwenToken: burst.lastQwenProposal,
                             position: committedPosition)
        }
        committedPosition = P + K + 1
    }

    // MARK: - Private

    private func mapSeed(_ gemmaId: Int32) -> Int32? {
        let q = vocabMap.gemma(gemmaId)
        return q >= 0 ? q : nil
    }

    private func stepQwen(qwenToken: Int32, position: Int) throws -> Int32 {
        return try step(qwenToken, position)
    }
}
