import XCTest
@testable import CoreMLLLM

/// Exercises the drafting state machine (draftBurst / applyCommit /
/// consume / fallback) using a scripted stepFn — no CoreML model loaded.
///
/// Convention in these tests:
///   * Qwen token id == Gemma token id (i.e. identity map over [0..31]).
///     This lets us reason about accept/reject purely in terms of token
///     ids without confusing translation layers.
///   * The scripted `stepFn` returns `argmaxTable[qwenToken]` — i.e.
///     feeding token T always produces argmax[T] as the next Qwen pick.
final class CrossVocabDraftTests: XCTestCase {

    private func identityMap(size: Int) throws -> CrossVocabMap {
        let qvs = size, gvs = size
        var data = Data()
        data.append(contentsOf: Array("QGVMAP01".utf8))
        withUnsafeBytes(of: UInt32(qvs).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(gvs).littleEndian) { data.append(contentsOf: $0) }
        let q2g = [Int32](0..<Int32(qvs))
        let g2q = [Int32](0..<Int32(gvs))
        q2g.withUnsafeBufferPointer { buf in
            data.append(UnsafeBufferPointer(start: buf.baseAddress, count: buf.count)
                .withMemoryRebound(to: UInt8.self) { Data($0) })
        }
        g2q.withUnsafeBufferPointer { buf in
            data.append(UnsafeBufferPointer(start: buf.baseAddress, count: buf.count)
                .withMemoryRebound(to: UInt8.self) { Data($0) })
        }
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qgvmap_id_\(UUID().uuidString).bin")
        try data.write(to: tmp)
        return try CrossVocabMap(url: tmp)
    }

    /// Drift test: under an identity argmax table (Qwen always picks
    /// token T+1 given token T, modulo 10), draftBurst should emit
    /// predictable proposals and advance committedPosition by K.
    func testDraftBurstAdvancesByK() throws {
        let map = try identityMap(size: 32)
        let K = 3
        var stepCalls: [(Int32, Int)] = []
        let step: CrossVocabDraft.StepFn = { token, pos in
            stepCalls.append((token, pos))
            return (token + 1) % 10  // deterministic successor
        }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: K, contextLength: 128)
        XCTAssertEqual(draft.committedPosition, 0)

        let burst = try draft.draftBurst(seed: 5)
        XCTAssertEqual(burst.startPosition, 0)
        XCTAssertEqual(burst.drafts.count, K)
        // seed=5 → argmax=6 → drafts[0]=6. Feed 6 → 7 → drafts[1]=7.
        // Feed 7 → 8 → drafts[2]=8. (Loop exits on i == K-1.)
        XCTAssertEqual(burst.drafts, [6, 7, 8])
        XCTAssertEqual(burst.lastQwenProposal, 8)
        // K steps run: seed + (K-1) forward feeds = K calls.
        XCTAssertEqual(stepCalls.count, K)
        XCTAssertEqual(draft.committedPosition, K)
        // Positions were 0, 1, 2.
        XCTAssertEqual(stepCalls.map { $0.1 }, [0, 1, 2])
        // Tokens fed were: seed=5, then nextQwen=6, then 7.
        XCTAssertEqual(stepCalls.map { $0.0 }, [5, 6, 7])
    }

    /// After a partial-accept cycle (matchCount < K), applyCommit must
    /// leave committedPosition at P + matchCount + 1 so the next cycle's
    /// seed lands at that slot.
    func testApplyCommitPartialAccept() throws {
        let map = try identityMap(size: 32)
        let K = 3
        let step: CrossVocabDraft.StepFn = { token, _ in (token + 1) % 10 }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: K, contextLength: 128)

        let burst = try draft.draftBurst(seed: 5)
        XCTAssertEqual(draft.committedPosition, 3)
        try draft.applyCommit(matchCount: 1, burst: burst)
        // P=0, matchCount=1 → committedPosition = 0 + 1 + 1 = 2.
        XCTAssertEqual(draft.committedPosition, 2)
    }

    /// All-accepted case: applyCommit must feed the lastQwenProposal so
    /// its KV joins state, and land committedPosition at P + K + 1.
    func testApplyCommitAllAccepted() throws {
        let map = try identityMap(size: 32)
        let K = 3
        var steps: [(Int32, Int)] = []
        let step: CrossVocabDraft.StepFn = { token, pos in
            steps.append((token, pos))
            return (token + 1) % 10
        }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: K, contextLength: 128)

        let burst = try draft.draftBurst(seed: 5)
        XCTAssertEqual(steps.count, K)
        try draft.applyCommit(matchCount: K, burst: burst)
        // applyCommit should have fed lastQwenProposal=8 at committedPosition=3.
        XCTAssertEqual(steps.count, K + 1)
        XCTAssertEqual(steps.last?.0, 8)
        XCTAssertEqual(steps.last?.1, 3)
        XCTAssertEqual(draft.committedPosition, K + 1)  // = 4
    }

    /// consume() advances by one and returns the Qwen argmax (identity
    /// map means Gemma token == Qwen token).
    func testConsumeAdvancesAndReturnsArgmax() throws {
        let map = try identityMap(size: 32)
        let step: CrossVocabDraft.StepFn = { token, _ in token + 100 }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: 3, contextLength: 128)

        let argmax = try draft.consume(gemmaToken: 7)
        XCTAssertEqual(argmax, 107)
        XCTAssertEqual(draft.committedPosition, 1)
    }

    /// Unmappable seed → draftBurst returns empty drafts, committedPosition
    /// unchanged, caller (speculative engine) falls back to single-step.
    func testDraftBurstUnmappableSeedShortCircuits() throws {
        let map = try identityMap(size: 8)  // ids 0..7 mapped; 12 is miss
        var stepCalls = 0
        let step: CrossVocabDraft.StepFn = { _, _ in
            stepCalls += 1
            return 3
        }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: 3, contextLength: 128)

        let burst = try draft.draftBurst(seed: 12)
        XCTAssertTrue(burst.drafts.isEmpty)
        XCTAssertEqual(stepCalls, 0, "no forward passes when seed is unmappable")
        XCTAssertEqual(draft.committedPosition, 0)
    }

    /// Multi-cycle trace mimicking production use: three back-to-back
    /// spec cycles with accept counts {all, partial, zero}, verifying
    /// every forward pass lands on the right position.
    func testMultiCycleTrace() throws {
        let map = try identityMap(size: 64)
        let K = 3
        // argmax(T) = T + 1 mod 30 — easy to predict what the drafter proposes.
        var steps: [(Int32, Int)] = []
        let step: CrossVocabDraft.StepFn = { token, pos in
            steps.append((token, pos))
            return (token + 1) % 30
        }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: K, contextLength: 128)

        // Prefill 2 tokens.
        _ = try draft.consume(gemmaToken: 10)
        _ = try draft.consume(gemmaToken: 11)
        XCTAssertEqual(draft.committedPosition, 2)

        // Cycle 1: seed=12, all K accepted.
        let b1 = try draft.draftBurst(seed: 12)
        XCTAssertEqual(b1.startPosition, 2)
        XCTAssertEqual(b1.drafts, [13, 14, 15])
        XCTAssertEqual(draft.committedPosition, 5)
        try draft.applyCommit(matchCount: K, burst: b1)
        // committedPosition = 2 + K + 1 = 6 (lastQwen fed at pos 5).
        XCTAssertEqual(draft.committedPosition, 6)
        XCTAssertEqual(steps.suffix(4).map { $0.1 }, [2, 3, 4, 5])

        // Cycle 2: seed=16 (bonus from last cycle), matchCount=1.
        let b2 = try draft.draftBurst(seed: 16)
        XCTAssertEqual(b2.startPosition, 6)
        XCTAssertEqual(b2.drafts, [17, 18, 19])
        XCTAssertEqual(draft.committedPosition, 9)
        try draft.applyCommit(matchCount: 1, burst: b2)
        // committedPosition = 6 + 1 + 1 = 8.
        XCTAssertEqual(draft.committedPosition, 8)

        // Cycle 3: seed=20 (correction), matchCount=0.
        let b3 = try draft.draftBurst(seed: 20)
        XCTAssertEqual(b3.startPosition, 8)
        XCTAssertEqual(draft.committedPosition, 11)
        try draft.applyCommit(matchCount: 0, burst: b3)
        // committedPosition = 8 + 0 + 1 = 9.
        XCTAssertEqual(draft.committedPosition, 9)

        // Positions fed across all 3 cycles. Write-through KV means a
        // partial-accept cycle can leave stale draft KV at positions
        // beyond the new committedPosition; the next cycle's seed
        // write at the same slot overwrites it. This manifests as
        // a repeated position in the trace — expected, not a bug.
        //
        // Trace:
        //   consume ×2:       pos 0, 1
        //   cycle 1 draft ×3: pos 2, 3, 4    (all-accepted → lastQwen fed too)
        //   cycle 1 commit:   pos 5          (K+1-th write)
        //   cycle 2 draft ×3: pos 6, 7, 8    (partial accept; commit rewinds
        //                                     committedPosition to 8)
        //   cycle 3 seed:     pos 8          (OVERWRITES stale entry from 8)
        //   cycle 3 draft ×2: pos 9, 10
        let positions = steps.map { $0.1 }
        XCTAssertEqual(positions,
                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10],
                       "position trace including expected stale-slot overwrite")
        // Sanity: the duplicate 8s are the only repeat; nothing moves
        // backwards by >1.
        for i in 1..<positions.count {
            XCTAssertGreaterThanOrEqual(positions[i], positions[i - 1] - 1)
        }
    }

    /// Full cycle trace: prefill 3 tokens via consume(), run one spec
    /// cycle with partial accept, verify all positions advance as expected.
    func testFullCycleTrace() throws {
        let map = try identityMap(size: 32)
        let K = 3
        var steps: [(Int32, Int)] = []
        let step: CrossVocabDraft.StepFn = { token, pos in
            steps.append((token, pos))
            return (token + 1) % 20
        }
        let draft = CrossVocabDraft(stepFn: step, vocabMap: map, K: K, contextLength: 128)

        // Bootstrap-style replay of 3 committed tokens.
        _ = try draft.consume(gemmaToken: 2)
        _ = try draft.consume(gemmaToken: 3)
        _ = try draft.consume(gemmaToken: 4)
        XCTAssertEqual(draft.committedPosition, 3)

        // Spec cycle: seed=5, matchCount=2 (partial).
        let burst = try draft.draftBurst(seed: 5)
        XCTAssertEqual(burst.startPosition, 3)
        XCTAssertEqual(burst.drafts, [6, 7, 8])
        XCTAssertEqual(draft.committedPosition, 6)
        try draft.applyCommit(matchCount: 2, burst: burst)
        // P=3, match=2 → committedPosition = 3 + 2 + 1 = 6.
        XCTAssertEqual(draft.committedPosition, 6)

        // Positions fed across the whole trace: 0,1,2 (consume) then
        // 3,4,5 (draft), none from applyCommit (partial-accept path).
        XCTAssertEqual(steps.map { $0.1 }, [0, 1, 2, 3, 4, 5])
    }
}
