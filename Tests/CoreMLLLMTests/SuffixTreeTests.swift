import XCTest
@testable import CoreMLLLM

/// Pure-algorithmic coverage for `SuffixTree` + `SuffixSpeculativeEngine`.
/// No CoreML models, no file I/O outside `NSTemporaryDirectory()`.
final class SuffixTreeTests: XCTestCase {

    // MARK: - Tree

    /// Two trajectories share prefix [A B C]; one continues with D twice,
    /// the other with E once. Drafting with suffix [A B C] must pick D
    /// (higher count).
    func testHigherCountWins() {
        let tree = SuffixTree(maxSuffixLen: 8, maxInsertDepth: 8, maxNodes: 1024)
        tree.insert(sequence: [1, 2, 3, 4])         // A B C D
        tree.insert(sequence: [1, 2, 3, 4])         // A B C D  (D count = 2)
        tree.insert(sequence: [1, 2, 3, 5])         // A B C E  (E count = 1)

        let drafts = tree.draft(suffix: [1, 2, 3], K: 1)
        XCTAssertEqual(drafts, [4], "expected D (count 2) over E (count 1)")
    }

    /// Suffix not in the tree returns empty. Ensures we don't fabricate
    /// drafts from unrelated branches.
    func testUnknownSuffixReturnsEmpty() {
        let tree = SuffixTree(maxSuffixLen: 8, maxInsertDepth: 8, maxNodes: 1024)
        tree.insert(sequence: [10, 20, 30, 40, 50])

        let drafts = tree.draft(suffix: [99, 98, 97], K: 4)
        XCTAssertTrue(drafts.isEmpty)
    }

    /// Multi-token draft walk follows most-frequent child at each step.
    func testMultiTokenDraftWalk() {
        let tree = SuffixTree(maxSuffixLen: 8, maxInsertDepth: 8, maxNodes: 1024)
        // A B C D E F (twice) beats A B C D X Y (once) after the shared
        // prefix, so drafting from [A B] should walk D → E → F.
        tree.insert(sequence: [1, 2, 3, 4, 5, 6])
        tree.insert(sequence: [1, 2, 3, 4, 5, 6])
        tree.insert(sequence: [1, 2, 3, 4, 7, 8])

        let drafts = tree.draft(suffix: [1, 2], K: 4)
        XCTAssertEqual(drafts, [3, 4, 5, 6])
    }

    /// Shorter-suffix fallback: full-length suffix misses, but a 1-token
    /// tail does match and produces a draft.
    func testSuffixFallback() {
        let tree = SuffixTree(maxSuffixLen: 8, maxInsertDepth: 8, maxNodes: 1024)
        tree.insert(sequence: [42, 7, 8, 9])

        // Suffix [99, 42] doesn't match as-is (no 99 root child), but the
        // tail [42] does — should produce [7].
        let drafts = tree.draft(suffix: [99, 42], K: 1)
        XCTAssertEqual(drafts, [7])
    }

    /// Persistence round-trip: encode → decode → draft should return the
    /// same tokens as the pre-encode tree.
    func testPersistenceRoundTrip() throws {
        let tree = SuffixTree(maxSuffixLen: 8, maxInsertDepth: 8, maxNodes: 1024)
        tree.insert(sequence: [1, 2, 3, 4, 5])
        tree.insert(sequence: [1, 2, 3, 4, 5])
        tree.insert(sequence: [1, 2, 3, 9])

        let original = tree.draft(suffix: [1, 2, 3], K: 2)
        XCTAssertEqual(original, [4, 5])

        let data = try tree.encodeSnapshot()
        let restored = try SuffixTree.decodeSnapshot(data)

        XCTAssertEqual(restored.draft(suffix: [1, 2, 3], K: 2), original)
        XCTAssertEqual(restored.approximateNodeCount, tree.approximateNodeCount)
    }

    /// File round-trip via `save` / `load` using a temp path.
    func testFileRoundTrip() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("suffix_tree_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let tree = SuffixTree()
        tree.insert(sequence: [10, 11, 12, 13])
        try tree.save(to: tmp)

        guard let loaded = try SuffixTree.load(from: tmp) else {
            XCTFail("load returned nil for freshly-saved tree"); return
        }
        XCTAssertEqual(loaded.draft(suffix: [10, 11], K: 2), [12, 13])
    }

    /// Purge keeps node count bounded. We shove enough distinct 4-grams
    /// into a small cap that auto-purge is triggered; an explicit final
    /// purge brings the tree down to the cap.
    func testPurgeBoundsMemory() {
        let cap = 64
        let tree = SuffixTree(maxSuffixLen: 8, maxInsertDepth: 4, maxNodes: cap)
        for i: Int32 in 0..<200 {
            tree.insert(sequence: [i, i &+ 1, i &+ 2, i &+ 3])
        }
        // Auto-purge keeps growth bounded — should be under a comfortable
        // ceiling even before a manual purge.
        let auto = tree.approximateNodeCount
        XCTAssertLessThanOrEqual(auto, 2 * cap + 64,
                                 "auto-purge kept tree bounded (got \(auto))")
        // Explicit purge brings us to the cap.
        tree.purge()
        XCTAssertLessThanOrEqual(tree.approximateNodeCount, cap,
                                 "manual purge drives tree to maxNodes")
    }

    // MARK: - Engine

    /// Engine wraps the tree and folds committed tokens back in.
    func testEngineCommitTrainsTree() {
        let tree = SuffixTree()
        let engine = SuffixSpeculativeEngine(tree: tree)

        // First generation — seed the tree via applyCommit.
        engine.setPrefillHistory([1, 2, 3])
        engine.applyCommit(tokens: [4, 5, 6])
        engine.applyCommit(tokens: [4, 5, 6])  // second time to bump count

        // Fresh generation with the same prefix should now draft [4, 5, 6].
        engine.reset()
        engine.setPrefillHistory([1, 2, 3])
        let drafts = engine.drawBurst(context: engine.history, K: 3)
        XCTAssertEqual(drafts, [4, 5, 6])
    }

    /// Rewind clamps history without touching the tree.
    func testEngineRewind() {
        let engine = SuffixSpeculativeEngine(tree: SuffixTree())
        engine.setPrefillHistory([1, 2, 3, 4, 5])
        engine.rewind(toPosition: 3)
        XCTAssertEqual(engine.history, [1, 2, 3])
        // Rewinding past the end is clamped.
        engine.rewind(toPosition: 99)
        XCTAssertEqual(engine.history, [1, 2, 3])
    }

    /// `drawBurst` with empty context returns empty without incrementing
    /// metrics beyond the burst counter.
    func testEngineEmptyContext() {
        let engine = SuffixSpeculativeEngine(tree: SuffixTree())
        let drafts = engine.drawBurst(context: [], K: 4)
        XCTAssertTrue(drafts.isEmpty)
        XCTAssertEqual(engine.totalProposed, 0)
    }
}
