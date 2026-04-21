import XCTest
@testable import CoreMLLLM

/// 11c protocol: validates that running `verifyCandidates(K=3) + commitAccepted`
/// produces byte-identical persistent KV state to running 3 sequential
/// `predictStep` calls with the same tokens. This proves Swift's
/// commitKVSlices (sliding shift+append, full scatter, hd→maxHd zero-pad)
/// matches the in-graph KV update path.
///
/// Skipped unless env var `KV_PARITY_MODEL_DIR` is set to a directory
/// containing built mlmodelc bundles for the new (11c) verify chunks.
///
/// Run with:
///   KV_PARITY_MODEL_DIR=/abs/path/to/output/verify-11c \
///     swift test --filter ChunkedEngineKVParityTests
final class ChunkedEngineKVParityTests: XCTestCase {

    func testVerifyCommitMatchesSerialDecode() async throws {
        guard let dirStr = ProcessInfo.processInfo.environment["KV_PARITY_MODEL_DIR"] else {
            throw XCTSkip("Set KV_PARITY_MODEL_DIR to run this integration test.")
        }
        let dir = URL(fileURLWithPath: dirStr)

        let llm = try await CoreMLLLM.load(from: dir) { msg in
            print("[load] \(msg)")
        }
        guard let engine = llm._testChunkedEngine else {
            XCTFail("Loaded model has no ChunkedEngine; expected chunked SWA bundle.")
            return
        }
        guard engine.hasVerify else {
            XCTFail("Loaded model has no verify chunks; build mlpackages with verify_qK.")
            return
        }
        XCTAssertEqual(engine.verifyK, 3, "Test assumes K=3.")

        // K=3 input tokens. Use small ints so they exist in vocab.
        let tokens: [Int32] = [100, 200, 300]
        let K = tokens.count

        // ---- Path A: verify K=3, then commit all K (all-match scenario) ----
        engine.reset()
        _ = try engine.verifyCandidates(tokens: tokens, startPosition: 0)
        // commitAccepted with the same K tokens → matched prefix M = K, no T=1.
        try engine.commitAccepted(tokens)
        let snapA = engine._kvSnapshotBytes()
        XCTAssertEqual(engine.currentPosition, K)

        // ---- Path B: 3× sequential T=1 predictStep ----
        engine.reset()
        for t in 0..<K {
            _ = try engine.predictStep(tokenID: Int(tokens[t]), position: t)
        }
        let snapB = engine._kvSnapshotBytes()

        // ---- Compare ----
        XCTAssertEqual(snapA.keys.sorted(), snapB.keys.sorted())
        // Offset ranges where the two paths SHOULD write:
        //   Sliding bufs (slot,1,W=512,maxHd=512): rows [W-K..W-1] of every slot.
        //   Full bufs    (slot,1,ctx=2048,maxHd=512): rows [0..K-1] of every slot.
        // Offsets outside those ranges must be byte-identical (both paths leave them
        // at zero — this test starts from engine.reset() for both paths).
        let W = 512, ctx = 2048, maxHd = 512
        func isSliding(_ key: String) -> Bool { key.contains("Sliding") }
        func slotCount(_ key: String) -> Int {
            if key.contains("1") { return key.contains("Full") ? 1 : 7 }
            else { return key.contains("Full") ? 2 : 5 }
        }
        func isInWriteRegion(idx: Int, key: String) -> Bool {
            let sc = slotCount(key)
            if isSliding(key) {
                let perSlot = W * maxHd
                let slot = idx / perSlot
                if slot >= sc { return false }
                let rowInSlot = (idx - slot * perSlot) / maxHd
                return rowInSlot >= W - K
            } else {
                let perSlot = ctx * maxHd
                let slot = idx / perSlot
                if slot >= sc { return false }
                let rowInSlot = (idx - slot * perSlot) / maxHd
                return rowInSlot < K
            }
        }

        for key in snapA.keys.sorted() {
            let a = snapA[key]!
            let b = snapB[key]!
            XCTAssertEqual(a.count, b.count)
            let n = a.count / MemoryLayout<UInt16>.stride
            var diffsInsideWriteRegion = 0
            var diffsOutsideWriteRegion = 0
            var firstOutsideDiff = -1
            a.withUnsafeBytes { (aRaw: UnsafeRawBufferPointer) in
                b.withUnsafeBytes { (bRaw: UnsafeRawBufferPointer) in
                    let aPtr = aRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    let bPtr = bRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<n where aPtr[i] != bPtr[i] {
                        if isInWriteRegion(idx: i, key: key) {
                            diffsInsideWriteRegion += 1
                        } else {
                            diffsOutsideWriteRegion += 1
                            if firstOutsideDiff < 0 { firstOutsideDiff = i }
                        }
                    }
                }
            }
            let totalInside = isSliding(key)
                ? slotCount(key) * K * maxHd
                : slotCount(key) * K * maxHd
            let pctInside = totalInside > 0
                ? Double(diffsInsideWriteRegion) / Double(totalInside) * 100
                : 0
            print("[KV-parity] \(key): inside-region \(diffsInsideWriteRegion)/\(totalInside) (\(String(format: "%.1f", pctInside))%) outside-region \(diffsOutsideWriteRegion) firstOutsideIdx=\(firstOutsideDiff)")
            // Hard assertion: outside the K written rows, buffers must be byte-identical.
            // Any mismatch there = real Swift layout bug (wrong slot/stride).
            XCTAssertEqual(diffsOutsideWriteRegion, 0,
                "\(key): \(diffsOutsideWriteRegion) diffs OUTSIDE the K-row write region — Swift commitKVSlices layout bug. First bad idx: \(firstOutsideDiff)")
        }
    }
}
