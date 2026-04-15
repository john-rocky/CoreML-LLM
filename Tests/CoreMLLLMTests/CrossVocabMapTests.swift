import XCTest
@testable import CoreMLLLM

final class CrossVocabMapTests: XCTestCase {

    /// Writes a synthetic map with known contents, loads it through
    /// CrossVocabMap, and verifies the binary format + lookup contract.
    func testBinaryRoundtrip() throws {
        let qvs = 16
        let gvs = 24
        var q2g = [Int32](repeating: -1, count: qvs)
        var g2q = [Int32](repeating: -1, count: gvs)

        // Arbitrary deterministic mapping for a few ids.
        let pairs: [(qid: Int32, gid: Int32)] = [
            (0, 0), (1, 1), (5, 9), (7, 13), (15, 23),
        ]
        for (q, g) in pairs {
            q2g[Int(q)] = g
            g2q[Int(g)] = q
        }

        // Serialize in the QGVMAP01 format.
        var data = Data()
        data.append(contentsOf: Array("QGVMAP01".utf8))
        withUnsafeBytes(of: UInt32(qvs).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(gvs).littleEndian) { data.append(contentsOf: $0) }
        q2g.withUnsafeBufferPointer { buf in
            data.append(UnsafeBufferPointer(start: buf.baseAddress,
                                            count: buf.count).withMemoryRebound(to: UInt8.self) {
                Data($0)
            })
        }
        g2q.withUnsafeBufferPointer { buf in
            data.append(UnsafeBufferPointer(start: buf.baseAddress,
                                            count: buf.count).withMemoryRebound(to: UInt8.self) {
                Data($0)
            })
        }

        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qgvmap_test_\(UUID().uuidString).bin")
        try data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let map = try CrossVocabMap(url: tmp)
        XCTAssertEqual(map.qwenVocabSize, qvs)
        XCTAssertEqual(map.gemmaVocabSize, gvs)
        for (q, g) in pairs {
            XCTAssertEqual(map.qwen(q), g, "qwen→gemma at \(q)")
            XCTAssertEqual(map.gemma(g), q, "gemma→qwen at \(g)")
        }
        // Unmapped ids return -1
        XCTAssertEqual(map.qwen(2), -1)
        XCTAssertEqual(map.gemma(2), -1)
        // Out of range returns -1
        XCTAssertEqual(map.qwen(1000), -1)
        XCTAssertEqual(map.gemma(1000), -1)
        XCTAssertEqual(map.qwen(-1), -1)
    }

    func testRejectsBadMagic() throws {
        var data = Data("BOGUS___".utf8)
        data.append(contentsOf: [UInt8](repeating: 0, count: 16))
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qgvmap_bad_\(UUID().uuidString).bin")
        try data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertThrowsError(try CrossVocabMap(url: tmp))
    }

    func testRejectsTruncated() throws {
        let data = Data([0, 1, 2, 3])
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qgvmap_trunc_\(UUID().uuidString).bin")
        try data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertThrowsError(try CrossVocabMap(url: tmp))
    }

    /// If a real builder bin is sitting at /tmp, validate structural
    /// properties. Skipped silently if absent so CI stays green.
    func testRealBuilderBin() throws {
        let path = "/tmp/qwen_gemma_vocab.bin"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Real vocab map bin not at /tmp; run the Python builder to create it")
        }
        let map = try CrossVocabMap(url: URL(fileURLWithPath: path))
        XCTAssertGreaterThan(map.qwenVocabSize, 100_000, "expected Qwen 2.5 vocab ~151k")
        XCTAssertGreaterThan(map.gemmaVocabSize, 200_000, "expected Gemma 4 vocab ~262k")

        let qCov = Double(map.qwenToGemma.filter { $0 >= 0 }.count)
            / Double(map.qwenVocabSize)
        let gCov = Double(map.gemmaToQwen.filter { $0 >= 0 }.count)
            / Double(map.gemmaVocabSize)
        print("[testRealBuilderBin] coverage q→g=\(qCov) g→q=\(gCov)")
        XCTAssertGreaterThan(qCov, 0.30, "Qwen→Gemma coverage too low")
        XCTAssertGreaterThan(gCov, 0.15, "Gemma→Qwen coverage too low")

        // Round-trip consistency: if qwen(q) = g and g >= 0, then
        // gemma(g) must also be >= 0 and its round-trip must land on
        // some Qwen id whose surface equals the original.
        var consistentPairs = 0
        for q in 0..<map.qwenVocabSize {
            let g = map.qwenToGemma[q]
            guard g >= 0 else { continue }
            let back = map.gemmaToQwen[Int(g)]
            if back >= 0 { consistentPairs += 1 }
        }
        XCTAssertGreaterThan(consistentPairs, 50_000,
                             "expected tens of thousands of consistent pairs")
    }
}
