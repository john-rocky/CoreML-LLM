import CryptoKit
import Foundation

/// On-disk cache of KV state at the end of prefilled prompt prefixes.
///
/// When a chat continues across turns, the next prompt usually starts
/// with the same tokens we just prefilled (system prompt + prior turn
/// content). Caching the KV cache at the end of the previous prefill
/// lets us restore that state and only re-prefill (or per-token decode)
/// the new delta tokens — converting a 13 s full prefill into a fraction
/// of that.
///
/// Storage layout (`directory`):
///   index.json — array of `Entry` metadata (token sequence, file name,
///                buffer sizes, position, last access).
///   <hash>.kv  — binary blob: 76-byte header + concatenated buffer bytes.
///
/// Eviction: LRU by `lastAccess` once total bytes exceed `capacityBytes`.
///
/// Concurrency: not thread-safe. Caller must serialize.
final class PrefixCache {

    // MARK: - Public types

    public struct Entry: Codable {
        public let tokenIDs: [Int]
        public let filename: String
        public let position: Int
        public let bufferSizes: [Int]
        public var lastAccess: Double
        public var totalBytes: Int

        var fileSize: Int { totalBytes }
    }

    /// Result of a successful lookup.
    public struct Match {
        public let entry: Entry
        /// Number of tokens matched (always equal to `entry.tokenIDs.count`).
        public let matchLen: Int
        public let blobURL: URL
    }

    // MARK: - State

    private let directory: URL
    private let capacityBytes: Int
    private let indexURL: URL
    private var entries: [Entry] = []

    // MARK: - Init

    init(directory: URL, capacityBytes: Int = 256 * 1024 * 1024) throws {
        self.directory = directory
        self.capacityBytes = capacityBytes
        self.indexURL = directory.appendingPathComponent("index.json")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        loadIndex()
    }

    // MARK: - Lookup

    /// Find the longest cached prefix of `tokenIDs`. Returns nil if no entry
    /// is a prefix of the input. On hit, updates the entry's lastAccess.
    func longestPrefixMatch(tokenIDs: [Int]) -> Match? {
        var best: Entry? = nil
        for entry in entries where entry.tokenIDs.count <= tokenIDs.count {
            // Prefix check
            var match = true
            for i in 0..<entry.tokenIDs.count {
                if entry.tokenIDs[i] != tokenIDs[i] { match = false; break }
            }
            if match {
                if best == nil || entry.tokenIDs.count > best!.tokenIDs.count {
                    best = entry
                }
            }
        }
        guard let entry = best else { return nil }

        // Touch lastAccess
        if let idx = entries.firstIndex(where: { $0.filename == entry.filename }) {
            entries[idx].lastAccess = Date().timeIntervalSince1970
            saveIndex()
        }
        return Match(entry: entry, matchLen: entry.tokenIDs.count,
                     blobURL: directory.appendingPathComponent(entry.filename))
    }

    // MARK: - Store

    /// Store a new snapshot. Replaces any existing entry whose tokenIDs
    /// match exactly. Triggers LRU eviction if `capacityBytes` exceeded.
    func store(tokenIDs: [Int], buffers: [Data], position: Int) throws {
        let hash = sha256Hex(tokenIDs)
        let filename = "\(hash).kv"
        let blobURL = directory.appendingPathComponent(filename)

        // Build blob: 76-byte header + buffers.
        // Header layout (little-endian, padded to 76 bytes for stability):
        //   bytes 0..3   "PCV1"
        //   bytes 4..7   int32 position
        //   bytes 8..11  int32 num_buffers (= buffers.count)
        //   bytes 12..75 int64[8] buffer sizes (64 bytes)
        var header = Data(count: 76)
        header.withUnsafeMutableBytes { raw in
            let p = raw.baseAddress!
            memcpy(p, "PCV1", 4)
            (p + 4).assumingMemoryBound(to: Int32.self).pointee = Int32(position)
            (p + 8).assumingMemoryBound(to: Int32.self).pointee = Int32(buffers.count)
            let sizesPtr = (p + 12).assumingMemoryBound(to: Int64.self)
            for (i, b) in buffers.enumerated() where i < 8 {
                sizesPtr[i] = Int64(b.count)
            }
        }
        var data = header
        for b in buffers { data.append(b) }
        try data.write(to: blobURL, options: .atomic)

        // Replace existing entry with same tokenIDs (same hash → same filename).
        entries.removeAll { $0.filename == filename }
        let entry = Entry(
            tokenIDs: tokenIDs,
            filename: filename,
            position: position,
            bufferSizes: buffers.map { $0.count },
            lastAccess: Date().timeIntervalSince1970,
            totalBytes: data.count)
        entries.append(entry)
        evictIfNeeded()
        saveIndex()
        print("[PrefixCache] stored len=\(tokenIDs.count) pos=\(position) " +
              "size=\(data.count / 1024)KB total=\(totalBytes() / (1024*1024))MB")
    }

    // MARK: - Maintenance

    func clear() {
        for entry in entries {
            try? FileManager.default.removeItem(
                at: directory.appendingPathComponent(entry.filename))
        }
        entries = []
        try? FileManager.default.removeItem(at: indexURL)
    }

    func totalBytes() -> Int {
        entries.reduce(0) { $0 + $1.totalBytes }
    }

    // MARK: - Private

    private func evictIfNeeded() {
        while totalBytes() > capacityBytes && !entries.isEmpty {
            entries.sort { $0.lastAccess < $1.lastAccess }
            let removed = entries.removeFirst()
            try? FileManager.default.removeItem(
                at: directory.appendingPathComponent(removed.filename))
            print("[PrefixCache] evicted len=\(removed.tokenIDs.count) " +
                  "size=\(removed.totalBytes / 1024)KB")
        }
    }

    private func loadIndex() {
        guard let data = try? Data(contentsOf: indexURL) else { return }
        if let loaded = try? JSONDecoder().decode([Entry].self, from: data) {
            // Drop entries whose blob file is missing (manual deletion etc).
            entries = loaded.filter {
                FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent($0.filename).path)
            }
        }
    }

    private func saveIndex() {
        if let data = try? JSONEncoder().encode(entries) {
            try? data.write(to: indexURL, options: .atomic)
        }
    }

    private func sha256Hex(_ tokenIDs: [Int]) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokenIDs.count * 4)
        for t in tokenIDs {
            let v = Int32(truncatingIfNeeded: t)
            bytes.append(UInt8(truncatingIfNeeded: v & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: (v >> 8) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: (v >> 16) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: (v >> 24) & 0xFF))
        }
        let digest = SHA256.hash(data: Data(bytes))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Blob reader (used by ChunkedEngine.restoreSnapshot)

extension PrefixCache {
    /// Read a blob from disk and split into per-buffer Data slices.
    /// Verifies header magic, position, and buffer count/sizes match the entry.
    static func readBlob(at url: URL, expecting entry: Entry) throws -> [Data] {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 76 else {
            throw NSError(domain: "PrefixCache", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "blob too short"])
        }
        try data.withUnsafeBytes { raw in
            let p = raw.baseAddress!
            let magic = String(data: Data(bytes: p, count: 4), encoding: .ascii)
            guard magic == "PCV1" else {
                throw NSError(domain: "PrefixCache", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "bad magic"])
            }
            let position = Int((p + 4).assumingMemoryBound(to: Int32.self).pointee)
            let numBuffers = Int((p + 8).assumingMemoryBound(to: Int32.self).pointee)
            guard position == entry.position, numBuffers == entry.bufferSizes.count else {
                throw NSError(domain: "PrefixCache", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "header mismatch"])
            }
        }
        var offset = 76
        var slices: [Data] = []
        for size in entry.bufferSizes {
            guard offset + size <= data.count else {
                throw NSError(domain: "PrefixCache", code: 4,
                              userInfo: [NSLocalizedDescriptionKey: "blob truncated"])
            }
            slices.append(data.subdata(in: offset..<(offset + size)))
            offset += size
        }
        return slices
    }
}
