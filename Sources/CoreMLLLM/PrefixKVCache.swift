//
//  PrefixKVCache.swift
//  CoreMLLLM
//
//  Persistent KV cache for repeated prompt prefixes (Approach E of
//  docs/UNEXPLORED_APPROACHES.md). For chat apps with a stable system
//  prompt, the prefill of that prefix is the dominant component of
//  cold-start latency. Caching the post-prefill KV state to disk cuts
//  cold-start TTFT from seconds (~13 s on 2K prompt) to milliseconds
//  on cache hit.
//
//  External literature: persistent Q4 KV cache reports 4-35× TTFT
//  speedup at 1-4K context, 136× at 32K (cache warm disk, fp16
//  prefix). INT4 quantization keeps each 2K cache ≈ 15 MB on disk,
//  so hundreds of prefix caches fit in a few GB of app sandbox storage.
//
//  Format:
//  Each cache entry lives in a directory named after a content hash
//  of (prefix_tokens, contextLength, model_id, library version):
//
//      Caches/PrefixKV/<hex-sha256>/
//        meta.json           { prefix_len, ctx, model_id, version, created_at }
//        k_sliding1.bin      IOSurface-compatible half-precision dumps
//        v_sliding1.bin
//        k_full1.bin
//        v_full1.bin
//        k_sliding2.bin
//        v_sliding2.bin
//        k_full2.bin
//        v_full2.bin
//        kv14.bin            (KV-shared layer anchor outputs for L15-34)
//
//  The runtime must be able to dump and restore these buffers through a
//  protocol `PrefixKVSnapshotable` that ChunkedEngine implements. That
//  protocol is declared here; the conforming implementation lives in
//  ChunkedEngine (bench-owned file) — see TODO in the adapter below.
//

import CoreML
import CryptoKit
import Foundation

public enum PrefixKVCacheError: Error, LocalizedError {
    case hashFailed
    case ioFailed(String)
    case metaInvalid(String)
    case notImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .hashFailed:           return "failed to compute prefix hash"
        case .ioFailed(let s):      return "prefix cache I/O: \(s)"
        case .metaInvalid(let s):   return "prefix cache meta: \(s)"
        case .notImplemented(let s):return "not implemented yet: \(s)"
        }
    }
}

/// Engine-side protocol: dump / restore KV state to/from a directory.
/// ChunkedEngine (bench-owned file) should conform to this.
public protocol PrefixKVSnapshotable: AnyObject {
    /// Current position / prefill length (in tokens).
    var currentPosition: Int { get }

    /// Write all KV buffers + state to `directory`. Caller creates the dir.
    func writeKVSnapshot(to directory: URL) throws

    /// Read KV buffers + state from `directory`, set the engine's position to
    /// the snapshot's position so subsequent decode steps continue from there.
    func readKVSnapshot(from directory: URL) throws
}

public final class PrefixKVCache {
    public let rootDirectory: URL
    public let modelId: String
    public let contextLength: Int
    public let libraryVersion: String

    /// Set to nil to cache indefinitely; set to a value to auto-evict old entries.
    public var maxEntries: Int? = 64

    public init(
        rootDirectory: URL,
        modelId: String,
        contextLength: Int,
        libraryVersion: String = "v0.5.1"
    ) throws {
        self.rootDirectory = rootDirectory
        self.modelId = modelId
        self.contextLength = contextLength
        self.libraryVersion = libraryVersion
        try FileManager.default.createDirectory(
            at: rootDirectory, withIntermediateDirectories: true)
    }

    // MARK: - Hashing

    /// Hash the prefix tokens + ctx + model id + version. Stable across runs.
    public func hashKey(prefixTokens: [Int32]) throws -> String {
        var hasher = SHA256()
        withUnsafeBytes(of: contextLength) { hasher.update(bufferPointer: $0) }
        if let idData = modelId.data(using: .utf8) { hasher.update(data: idData) }
        if let verData = libraryVersion.data(using: .utf8) { hasher.update(data: verData) }
        prefixTokens.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func directory(for key: String) -> URL {
        rootDirectory.appendingPathComponent(key, isDirectory: true)
    }

    // MARK: - Lookup / store / evict

    public func hasEntry(for key: String) -> Bool {
        let dir = directory(for: key)
        let meta = dir.appendingPathComponent("meta.json")
        return FileManager.default.fileExists(atPath: meta.path)
    }

    /// Try to restore. Returns true on cache hit, false on miss (no error).
    @discardableResult
    public func restore(into engine: PrefixKVSnapshotable, key: String) throws -> Bool {
        let dir = directory(for: key)
        guard hasEntry(for: key) else { return false }
        // Read meta, sanity check
        let metaURL = dir.appendingPathComponent("meta.json")
        let data = try Data(contentsOf: metaURL)
        guard let meta = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelIdMeta = meta["model_id"] as? String,
              let ctxMeta = meta["ctx"] as? Int else {
            throw PrefixKVCacheError.metaInvalid("missing fields")
        }
        guard modelIdMeta == modelId, ctxMeta == contextLength else {
            // Not a hit after all — stale cache
            try? FileManager.default.removeItem(at: dir)
            return false
        }
        try engine.readKVSnapshot(from: dir)
        touch(dir)
        return true
    }

    /// Store current engine KV state under `key`.
    public func store(from engine: PrefixKVSnapshotable, key: String,
                      prefixLen: Int) throws {
        let dir = directory(for: key)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try engine.writeKVSnapshot(to: dir)
        let meta: [String: Any] = [
            "prefix_len":       prefixLen,
            "ctx":              contextLength,
            "model_id":         modelId,
            "version":          libraryVersion,
            "created_at":       Date().timeIntervalSince1970,
        ]
        let data = try JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted)
        try data.write(to: dir.appendingPathComponent("meta.json"))
        try evictIfNeeded()
    }

    private func touch(_ url: URL) {
        try? FileManager.default.setAttributes(
            [.modificationDate: Date()], ofItemAtPath: url.path)
    }

    private func evictIfNeeded() throws {
        guard let max = maxEntries else { return }
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(
            at: rootDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        if items.count <= max { return }
        let ranked = items.sorted { a, b in
            let ad = (try? a.resourceValues(forKeys: [.contentModificationDateKey])
                      .contentModificationDate) ?? .distantPast
            let bd = (try? b.resourceValues(forKeys: [.contentModificationDateKey])
                      .contentModificationDate) ?? .distantPast
            return ad < bd
        }
        for url in ranked.prefix(items.count - max) {
            try? fm.removeItem(at: url)
        }
    }

    /// Clear every cache entry.
    public func clearAll() throws {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: rootDirectory, includingPropertiesForKeys: nil) else { return }
        for url in items { try? fm.removeItem(at: url) }
    }
}

// MARK: - Default Caches/ location

public extension PrefixKVCache {
    /// Standard location: <App Caches>/PrefixKV/<modelId slug>/.
    static func defaultRoot(for modelId: String) throws -> URL {
        let fm = FileManager.default
        let caches = try fm.url(for: .cachesDirectory,
                                 in: .userDomainMask,
                                 appropriateFor: nil,
                                 create: true)
        let slug = modelId.replacingOccurrences(of: "/", with: "_")
        return caches.appendingPathComponent("PrefixKV")
            .appendingPathComponent(slug)
    }
}
