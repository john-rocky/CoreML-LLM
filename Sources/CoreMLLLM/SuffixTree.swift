//
//  SuffixTree.swift
//  CoreMLLLM
//
//  Token-level suffix tree for SuffixDecoding draft candidate generation.
//
//  Stores n-gram frequencies from prior model outputs as a trie keyed by
//  token IDs. Lookup is O(k + K) where k = context length, K = draft length.
//  Memory: ~100 bytes/node typical.
//
//  Not thread-safe — caller must serialize access.
//

import Foundation

// MARK: - Errors

public enum SuffixDecodingError: LocalizedError {
    case corruptedStore(String)

    public var errorDescription: String? {
        switch self {
        case .corruptedStore(let detail):
            return "Corrupted suffix tree store: \(detail)"
        }
    }
}

// MARK: - SuffixTree

public final class SuffixTree {

    final class Node {
        var children: [Int32: Node] = [:]
        var count: Int32 = 0
    }

    let root = Node()
    private let lock = NSLock()

    /// Maximum suffix depth to insert (bounds memory growth per sequence).
    public let maxDepth: Int

    /// Total number of sequences ingested via insert(sequence:).
    public private(set) var totalSequences: Int = 0

    /// Total number of nodes in the tree (including root).
    public private(set) var nodeCount: Int = 1

    public init(maxDepth: Int = 16) {
        self.maxDepth = maxDepth
    }

    // MARK: - Insert

    /// Ingest a completed token sequence.
    ///
    /// Inserts all suffixes of length 1…maxDepth. For a sequence of length N
    /// this is O(N × maxDepth) node visits.
    public func insert(sequence: [Int32]) {
        guard !sequence.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for start in 0..<sequence.count {
            let end = min(start + maxDepth, sequence.count)
            var node = root
            for i in start..<end {
                let token = sequence[i]
                if let child = node.children[token] {
                    child.count += 1
                    node = child
                } else {
                    let child = Node()
                    child.count = 1
                    node.children[token] = child
                    nodeCount += 1
                    node = child
                }
            }
        }
        totalSequences += 1
    }

    // MARK: - Lookup

    /// Look up the most frequent continuation given a token context.
    ///
    /// Tries the full context first, then progressively shorter suffixes.
    /// Returns up to `maxLength` continuation tokens via greedy most-frequent
    /// child selection. Ties are broken by lower token ID for determinism.
    ///
    /// - Parameters:
    ///   - context: Recent token history (last k tokens).
    ///   - maxLength: Maximum continuation tokens to return (K).
    ///   - minCount: Minimum node frequency to consider (default 1).
    /// - Returns: Draft token candidates, possibly empty.
    public func lookup(context: [Int32], maxLength: Int,
                       minCount: Int32 = 1) -> [Int32] {
        guard !context.isEmpty else { return [] }
        lock.lock()
        defer { lock.unlock() }

        let maxCtx = min(context.count, maxDepth - 1)

        for contextLen in stride(from: maxCtx, through: 1, by: -1) {
            let start = context.count - contextLen

            // Walk tree matching context suffix
            var node = root
            var matched = true
            for i in start..<context.count {
                guard let child = node.children[context[i]] else {
                    matched = false
                    break
                }
                node = child
            }
            guard matched else { continue }

            // Greedily follow highest-count children
            let candidates = greedyContinuation(from: node,
                                                maxLength: maxLength,
                                                minCount: minCount)
            if !candidates.isEmpty { return candidates }
        }

        return []
    }

    private func greedyContinuation(from node: Node, maxLength: Int,
                                    minCount: Int32) -> [Int32] {
        var result: [Int32] = []
        result.reserveCapacity(maxLength)
        var current = node

        for _ in 0..<maxLength {
            // Pick child with highest count; break ties by lower token ID
            guard let (bestToken, bestChild) = current.children.max(by: {
                a, b in
                if a.value.count != b.value.count {
                    return a.value.count < b.value.count
                }
                return a.key > b.key
            }), bestChild.count >= minCount else {
                break
            }
            result.append(bestToken)
            current = bestChild
        }

        return result
    }

    // MARK: - Maintenance

    /// Remove all nodes with count below `minCount`. Returns count of removed nodes.
    @discardableResult
    public func prune(minCount: Int32) -> Int {
        let before = nodeCount
        pruneRecursive(node: root, minCount: minCount)
        nodeCount = 1 + countSubtreeNodes(root)
        return before - nodeCount
    }

    private func pruneRecursive(node: Node, minCount: Int32) {
        for child in node.children.values {
            pruneRecursive(node: child, minCount: minCount)
        }
        let toRemove = node.children.keys.filter { node.children[$0]!.count < minCount }
        for token in toRemove {
            node.children.removeValue(forKey: token)
        }
    }

    private func countSubtreeNodes(_ node: Node) -> Int {
        var count = 0
        for child in node.children.values {
            count += 1 + countSubtreeNodes(child)
        }
        return count
    }

    /// Estimated heap memory in bytes.
    public var estimatedMemoryBytes: Int {
        // ~100 bytes per node (object header + Dictionary + count)
        nodeCount * 100
    }

    // MARK: - Binary Persistence

    //  File layout:
    //    magic   : UInt32  "SFTX"
    //    version : UInt32  1
    //    maxDepth: Int32
    //    totalSeq: Int32
    //    nodeCount: Int32
    //    root node (recursive DFS — see serializeNode)

    private static let magic: UInt32 = 0x5346_5458   // "SFTX"
    private static let version: UInt32 = 1

    /// Serialize the tree to a binary file.
    public func save(to url: URL) throws {
        var data = Data()
        data.reserveCapacity(nodeCount * 12)

        withUnsafeBytes(of: Self.magic)   { data.append(contentsOf: $0) }
        withUnsafeBytes(of: Self.version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: Int32(maxDepth))        { data.append(contentsOf: $0) }
        withUnsafeBytes(of: Int32(totalSequences))  { data.append(contentsOf: $0) }
        withUnsafeBytes(of: Int32(nodeCount))       { data.append(contentsOf: $0) }

        serializeNode(root, into: &data)
        try data.write(to: url, options: .atomic)
    }

    private func serializeNode(_ node: Node, into data: inout Data) {
        withUnsafeBytes(of: node.count)                   { data.append(contentsOf: $0) }
        withUnsafeBytes(of: Int32(node.children.count))   { data.append(contentsOf: $0) }

        // Sorted by token ID for deterministic output
        for (token, child) in node.children.sorted(by: { $0.key < $1.key }) {
            withUnsafeBytes(of: token) { data.append(contentsOf: $0) }
            serializeNode(child, into: &data)
        }
    }

    /// Deserialize a tree from a binary file.
    public static func load(from url: URL) throws -> SuffixTree {
        let data = try Data(contentsOf: url)
        var offset = 0

        func read<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
            let size = MemoryLayout<T>.size
            guard offset + size <= data.count else {
                throw SuffixDecodingError.corruptedStore("unexpected EOF at offset \(offset)")
            }
            let value: T = data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: T.self)
            }
            offset += size
            return value
        }

        let fileMagic = try read(UInt32.self)
        guard fileMagic == magic else {
            throw SuffixDecodingError.corruptedStore("bad magic 0x\(String(fileMagic, radix: 16))")
        }
        let fileVersion = try read(UInt32.self)
        guard fileVersion == version else {
            throw SuffixDecodingError.corruptedStore("unsupported version \(fileVersion)")
        }
        let depth = try read(Int32.self)
        let seqs  = try read(Int32.self)
        let nodes = try read(Int32.self)

        let tree = SuffixTree(maxDepth: Int(depth))
        tree.totalSequences = Int(seqs)
        tree.nodeCount = Int(nodes)

        func deserializeNode() throws -> Node {
            let node = Node()
            node.count = try read(Int32.self)
            let numChildren = try read(Int32.self)
            node.children.reserveCapacity(Int(numChildren))
            for _ in 0..<numChildren {
                let token = try read(Int32.self)
                node.children[token] = try deserializeNode()
            }
            return node
        }

        let loaded = try deserializeNode()
        tree.root.count = loaded.count
        tree.root.children = loaded.children

        return tree
    }
}
