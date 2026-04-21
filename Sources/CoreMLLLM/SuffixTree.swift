//
//  SuffixTree.swift
//  CoreMLLLM
//
//  Cross-session suffix trie for SuffixDecoding (arXiv 2411.04975,
//  Snowflake ArcticInference). Given a rolling token history from prior
//  generations, the tree remembers which continuations followed each
//  observed suffix. At draft time we locate the longest suffix of the
//  current context that exists in the tree and walk the most-frequent
//  child path for up to K tokens — the zero-model-cost drafter.
//
//  The tree is:
//    * counted   — each node stores how many times the (parent-suffix → token)
//                  transition was observed, so `argmax_frequency` at each node
//                  is a simple `max` over children counts;
//    * bounded   — `purge(maxNodes:)` evicts low-count leaves when the node
//                  cap is exceeded, keeping long-term memory footprint flat;
//    * thread-safe — all mutations + reads go through a serial
//                  `DispatchQueue`. Drafts happen on the drafter queue while
//                  commits happen on the main decode queue, and both can
//                  race during fast bursts;
//    * persistable — `Codable` snapshot written to `Documents/suffix_tree.json`
//                  so the knowledge base survives app restart. The paper's
//                  cross-session claim hinges on this persistence.
//
//  Design notes:
//    * Children are stored in a `[Int32: Node]` map keyed by token id.
//      Token vocabularies are large (>100k for Gemma), but any one
//      subtree only sees the small set of tokens that actually followed
//      the suffix in practice, so dict scaling is fine.
//    * `insert(sequence:)` adds every suffix of the sequence up to
//      `maxInsertDepth` — this is the standard suffix-trie trick so a
//      single committed generation trains every length-≤d match window.
//    * Accesses record a monotonic `lastTouched` timestamp used by
//      `purge` as an LRU tiebreaker when counts are equal.
//

import Foundation

/// Bounded-memory suffix trie keyed by token ids, designed for
/// SuffixDecoding-style drafting. See file header for design notes.
public final class SuffixTree {

    // MARK: - Node

    /// A node in the suffix trie. Public for `Codable` conformance but
    /// callers should go through `insert` / `draft` / `purge` rather than
    /// mutating nodes directly (the lock lives on the outer tree).
    public final class Node: Codable {
        /// Number of times the (parent-suffix → token) transition was seen.
        public var count: UInt32
        /// Monotonic access counter, used by `purge` as an LRU tiebreaker.
        public var lastTouched: UInt64
        /// Children keyed by token id.
        public var children: [Int32: Node]

        public init(count: UInt32 = 0, lastTouched: UInt64 = 0,
                    children: [Int32: Node] = [:]) {
            self.count = count
            self.lastTouched = lastTouched
            self.children = children
        }
    }

    // MARK: - Config

    /// Longest suffix (in tokens) that will be used when matching against
    /// the tree at draft time. Longer suffixes are more specific but less
    /// likely to appear; 32 is a reasonable default per the paper.
    public let maxSuffixLen: Int

    /// Longest suffix (in tokens) that will be inserted per committed
    /// trajectory. Insert cost is O(min(depth, len)·len) per suffix so
    /// keeping this bounded keeps commit-time cost linear.
    public let maxInsertDepth: Int

    /// Soft node cap. `purge` brings the tree down to roughly this many
    /// nodes by evicting least-useful branches (lowest count, then oldest
    /// `lastTouched`). 50k nodes is about 3-5 MB of resident data.
    public let maxNodes: Int

    // MARK: - State

    private let root: Node
    private var nodeCount: Int
    private var tick: UInt64
    private let queue: DispatchQueue

    // MARK: - Init

    public init(maxSuffixLen: Int = 32,
                maxInsertDepth: Int = 16,
                maxNodes: Int = 50_000,
                queueLabel: String = "com.coremlllm.suffixtree") {
        self.maxSuffixLen = max(maxSuffixLen, 1)
        self.maxInsertDepth = max(maxInsertDepth, 1)
        // Floor just high enough to hold the root + a handful of
        // children; callers that want a tiny cap (tests, memory-probes)
        // should still be honoured.
        self.maxNodes = max(maxNodes, 16)
        self.root = Node()
        self.nodeCount = 1
        self.tick = 0
        self.queue = DispatchQueue(label: queueLabel)
    }

    // MARK: - Mutation

    /// Insert a trajectory. Every suffix of `sequence` of length up to
    /// `maxInsertDepth` is added, each incrementing counts on its path.
    /// Safe to call from any thread — serialised on the internal queue.
    public func insert(sequence: [Int32]) {
        guard !sequence.isEmpty else { return }
        queue.sync { self._insertLocked(sequence) }
    }

    private func _insertLocked(_ sequence: [Int32]) {
        let n = sequence.count
        // For each starting index i, walk forward up to maxInsertDepth
        // tokens, creating/updating nodes.
        sequence.withUnsafeBufferPointer { buf in
            let p = buf.baseAddress!
            for i in 0..<n {
                var node = root
                let end = min(i + maxInsertDepth, n)
                for j in i..<end {
                    let tok = p[j]
                    let next: Node
                    if let existing = node.children[tok] {
                        next = existing
                    } else {
                        next = Node()
                        node.children[tok] = next
                        nodeCount += 1
                    }
                    tick &+= 1
                    next.count &+= 1
                    next.lastTouched = tick
                    node = next
                }
            }
        }
        if nodeCount > maxNodes {
            _purgeLocked(targetNodes: maxNodes)
        }
    }

    // MARK: - Draft

    /// Return up to `K` draft tokens continuing the tree path matched by
    /// the longest suffix of `context` that exists in the tree. Returns
    /// `[]` when no suffix matches or the match has no children.
    /// Safe to call from any thread.
    public func draft(suffix context: [Int32], K: Int) -> [Int32] {
        guard K > 0, !context.isEmpty else { return [] }
        return queue.sync { self._draftLocked(context: context, K: K) }
    }

    private func _draftLocked(context: [Int32], K: Int) -> [Int32] {
        // Try the longest suffix first; fall back to shorter suffixes
        // until one matches.
        let maxLen = min(context.count, maxSuffixLen)
        for len in stride(from: maxLen, through: 1, by: -1) {
            let start = context.count - len
            if let node = _walk(from: root, tokens: context, start: start, len: len) {
                let drafts = _bestPath(from: node, K: K)
                if !drafts.isEmpty { return drafts }
            }
        }
        return []
    }

    /// Walk the tree from `from` along `tokens[start..<start+len]`, returning
    /// the final node or nil if any transition is missing. Touches each
    /// node so frequent lookups keep the branch hot for LRU purging.
    private func _walk(from start: Node, tokens: [Int32],
                       start startIdx: Int, len: Int) -> Node? {
        var node = start
        for i in 0..<len {
            guard let child = node.children[tokens[startIdx + i]] else { return nil }
            tick &+= 1
            child.lastTouched = tick
            node = child
        }
        return node
    }

    /// Walk most-frequent children for up to K steps, breaking ties by
    /// `lastTouched` (most recently useful wins). Stops early if the
    /// current node has no children.
    private func _bestPath(from node: Node, K: Int) -> [Int32] {
        var result: [Int32] = []
        result.reserveCapacity(K)
        var cur = node
        for _ in 0..<K {
            guard let (tok, child) = _bestChild(of: cur) else { break }
            result.append(tok)
            cur = child
        }
        return result
    }

    private func _bestChild(of node: Node) -> (Int32, Node)? {
        var best: (Int32, Node)? = nil
        var bestCount: UInt32 = 0
        var bestTouched: UInt64 = 0
        for (tok, child) in node.children {
            if child.count > bestCount
                || (child.count == bestCount && child.lastTouched > bestTouched) {
                best = (tok, child)
                bestCount = child.count
                bestTouched = child.lastTouched
            }
        }
        return best
    }

    // MARK: - Purge

    /// Evict low-count / old nodes until node count ≤ `maxNodes`.
    /// Safe to call manually; also invoked automatically by `insert`
    /// when the cap is exceeded.
    public func purge(maxNodes: Int? = nil) {
        let target = maxNodes ?? self.maxNodes
        queue.sync { self._purgeLocked(targetNodes: target) }
    }

    private func _purgeLocked(targetNodes: Int) {
        guard nodeCount > targetNodes else { return }
        // Iteratively drop the worst leaf until we are under the cap.
        // A single leaf-sweep is cheaper than the full branch-cost
        // computation every iteration and keeps the invariant that
        // pruning never strands a parent without its high-value child
        // — once a leaf is pruned, its parent may itself become a leaf
        // and be eligible next pass.
        struct Entry { let parent: Node; let key: Int32; let child: Node }
        // Guard against pathological inputs.
        var safetyIter = nodeCount
        while nodeCount > targetNodes && safetyIter > 0 {
            safetyIter -= 1
            var leaves: [Entry] = []
            var stack: [Node] = [root]
            while let n = stack.popLast() {
                for (k, c) in n.children {
                    if c.children.isEmpty {
                        leaves.append(Entry(parent: n, key: k, child: c))
                    } else {
                        stack.append(c)
                    }
                }
            }
            if leaves.isEmpty { break }
            // Sort worst-first: lowest count, then oldest lastTouched.
            leaves.sort { a, b in
                if a.child.count != b.child.count { return a.child.count < b.child.count }
                return a.child.lastTouched < b.child.lastTouched
            }
            let budget = nodeCount - targetNodes
            let cull = min(budget, leaves.count)
            for i in 0..<cull {
                let e = leaves[i]
                if e.parent.children[e.key] === e.child {
                    e.parent.children.removeValue(forKey: e.key)
                    nodeCount -= 1
                }
            }
        }
    }

    // MARK: - Introspection

    public var approximateNodeCount: Int {
        queue.sync { self.nodeCount }
    }

    public func clear() {
        queue.sync {
            root.children.removeAll()
            nodeCount = 1
            tick = 0
        }
    }

    // MARK: - Persistence

    /// Snapshot serialised to JSON. We keep the root + accounting only —
    /// the queue and config are re-supplied when rehydrating.
    private struct Snapshot: Codable {
        let root: Node
        let nodeCount: Int
        let tick: UInt64
        let maxSuffixLen: Int
        let maxInsertDepth: Int
        let maxNodes: Int
    }

    /// Encode the current tree state to JSON. Thread-safe.
    public func encodeSnapshot() throws -> Data {
        try queue.sync {
            let snap = Snapshot(root: root, nodeCount: nodeCount, tick: tick,
                                maxSuffixLen: maxSuffixLen,
                                maxInsertDepth: maxInsertDepth,
                                maxNodes: maxNodes)
            return try JSONEncoder().encode(snap)
        }
    }

    /// Load tree state from a JSON snapshot. Returns a fresh tree — we
    /// don't mutate `self` in place because `maxSuffixLen` / `maxNodes`
    /// are `let` by design.
    public static func decodeSnapshot(_ data: Data,
                                      queueLabel: String = "com.coremlllm.suffixtree")
    throws -> SuffixTree {
        let snap = try JSONDecoder().decode(Snapshot.self, from: data)
        let tree = SuffixTree(maxSuffixLen: snap.maxSuffixLen,
                              maxInsertDepth: snap.maxInsertDepth,
                              maxNodes: snap.maxNodes,
                              queueLabel: queueLabel)
        tree.queue.sync {
            // Replace root contents — tree.root itself must stay the
            // same object so any external reference is still valid.
            tree.root.children = snap.root.children
            tree.root.count = snap.root.count
            tree.root.lastTouched = snap.root.lastTouched
            tree.nodeCount = snap.nodeCount
            tree.tick = snap.tick
        }
        return tree
    }

    /// Persist the tree to the user's Documents directory (or the URL
    /// supplied). Returns the file URL written. Errors propagate.
    @discardableResult
    public func save(to url: URL? = nil) throws -> URL {
        let dest = try url ?? Self.defaultPersistURL()
        let data = try encodeSnapshot()
        try data.write(to: dest, options: .atomic)
        return dest
    }

    /// Load a previously-saved tree. Returns `nil` if the file does not
    /// exist yet (fresh install). Decode errors propagate.
    public static func load(from url: URL? = nil) throws -> SuffixTree? {
        let src = try url ?? Self.defaultPersistURL()
        guard FileManager.default.fileExists(atPath: src.path) else { return nil }
        let data = try Data(contentsOf: src)
        return try decodeSnapshot(data)
    }

    /// Default persistence path: `<Documents>/suffix_tree.json`.
    public static func defaultPersistURL() throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true)
        return docs.appendingPathComponent("suffix_tree.json")
    }
}
