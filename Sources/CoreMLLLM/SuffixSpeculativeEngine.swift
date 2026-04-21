//
//  SuffixSpeculativeEngine.swift
//  CoreMLLLM
//
//  SuffixDecoding drafter (arXiv 2411.04975, Snowflake ArcticInference).
//
//  Zero-model-cost drafter backed by `SuffixTree`. At draft time the
//  current tail of the running context is matched against the tree and
//  the most-frequent continuation is proposed as a K-token burst. On
//  commit, the newly committed tokens are folded back into the tree so
//  the knowledge base grows monotonically across sessions.
//
//  Interface mirrors the Phase B drafters (`PromptLookupDraft`,
//  `CrossVocabDraft`) in spirit — pure CPU, no MLModel assets, suitable
//  for inclusion in `DrafterUnion` as an additional zero-cost source.
//
//  Distinct from PLD: PLD searches the current prompt/response for a
//  matching n-gram. SuffixDecoding searches a persistent cross-session
//  trie, so repeated patterns across previous generations (e.g. code
//  boilerplate, structured-output templates) accelerate drafting on the
//  first occurrence in the new generation.
//

import Foundation

/// SuffixDecoding drafter. See file header.
public final class SuffixSpeculativeEngine {

    // MARK: - Config

    /// Underlying suffix trie. Shared instance is safe to reuse across
    /// generations (that is the whole point — cross-session persistence).
    public let tree: SuffixTree

    /// Optional URL to persist the tree to. When set, `applyCommit`
    /// writes the snapshot every `saveEvery` commits; `.save()` is also
    /// safe to call explicitly (e.g. from app background hooks).
    public var persistURL: URL?

    /// Commit count between automatic persist writes. Defaults to 64 —
    /// small enough that a crash loses at most a few dozen tokens of
    /// learnt state, large enough that steady-state IO is negligible.
    public var saveEvery: Int = 64

    // MARK: - Metrics

    private(set) public var totalBursts: Int = 0
    private(set) public var totalProposed: Int = 0
    private(set) public var totalAccepted: Int = 0
    private(set) public var totalCommits: Int = 0

    /// Cumulative accept ratio = accepted / proposed. 0 until the first
    /// burst reports results.
    public var acceptanceRate: Double {
        totalProposed == 0 ? 0 : Double(totalAccepted) / Double(totalProposed)
    }

    // MARK: - State

    /// Monotonic committed history (prompt + emitted tokens). We keep our
    /// own copy rather than depending on the outer engine because the
    /// drafter wants to seed the tree with the final committed prefix
    /// regardless of which engine drove the commit. `applyCommit` appends,
    /// `rewind` truncates.
    private(set) public var history: [Int32] = []

    /// Last burst's proposals — captured for `drawBurst` callers that
    /// want to compare against target argmax themselves (DrafterUnion
    /// already builds its own compare loop).
    private(set) public var lastProposals: [Int32] = []

    private var commitsSinceSave: Int = 0

    // MARK: - Init

    /// - Parameters:
    ///   - tree: pre-built `SuffixTree`. Pass a shared instance to keep
    ///     state warm across generations.
    ///   - persistURL: where to save snapshots. `nil` disables autosave.
    public init(tree: SuffixTree, persistURL: URL? = nil) {
        self.tree = tree
        self.persistURL = persistURL
    }

    /// Convenience: build a fresh engine backed by a default tree persisted
    /// to `<Documents>/suffix_tree.json`. Loads existing state if present.
    public static func loadDefault() throws -> SuffixSpeculativeEngine {
        let url = try SuffixTree.defaultPersistURL()
        let tree = (try? SuffixTree.load(from: url)) ?? SuffixTree()
        return SuffixSpeculativeEngine(tree: tree, persistURL: url)
    }

    // MARK: - Reset

    /// Wipe per-generation state. The tree itself is preserved —
    /// cross-session knowledge is the whole point.
    public func reset() {
        history.removeAll()
        lastProposals.removeAll()
        totalBursts = 0
        totalProposed = 0
        totalAccepted = 0
        totalCommits = 0
        commitsSinceSave = 0
    }

    /// Seed the engine with prompt tokens before the first draw. Does
    /// not insert into the tree — only commits learn.
    public func setPrefillHistory(_ tokens: [Int32]) {
        history = tokens
        lastProposals.removeAll()
    }

    // MARK: - Draft

    /// Propose up to `K` draft tokens. Returns `[]` when the current
    /// context has no matching suffix in the tree.
    ///
    /// - Parameters:
    ///   - context: the committed token history to draft against. Callers
    ///     that track `tTokNext` should append it before calling (matches
    ///     PLD's `lookupHist` convention in `DrafterUnion`).
    ///   - K: max draft length.
    public func drawBurst(context: [Int32], K: Int) -> [Int32] {
        guard K > 0, !context.isEmpty else {
            lastProposals = []
            return []
        }
        let drafts = tree.draft(suffix: context, K: K)
        lastProposals = drafts
        totalBursts += 1
        totalProposed += drafts.count
        return drafts
    }

    // MARK: - Commit

    /// Feed newly committed tokens back into the tree so future drafts
    /// benefit. `tokens` must be the authoritative committed prefix
    /// extension (the caller's source of truth, not the drafter's
    /// proposals). Idempotent w.r.t. repeat calls when the same tokens
    /// are passed twice; callers should pass only the delta.
    public func applyCommit(tokens: [Int32]) {
        guard !tokens.isEmpty else { return }
        // Estimate accepted = prefix of lastProposals that matches
        // tokens, purely for metrics. The authoritative accept count
        // lives in the orchestrator (DrafterUnion / SpeculativeLoop).
        let accepted = Self.prefixMatch(lastProposals, tokens)
        totalAccepted += accepted
        lastProposals = []

        history.append(contentsOf: tokens)
        // Insert using the tail of the committed history so transitions
        // leading into the new tokens are captured too.
        let tailLen = min(history.count, tree.maxInsertDepth * 2)
        let tailStart = history.count - tailLen
        let slice = Array(history[tailStart..<history.count])
        tree.insert(sequence: slice)

        totalCommits += 1
        commitsSinceSave += 1
        if let url = persistURL, commitsSinceSave >= saveEvery {
            commitsSinceSave = 0
            try? tree.save(to: url)
        }
    }

    // MARK: - Rewind

    /// Truncate the drafter's committed-history view to `position` tokens.
    /// Used on verifier mispredicts where the orchestrator rewinds shared
    /// state. Tree contents are unaffected — we only drop in-flight
    /// proposals and shorten the local history window.
    public func rewind(toPosition position: Int) {
        let clamped = max(0, min(position, history.count))
        if clamped < history.count {
            history.removeLast(history.count - clamped)
        }
        lastProposals.removeAll()
    }

    // MARK: - Persistence

    /// Force a snapshot write. Safe to call from app-background hooks
    /// even if `persistURL` is nil (becomes a no-op).
    public func save() throws {
        guard let url = persistURL else { return }
        try tree.save(to: url)
        commitsSinceSave = 0
    }

    // MARK: - Utility

    private static func prefixMatch(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var m = 0
        while m < n && a[m] == b[m] { m += 1 }
        return m
    }
}
