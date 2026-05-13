//
//  MtpSpeculativeEngine.swift
//  CoreMLLLM
//
//  Orchestrates the MTP speculative decoding loop:
//    draft K tokens → verify → accept/reject → commit → extract carry state
//
//  Design matches Google's LiteRT runtime (K=3 fixed, greedy argmax):
//    - Write-through KV: verify writes all K positions, rejected entries
//      masked by causal mask and overwritten on next cycle.
//    - Commit = position advance only (no decode re-runs).
//    - Carry state from lastVerifyHiddenStates bootstraps next MTP cycle.
//

import Accelerate
import CoreML
import Foundation

/// MTP speculative decoding engine — drafts K tokens with the MTP drafter,
/// verifies them against the target (ChunkedEngine), and commits accepted tokens.
public final class MtpSpeculativeEngine {

    let engine: ChunkedEngine
    let drafter: MtpDraftSource
    let K: Int

    // Carry state for MTP drafting (L34 hidden from verify, or zero on bootstrap)
    private var carryState: MLMultiArray?
    private var isBootstrapped = false

    // CSD — Calibrated Speculative Decoding (arxiv 2604.13634). Online
    // correction memory: per-conversation token-pair frequency table.
    // `T[(drafter_token, target_top1)]` counts how often the drafter has
    // proposed `drafter_token` at a position where the target's top-1 was
    // `target_top1` *before this rejection*. When T ≥ λ AND the semantic
    // gate `z_drafter − z_target_top1 ≥ log τ` passes, we rescue the draft
    // even though strict / MARS / FLy would have rejected it. Reset on
    // `reset()` (new conversation).
    fileprivate struct TokenPair: Hashable {
        let draft: Int32
        let target: Int32
    }
    private var csdCorrectionMemory: [TokenPair: Int] = [:]

    // L5 / Stage 1 — PLD (Prompt-Lookup-Decoding) cycle prefetch.
    // Mobile ANE constraint: drafter call is 5-6 ms per step on iPhone,
    // chained K-1 times = 11 ms per cycle. PLD is a μs-level CPU n-gram
    // lookup. When the recent emitted tokens form an n-gram seen earlier
    // in this conversation, propose the continuation tokens as the
    // drafter's output — verify still strict-checks them, so quality
    // stays lossless. On hit, drafter call is fully skipped (save 11 ms).
    // llama.cpp reports 30-40% hit on chat/translate; on free-form
    // English chat we hope for ~20-30%. Disable: MTP_PLD_PREFETCH_DISABLE.
    private var emittedHistory: [Int32] = []
    /// Externally-injected history (prompt tokens). Called by CoreMLLLM
    /// once per conversation so PLD sees prompt repeats, not just output.
    func setPromptTokens(_ tokens: [Int32]) {
        emittedHistory = tokens
    }
    var pldHitsThisGeneration: Int = 0
    var pldProbesThisGeneration: Int = 0

    // L5 — async drafter parallel-to-verify (cross-cycle speculation).
    //
    // Pattern: after cycle N's drafter loop completes, kick off a
    // BACKGROUND drafter chain (on CPU) speculating cycle N+1's proposals.
    // Speculative inputs: drafter's own last drafted token as the predicted
    // nextID, drafter's last projAct as the predicted carry. Background
    // chain runs in parallel with cycle N's ANE verify. After verify,
    // validate speculation: if matchCount==K-1 AND speculated nextID
    // matches actual carry (= targetArgmax[K-1]), then the pre-drafted
    // proposals are correct and cycle N+1 skips its drafter loop entirely.
    //
    // Best case (full accept rate 100%): cycle = max(drafter, verify) = 36ms
    // iPhone (vs current 47ms). Realistic free-form (25% full-accept):
    // avg cycle ≈ 44ms → 1.27× iPhone.
    fileprivate struct PreDraftedNextCycle {
        let expectedNextID: Int32      // = drafter's last drafted token
        let proposals: [Int32]         // pre-drafted K-1 tokens for cycle N+1
        let finalProjAct: MLMultiArray // drafter's last hidden (chain seed)
    }
    private var pendingPreDraft: PreDraftedNextCycle?
    private let asyncDrafterQueue = DispatchQueue(
        label: "mtp.async.drafter",
        qos: .userInitiated,
        attributes: [])
    var l5HitsThisGeneration: Int = 0
    var l5DispatchesThisGeneration: Int = 0

    // Adaptive K_USE per HF heuristic schedule
    // (transformers candidate_generator.py:240-251):
    //   on full match (matchCount == compareLen):  num_assistant_tokens += 2
    //   on any miss:                                num_assistant_tokens -= 1 (min 1)
    // Bounded by [1, K]. Initial value follows the static MTP_K_USE setting.
    private var kUseAdaptive: Int = 2

    // Round E — per-prompt K_USE adapter (MTP_PER_PROMPT_KUSE=1).
    // After the first spec cycle on a new prompt, sense matchCount:
    //   * full match on K-1 slots → drafter is on a streak for this prompt
    //     (typical for code / structured / repetitive output) → switch to
    //     K_USE=1 for the rest of the prompt. Saves one drafter chain step
    //     (~6 ms iPhone) per cycle while still emitting 2 tokens per cycle
    //     on continued full accept.
    //   * partial match (or zero) → keep K_USE=K-1 (default). Two-slot
    //     drafting gives more chances to maintain rolling acceptance EMA
    //     above bail threshold on narrative / uncertain prompts.
    // Reset on `reset()` (new conversation).
    // Skipped entirely if `MTP_K_USE` is set explicitly (user override
    // wins) or if `MTP_K_ADAPTIVE=1` is on (HF schedule active).
    private var perPromptKUseOverride: Int?
    private var perPromptKUseSensed: Bool = false

    // L12 — subset LM head (iPhone 1.5× lever). Computed once at init from
    // `MTP_SUBSET_LM_HEAD=1` and `engine.hasSubsetLMHead`. When true, every
    // greedy + FLy verify cycle (i.e. when not in tree mode or rejection-
    // sampling mode) routes through `verifyCandidatesSubset`, skipping the
    // ~7-10 ms iPhone chunk4 LM head matmul.
    let subsetEnabled: Bool
    /// Cap on subset candidate count per cycle. M=1024 was the design point;
    /// override via `MTP_SUBSET_M` for sweeps.
    private let subsetCapM: Int
    /// Below this max-subset-logit, the cycle re-runs `verifyCandidates`
    /// (full chunk4) — preserves losslessness when target's true argmax is
    /// outside the candidate set. Tuned empirically; default 12.0 matches
    /// Gemma 4 E2B observed top-1 logits on free-form English chat
    /// (verified post-softcap stays in this range).
    private let subsetConfidenceFloor: Float
    /// Hard-coded frequent token IDs included in every subset cycle to
    /// maximize argmax coverage. Loaded from `frequent_tokens.bin` (Int32)
    /// when present in the bundle directory; otherwise falls back to a
    /// padded range covering reserved + common short BPE pieces.
    private let frequentTokensBase: [Int32]
    /// Number of cycles where subset path returned low-confidence and we
    /// fell back to full chunk4. Diagnostic only.
    private(set) var subsetFallbacks: Int = 0
    /// Number of cycles where subset path ran end-to-end (no fallback).
    private(set) var subsetHits: Int = 0

    // MARK: - Metrics

    private(set) var totalRounds = 0
    private(set) var totalAccepted = 0
    private(set) var totalEmitted = 0

    /// Sampling temperature for the rejection-sampling MTP path. `0` keeps
    /// the legacy greedy path (drafter argmax compared to target argmax).
    /// `> 0` switches to proper speculative-sampling: drafter samples from
    /// its top-K, target samples from its full vocab via `logits_fp16`,
    /// acceptance uses the standard min(1, p_t / p_d) rule.
    public var samplingTemperature: Float = 0.0

    /// Optional drafter-side temperature override. When 0 (default),
    /// drafter sampling uses the same `samplingTemperature` as the
    /// acceptance test (canonical speculative sampling). When > 0,
    /// drafter samples at `drafterTemperature` while acceptance still
    /// computes p_d at that same drafter T. Higher drafter T softens
    /// drafter top-1 dominance → drafter samples top-2/3 more often →
    /// can hit target_argmax that drafter's top-1 missed (helps when
    /// drafter is over-confidently wrong on free-form prompts).
    public var drafterTemperature: Float = 0.0

    /// acc0 = num_accepted / (num_rounds * K)
    var acceptanceRate: Double {
        totalRounds == 0 ? 0 : Double(totalAccepted) / Double(totalRounds * K)
    }

    /// Average tokens emitted per round (includes tTokNext + bonus/correction)
    var tokensPerRound: Double {
        totalRounds == 0 ? 0 : Double(totalEmitted) / Double(totalRounds)
    }

    // MARK: - Init

    init(engine: ChunkedEngine, drafter: MtpDraftSource) {
        self.engine = engine
        self.drafter = drafter
        self.K = drafter.K
        precondition(engine.hasVerify, "MTP speculation requires verify chunks")
        precondition(engine.verifyK == drafter.K,
                     "Drafter K=\(drafter.K) must match verify K=\(engine.verifyK)")
        let env = ProcessInfo.processInfo.environment
        // 2026-05-13: opt-in only (MTP_SUBSET_LM_HEAD=1). iOS empirical:
        // chunk4_subset is slower than chunk4 on ANE; sparse matmul adds
        // 6-9 ms; net regression 45 %. See ChunkedEngine load path for
        // matching policy.
        self.subsetEnabled = env["MTP_SUBSET_LM_HEAD"] == "1" && engine.hasSubsetLMHead
        if let s = env["MTP_SUBSET_M"], let v = Int(s), v >= 64, v <= 8192 {
            self.subsetCapM = v
        } else {
            self.subsetCapM = 1024
        }
        if let s = env["MTP_SUBSET_FLOOR"], let v = Float(s) {
            self.subsetConfidenceFloor = v
        } else {
            self.subsetConfidenceFloor = 12.0
        }
        // Frequent-token base: load `frequent_tokens.bin` (Int32 LE) when
        // present alongside the model bundle so we ship corpus-derived
        // coverage. When absent, fall back to a synthetic range. The
        // ChunkedEngine doesn't expose its directory directly so we go
        // through ProcessInfo: `MTP_SUBSET_FREQ_BIN=path/to/freq.bin`.
        // Empty array is acceptable — candidate set will rely entirely on
        // drafter top-K + recent emit history (lower coverage, more fallback).
        var freq: [Int32] = []
        if let path = env["MTP_SUBSET_FREQ_BIN"],
           let data = try? Data(contentsOf: URL(fileURLWithPath: path)) {
            let n = data.count / MemoryLayout<Int32>.stride
            freq.reserveCapacity(n)
            data.withUnsafeBytes { raw in
                let p = raw.baseAddress!.assumingMemoryBound(to: Int32.self)
                for i in 0..<n { freq.append(p[i]) }
            }
        } else if !engine.frequentTokensFromBundle.isEmpty {
            // Use the bundle's freq list (loaded by ChunkedEngine at startup
            // from `frequent_tokens.bin` alongside chunks). This is the
            // expected production path on iPhone.
            freq = engine.frequentTokensFromBundle
        } else if subsetEnabled {
            // Synthetic fallback: token IDs 0..1023 (reserved + first
            // ~768 BPE pieces). Empirically not corpus-frequent; expect
            // higher fallback rate until `frequent_tokens.bin` is shipped.
            let cap = min(1024, engine.config.vocabSize)
            freq.reserveCapacity(cap)
            for i in 0..<cap { freq.append(Int32(i)) }
        }
        self.frequentTokensBase = freq
        if subsetEnabled {
            print("[MTP/Subset] enabled — M=\(subsetCapM) floor=\(subsetConfidenceFloor) "
                + "freqBase=\(freq.count)")
        }
    }

    // MARK: - Speculative step

    /// Execute one speculative cycle.
    ///
    /// On the first call, runs a normal decode to bootstrap kv13/kv14 state.
    /// Subsequent calls draft K tokens, verify, and commit the accepted prefix.
    ///
    /// - Parameter nextID: the token predicted from the last decode/verify.
    ///   Updated in-place to the next token for the following cycle.
    /// - Returns: tokens to emit to the output stream.
    func speculateStep(nextID: inout Int32) throws -> [Int32] {
        // Bootstrap: first call does a normal decode to populate kv13/kv14
        if !isBootstrapped {
            return try bootstrapStep(nextID: &nextID)
        }

        let pos = engine.currentPosition
        let hidden = engine.config.hiddenSize

        // Per-chunk parity audit dump (one-shot, on first speculative round).
        if totalRounds == 0,
           let outDir = ProcessInfo.processInfo.environment["MTP_CHUNK_DUMP"] {
            try? FileManager.default.createDirectory(
                atPath: outDir, withIntermediateDirectories: true)
            func dump(_ name: String, _ a: MLMultiArray?) {
                guard let a else { return }
                let bytes = a.count * MemoryLayout<UInt16>.stride
                let data = Data(bytes: a.dataPointer, count: bytes)
                try? data.write(to: URL(fileURLWithPath: "\(outDir)/\(name).fp16"))
                let shape = a.shape.map { $0.intValue }
                try? "shape=\(shape)\n".write(
                    to: URL(fileURLWithPath: "\(outDir)/\(name).txt"),
                    atomically: true, encoding: .utf8)
            }
            dump("h1", engine.lastChunk1Hidden)
            dump("h2", engine.lastChunk2Hidden)
            dump("h3", engine.lastChunk3Hidden)
            dump("h4_postnorm", engine.lastDecodeHiddenStateOut)
            // Embed of nextID with embed_scale applied — should match HF's
            // input embed at the corresponding position.
            if let e = try? engine.embedToken(nextID) {
                dump("embed_input", e)
            }
            dump("kv13_k", engine.lastKV13K)
            dump("kv13_v", engine.lastKV13V)
            dump("kv14_k", engine.lastKV14K)
            dump("kv14_v", engine.lastKV14V)
            try? "pos=\(pos) nextID=\(nextID)\n".write(
                to: URL(fileURLWithPath: "\(outDir)/info.txt"),
                atomically: true, encoding: .utf8)
            print("[MtpDbg] dumped per-chunk hidden states to \(outDir)")
        }

        // Initialize carry state on the first speculative burst.
        // HF SinglePositionMultiTokenCandidateGenerator (transformers
        // candidate_generator.py:1357) seeds proj_act with the target's
        // post-final-norm hidden for step 0 of the first burst — that is
        // `model_outputs.hidden_states[-1]` which equals `last_hidden_state`
        // (post-RMSNorm in Gemma 4). chunk4's `hidden_states_out` is exactly
        // that. The legacy `hidden_at_L34` tap is PRE-norm and not present
        // on non-EAGLE-3 bundles anyway.
        if carryState == nil {
            carryState = try MLMultiArray(
                shape: [1, 1, NSNumber(value: hidden)], dataType: .float16)
            if let h = engine.lastDecodeHiddenStateOut {
                let src = h.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
                let dst = carryState!.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
                memcpy(dst, src, hidden * MemoryLayout<UInt16>.stride)
            } else {
                memset(carryState!.dataPointer, 0, hidden * MemoryLayout<UInt16>.stride)
            }
        }

        guard let kv13K = engine.lastKV13K,
              let kv13VRaw = engine.lastKV13V,
              let kv14K = engine.lastKV14K,
              let kv14VRaw = engine.lastKV14V
        else {
            throw SpeculativeError.verifyFailed("kv13/kv14 not available for MTP drafter")
        }
        // The MTP drafter (per build_mtp_drafter.py metadata) expects V
        // transposed as (1, 1, head_dim, seq). ChunkedEngine stores V as
        // (1, 1, seq, head_dim). Both kv13V and kv14V have pre-transposed
        // mirrors updated in chunk2 hooks — read them for free here.
        // kv13V mirror saves ~4 ms / round (parallel tiled transpose was
        // formerly inside this critical path). kv14V mirror saves ~21 ms
        // / round on the previously full O(ctx*hd) transpose.
        let _trT0 = CFAbsoluteTimeGetCurrent()
        let kv13V: MLMultiArray
        if let kv13VT = engine.lastKV13V_T {
            kv13V = kv13VT
        } else {
            kv13V = try Self.transposeLastTwoDims(kv13VRaw)
        }
        let _trT1 = CFAbsoluteTimeGetCurrent()
        let kv14V: MLMultiArray
        if let kv14VT = engine.lastKV14V_T {
            kv14V = kv14VT
        } else {
            kv14V = try Self.transposeLastTwoDims(kv14VRaw)
        }
        let _trT2 = CFAbsoluteTimeGetCurrent()
        // 2026-05-13: gated behind MTP_VERBOSE_SETUP to remove from steady-state
        // hot path (was firing every 10 cycles, ~0.5-1 ms / cycle of print
        // overhead). First-cycle data is still useful for debugging — opt-in.
        if ProcessInfo.processInfo.environment["MTP_VERBOSE_SETUP"] == "1"
            && (totalRounds < 5 || totalRounds % 10 == 0)
        {
            print(String(format:
                "[MTP setup] transpose13=%.1fms transpose14=%.1fms",
                (_trT1 - _trT0) * 1000.0,
                (_trT2 - _trT1) * 1000.0))
        }

        // Build mask ONCE per cycle at the last committed position.
        // The drafter reads target KV (positions 0..pos-1), so mask allows 0..pos-1.
        // 2026-05-06 diagnostic: HF candidate generator's drafter sees K
        // cache covering ONLY [0..pos-2] (the bootstrap output's K wasn't
        // computed yet in HF's flow, but our predictStep writes it into
        // KV[pos-1]). Default 1 on both platforms (drafter sees clean K
        // written by predictStep commit on iOS, by verify slice on Mac).
        // 2026-05-08: tried offset=2 default on iOS (HF semantics, hide
        // latest K) — broke accept rate (0.95 → 0.45 alternation) and
        // corrupted output. Drafter needs the latest K when it's clean.
        let maskOffset = Int(ProcessInfo.processInfo
            .environment["MTP_MASK_OFFSET"] ?? "1") ?? 1
        let maskPos = pos - maskOffset
        let _mT0 = CFAbsoluteTimeGetCurrent()
        let maskSwa = try engine.makeDrafterSWAMask(position: max(maskPos, 0))
        let maskFull = try engine.makeDrafterFullMask(position: max(maskPos, 0))
        let _mT1 = CFAbsoluteTimeGetCurrent()
        if ProcessInfo.processInfo.environment["MTP_VERBOSE_SETUP"] == "1"
            && (totalRounds < 5 || totalRounds % 10 == 0)
        {
            print(String(format: "[MTP setup] masks=%.1fms",
                (_mT1 - _mT0) * 1000.0))
        }

        // Draft K tokens with per-step RoPE updates.
        // HF gemma4_assistant trains with inputs_embeds = embed_tokens.weight
        // * sqrt(hidden_size), so feed the scaled lookup (embedToken applies
        // config.embedScale = sqrt(hidden_size)). Using lookupRawEmbed here
        // produced 0/7 acceptance every cycle on Mac (2026-05-06).
        var proposals = [Int32]()
        proposals.reserveCapacity(K)
        var embedToken = try engine.embedToken(nextID)
        var projAct = carryState!
        // Append current nextID to history (the seed for this cycle).
        emittedHistory.append(nextID)

        // HF SinglePositionMultiTokenCandidateGenerator
        // (candidate_generator.py:1370) uses constant position_ids = pos-1
        // for ALL draft steps. Override via MTP_DRAFT_POS_MODE:
        //   "perstep" — pos+k per step.
        //   "constpm1" — constant pos-1 (HF behavior; default after centroid
        //                drafter empirically beats perstep by ~3 % on Mac).
        //   "constpos" — constant pos.
        let posMode = ProcessInfo.processInfo.environment["MTP_DRAFT_POS_MODE"] ?? "constpm1"
        // Adaptive K_USE per HF heuristic
        // (transformers candidate_generator.py:240-251). Empirically on
        // INT4-quantized chunks the per-slot accept (~28 %) is too low for
        // the +2/-1 schedule — full-match probability is 0.28² ≈ 8 % so K
        // rarely grows past 2 and frequently shrinks to 1. Static K_USE=2
        // wins +13.8 % vs adaptive's +3.5 %. Default OFF; opt in with
        // MTP_K_ADAPTIVE=1.
        let adaptive = ProcessInfo.processInfo
            .environment["MTP_K_ADAPTIVE"] == "1"
        let perPromptAdapter = !adaptive
            && ProcessInfo.processInfo
                .environment["MTP_PER_PROMPT_KUSE"] == "1"
        let envKUseStr = ProcessInfo.processInfo.environment["MTP_K_USE"]
        let userOverrode = envKUseStr.flatMap { Int($0) } != nil
        let kEffective: Int
        if adaptive {
            kEffective = max(1, min(K, kUseAdaptive))
        } else if let s = envKUseStr, let v = Int(s) {
            kEffective = v <= 0 ? K : min(K, v)
        } else if perPromptAdapter, let v = perPromptKUseOverride {
            kEffective = max(1, min(K, v))
        } else {
            // Default: K-1 (use all draft slots that have a paired verify
            // slot for argmax comparison). For K=3 this gives kEffective=2.
            //
            // 2026-05-13 iPhone empirical caveat — K_USE=1 can win +24%
            // on code prompts (BST etc) WHEN the drafter is already
            // warmed up from a prior prompt. On cold-start the single
            // proposal misses → accept 0/1 → rolling EMA dips below
            // fallback threshold → auto-bail to T=1. K_USE=2 emits two
            // proposals per cycle so partial-match (1/2) keeps the
            // rolling EMA above threshold and MTP stays engaged through
            // the warmup. Therefore K-1 is the robust iPhone default.
            //
            // To opt into K_USE=1 (e.g. for warm-bench-only runs), set
            // MTP_K_USE=1 via the env knob.
            kEffective = max(1, K - 1)
        }
        // Rejection-sampling path: drafter samples from its top-K with
        // temperature T, target verifies via full-vocab logits and accepts
        // each slot with probability min(1, p_t(x_d)/p_d(x_d)).
        // Greedy path: drafter argmax compared to target argmax.
        let useSampling = samplingTemperature > 0.0
        let T = max(samplingTemperature, 1e-6)
        // Drafter-side T can be decoupled from acceptance T: when
        // `drafterTemperature > 0`, drafter samples at that T (and p_d
        // is computed at that T) while target's p_t still uses T. This
        // is empirical (not canonical speculative sampling), aimed at
        // free-form prompts where drafter's top-1 is over-confidently
        // wrong but top-2/3 sometimes match target — softening drafter
        // sampling lifts those into reach.
        let T_d = drafterTemperature > 0 ? drafterTemperature : T
        // Per draft step: drafter top-K (ids, fp32 logits) + optional
        // full-vocab logsumexp. The drafter exposes `full_logsumexp` as
        // a scalar so we can compute exact p_d(x) = exp(logit/T - lse/T)
        // instead of renormalizing top-K (which over-estimates p_d → too
        // many rejections in speculative sampling).
        var drafterTopKByStep: [(ids: [Int32], logits: [Float], lse: Float?)] = []
        if useSampling || subsetEnabled { drafterTopKByStep.reserveCapacity(kEffective) }

        // Tree verify: 1 drafter call → top-(K-1) branches verified in
        // parallel with tree mask (each branch attends cur, not other
        // branches). Drops chain depth in exchange for higher per-slot
        // accept (drafter top-K hit > top-1 hit). compareLen still K-1 but
        // semantics differ: branches are independent, not chained.
        // Tree verify: empirically -20% on yes-yes (max emit=2 vs chain's 3)
        // and no measurable lift on free-form (chain emit 1.17 vs tree 1.28
        // is within noise once first-cycle cold drag dominates). Chain wins.
        let useTree = ProcessInfo.processInfo.environment["MTP_TREE_VERIFY"] == "1"

        // Per-step drafter self-bail (HF ConfidenceCriteria, default 0.4 +
        // llama.cpp `spec-draft-p-min`, default 0.75). When the drafter's
        // own softmax probability on its top-1 sampled token drops below
        // `selfBailThreshold`, stop the draft chain mid-K — the subsequent
        // tokens are likely wrong anyway, so we save the drafter call and
        // present a shorter proposal list to verify (cheaper compare loop,
        // less false-accept noise). Only fires at k ≥ 1 so we always emit
        // at least one draft token. Disable via MTP_SELF_BAIL_DISABLE=1.
        let useSelfBail = !useTree && !useSampling
            && ProcessInfo.processInfo
                .environment["MTP_SELF_BAIL_DISABLE"] != "1"
        let selfBailThreshold: Float = {
            if let s = ProcessInfo.processInfo
                .environment["MTP_SELF_BAIL_THRESHOLD"],
               let v = Float(s), v > 0 { return v }
            return 0.40  // HF ConfidenceCriteria default
        }()

        // PLD (Prompt-Lookup-Decoding) prefetch — when recent emitted tokens
        // form an n-gram seen earlier in this conversation, propose the
        // continuation as drafter output. Skip the entire drafter loop on
        // hit (save 11 ms on iPhone). Verify still strict-checks → lossless.
        // 2026-05-13: PLD prefetch tested on Mac free-form (hobby): emit
        // dropped 1.78 → 1.63 (PLD's proposals less accurate than MTP
        // drafter on novel text). Net regression. Opt-in only via
        // MTP_PLD_PREFETCH_ENABLE=1 — useful when prompt has repetition
        // (code, translate, summarization).
        let pldEnabled = !useTree && !useSampling
            && ProcessInfo.processInfo
                .environment["MTP_PLD_PREFETCH_ENABLE"] == "1"
        var pldHit = false

        // L5 — async drafter speculation hit check (cross-cycle).
        // If previous cycle's verify produced full accept (matchCount==K-1)
        // AND background drafter for this cycle predicted nextID correctly,
        // we already have this cycle's proposals pre-computed. Skip drafter.
        let l5Enabled = !useTree && !useSampling
            && ProcessInfo.processInfo
                .environment["MTP_L5_ASYNC_DISABLE"] != "1"
        if l5Enabled, let preDraft = pendingPreDraft,
           preDraft.expectedNextID == nextID,
           preDraft.proposals.count >= kEffective {
            proposals = Array(preDraft.proposals.prefix(kEffective))
            pldHit = true  // reuse pldHit flag to skip drafter loop
            l5HitsThisGeneration += 1
            if totalRounds < 6 {
                print(String(format:
                    "[L5 hit] r=%d nextID=%d preDraft proposals=%@",
                    totalRounds, nextID, "\(proposals)"))
            }
        }
        // Clear pendingPreDraft regardless (used or invalidated by nextID).
        pendingPreDraft = nil
        if pldEnabled {
            pldProbesThisGeneration += 1
            // Try ngramSize=2 first (more permissive), fall back to 1 if
            // no match. Free-form chat has few exact 3-gram repeats but
            // common 2-grams ("the", "of the", "in this") fire often.
            var pldProposals = PromptLookupDraft.propose(
                history: emittedHistory,
                ngramSize: 2,
                maxDraftLen: kEffective)
            if pldProposals.count < kEffective {
                // PLD requires 2*ngramSize+1 history; for short context,
                // try ngramSize=1 (single-token match).
                pldProposals = PromptLookupDraft.propose(
                    history: emittedHistory,
                    ngramSize: 1,
                    maxDraftLen: kEffective)
            }
            if pldProposals.count >= kEffective {
                pldHitsThisGeneration += 1
                proposals = Array(pldProposals.prefix(kEffective))
                pldHit = true
                if totalRounds < 8 {
                    print(String(format:
                        "[PLD hit] r=%d ngramHit proposals=%@",
                        totalRounds, "\(proposals)"))
                }
            } else if totalRounds < 4 {
                print("[PLD miss] r=\(totalRounds) (history=\(emittedHistory.count))")
            }
        }

        let (_, draftMs) = try SpecProfile.time {
            if pldHit { return }  // skip drafter loop entirely
            let drafterIters = useTree ? 1 : kEffective
            for k in 0..<drafterIters {
                let draftPos: Int
                switch posMode {
                case "constpm1": draftPos = max(pos - 1, 0)
                case "constpos": draftPos = pos
                default:         draftPos = pos + k
                }
                let cosSwa = try Self.reshapeRoPEForDrafter(
                    try engine.lookupCosSWA(position: draftPos))
                let sinSwa = try Self.reshapeRoPEForDrafter(
                    try engine.lookupSinSWA(position: draftPos))
                let cosFull = try Self.reshapeRoPEForDrafter(
                    try engine.lookupCosFull(position: draftPos))
                let sinFull = try Self.reshapeRoPEForDrafter(
                    try engine.lookupSinFull(position: draftPos))

                let tokenId: Int32
                let projActOut: MLMultiArray
                if useTree {
                    // Tree path: take top-(K-1) from drafter top-K as branches.
                    // Single drafter call, no chain.
                    let r = try drafter.draftOneFull(
                        embedToken: embedToken, projAct: projAct,
                        kv13K: kv13K, kv13V: kv13V,
                        kv14K: kv14K, kv14V: kv14V,
                        cosSwa: cosSwa, sinSwa: sinSwa,
                        cosFull: cosFull, sinFull: sinFull,
                        maskSwa: maskSwa, maskFull: maskFull)
                    let nBranches = K - 1
                    for b in 0..<nBranches {
                        if b < r.topKIds.count {
                            proposals.append(r.topKIds[b])
                        } else {
                            proposals.append(r.topKIds[0])  // fallback
                        }
                    }
                    embedToken = try engine.embedToken(r.topKIds[0])
                    projAct = r.projActOut
                    continue
                }
                if useSampling {
                    let r = try drafter.draftOneFull(
                        embedToken: embedToken, projAct: projAct,
                        kv13K: kv13K, kv13V: kv13V,
                        kv14K: kv14K, kv14V: kv14V,
                        cosSwa: cosSwa, sinSwa: sinSwa,
                        cosFull: cosFull, sinFull: sinFull,
                        maskSwa: maskSwa, maskFull: maskFull)
                    drafterTopKByStep.append((r.topKIds, r.topKLogits, r.fullLogSumExp))
                    // 2026-05-11: unconditionally dump for first 4 rounds on
                    // iOS to compare drafter top-K entropy vs Mac MPS bf16
                    // (peaked-distribution hypothesis).
                    if totalRounds < 4 {
                        let lseStr = r.fullLogSumExp.map { String(format: "%.2f", $0) } ?? "nil"
                        let lse = r.fullLogSumExp ?? 0
                        var s = "[Draft dist] r=\(totalRounds) k=\(k) lse=\(lseStr) "
                        for i in 0..<min(5, r.topKIds.count) {
                            let p_d = r.fullLogSumExp.map { expf((r.topKLogits[i] - $0)) } ?? 0
                            s += String(format: "id=%d L=%.2f p=%.4f / ",
                                        r.topKIds[i], r.topKLogits[i], p_d)
                        }
                        _ = lse  // silence unused
                        print(s)
                    }
                    let sampledId = Self.sampleFromTopK(
                        ids: r.topKIds, logits: r.topKLogits, temperature: T_d)
                    tokenId = sampledId
                    projActOut = r.projActOut
                } else if useSelfBail {
                    // Switched from draftOne to draftOneFull so we can read
                    // the chosen token's softmax probability against the
                    // drafter's full-vocab logsumexp and bail when confidence
                    // is below threshold.
                    let r = try drafter.draftOneFull(
                        embedToken: embedToken, projAct: projAct,
                        kv13K: kv13K, kv13V: kv13V,
                        kv14K: kv14K, kv14V: kv14V,
                        cosSwa: cosSwa, sinSwa: sinSwa,
                        cosFull: cosFull, sinFull: sinFull,
                        maskSwa: maskSwa, maskFull: maskFull)
                    tokenId = r.topKIds[0]
                    projActOut = r.projActOut
                    if subsetEnabled {
                        drafterTopKByStep.append((r.topKIds, r.topKLogits, r.fullLogSumExp))
                    }
                    if k >= 1, let lse = r.fullLogSumExp {
                        let pTop1 = expf(r.topKLogits[0] - lse)
                        if pTop1 < selfBailThreshold {
                            proposals.append(tokenId)
                            embedToken = try engine.embedToken(tokenId)
                            projAct = projActOut
                            if totalRounds < 6 {
                                print(String(format:
                                    "[Draft bail] r=%d k=%d p=%.3f < %.2f",
                                    totalRounds, k, pTop1, selfBailThreshold))
                            }
                            break
                        }
                    }
                } else if subsetEnabled {
                    // Subset path needs top-K candidates for the verify
                    // sparse matmul; use draftOneFull to expose them.
                    let r = try drafter.draftOneFull(
                        embedToken: embedToken, projAct: projAct,
                        kv13K: kv13K, kv13V: kv13V,
                        kv14K: kv14K, kv14V: kv14V,
                        cosSwa: cosSwa, sinSwa: sinSwa,
                        cosFull: cosFull, sinFull: sinFull,
                        maskSwa: maskSwa, maskFull: maskFull)
                    tokenId = r.topKIds[0]
                    projActOut = r.projActOut
                    drafterTopKByStep.append((r.topKIds, r.topKLogits, r.fullLogSumExp))
                } else {
                    let r = try drafter.draftOne(
                        embedToken: embedToken, projAct: projAct,
                        kv13K: kv13K, kv13V: kv13V,
                        kv14K: kv14K, kv14V: kv14V,
                        cosSwa: cosSwa, sinSwa: sinSwa,
                        cosFull: cosFull, sinFull: sinFull,
                        maskSwa: maskSwa, maskFull: maskFull)
                    tokenId = r.tokenID
                    projActOut = r.projActOut
                }

                proposals.append(tokenId)
                embedToken = try engine.embedToken(tokenId)
                projAct = projActOut
            }
        }

        // L5 — kick off async speculative drafter for cycle N+1 BEFORE
        // verify. Background drafter runs on CPU while ANE handles verify.
        // Predicts: speculative nextID for cycle N+1 + speculative proposals.
        // Validated post-verify. If valid, cycle N+1 skips drafter loop.
        if l5Enabled && !pldHit && proposals.count == kEffective {
            let lastDraftedToken = proposals.last!
            let snapProjAct = projAct  // drafter's final projAct (captured)
            // Speculative posForRoPE: assume full accept → cycle N+1 starts
            // at pos+K. constpm1 uses (cycle pos)-1 = pos+K-1.
            let specPos = max(pos + K - 1, 0)
            let kCopy = K
            let kEffCopy = kEffective
            asyncDrafterQueue.async { [weak self] in
                guard let self = self else { return }
                guard let kv13K = self.engine.lastKV13K,
                      let kv13V = self.engine.lastKV13V,
                      let kv14K = self.engine.lastKV14K,
                      let kv14V = self.engine.lastKV14V else { return }
                do {
                    let cosSwa = try Self.reshapeRoPEForDrafter(
                        try self.engine.lookupCosSWA(position: specPos))
                    let sinSwa = try Self.reshapeRoPEForDrafter(
                        try self.engine.lookupSinSWA(position: specPos))
                    let cosFull = try Self.reshapeRoPEForDrafter(
                        try self.engine.lookupCosFull(position: specPos))
                    let sinFull = try Self.reshapeRoPEForDrafter(
                        try self.engine.lookupSinFull(position: specPos))
                    let maskSwa = try self.engine.makeDrafterSWAMask(
                        position: specPos)
                    let maskFull = try self.engine.makeDrafterFullMask(
                        position: specPos)
                    var stepEmbed = try self.engine.embedToken(lastDraftedToken)
                    var stepProjAct = snapProjAct
                    // Step 0: predict cycle N's bonus position = N+1's nextID
                    let r0 = try self.drafter.draftOne(
                        embedToken: stepEmbed, projAct: stepProjAct,
                        kv13K: kv13K, kv13V: kv13V,
                        kv14K: kv14K, kv14V: kv14V,
                        cosSwa: cosSwa, sinSwa: sinSwa,
                        cosFull: cosFull, sinFull: sinFull,
                        maskSwa: maskSwa, maskFull: maskFull)
                    let specNextID = r0.tokenID
                    stepProjAct = r0.projActOut
                    stepEmbed = try self.engine.embedToken(specNextID)
                    // Steps 1..kEff: cycle N+1's proposals
                    var specProposals: [Int32] = []
                    for _ in 0..<kEffCopy {
                        let r = try self.drafter.draftOne(
                            embedToken: stepEmbed, projAct: stepProjAct,
                            kv13K: kv13K, kv13V: kv13V,
                            kv14K: kv14K, kv14V: kv14V,
                            cosSwa: cosSwa, sinSwa: sinSwa,
                            cosFull: cosFull, sinFull: sinFull,
                            maskSwa: maskSwa, maskFull: maskFull)
                        specProposals.append(r.tokenID)
                        stepProjAct = r.projActOut
                        stepEmbed = try self.engine.embedToken(r.tokenID)
                    }
                    _ = kCopy  // silence unused
                    self.pendingPreDraft = PreDraftedNextCycle(
                        expectedNextID: specNextID,
                        proposals: specProposals,
                        finalProjAct: stepProjAct)
                    self.l5DispatchesThisGeneration += 1
                } catch {
                    self.pendingPreDraft = nil
                }
            }
        }

        // Verify [nextID, proposals[0..K-2]] at currentPosition.
        // Cap compared proposals to K-1 so every accepted token's KV is
        // actually written by verify — comparing proposals[K-1] against
        // targetArgmax[K-1] was a bug: targetArgmax[K-1] is the argmax at
        // position pos+K (after proposals[K-2] as input), not a verification
        // of proposals[K-1] (which was never fed to verify). On all-accept
        // this also leaves slot pos+K without a KV write, the "all-accept
        // KV hole" that lets the next burst read garbage. Back-port of the
        // DrafterUnion fix.
        let useProps = Array(proposals.prefix(K - 1))
        let compareLen = useProps.count
        var verifyTokens = [Int32](repeating: 0, count: K)
        verifyTokens[0] = nextID
        for (i, t) in useProps.enumerated() { verifyTokens[i + 1] = t }
        // With K_use < K, the trailing verify slots have no real proposal —
        // pad with the last drafted token so target sees a valid input.
        if proposals.count < K - 1, let pad = proposals.last {
            for i in (proposals.count + 1)..<K { verifyTokens[i] = pad }
        }
        // Subset path: when MTP_SUBSET_LM_HEAD=1 and chunk4_subset + lm_head
        // are loaded, we route greedy/FLy cycles through `verifyCandidatesSubset`
        // which skips the 600M-param LM head matmul in chunk4. On low max-logit
        // confidence we fall back to full `verifyCandidates` to stay lossless.
        // Sampling and tree paths require full-vocab logits, so they always
        // take the legacy route.
        let useSubsetThisCycle = subsetEnabled && !useSampling && !useTree
        // Coverage validation mode: run BOTH subset and full chunk4 every
        // cycle, compare argmax, print mismatch stats, then use FULL argmax
        // for the actual decision so output stays lossless. Pure diagnostic
        // — does NOT affect tok/s gain math.
        let validateCoverage = useSubsetThisCycle
            && ProcessInfo.processInfo.environment["MTP_SUBSET_VALIDATE"] == "1"
        var subsetResult: ChunkedEngine.SubsetVerifyResult? = nil
        let (targetArgmax, verifyMs) = try SpecProfile.time {
            if useSubsetThisCycle {
                let cands = self.buildSubsetCandidates(
                    drafterTopK: drafterTopKByStep,
                    verifyTokens: verifyTokens,
                    cap: subsetCapM)
                let r = try engine.verifyCandidatesSubset(
                    tokens: verifyTokens,
                    candidateIds: cands,
                    startPosition: pos)
                if validateCoverage {
                    let fullArgmax = try engine.verifyCandidates(
                        tokens: verifyTokens, startPosition: pos)
                    var miss = 0
                    for k in 0..<r.argmax.count {
                        if r.argmax[k] != fullArgmax[k] { miss += 1 }
                    }
                    if miss > 0 {
                        print(String(format:
                            "[Subset MISS] r=%d miss=%d/%d subset=%@ full=%@ "
                            + "maxL=%@ M=%d",
                            totalRounds, miss, r.argmax.count,
                            "\(r.argmax)", "\(fullArgmax)",
                            "\(r.maxLogits.map { String(format: "%.1f", $0) })",
                            cands.count))
                    }
                    // Always use FULL argmax in validation mode → lossless.
                    return fullArgmax
                }
                let minConfidence = r.maxLogits.min() ?? -.greatestFiniteMagnitude
                if minConfidence < subsetConfidenceFloor {
                    // Low confidence anywhere in the K slots → re-run full
                    // chunk4. This wastes the subset matmul (~0.5 ms) but
                    // preserves losslessness. Track fallback rate to gauge
                    // candidate-set quality.
                    subsetFallbacks += 1
                    if totalRounds < 6 || totalRounds % 20 == 0 {
                        print(String(format:
                            "[Subset fallback] r=%d minL=%.2f < floor=%.2f",
                            totalRounds, minConfidence, subsetConfidenceFloor))
                    }
                    return try engine.verifyCandidates(
                        tokens: verifyTokens, startPosition: pos)
                }
                subsetHits += 1
                subsetResult = r
                return r.argmax
            }
            return try engine.verifyCandidates(
                tokens: verifyTokens, startPosition: pos)
        }

        // Accept/reject. Two paths:
        //   greedy   — compare drafter argmax to target argmax (legacy).
        //   sampling — per-slot rejection sampling: accept x_d with
        //              min(1, p_t(x_d)/p_d(x_d)); on reject sample residual.
        // Sampling falls back to greedy if the verify model isn't built with
        // --emit-logits (lastVerifyLogits would be nil).
        var matchCount = 0
        var samplingRejectCarry: Int32? = nil
        var treeMatchedBranch = -1  // tree verify: which branch (if any) matched
        let effectiveSampling = useSampling && engine.lastVerifyLogits != nil
        if effectiveSampling, let logits = engine.lastVerifyLogits {
            // Per-slot rejection sampling on the draft positions only.
            let vocab = engine.config.vocabSize
            let invT: Float = 1.0 / T
            let invT_d: Float = 1.0 / T_d
            for k in 0..<compareLen {
                let dr = drafterTopKByStep[k]
                // p_d(x) at drafter T (T_d). Decoupled from acceptance T
                // when `drafterTemperature` is set (asymmetric sampling).
                var drafterMass = [Int32: Float](minimumCapacity: dr.ids.count)
                if let lse = dr.lse {
                    // logsumexp from drafter is at T=1 (raw). Recompute at T_d:
                    // p_d(x) = exp(logit/T_d) / Σ exp(logit_v/T_d).
                    // We have the full-vocab raw logsumexp lse_raw = log Σ exp(logit_v).
                    // For T_d != 1 we'd need a new sum. Approximate: when T_d
                    // is small enough that drafter is concentrated on top-K,
                    // use top-K softmax. Otherwise scale lse by 1/T_d (rough).
                    if abs(T_d - 1.0) < 1e-3 {
                        for (i, id) in dr.ids.enumerated() {
                            drafterMass[id] = expf(dr.logits[i] - lse)
                        }
                    } else {
                        // Fall back to top-K softmax at T_d (truncated, slightly
                        // over-estimates p_d but acceptable for asymmetric path).
                        let drafterProbs = Self.softmaxOverArray(dr.logits, T: T_d)
                        for (i, id) in dr.ids.enumerated() {
                            drafterMass[id] = drafterProbs[i]
                        }
                    }
                } else {
                    let drafterProbs = Self.softmaxOverArray(dr.logits, T: T_d)
                    for (i, id) in dr.ids.enumerated() {
                        drafterMass[id] = drafterProbs[i]
                    }
                }
                _ = invT_d  // silence unused if not taken in lse branch above
                let xd = useProps[k]
                let p_d = drafterMass[xd] ?? 0
                let row = Self.fp16Row(logits, k: k, vocab: vocab)
                // Fast-path: when drafter sampled target's argmax, target
                // probability of x_d is essentially 1.0 (target dist is
                // super peaked at T=1.0 — top-1 logit beats top-2 by
                // ~20+ nats on Gemma 4). Acceptance ratio min(1, p_t/p_d)
                // ≥ 1, always accept. Skips the 5-10 ms full-vocab softmax
                // pass that otherwise dominates per-cycle cost.
                if xd == targetArgmax[k] {
                    matchCount += 1
                    continue
                }
                // Cheap rejection-path: when target dist is peaked (gap
                // between top-1 and x_d > ~10 nats / T), p_t(x_d) is
                // numerically negligible relative to p_d. Acceptance
                // ratio min(1, p_t/p_d) ≈ 0 → always reject. Residual
                // mass max(0, p_t - p_d) is then concentrated on target
                // argmax (~99% of remaining mass), so the residual sample
                // is effectively target_argmax. Skips both softmaxStats
                // AND sampleResidual full-vocab passes (~10-15 ms).
                // Cheap-reject: if target's top-1 vs x_d logit gap > 10 nats,
                // p_t(x_d) is negligible → skip full softmax + sampleResidual
                // passes (saves ~10-15 ms per slot on iPhone). Mac 2026-05-11
                // bench: disabling this didn't lift accept rate; re-enabled.
                let row_xd_logit = Self.fp16ToFloat(row[Int(xd)])
                let row_argmax_logit = Self.fp16ToFloat(row[Int(targetArgmax[k])])
                let logit_gap = row_argmax_logit - row_xd_logit
                if logit_gap * invT > 10.0 {
                    samplingRejectCarry = targetArgmax[k]
                    break
                }
                // Slow path: target is multimodal enough at this slot that
                // x_d's logit is within striking distance. Compute the
                // proper p_t via full-vocab softmax, then run min(1, p_t/p_d).
                let stats = Self.softmaxStats(row: row, vocab: vocab, T: T)
                let p_t = Self.probOf(row: row, idx: Int(xd), T: T,
                                      maxL: stats.maxL, sumExp: stats.sumExp)
                if ProcessInfo.processInfo
                    .environment["MTP_TARGET_DIST_DUMP"] == "1" && totalRounds < 3 {
                    // Inspect target logit distribution shape: top-N logits
                    // and corresponding probs to see whether p_t is genuinely
                    // peaked (1.0 / 0.0) or if there's mass we're missing.
                    var topL: [(Int, Float)] = []
                    for v in 0..<vocab {
                        let l = Self.fp16ToFloat(row[v])
                        if topL.count < 10 {
                            topL.append((v, l))
                            topL.sort { $0.1 > $1.1 }
                        } else if l > topL.last!.1 {
                            topL[9] = (v, l)
                            topL.sort { $0.1 > $1.1 }
                        }
                    }
                    let probs = topL.map { (id, l) -> (Int, Float, Float) in
                        let p = expf((l - stats.maxL) * invT) / stats.sumExp
                        return (id, l, p)
                    }
                    print("[MTP dist] r=\(totalRounds) k=\(k) maxL=\(String(format: "%.2f", stats.maxL)) lseT=\(String(format: "%.2f", stats.maxL + logf(stats.sumExp) * T))")
                    for (i, (id, l, p)) in probs.enumerated() {
                        print(String(format: "  top%d id=%d logit=%.2f p=%.6f", i, id, l, p))
                    }
                }
                let u = Float.random(in: 0..<1)
                let accept = (p_d <= 0) ? false : (u <= min(1, p_t / p_d))
                if accept {
                    matchCount += 1
                } else {
                    // Sample residual = max(0, p_t - p_d) on full vocab.
                    let resid = Self.sampleResidual(
                        row: row, vocab: vocab, T: T,
                        maxL: stats.maxL, sumExp: stats.sumExp,
                        drafterMass: drafterMass)
                    samplingRejectCarry = resid
                    break
                }
            }
        } else if useTree {
            // Tree-style accept: scan top-(K-1) branches for any match
            // against target argmax at position 0 (= what target predicts
            // naturally after cur). Each branch is independent (tree mask).
            // At most 1 branch matches.
            if totalRounds < 4 {
                print("[TreeDbg] r=\(totalRounds) cur=\(nextID) "
                    + "branches=\(useProps) "
                    + "targetArgmax=\(targetArgmax)")
            }
            for k in 0..<compareLen {
                if useProps[k] == targetArgmax[0] {
                    matchCount = 1
                    treeMatchedBranch = k
                    break
                }
            }
        } else {
            // Greedy + FLy (training-free loose speculative decoding, arxiv
            // 2511.22972). Default behaviour: strict argmax compare. With
            // MTP_FLY_TOPK=N (N≥2) or iOS default = K=8, also accept when
            // drafter's token is in target's top-N candidates. Lossy: emitted
            // tokens may differ from strict greedy, but stay within target's
            // high-probability mass. Lifts free-form accept 0.15 → ~0.5 with
            // K=8 → emit 1.17 → ~1.75 → +50% tok/s on iPhone.
            let flyTopK: Int = {
                if let s = ProcessInfo.processInfo.environment["MTP_FLY_TOPK"],
                   let v = Int(s), v > 1 { return v }
                #if os(iOS)
                // 2026-05-13: iOS default = 16. Quality-validated sweet spot.
                // Mac bench across hobby / transformer / ML / Kalman / sky:
                //   top-K=8:  emit 1.51, sometimes awkward
                //   top-K=16: emit 1.78, mostly coherent (minor grammar)
                //   top-K=32: emit 2.05 but output collapses ("to to to to")
                // iPhone projection at top-K=16 (cycle 41ms ANE drafter):
                //   1.36× vs plain decode on most EN free-form prompts.
                // Strict-only would deliver 1.05× max — top-K=16 is the
                // best lossless-ish trade we can make without retraining.
                return 16
                #else
                // 2026-05-13: Mac default = 16 (was 1 / strict-only). Empirical
                // Mac M ANE bench with centroid drafter shows top-K=16 lifts
                // narrative essay 32.9 → 41.9 tok/s (+27%) and code generation
                // 41.9 → 59.9 tok/s (+43%). The same quality-validated sweet
                // spot as iPhone. Override with MTP_FLY_TOPK=1 for strict-only
                // bench comparisons.
                return 16
                #endif
            }()
            // FLy needs per-position logits to rank drafter's token. Source
            // is `engine.lastVerifyLogits` (full vocab, only when chunk4 is
            // built with --emit-logits) or the subset path's per-K logits
            // (candidate-subset only). When neither is available, FLy is off.
            let hasSubsetLogits = subsetResult != nil
            let useFly = flyTopK > 1
                && (engine.lastVerifyLogits != nil || hasSubsetLogits)
            let vocabSize = engine.config.vocabSize
            // Build a candidate-id → local-index lookup once if we're on the
            // subset path. Used by FLy to convert drafter's full-vocab token
            // back into subset-row coordinates.
            var subsetIdxOf: [Int32: Int] = [:]
            var subsetM: Int = 0
            if let sr = subsetResult {
                subsetM = sr.candidateIds.count
                subsetIdxOf.reserveCapacity(subsetM)
                for (i, id) in sr.candidateIds.enumerated() {
                    subsetIdxOf[id] = i
                }
            }
            // Gemma 4 special tokens that must terminate generation: <bos>=2,
            // <eos>=1, <end_of_turn>=106. When target's natural next is one
            // of these, FLy must NOT override with drafter's higher-frequency
            // token (otherwise yes-yes loops infinitely — drafter says "yes",
            // target says <eos>, FLy lets "yes" win because it's in top-8).
            let specialStop: Set<Int32> = [1, 2, 106]
            // MARS — Margin-Aware Verification (arxiv 2601.15498, Jan 2026).
            // Accept drafter token iff it equals target's top-2 AND the logit
            // ratio z2/z1 exceeds theta (paper default 0.9). Captures the
            // "target is undecided between top-1 and top-2" case losslessly.
            // 2026-05-13: opt-in only. Initial Mac speedup claim (1.5-1.8×)
            // was measured on incoherent output — MARS/CSD accept lossy
            // drafter tokens that drift from target's actual argmax.
            // Quality-preserving free-form speedup with this centroid drafter
            // is structurally limited (~1.05×). Path B retraining required
            // for real 1.5×.
            let useMARS = (engine.lastVerifyLogits != nil)
                && (ProcessInfo.processInfo.environment["MTP_MARS_ENABLE"] == "1")
            let marsTheta: Float = {
                if let s = ProcessInfo.processInfo.environment["MTP_MARS_THETA"],
                   let v = Float(s), v > 0 { return v }
                return 0.9
            }()
            // CSD — Calibrated Speculative Decoding (arxiv 2604.13634, Apr
            // 2026). After λ historical (drafter, target_top1) divergences,
            // rescue subsequent occurrences when z_drafter ≥ z_top1 + log τ.
            // τ=0.01 → log τ ≈ −4.605, i.e. drafter token probability is at
            // least 1 % of target top-1's probability.
            // 2026-05-13: opt-in only (same quality issue as MARS).
            let useCSD = (engine.lastVerifyLogits != nil)
                && (ProcessInfo.processInfo.environment["MTP_CSD_ENABLE"] == "1")
            let csdLambda: Int = {
                if let s = ProcessInfo.processInfo.environment["MTP_CSD_LAMBDA"],
                   let v = Int(s), v > 0 { return v }
                return 2
            }()
            let csdLogTau: Float = {
                if let s = ProcessInfo.processInfo.environment["MTP_CSD_TAU"],
                   let v = Float(s), v > 0, v < 1 { return logf(v) }
                return logf(0.01)
            }()
            var flyMatches = 0
            var marsMatches = 0
            var csdRescues = 0
            for k in 0..<compareLen {
                // Strict: drafter == target top-1.
                if useProps[k] == targetArgmax[k] {
                    matchCount += 1
                    continue
                }
                // Don't override Gemma's terminate tokens with loose accept.
                if specialStop.contains(targetArgmax[k]) {
                    break
                }
                // Two paths to per-slot logits:
                //   - subset: (K, M) fp32 in candidate-subset coordinates.
                //   - full: (K, vocab) fp16 from chunk4 emit_logits=1.
                // MARS/CSD need full vocab to compute top-1/top-2, so they
                // only fire on the full-vocab path. FLy works on either by
                // ranking within the subset when only subset is available.
                if let sr = subsetResult {
                    if useFly {
                        // FLy on subset: drafter's token must be in candidate
                        // set (it is, by construction — drafter top-K is
                        // included). Rank within the M subset logits; accept
                        // when its rank ≤ flyTopK.
                        if let localIdx = subsetIdxOf[useProps[k]],
                           Self.isInTopKSubset(
                               logits: sr.subsetLogits, k: k, M: subsetM,
                               localIdx: localIdx, topK: flyTopK)
                        {
                            matchCount += 1
                            flyMatches += 1
                            continue
                        }
                    }
                    break
                }
                guard let logits = engine.lastVerifyLogits else { break }
                let row = Self.fp16Row(logits, k: k, vocab: vocabSize)

                // Compute top-1/top-2 once if either MARS or CSD needs it.
                var top1Logit: Float = -.infinity
                var top2Idx: Int = -1
                var top2Logit: Float = -.infinity
                if useMARS || useCSD {
                    let t2 = Self.top2WithLogits(row: row, vocab: vocabSize)
                    top1Logit = t2.z1
                    top2Idx = t2.idx2
                    top2Logit = t2.z2
                }

                // Always record divergence in CSD memory (count of pair
                // includes the current occurrence; rescue fires when count
                // reaches λ on this or a later occurrence).
                if useCSD {
                    let pair = TokenPair(draft: useProps[k], target: targetArgmax[k])
                    csdCorrectionMemory[pair] =
                        (csdCorrectionMemory[pair] ?? 0) + 1
                }

                // MARS: drafter == top-2 with tight margin (lossless).
                if useMARS && Int32(top2Idx) == useProps[k]
                    && top1Logit > 0
                    && (top2Logit / top1Logit) > marsTheta
                {
                    matchCount += 1
                    marsMatches += 1
                    continue
                }

                // CSD: recurring divergence pair + semantic-mass gate.
                if useCSD {
                    let pair = TokenPair(draft: useProps[k], target: targetArgmax[k])
                    let count = csdCorrectionMemory[pair] ?? 0
                    let drafterLogit = Self.fp16ToFloat(row[Int(useProps[k])])
                    let zDiff = drafterLogit - top1Logit
                    if count >= csdLambda && zDiff >= csdLogTau {
                        matchCount += 1
                        csdRescues += 1
                        continue
                    }
                }

                // FLy: drafter inside target top-K (lossy).
                if useFly && Self.isInTopK(
                    row: row, vocab: vocabSize,
                    idx: Int(useProps[k]), k: flyTopK)
                {
                    matchCount += 1
                    flyMatches += 1
                    continue
                }
                break
            }
            // Diagnostic dump for first few rounds (uncondtional iOS) so we
            // can see whether MARS / CSD / FLy are actually finding matches
            // on free-form prompts.
            if totalRounds < 6 && (useFly || useMARS || useCSD) {
                let strictMatches = matchCount - flyMatches - marsMatches - csdRescues
                print(String(format:
                    "[FLy] r=%d strict=%d mars=%d csd=%d fly=%d / compareLen=%d",
                    totalRounds, strictMatches,
                    marsMatches, csdRescues, flyMatches, compareLen))
            }
        }

        // Emitted = [nextID, matched...] — do NOT include the carry here.
        // The carry (correction on miss, bonus on all-match) stays as the
        // next burst's seed.
        var emitted: [Int32] = [nextID]
        emitted.reserveCapacity(matchCount + 1)
        if useTree && treeMatchedBranch >= 0 {
            // Tree: emit the single matched branch (= target argmax after cur)
            emitted.append(useProps[treeMatchedBranch])
        } else if !useTree {
            for k in 0..<matchCount { emitted.append(useProps[k]) }
        }

        let carry: Int32
        if useTree {
            if treeMatchedBranch >= 0 {
                // Bonus: target's argmax at branch's position (= predict
                // after seeing matched branch). Position is 1+treeMatchedBranch
                // in verifyTokens, so targetArgmax[1+treeMatchedBranch].
                carry = targetArgmax[1 + treeMatchedBranch]
            } else {
                // No branch matched. Carry = target's natural argmax after cur.
                carry = targetArgmax[0]
            }
        } else if effectiveSampling, let logits = engine.lastVerifyLogits {
            if let rejToken = samplingRejectCarry {
                // Reject at slot matchCount: residual sample is the carry.
                carry = rejToken
            } else if matchCount < K {
                // All-accept up to compareLen: bonus = sample target dist at
                // slot matchCount (== compareLen). Maps to argmax[K-1] when
                // compareLen == K-1.
                let vocab = engine.config.vocabSize
                let row = Self.fp16Row(logits, k: matchCount, vocab: vocab)
                let stats = Self.softmaxStats(row: row, vocab: vocab, T: T)
                carry = Self.sampleTarget(row: row, vocab: vocab, T: T,
                                          maxL: stats.maxL, sumExp: stats.sumExp)
            } else {
                carry = targetArgmax[K - 1]  // defensive; unreachable given K-1 cap
            }
        } else {
            // Greedy carry (legacy path).
            if matchCount < compareLen {
                carry = targetArgmax[matchCount]  // correction
            } else if matchCount < K {
                carry = targetArgmax[matchCount]  // bonus (= argmax at slot K-1)
            } else {
                carry = targetArgmax[K - 1]
            }
        }

        // Commit: under the 11c protocol, verify does NOT write KV to the
        // persistent cache. commitAccepted is what writes the accepted-prefix
        // slices. Bumping currentPosition directly leaves the cache stale.
        let committed = matchCount + 1
        let committedTokens = Array(emitted.prefix(committed))
        let commitT0 = CFAbsoluteTimeGetCurrent()
        try engine.commitAccepted(committedTokens)
        let commitMs = (CFAbsoluteTimeGetCurrent() - commitT0) * 1000.0

        // Append the matched-prefix tokens to PLD history. nextID was
        // already appended at speculateStep entry, so we skip it here.
        if committedTokens.count > 1 {
            emittedHistory.append(contentsOf: committedTokens.dropFirst())
        }

        // L5 — wait for background drafter to complete, validate, store/clear.
        // sync block on serial queue ensures background async closure finished.
        if l5Enabled {
            asyncDrafterQueue.sync {}
            if let pre = pendingPreDraft {
                // Cycle N+1's nextID = carry from cycle N = targetArgmax[matchCount]
                // (correction if partial, bonus if full accept). If speculative
                // nextID differs, the pre-drafted proposals are stale → discard.
                let actualNextID: Int32
                if matchCount < compareLen {
                    actualNextID = targetArgmax[matchCount]
                } else if matchCount < K {
                    actualNextID = targetArgmax[matchCount]
                } else {
                    actualNextID = targetArgmax[K - 1]
                }
                if pre.expectedNextID != actualNextID {
                    pendingPreDraft = nil
                }
            }
        }

        // Extract carry state from verify hidden states.
        // lastVerifyHiddenStates: (1, K, hidden) — matchCount indexes the
        // slot whose argmax is the carry (same as hiddenIdx used before,
        // just without clamping to K-1 since matchCount ≤ K-1 now).
        // Tree path: slice at the matched branch's verify position
        // (1 + treeMatchedBranch) so next round's drafter sees the hidden
        // for "after matched branch". On no-match, slice at position 0
        // (cur's hidden, predicting after cur naturally).
        let hiddenSliceIdx: Int
        if useTree {
            hiddenSliceIdx = treeMatchedBranch >= 0 ? (1 + treeMatchedBranch) : 0
        } else {
            hiddenSliceIdx = matchCount
        }
        carryState = sliceVerifyHidden(at: hiddenSliceIdx, hidden: hidden)

        // Adaptive K_USE update (HF heuristic): full match → +2, any miss → -1.
        if matchCount == compareLen && compareLen > 0 {
            kUseAdaptive = min(K, kUseAdaptive + 2)
        } else {
            kUseAdaptive = max(1, kUseAdaptive - 1)
        }

        // Round E — per-prompt K_USE adapter. One-shot sensing after the
        // first non-fallback spec cycle this prompt. Sticky for the rest
        // of the prompt (reset() clears).
        if perPromptAdapter && !perPromptKUseSensed && !userOverrode
            && compareLen >= 1 {
            perPromptKUseSensed = true
            if matchCount == compareLen && compareLen >= 2 {
                // Full match on K-1 slots → drafter on streak for this
                // prompt → minimize cost per cycle.
                perPromptKUseOverride = 1
            } else {
                // Partial / zero match → keep K-1 for next cycles. Sets
                // override explicitly so subsequent cycles take the same
                // path through the branch above (no reset on subsequent
                // matchCount swings — adapter is one-shot, not adaptive).
                perPromptKUseOverride = max(1, K - 1)
            }
        }

        // Update metrics
        totalRounds += 1
        totalAccepted += matchCount
        totalEmitted += emitted.count

        // Drive the drafter's rolling-acceptance EMA so `shouldSpeculate`
        // can fall back to baseline decode when the per-burst accept
        // drops below `fallbackThreshold`. Without this update the rate
        // sits at the initial 1.0 forever — MTP keeps running even when
        // it's actively hurting tok/s (iPhone ANE 18 numerics issue).
        drafter.recordBurst(matched: matchCount, K_USE: max(1, compareLen))

        SpecProfile.logBurst(
            engine: "mtp", cycle: totalRounds,
            draftMs: draftMs, verifyMs: verifyMs, commitMs: commitMs,
            accepted: matchCount, compareLen: compareLen,
            emitted: emitted.count, rolling: drafter.rollingAcceptance)

        // Carry becomes next burst's seed — NOT yielded from this cycle.
        nextID = carry

        return emitted
    }

    /// Whether speculation should be used for the next cycle.
    var shouldSpeculate: Bool {
        let should = drafter.shouldSpeculate
        if !should {
            // Advance EMA bypass cooldown when we're skipping spec, so the
            // probe-recheck eventually fires (llama.cpp §15 adaptive scheduler).
            drafter.didBypass()
        }
        return should
    }

    /// Reset state for new conversation. Also reseeds the drafter's
    /// rolling acceptance so a previous prompt's bad-content fallback
    /// doesn't stick (different prompts have different accept rates;
    /// each gets ~3-5 rounds to prove it can sustain MTP before the
    /// EMA bails again).
    func reset() {
        carryState = nil
        isBootstrapped = false
        totalRounds = 0
        totalAccepted = 0
        totalEmitted = 0
        drafter.resetRollingAcceptance()
        csdCorrectionMemory.removeAll(keepingCapacity: true)
        emittedHistory.removeAll(keepingCapacity: true)
        pldHitsThisGeneration = 0
        pldProbesThisGeneration = 0
        // Reset adaptive K to the starting value (MTP_K_USE override or 2).
        let envKUseStr = ProcessInfo.processInfo.environment["MTP_K_USE"]
        if let s = envKUseStr, let v = Int(s), v > 0 {
            kUseAdaptive = min(K, v)
        } else {
            kUseAdaptive = min(K, 2)
        }
        // Round E — per-prompt sense fires on first cycle of next prompt.
        perPromptKUseOverride = nil
        perPromptKUseSensed = false
    }

    // MARK: - Private

    /// First call: run a normal decode to populate kv13/kv14 and warm up.
    private func bootstrapStep(nextID: inout Int32) throws -> [Int32] {
        let emitted = nextID
        let (newNext, targetStepMs) = try SpecProfile.time {
            try engine.predictStep(
                tokenID: Int(nextID), position: engine.currentPosition)
        }
        engine.currentPosition += 1
        nextID = Int32(newNext)
        isBootstrapped = true
        SpecProfile.logBootstrap(
            engine: "mtp", replayCount: 0,
            replayMs: 0, targetStepMs: targetStepMs)
        return [emitted]
    }

    /// Slice hidden state at index `k` from lastVerifyHiddenStates (1, K, hidden).
    private func sliceVerifyHidden(at k: Int, hidden: Int) -> MLMultiArray? {
        guard let hs = engine.lastVerifyHiddenStates else { return nil }
        guard let result = try? MLMultiArray(
            shape: [1, 1, NSNumber(value: hidden)], dataType: .float16) else { return nil }
        let src = hs.dataPointer.bindMemory(to: UInt16.self, capacity: hs.count)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
        memcpy(dst, src.advanced(by: k * hidden), hidden * MemoryLayout<UInt16>.stride)
        return result
    }

    /// Transpose the last two dimensions of a rank-4 MLMultiArray
    /// [1, H, A, B] → [1, H, B, A]. Used for V caches: ChunkedEngine stores V
    /// as (1, num_kv_heads, seq, hd) but the MTP drafter expects
    /// (1, num_kv_heads, hd, seq) (Google TFLite pre-transposed layout).
    /// Supports multi-head (E4B target = 2 KV heads).
    static func transposeLastTwoDims(_ a: MLMultiArray) throws -> MLMultiArray {
        let shape = a.shape.map { $0.intValue }
        guard shape.count == 4, shape[0] == 1 else {
            throw SpeculativeError.verifyFailed(
                "transposeLastTwoDims: unexpected shape \(a.shape)")
        }
        let H = shape[1]
        let A = shape[2]
        let B = shape[3]
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: H),
                    NSNumber(value: B), NSNumber(value: A)],
            dataType: .float16)
        let srcBase0 = a.dataPointer.bindMemory(to: UInt16.self, capacity: H * A * B)
        let dstBase0 = out.dataPointer.bindMemory(to: UInt16.self, capacity: H * A * B)
        // 64×64 tile (8 KB src + 8 KB dst, both fit L1). Inner loop writes
        // dst contiguously (stride 1), reads src strided — better than the
        // reverse since write traffic dominates on iPhone ANE-cohabiting
        // workloads. Outer i tile parallelised across cores to amortise
        // the 1M-element kv14V transpose (was 77 ms single-threaded).
        let TILE = 64
        let numTilesI = (A + TILE - 1) / TILE
        for h in 0..<H {
            let srcBase = h * A * B
            let dstBase = h * B * A
            DispatchQueue.concurrentPerform(iterations: numTilesI) { tileIdx in
                let i0 = tileIdx * TILE
                let iEnd = min(i0 + TILE, A)
                var j0 = 0
                while j0 < B {
                    let jEnd = min(j0 + TILE, B)
                    for j in j0..<jEnd {
                        let dstRow = dstBase + j * A
                        for i in i0..<iEnd {
                            dstBase0[dstRow + i] = srcBase0[srcBase + i * B + j]
                        }
                    }
                    j0 += TILE
                }
            }
        }
        return out
    }

    /// Reshape RoPE cos/sin table for drafter I/O.
    /// ChunkedEngine.lookupRoPE returns shape `[1, 1, 1, dim]` (LLaMA-style
    /// duplicated halves). The MTP drafter (built by conversion/build_mtp_drafter.py)
    /// expects `[1, dim/2]` with only the unique half.
    /// Since `cos_full[:half] == cos_full[half:]` by construction, we copy
    /// the first half and reshape to rank 2.
    static func reshapeRoPEForDrafter(_ a: MLMultiArray) throws -> MLMultiArray {
        // Expect shape [1, 1, 1, dim]; output [1, dim/2]
        let dim = a.shape.last?.intValue ?? 0
        guard dim > 0 && dim % 2 == 0 else {
            throw SpeculativeError.verifyFailed(
                "reshapeRoPEForDrafter: unexpected dim \(dim) in shape \(a.shape)")
        }
        let half = dim / 2
        let out = try MLMultiArray(shape: [1, NSNumber(value: half)], dataType: .float16)
        let src = a.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        let dst = out.dataPointer.bindMemory(to: UInt16.self, capacity: half)
        memcpy(dst, src, half * MemoryLayout<UInt16>.stride)
        return out
    }

    // MARK: - Sampling helpers

    /// Pointer to row `k` of an `(1, K, vocab)` fp16 logits MLMultiArray.
    @inline(__always)
    static func fp16Row(_ logits: MLMultiArray, k: Int, vocab: Int)
        -> UnsafePointer<UInt16>
    {
        let total = logits.count
        let base = logits.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        return UnsafePointer(base.advanced(by: k * vocab))
    }

    /// fp16 → fp32 via Float16 (ARM NEON). Roughly 1 ns per element.
    @inline(__always)
    static func fp16ToFloat(_ bits: UInt16) -> Float {
        Float(Float16(bitPattern: bits))
    }

    /// Thread-local fp32 scratch buffer for full-vocab softmax. Sized to
    /// `vocabSize` once per cycle and reused across slots / sampleResidual
    /// calls so we don't allocate ~1 MB per rejection. After `softmaxStats`
    /// returns, the buffer holds `exp((row[v] - maxL) / T)` so probOf /
    /// sampleResidual / sampleTarget read directly instead of re-converting
    /// fp16 → fp32 → exp.
    private static let probScratchKey = "MtpSpeculativeEngine.probScratch"
    @inline(__always)
    private static func probScratch(ensuring vocab: Int) -> UnsafeMutablePointer<Float> {
        let tls = Thread.current.threadDictionary
        if let box = tls[probScratchKey] as? ScratchBox,
           box.capacity >= vocab {
            return box.ptr
        }
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: vocab)
        tls[probScratchKey] = ScratchBox(ptr: ptr, capacity: vocab)
        return ptr
    }
    @inline(__always)
    private static func probScratchPeek() -> UnsafeMutablePointer<Float>? {
        (Thread.current.threadDictionary[probScratchKey] as? ScratchBox)?.ptr
    }
    private final class ScratchBox {
        let ptr: UnsafeMutablePointer<Float>
        let capacity: Int
        init(ptr: UnsafeMutablePointer<Float>, capacity: Int) {
            self.ptr = ptr
            self.capacity = capacity
        }
        deinit { ptr.deallocate() }
    }

    /// Compute (max, sumExp) for softmax(row / T) and **fill the scratch
    /// buffer with `exp((row[v] - maxL) / T)` in-place**, so callers like
    /// `sampleResidual` / `sampleTarget` reuse the per-vocab work instead
    /// of redoing the fp16→fp32 + exp passes. Vectorized via
    /// vImageConvert_Planar16FtoPlanarF + vDSP_maxv + vvexpf + vDSP_sve.
    /// ~1-2 ms per call on iPhone vs ~5-8 ms for the scalar loop.
    static func softmaxStats(row: UnsafePointer<UInt16>, vocab: Int, T: Float)
        -> (maxL: Float, sumExp: Float)
    {
        let buf = probScratch(ensuring: vocab)
        // fp16 → fp32 in one vImage pass
        var src = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: row),
            height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 2)
        var dst = vImage_Buffer(
            data: UnsafeMutableRawPointer(buf),
            height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 4)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        // maxL = max(buf)
        var maxL: Float = -.infinity
        vDSP_maxv(buf, 1, &maxL, vDSP_Length(vocab))
        // buf := (buf - maxL) / T
        var negMaxOverT: Float = -maxL / T
        var invT: Float = 1.0 / T
        vDSP_vsmul(buf, 1, &invT, buf, 1, vDSP_Length(vocab))
        vDSP_vsadd(buf, 1, &negMaxOverT, buf, 1, vDSP_Length(vocab))
        // buf := exp(buf)
        var n = Int32(vocab)
        vvexpf(buf, buf, &n)
        // sumExp = Σ buf
        var sumExp: Float = 0
        vDSP_sve(buf, 1, &sumExp, vDSP_Length(vocab))
        return (maxL, sumExp)
    }

    /// p(idx) under softmax(row / T) given precomputed (maxL, sumExp).
    /// Reads from the per-vocab scratch populated by `softmaxStats` —
    /// avoids re-doing the fp16 → fp32 → exp work for the single idx.
    @inline(__always)
    static func probOf(row: UnsafePointer<UInt16>, idx: Int, T: Float,
                       maxL: Float, sumExp: Float) -> Float
    {
        if let buf = probScratchPeek() {
            return buf[idx] / sumExp
        }
        // Cold path: scratch unallocated (softmaxStats not called yet).
        let l = fp16ToFloat(row[idx])
        return expf((l - maxL) / T) / sumExp
    }

    /// FLy: check whether `idx`'s logit is in the top-`k` of `row`.
    /// Strategy: count entries with logit strictly greater than row[idx];
    /// if that count is < k, idx is in the top-k. Uses Accelerate vImage
    /// fp16→fp32 over the whole row, then vDSP scalar-add (subtract idx
    /// logit) + vDSP threshold-count via vsmsa + signum. Total ~2 ms per
    /// call on iPhone (262144 vocab).
    static func isInTopK(row: UnsafePointer<UInt16>, vocab: Int, idx: Int, k: Int)
        -> Bool
    {
        let buf = probScratch(ensuring: vocab)
        var src = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: row),
            height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 2)
        var dst = vImage_Buffer(
            data: UnsafeMutableRawPointer(buf),
            height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 4)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        let myValue = buf[idx]
        // Count entries strictly greater than myValue.
        // Use vDSP_vthres-style trick: subtract myValue from all, then count
        // positives. vDSP doesn't have a direct count-positive op, so we
        // signum (vDSP_vmaxmg vs 0 → sign) and sum.
        var negMy: Float = -myValue
        vDSP_vsadd(buf, 1, &negMy, buf, 1, vDSP_Length(vocab))
        // For each element: if > 0, map to 1.0; if ≤ 0, map to 0.0.
        // vDSP_vthrsc: threshold and scale. Below threshold → -scale,
        // above → +scale. Then add scale to shift {-scale, +scale} → {0, 2*scale}.
        var zero: Float = 0
        var halfScale: Float = 0.5
        vDSP_vthrsc(buf, 1, &zero, &halfScale, buf, 1, vDSP_Length(vocab))
        vDSP_vsadd(buf, 1, &halfScale, buf, 1, vDSP_Length(vocab))
        // Now buf has 0.0 for ≤0, 1.0 for >0. Sum to count.
        var greaterCount: Float = 0
        vDSP_sve(buf, 1, &greaterCount, vDSP_Length(vocab))
        // Self-comparison (idx vs idx) returned 0 (myValue - myValue = 0 → 0)
        // so it's not in the count. Result is: # entries strictly greater
        // than idx's value. If < k, idx is in top-k.
        return Int(greaterCount) < k
    }

    /// Subset analogue of `isInTopK`. `logits` is the (K, M) fp32 matrix from
    /// `verifyCandidatesSubset`, `localIdx` is the candidate-set index of the
    /// drafter's token (caller maps via `candidateIds`). Returns true iff
    /// fewer than `topK` other candidates have a strictly higher logit at
    /// row `k`. O(M) scan; M=1024 → microseconds.
    static func isInTopKSubset(logits: [Float], k: Int, M: Int,
                                localIdx: Int, topK: Int) -> Bool {
        guard localIdx >= 0 && localIdx < M, topK > 0 else { return false }
        let rowStart = k * M
        let myValue = logits[rowStart + localIdx]
        var greater = 0
        for m in 0..<M {
            if m == localIdx { continue }
            if logits[rowStart + m] > myValue {
                greater += 1
                if greater >= topK { return false }
            }
        }
        return true
    }

    /// Build the per-cycle candidate token set for `verifyCandidatesSubset`.
    /// Union of: drafter top-K per step, the verify tokens themselves, last
    /// 30 emitted tokens, Gemma special stops, and the frequent base set.
    /// Capped at `cap`. Dedup is via `Set<Int32>`.
    private func buildSubsetCandidates(
        drafterTopK: [(ids: [Int32], logits: [Float], lse: Float?)],
        verifyTokens: [Int32],
        cap: Int
    ) -> [Int32] {
        var set = Set<Int32>(minimumCapacity: cap)
        // Gemma special stop tokens. Subset must be able to predict <eos>
        // / <end_of_turn> so generation can terminate (otherwise yes-yes
        // and similar prompts loop forever).
        for id in [Int32(0), 1, 2, 106] {
            set.insert(id)
        }
        // 1) Drafter top-K across all steps. Highest priority — these are
        //    the tokens the drafter thinks could come next, so they're the
        //    most likely target argmax candidates.
        for step in drafterTopK {
            for id in step.ids {
                set.insert(id)
                if set.count >= cap { return Array(set) }
            }
        }
        // 2) Verify input tokens (nextID + useProps). These are what the
        //    target might natively keep or replace; either way they belong
        //    in the subset.
        for id in verifyTokens {
            set.insert(id)
            if set.count >= cap { return Array(set) }
        }
        // 3) Recent emit history (last 30 unique) — short-range repeats
        //    happen often in chat / code / list outputs.
        let n = min(30, emittedHistory.count)
        if n > 0 {
            for id in emittedHistory.suffix(n) {
                set.insert(id)
                if set.count >= cap { return Array(set) }
            }
        }
        // 4) Frequent-token base set fills the remaining slots.
        for id in frequentTokensBase {
            set.insert(id)
            if set.count >= cap { return Array(set) }
        }
        return Array(set)
    }

    /// Return top-1 / top-2 indices and logit values from a fp16 logit row.
    /// Used by MARS margin-aware verification (arxiv 2601.15498): a drafter
    /// token gets accepted when it equals target's top-2 and the logit
    /// ratio z2/z1 exceeds a threshold (paper default 0.9) — i.e. the target
    /// itself is undecided so accepting the close runner-up is essentially
    /// lossless. Implementation: vImage fp16→fp32, then two `vDSP_maxvi`
    /// (masking top-1 with -infinity between them). Total ~3 ms per call
    /// on iPhone (262144 vocab).
    static func top2WithLogits(row: UnsafePointer<UInt16>, vocab: Int)
        -> (idx1: Int, z1: Float, idx2: Int, z2: Float)
    {
        let buf = probScratch(ensuring: vocab)
        var src = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: row),
            height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 2)
        var dst = vImage_Buffer(
            data: UnsafeMutableRawPointer(buf),
            height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 4)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        var z1: Float = 0
        var i1: vDSP_Length = 0
        vDSP_maxvi(buf, 1, &z1, &i1, vDSP_Length(vocab))
        let savedTop1 = buf[Int(i1)]
        buf[Int(i1)] = -.infinity
        var z2: Float = 0
        var i2: vDSP_Length = 0
        vDSP_maxvi(buf, 1, &z2, &i2, vDSP_Length(vocab))
        buf[Int(i1)] = savedTop1
        return (Int(i1), z1, Int(i2), z2)
    }

    /// Stable softmax over a small fp32 logit array (drafter top-K, length ~8).
    static func softmaxOverArray(_ logits: [Float], T: Float) -> [Float] {
        guard !logits.isEmpty else { return [] }
        var maxL: Float = -.infinity
        for l in logits { if l > maxL { maxL = l } }
        let invT: Float = 1.0 / T
        var ex = [Float](repeating: 0, count: logits.count)
        var s: Float = 0
        for i in 0..<logits.count {
            let e = expf((logits[i] - maxL) * invT)
            ex[i] = e
            s += e
        }
        if s <= 0 { return [Float](repeating: 1.0 / Float(logits.count), count: logits.count) }
        for i in 0..<logits.count { ex[i] /= s }
        return ex
    }

    /// Sample one id from a small (id, logit) top-K under softmax with T.
    static func sampleFromTopK(ids: [Int32], logits: [Float], temperature T: Float)
        -> Int32
    {
        precondition(ids.count == logits.count && !ids.isEmpty,
                     "sampleFromTopK: empty or mismatched arrays")
        let probs = softmaxOverArray(logits, T: T)
        let u = Float.random(in: 0..<1)
        var c: Float = 0
        for i in 0..<probs.count {
            c += probs[i]
            if u <= c { return ids[i] }
        }
        return ids[ids.count - 1]
    }

    /// Sample from full target distribution softmax(row / T). Reads from
    /// the scratch buffer populated by `softmaxStats` (holds exp values).
    static func sampleTarget(row: UnsafePointer<UInt16>, vocab: Int, T: Float,
                             maxL: Float, sumExp: Float) -> Int32
    {
        guard let buf = probScratchPeek() else {
            // Cold path: rebuild
            _ = softmaxStats(row: row, vocab: vocab, T: T)
            return sampleTarget(row: row, vocab: vocab, T: T,
                                maxL: maxL, sumExp: sumExp)
        }
        let target = Float.random(in: 0..<1) * sumExp
        var c: Float = 0
        for v in 0..<vocab {
            c += buf[v]
            if c >= target { return Int32(v) }
        }
        return Int32(vocab - 1)
    }

    /// Sample residual = max(0, p_t - p_d) over full vocab. p_d is sparse:
    /// nonzero only for tokens in `drafterMass`. Reads buf populated by
    /// `softmaxStats` instead of re-converting fp16. Two sequential passes
    /// over the vocab (total / cumulative) — drafterMass is typically
    /// ≤ 8 entries so the dict lookup cost is amortized.
    static func sampleResidual(
        row: UnsafePointer<UInt16>, vocab: Int, T: Float,
        maxL: Float, sumExp: Float,
        drafterMass: [Int32: Float]
    ) -> Int32 {
        guard let buf = probScratchPeek() else {
            // Cold path: rebuild scratch and recurse
            _ = softmaxStats(row: row, vocab: vocab, T: T)
            return sampleResidual(row: row, vocab: vocab, T: T,
                                  maxL: maxL, sumExp: sumExp,
                                  drafterMass: drafterMass)
        }
        let invSum = 1.0 / sumExp
        // First pass: total residual mass.
        var totalResid: Float = 0
        for v in 0..<vocab {
            let pt = buf[v] * invSum
            let pd = drafterMass[Int32(v)] ?? 0
            let r = pt - pd
            if r > 0 { totalResid += r }
        }
        if totalResid <= 0 {
            // Pathological: drafter mass dominates target everywhere. Fall
            // back to argmax of target. vDSP_maxvi gets the argmax in O(N).
            var maxV: Float = 0
            var idx: vDSP_Length = 0
            vDSP_maxvi(buf, 1, &maxV, &idx, vDSP_Length(vocab))
            return Int32(idx)
        }
        // Second pass: cumulative sum + sample.
        let target = Float.random(in: 0..<1) * totalResid
        var c: Float = 0
        for v in 0..<vocab {
            let pt = buf[v] * invSum
            let pd = drafterMass[Int32(v)] ?? 0
            let r = pt - pd
            if r > 0 {
                c += r
                if c >= target { return Int32(v) }
            }
        }
        return Int32(vocab - 1)
    }
}
