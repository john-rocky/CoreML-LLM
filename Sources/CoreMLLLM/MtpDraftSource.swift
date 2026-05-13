//
//  MtpDraftSource.swift
//  CoreMLLLM
//
//  MTP drafter integration — uses Google's pre-trained 4-layer mini-transformer
//  to produce K draft tokens for speculative decoding.
//
//  Architecture: reads target model's kv13 (sliding) and kv14 (full attention)
//  caches directly. No separate drafter KV cache needed.
//
//  I/O contract (matches build_mtp_drafter.py):
//    Inputs:  embed_token (1,1,1536), proj_act (1,1,1536),
//             kv13_k/v, kv14_k/v, cos/sin tables, masks
//    Outputs: top_k_indices (8,), top_k_values (8,), proj_act_out (1,1,1536)
//

import CoreML
import Foundation

/// MTP drafter for speculative decoding, using Google's pre-trained weights.
public final class MtpDraftSource {

    // MARK: - Model

    private let model: MLModel

    /// Number of draft tokens per burst.
    public let K: Int

    /// Rolling acceptance rate for adaptive fallback. Initialised at the
    /// fallback threshold + small margin so the very first miss-streak
    /// fires fallback within ~3-5 rounds (instead of 17 when starting
    /// from 1.0). EMA alpha=0.20 — half-life ~3 rounds:
    ///   * Two consecutive 0/2 rounds from 0.55 → drops to 0.354 (bail).
    ///   * Strict 0/2-2/2 alternation stabilises around 0.50 (stay in MTP).
    /// This gives MTP a fair shot when accept rate is high while bailing
    /// fast on iPhone-style patterns where most rounds are 0/2.
    private(set) public var rollingAcceptance: Double = 0.55
    private let rollingAlpha: Double = 0.20
    /// Bail threshold. Platform-aware because MTP cycle latency differs:
    ///   Mac:   cycle ≈ 30-35 ms, break-even accept ≈ 0.05
    ///   iPhone: cycle ≈ 45-50 ms, break-even accept ≈ 0.20
    /// 2026-05-11 Mac bench: lowering Mac threshold 0.35 → 0.10 recovers
    /// the ~10 tok/s lost to premature bail on translate (44 forced vs 33
    /// auto-bail). iPhone stays conservative because free-form accept
    /// typically 0.11-0.16, below iPhone break-even.
    // 2026-05-13: unified never-bail for iOS too. Earlier iOS=0.25 threshold
    // killed FLy top-K=16 wins after 7-8 rounds of intermittent accepts (the
    // EMA drifts below 0.25 even when FLy hits 50% of slots), causing the
    // iPhone to silently fall back to plain decode. Mac never-bail behaviour
    // sustains MTP for 45+ rounds at avg 38% accept — that's the source of
    // Mac's 1.5× number. Pay 15ms per miss cycle to capture FLy wins.
    public var fallbackThreshold: Double = 0.0
    /// Hard bail: N consecutive `matched=0` rounds short-circuit the EMA.
    /// Platform-aware:
    ///   iPhone: 2 (cycle 45-50ms, drafter zero-rounds costly)
    ///   Mac:    disabled (Int.max) — Mac cycle (30ms) tolerates any zero
    ///   streak; bail loses the next-accept that recoups overhead. Once
    ///   bailed, no recovery path → only bail iPhone where overhead > gain.
    private(set) public var consecutiveZeroRounds: Int = 0
    // 2026-05-13: never-bail on both platforms (was iOS=2). See
    // `fallbackThreshold` comment.
    public var consecutiveZeroBailLimit: Int = Int.max

    /// Bail recovery counter (2026-05-12): after we hit the hard bail above
    /// the original logic kept MTP off for the rest of the generation. Free-
    /// form chat output drifts in/out of drafter-aligned regions (普通名詞
    /// followed by 述語など), so a permanent latch loses recoverable wins.
    /// `tokensSinceBail` counts post-bail target steps and, when it crosses
    /// `bailRecoveryInterval`, we let MTP probe again. The next round resets
    /// the counter if matched > 0; otherwise we re-bail and wait another
    /// interval. Net cost per failed retry: ~10 ms (one wasted draft+verify).
    private(set) public var tokensSinceBail: Int = 0
    public var bailRecoveryInterval: Int = 16

    /// EMA adaptive bypass (llama.cpp §15 pattern, M4 Max-validated). When
    /// rolling acceptance drops below `emaBypassThreshold`, skip the drafter
    /// for `emaBypassCooldown` rounds, then recheck on the next round. Lifted
    /// fiction prompts from 0.36× to 0.85× and idiomatic from 1.21× to 1.34×
    /// in the llama.cpp E2B Q4_K_M bench. Different from `consecutiveZeroBail`:
    /// EMA-based recovery lets drafter probe periodically instead of dying
    /// permanently, so a prompt that warms up after a slow start can rejoin.
    public var emaBypassThreshold: Double = 0.30
    public var emaBypassCooldown: Int = 5
    public var emaWarmupRounds: Int = 4
    private(set) public var skipCooldown: Int = 0
    private(set) public var roundsSeen: Int = 0

    /// External update hook for engines that drive their own
    /// draft/verify loop (MtpSpeculativeEngine). They compute matchCount
    /// per round and call this so `shouldSpeculate` can fall back to
    /// non-speculative decode when accept collapses on a given device.
    public func recordBurst(matched: Int, K_USE: Int) {
        guard K_USE > 0 else { return }
        let rate = Double(matched) / Double(K_USE)
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance
        roundsSeen += 1
        if matched == 0 {
            consecutiveZeroRounds += 1
        } else {
            consecutiveZeroRounds = 0
            tokensSinceBail = 0
        }
    }

    /// EMA bypass bookkeeping. Called by the engine when `shouldSpeculate`
    /// returns false so the cooldown counter advances and a recheck fires
    /// when it reaches zero.
    public func tickBypassCooldown() {
        if skipCooldown > 0 { skipCooldown -= 1 }
    }

    /// Reset the rolling acceptance EMA back to the init value so a new
    /// conversation/prompt can earn its way back into the MTP path even
    /// if the previous prompt collapsed accept rate to zero.
    public func resetRollingAcceptance() {
        rollingAcceptance = 0.55
        consecutiveZeroRounds = 0
        skipCooldown = 0
        roundsSeen = 0
        tokensSinceBail = 0
    }

    // MARK: - Init

    public init(
        modelURL: URL,
        K: Int = 3,
        configuration: MLModelConfiguration? = nil
    ) throws {
        let cfg = configuration ?? {
            let c = MLModelConfiguration()
            // 2026-05-06 diagnostic: ANE fp16/INT4 numerics differ enough
            // from CPU fp32 to crater drafter accept on Mac. Set
            // MTP_DRAFTER_DEVICE=cpu / gpu / ane to override.
            // 2026-05-13: iOS default flipped back to ANE. Real-device
            // measurement showed CPU draft = 10 ms per call vs ANE ~6 ms
            // historically. Mac bench projection: CPU drafter → iPhone 1.4×
            // avg / 2-of-5 prompts ≥1.5× ; ANE drafter → 1.5× avg / 3-of-5
            // ≥1.5×. The earlier "Mac CPU 2.2ms beat ANE" (task #57)
            // measurement was Mac M3 CPU, which has much higher single-
            // thread throughput than iPhone A18 P-core. ANE wins on iPhone.
            switch ProcessInfo.processInfo.environment["MTP_DRAFTER_DEVICE"] {
            case "cpu": c.computeUnits = .cpuOnly
            case "gpu": c.computeUnits = .cpuAndGPU
            case "ane": c.computeUnits = .cpuAndNeuralEngine
            default:    c.computeUnits = .cpuAndNeuralEngine
            }
            return c
        }()
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw SpeculativeError.assetMissing(modelURL.lastPathComponent)
        }
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
        self.K = K
    }

    /// Pre-compile the drafter's ANE graph by issuing a single dummy
    /// inference. Without this, the first drafter call after app launch
    /// (or after iPhone ANE goes idle) takes 60-70 ms instead of warm
    /// 13 ms — adding ~50 ms to MTP cycle #1 cold-start latency.
    /// All-zero inputs are sufficient to exercise the compile path; the
    /// returned tokens are discarded.
    public func prewarm(
        targetHidden: Int = 1536,
        slidingWindow: Int = 512,
        contextLength: Int = 2048,
        slidingHeadDim: Int = 256,
        fullHeadDim: Int = 512
    ) throws {
        func zerosFP16(_ shape: [Int]) throws -> MLMultiArray {
            let arr = try MLMultiArray(
                shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
            memset(arr.dataPointer, 0,
                   arr.count * MemoryLayout<UInt16>.stride)
            return arr
        }
        let H = targetHidden
        let W = slidingWindow
        let C = contextLength
        let hdSwa = slidingHeadDim
        let hdFull = fullHeadDim
        let embed = try zerosFP16([1, 1, H])
        let proj = try zerosFP16([1, 1, H])
        let kv13K = try zerosFP16([1, 1, W, hdSwa])
        let kv13V = try zerosFP16([1, 1, hdSwa, W])
        let kv14K = try zerosFP16([1, 1, C, hdFull])
        let kv14V = try zerosFP16([1, 1, hdFull, C])
        let cosSwa = try zerosFP16([1, hdSwa / 2])
        let sinSwa = try zerosFP16([1, hdSwa / 2])
        let cosFull = try zerosFP16([1, hdFull / 2])
        let sinFull = try zerosFP16([1, hdFull / 2])
        let maskSwa = try zerosFP16([1, 1, 1, W])
        let maskFull = try zerosFP16([1, 1, 1, C])
        _ = try draftOne(
            embedToken: embed, projAct: proj,
            kv13K: kv13K, kv13V: kv13V,
            kv14K: kv14K, kv14V: kv14V,
            cosSwa: cosSwa, sinSwa: sinSwa,
            cosFull: cosFull, sinFull: sinFull,
            maskSwa: maskSwa, maskFull: maskFull)
    }

    // MARK: - Drafting

    /// Execute one MTP drafting burst.
    ///
    /// - Parameters:
    ///   - target: the target model (ChunkedEngine conforming to SpeculativeTarget)
    ///   - tTokNext: target's argmax from the previous decode step
    ///   - tokenEmbed: closure producing raw embedding (1,1,1536) fp16 for a token
    ///   - lastHidden: target's last hidden state from L34 (1,1,1536) fp16
    ///   - kv13K: target's sliding K cache (1,1,W,256) fp16
    ///   - kv13V: target's sliding V cache (1,1,256,W) fp16
    ///   - kv14K: target's full K cache (1,1,C,512) fp16
    ///   - kv14V: target's full V cache (1,1,512,C) fp16
    ///   - cosSwa: precomputed cos table for current position, SWA head_dim
    ///   - sinSwa: precomputed sin table
    ///   - cosFull: precomputed cos table, full head_dim
    ///   - sinFull: precomputed sin table
    ///   - maskSwa: sliding window causal mask (1,1,1,W)
    ///   - maskFull: full context causal mask (1,1,1,C)
    ///
    /// - Returns: array of draft token ids (length K), plus their projected_activations
    public func draft(
        tTokNext: Int32,
        tokenEmbed: (Int32) throws -> MLMultiArray,
        lastHidden: MLMultiArray,
        kv13K: MLMultiArray,
        kv13V: MLMultiArray,
        kv14K: MLMultiArray,
        kv14V: MLMultiArray,
        cosSwa: MLMultiArray,
        sinSwa: MLMultiArray,
        cosFull: MLMultiArray,
        sinFull: MLMultiArray,
        maskSwa: MLMultiArray,
        maskFull: MLMultiArray
    ) throws -> (tokens: [Int32], projectedActivations: MLMultiArray) {

        var proposals: [Int32] = []
        proposals.reserveCapacity(K)

        // Step 0: first call uses target's last hidden as proj_act,
        // and embed(tTokNext) as embed_token
        var embedToken = try tokenEmbed(tTokNext)
        var projAct: MLMultiArray = lastHidden

        for _ in 0..<K {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "embed_token": embedToken,
                "proj_act": projAct,
                "kv13_k": kv13K,
                "kv13_v": kv13V,
                "kv14_k": kv14K,
                "kv14_v": kv14V,
                "cos_swa": cosSwa,
                "sin_swa": sinSwa,
                "cos_full": cosFull,
                "sin_full": sinFull,
                "mask_swa": maskSwa,
                "mask_full": maskFull,
            ])

            let output = try model.prediction(from: input)

            // Extract top-k indices
            guard let topKIds = output.featureValue(for: "top_k_indices")?.multiArrayValue,
                  let projActOut = output.featureValue(for: "proj_act_out")?.multiArrayValue
            else {
                throw SpeculativeError.verifyFailed("MTP drafter missing outputs")
            }

            // Take argmax from top-k (index 0 = highest logit)
            let tokenId: Int32 = topKIds.dataPointer
                .bindMemory(to: Int32.self, capacity: 1)
                .pointee

            proposals.append(tokenId)

            // Feed back: embed(drafted_token) and projected_activations
            embedToken = try tokenEmbed(tokenId)
            projAct = projActOut
        }

        return (proposals, projAct)
    }

    /// Single-step draft: run the drafter model once with given inputs.
    /// Used by MtpSpeculativeEngine for per-step RoPE/mask updates.
    public func draftOne(
        embedToken: MLMultiArray,
        projAct: MLMultiArray,
        kv13K: MLMultiArray,
        kv13V: MLMultiArray,
        kv14K: MLMultiArray,
        kv14V: MLMultiArray,
        cosSwa: MLMultiArray,
        sinSwa: MLMultiArray,
        cosFull: MLMultiArray,
        sinFull: MLMultiArray,
        maskSwa: MLMultiArray,
        maskFull: MLMultiArray
    ) throws -> (tokenID: Int32, projActOut: MLMultiArray) {
        let r = try draftOneFull(
            embedToken: embedToken, projAct: projAct,
            kv13K: kv13K, kv13V: kv13V, kv14K: kv14K, kv14V: kv14V,
            cosSwa: cosSwa, sinSwa: sinSwa, cosFull: cosFull, sinFull: sinFull,
            maskSwa: maskSwa, maskFull: maskFull)
        return (r.topKIds[0], r.projActOut)
    }

    /// Single-step draft that returns the full top-K (id, logit) pairs in
    /// addition to the carry. Used by the rejection-sampling path so the
    /// engine can compute drafter probabilities for the sampled token.
    public func draftOneFull(
        embedToken: MLMultiArray,
        projAct: MLMultiArray,
        kv13K: MLMultiArray,
        kv13V: MLMultiArray,
        kv14K: MLMultiArray,
        kv14V: MLMultiArray,
        cosSwa: MLMultiArray,
        sinSwa: MLMultiArray,
        cosFull: MLMultiArray,
        sinFull: MLMultiArray,
        maskSwa: MLMultiArray,
        maskFull: MLMultiArray
    ) throws -> (topKIds: [Int32], topKLogits: [Float],
                 projActOut: MLMultiArray, fullLogSumExp: Float?) {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "embed_token": embedToken,
            "proj_act": projAct,
            "kv13_k": kv13K,
            "kv13_v": kv13V,
            "kv14_k": kv14K,
            "kv14_v": kv14V,
            "cos_swa": cosSwa,
            "sin_swa": sinSwa,
            "cos_full": cosFull,
            "sin_full": sinFull,
            "mask_swa": maskSwa,
            "mask_full": maskFull,
        ])
        let output = try model.prediction(from: input)
        guard let topKIds = output.featureValue(for: "top_k_indices")?.multiArrayValue,
              let topKValues = output.featureValue(for: "top_k_values")?.multiArrayValue,
              let projActOut = output.featureValue(for: "proj_act_out")?.multiArrayValue
        else {
            throw SpeculativeError.verifyFailed("MTP drafter missing outputs")
        }
        let n = topKIds.count
        let idsPtr = topKIds.dataPointer.bindMemory(to: Int32.self, capacity: n)
        let valsPtr = topKValues.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        var ids = [Int32](repeating: 0, count: n)
        var logits = [Float](repeating: 0, count: n)
        for i in 0..<n {
            ids[i] = idsPtr[i]
            logits[i] = Float(_floatFromFP16(valsPtr[i]))
        }
        // `full_logsumexp` is now `log(Σ exp(logit - max))` — the
        // residual after subtracting the max logit, so it always fits
        // fp16 (∈ [0, log(N)] ≈ [0, 8.3] for N=4096). To recover the
        // true LSE, add the max logit back, which is the first entry of
        // `top_k_values` (the drafter sorts top-K by descending logit).
        var lse: Float? = nil
        if let lseArr = output.featureValue(for: "full_logsumexp")?.multiArrayValue,
           lseArr.count > 0, !logits.isEmpty {
            let lsePtr = lseArr.dataPointer.bindMemory(to: UInt16.self, capacity: 1)
            let residual = Float(_floatFromFP16(lsePtr[0]))
            lse = logits[0] + residual
        }
        return (ids, logits, projActOut, lse)
    }

    @inline(__always)
    private func _floatFromFP16(_ bits: UInt16) -> Float {
        // Standard fp16 → fp32 conversion via bit manipulation.
        let sign = UInt32(bits & 0x8000) << 16
        let exp  = UInt32(bits & 0x7C00) >> 10
        let mant = UInt32(bits & 0x03FF)
        let result: UInt32
        if exp == 0 {
            if mant == 0 { result = sign }
            else {
                // subnormal
                var m = mant
                var e: UInt32 = 0
                while (m & 0x0400) == 0 { m <<= 1; e &+= 1 }
                m &= 0x03FF
                result = sign | ((127 &- 15 &- e) << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            result = sign | 0x7F800000 | (mant << 13)
        } else {
            result = sign | ((exp &+ (127 &- 15)) << 23) | (mant << 13)
        }
        return Float(bitPattern: result)
    }

    /// Full speculative decode burst: draft → verify → accept.
    public func drawBurst(
        target: SpeculativeTarget,
        tTokNext: Int32,
        tokenEmbed: (Int32) throws -> MLMultiArray,
        lastHidden: MLMultiArray,
        kv13K: MLMultiArray,
        kv13V: MLMultiArray,
        kv14K: MLMultiArray,
        kv14V: MLMultiArray,
        cosSwa: MLMultiArray,
        sinSwa: MLMultiArray,
        cosFull: MLMultiArray,
        sinFull: MLMultiArray,
        maskSwa: MLMultiArray,
        maskFull: MLMultiArray
    ) throws -> [Int32] {

        // 1. Draft K tokens
        let (proposals, _) = try draft(
            tTokNext: tTokNext,
            tokenEmbed: tokenEmbed,
            lastHidden: lastHidden,
            kv13K: kv13K, kv13V: kv13V,
            kv14K: kv14K, kv14V: kv14V,
            cosSwa: cosSwa, sinSwa: sinSwa,
            cosFull: cosFull, sinFull: sinFull,
            maskSwa: maskSwa, maskFull: maskFull
        )

        // 2. Verify: target checks [tTokNext, proposals[0..K-2]]
        var verifyTokens = [tTokNext]
        verifyTokens.append(contentsOf: proposals.dropLast())
        let targetArgmax = try target.verifyCandidates(verifyTokens, K: K)
        guard targetArgmax.count == K else {
            throw SpeculativeError.verifyFailed(
                "verify returned \(targetArgmax.count), expected \(K)")
        }

        // 3. Accept prefix up to first disagreement
        var accepted: [Int32] = [tTokNext]
        var matched = 0
        for k in 0..<K {
            if proposals[k] == targetArgmax[k] {
                accepted.append(proposals[k])
                matched += 1
            } else {
                accepted.append(targetArgmax[k])
                break
            }
        }

        // 4. Commit to target
        try target.commitAccepted(accepted)

        // 5. Update rolling acceptance
        let rate = Double(matched) / Double(K)
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance

        return accepted
    }

    /// Whether to use speculative path for the next burst.
    /// Set MTP_FORCE_SPECULATE=1 to bypass the fallback threshold (debugging /
    /// raw accept measurement). Production should leave it on so the engine
    /// stops paying drafter cost when accept collapses.
    public var shouldSpeculate: Bool {
        if ProcessInfo.processInfo.environment["MTP_FORCE_SPECULATE"] == "1" {
            return true
        }
        // Hard bail on consecutive 0-accept rounds: stops drafter cost
        // accumulation when the prompt domain is clearly off. Recovery probe:
        // every `bailRecoveryInterval` committed tokens we let the drafter
        // re-test the prompt — long generations can pass through drafter-
        // aligned regions (反復語, particles, common suffixes) that we'd
        // otherwise miss with the original permanent latch.
        if consecutiveZeroRounds >= consecutiveZeroBailLimit {
            tokensSinceBail += 1
            if tokensSinceBail >= bailRecoveryInterval {
                tokensSinceBail = 0
                consecutiveZeroRounds = 0  // allow a fresh probe round
                return true
            }
            return false
        }
        // Per-run override for the rolling-acceptance auto-bail floor.
        // Default `fallbackThreshold` is 0.0 (never-bail) — that helps
        // FLy top-K paths where intermittent accept still pays. For
        // narrative free-form, drafter cost > 1-accept gain, so set
        // `MTP_FALLBACK_THRESHOLD=0.25` to let MTP fall back to T=1.
        let effectiveFloor: Double
        if let s = ProcessInfo.processInfo.environment["MTP_FALLBACK_THRESHOLD"],
           let v = Double(s) {
            effectiveFloor = v
        } else {
            effectiveFloor = fallbackThreshold
        }
        return rollingAcceptance >= effectiveFloor
    }

    /// No-op kept for ABI compatibility with engine that calls it after bail.
    public func didBypass() {}
}
