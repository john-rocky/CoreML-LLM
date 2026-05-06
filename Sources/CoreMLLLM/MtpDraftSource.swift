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

    /// Rolling acceptance rate for adaptive fallback.
    private(set) public var rollingAcceptance: Double = 1.0
    private let rollingAlpha: Double = 0.05
    public var fallbackThreshold: Double = 0.30

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
              let projActOut = output.featureValue(for: "proj_act_out")?.multiArrayValue
        else {
            throw SpeculativeError.verifyFailed("MTP drafter missing outputs")
        }
        let tokenId: Int32 = topKIds.dataPointer
            .bindMemory(to: Int32.self, capacity: 1)
            .pointee
        return (tokenId, projActOut)
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
        return rollingAcceptance >= fallbackThreshold
    }
}
