//
//  SpeculativeLoop.swift
//  CoreMLLLM
//
//  EAGLE-3 speculative decoding loop for iPhone ANE.
//
//  Integration: this file is self-contained and does NOT depend on
//  ChunkedEngine directly. The target-side operations (multi-layer hidden
//  fetch, K-token verify, commit) are expressed as a protocol,
//  `SpeculativeTarget`. ChunkedEngine should be made to conform to this
//  protocol as part of the bench session's integration work.
//
//  See docs/experiments/EAGLE3_DEPLOY.md for the full contract.
//

import CoreML
import Foundation

public enum SpeculativeError: Error {
    case missingModel(String)
    case verifyFailed(String)
    case assetMissing(String)
}

/// Target-side operations needed by the speculative decoding loop.
/// Implement this on whatever class owns the target chunks (ChunkedEngine).
public protocol SpeculativeTarget: AnyObject {
    /// Hidden states at the specified layer indices from the most recent
    /// decode step. Layer order matches the request order. Each array is
    /// shape (1, 1, H) fp16.
    func lastHiddenMulti(at layerIndices: [Int]) throws -> [MLMultiArray]

    /// Run verify chunks (EnumeratedShapes seq_dim = K) on `candidates`.
    /// MUST NOT commit the KV cache; the caller decides which prefix to accept.
    /// Returns the target's argmax at each of the K positions.
    func verifyCandidates(_ candidates: [Int32], K: Int) throws -> [Int32]

    /// Variant that also returns top-N (tokenID, logit) per verified position.
    /// Used by tolerance-based acceptance. Default implementation just calls
    /// `verifyCandidates` and wraps each argmax in a single-entry top list —
    /// conforming types that have cheap access to full logits should override
    /// with the true top-N extraction.
    func verifyCandidatesTopN(_ candidates: [Int32], K: Int, topN: Int)
        throws -> (argmax: [Int32], topN: [[Int32]])

    /// Commit `tokens` to the running KV cache. After this call, subsequent
    /// decode steps MUST see position advanced by `tokens.count`.
    func commitAccepted(_ tokens: [Int32]) throws
}

extension SpeculativeTarget {
    public func verifyCandidatesTopN(_ candidates: [Int32], K: Int, topN: Int)
        throws -> (argmax: [Int32], topN: [[Int32]])
    {
        let ax = try verifyCandidates(candidates, K: K)
        return (ax, ax.map { [$0] })
    }
}

public final class SpeculativeLoop {
    // MARK: - Assets

    /// Fusion: (h_low, h_mid, h_high) -> h_fused. Called once per drafting burst.
    public let fusion: MLModel

    /// Draft: (h_prev, e_next) -> (h_out, token:int32, logit:fp16). Called K times.
    public let draft: MLModel

    /// K tokens proposed per draft burst (fixed for v1; dynamic tree in v2).
    public let K: Int

    /// Target fusion layer indices (must match `eagle3_config.json` used for training).
    public let fusionLayers: [Int]

    /// Token embedding scale = sqrt(hidden). Matches training.
    public let embedScale: Float

    /// Rolling acceptance rate, used for adaptive fallback to K=1.
    private(set) public var rollingAcceptance: Double = 1.0
    private let rollingAlpha: Double = 0.05

    /// Disable speculative path when rolling acceptance drops below this.
    public var fallbackThreshold: Double = 0.30

    /// Tolerance for accepting a draft proposal: accept if it appears in the
    /// target's top-`tolerance` predictions at that position (1 = strict
    /// argmax match). Higher values relax matching to salvage bursts when
    /// fp16 on-device numerics flip argmax on tight-margin positions but the
    /// intended token is still near the top. Set via env var
    /// LLM_EAGLE3_TOLERANCE (default 1).
    public var tolerance: Int = {
        Int(ProcessInfo.processInfo.environment["LLM_EAGLE3_TOLERANCE"] ?? "") ?? 1
    }()

    // MARK: - Init

    public init(
        fusionURL: URL,
        draftURL: URL,
        K: Int = 3,
        fusionLayers: [Int],
        embedScale: Float,
        configuration: MLModelConfiguration? = nil
    ) throws {
        let cfg = configuration ?? {
            let c = MLModelConfiguration()
            c.computeUnits = .cpuAndNeuralEngine
            return c
        }()
        guard FileManager.default.fileExists(atPath: fusionURL.path) else {
            throw SpeculativeError.assetMissing(fusionURL.lastPathComponent)
        }
        guard FileManager.default.fileExists(atPath: draftURL.path) else {
            throw SpeculativeError.assetMissing(draftURL.lastPathComponent)
        }
        self.fusion = try MLModel(contentsOf: fusionURL, configuration: cfg)
        self.draft = try MLModel(contentsOf: draftURL, configuration: cfg)
        self.K = K
        self.fusionLayers = fusionLayers
        self.embedScale = embedScale
    }

    // MARK: - Public entry point

    /// Execute one decoding burst. Returns accepted tokens (≥ 1, ≤ K+1).
    ///
    /// - Parameters:
    ///   - target: the target model interface (typically ChunkedEngine).
    ///   - tTokNext: target's own argmax for the next position, taken from
    ///               the previous decode step's argmax output.
    ///   - tokenEmbed: closure producing `(1, 1, H) fp16` = embed(token) * embedScale.
    public func drawBurst(
        target: SpeculativeTarget,
        tTokNext: Int32,
        tokenEmbed: (Int32) throws -> MLMultiArray
    ) throws -> [Int32] {

        // Verbose one-shot diagnostic of the very first burst only, to let us
        // correlate on-device zero-accept with the actual values flowing
        // through fusion → draft → verify.
        let verbose = (burstCount < 2) || (ProcessInfo.processInfo
            .environment["LLM_EAGLE3_VERBOSE_BURST"] != nil)

        // 1. Fetch multi-layer hidden states at fusionLayers from the target's
        //    last decode step.
        let hs = try target.lastHiddenMulti(at: fusionLayers)
        guard hs.count == fusionLayers.count else {
            throw SpeculativeError.verifyFailed(
                "target returned \(hs.count) hiddens, expected \(fusionLayers.count)")
        }

        if verbose {
            for (i, h) in hs.enumerated() {
                let p = h.dataPointer.bindMemory(to: UInt16.self, capacity: 8)
                let vals = (0..<8).map { fp16BitsToFloat(p[$0]) }
                let norm = hiddenL2Norm(h)
                print(String(format: "[Spec/dbg] h_fuse_in[%d] L%d  first8=%@  |h|=%.3f",
                             i, fusionLayers[i], vals.description, norm))
            }
        }

        // 2. Fuse them. Names must match eagle3_fusion.mlpackage I/O contract.
        var fusionInputs: [String: Any] = [:]
        let names = ["h_low", "h_mid", "h_high"]
        for (i, name) in names.enumerated() where i < hs.count {
            fusionInputs[name] = hs[i]
        }
        let fusionIn = try MLDictionaryFeatureProvider(dictionary: fusionInputs)
        let fusionOut = try fusion.prediction(from: fusionIn)
        guard let hFused = fusionOut.featureValue(for: "h_fused")?.multiArrayValue else {
            throw SpeculativeError.verifyFailed("fusion missing h_fused output")
        }
        if verbose {
            let p = hFused.dataPointer.bindMemory(to: UInt16.self, capacity: 8)
            let vals = (0..<8).map { fp16BitsToFloat(p[$0]) }
            print(String(format: "[Spec/dbg] h_fused     first8=%@  |h|=%.3f",
                         vals.description, hiddenL2Norm(hFused)))
        }

        // 3. Draft K tokens autoregressively.
        var hPrev: MLMultiArray = hFused
        var eNext: MLMultiArray = try tokenEmbed(tTokNext)
        if verbose {
            let p = eNext.dataPointer.bindMemory(to: UInt16.self, capacity: 8)
            let vals = (0..<8).map { fp16BitsToFloat(p[$0]) }
            print(String(format: "[Spec/dbg] embed(tTokNext=%d) first8=%@  |e|=%.3f",
                         tTokNext, vals.description, hiddenL2Norm(eNext)))
        }
        var proposals: [Int32] = []
        proposals.reserveCapacity(K)

        for k in 0..<K {
            let draftIn = try MLDictionaryFeatureProvider(dictionary: [
                "h_prev": hPrev,
                "e_next": eNext
            ])
            let draftOut = try draft.prediction(from: draftIn)
            guard
                let hOut = draftOut.featureValue(for: "h_out")?.multiArrayValue,
                let tok = draftOut.featureValue(for: "token")?.multiArrayValue
            else {
                throw SpeculativeError.verifyFailed("draft missing outputs")
            }
            let pred: Int32 = tok.dataPointer
                .bindMemory(to: Int32.self, capacity: 1)
                .pointee
            proposals.append(pred)
            if verbose {
                let p = hOut.dataPointer.bindMemory(to: UInt16.self, capacity: 8)
                let vals = (0..<8).map { fp16BitsToFloat(p[$0]) }
                print(String(format: "[Spec/dbg] draft step %d → token=%d h_out[0..8]=%@  |h|=%.3f",
                             k, pred, vals.description, hiddenL2Norm(hOut)))
            }
            hPrev = hOut
            eNext = try tokenEmbed(pred)
        }

        // 4. Run target verify on [tTokNext, proposals[0..K-2]]. Target's argmax
        //    (and top-N for tolerance) at each of those K positions is what it
        //    "would" emit next.
        var verifyTokens = [tTokNext]
        verifyTokens.append(contentsOf: proposals.dropLast())
        let useTolerance = tolerance > 1
        let (targetArgmax, targetTopN): ([Int32], [[Int32]])
        if useTolerance {
            let r = try target.verifyCandidatesTopN(verifyTokens, K: K, topN: tolerance)
            targetArgmax = r.argmax
            targetTopN = r.topN
        } else {
            targetArgmax = try target.verifyCandidates(verifyTokens, K: K)
            targetTopN = targetArgmax.map { [$0] }
        }
        guard targetArgmax.count == K else {
            throw SpeculativeError.verifyFailed(
                "verify returned \(targetArgmax.count), expected \(K)")
        }
        if verbose {
            print("[Spec/dbg] verifyTokens = \(verifyTokens)")
            print("[Spec/dbg] targetArgmax = \(targetArgmax)")
            print("[Spec/dbg] proposals    = \(proposals)")
        }

        // 5. Accept prefix up to first disagreement. tTokNext is always accepted
        //    (it is target's own pick from the previous step's argmax).
        //    Tolerance: treat proposal as matching if it appears anywhere in
        //    target's top-N at that position.
        var accepted: [Int32] = [tTokNext]
        var matched = 0
        for k in 0..<K {
            let isMatch: Bool
            if useTolerance {
                isMatch = targetTopN[k].contains(proposals[k])
            } else {
                isMatch = proposals[k] == targetArgmax[k]
            }
            if isMatch {
                // Commit the draft proposal (which target also ranks highly)
                // rather than target's argmax — keeps output closer to draft's
                // distribution when tolerance > 1 is used as a relaxation.
                accepted.append(proposals[k])
                matched += 1
            } else {
                accepted.append(targetArgmax[k])
                break
            }
        }
        // If all K matched, no bonus token is appended here; the next burst
        // will re-derive multi-layer hiddens from the extended prefix.

        // 6. Commit accepted tokens to target's running state.
        try target.commitAccepted(accepted)

        // 7. Update rolling acceptance for adaptive fallback.
        let rate = Double(matched) / Double(K)
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance

        burstCount += 1
        if burstCount % 10 == 0 || burstCount <= 5 {
            let topNSize = useTolerance ? (targetTopN.first?.count ?? 1) : 1
            let inTop: Bool = useTolerance && (0..<K).contains { k in
                targetTopN[k].dropFirst().contains(proposals[k])
            }
            print(String(format: "[Spec] burst #%d accept=%d/%d emitted=%d rolling=%.1f%% tol=%d topNsize=%d lookAhead=%@",
                         burstCount, matched, K, accepted.count, rollingAcceptance * 100,
                         tolerance, topNSize, inTop ? "anyDraftInTopNexcArgmax" : "no"))
        }
        return accepted
    }

    private var burstCount: Int = 0

    /// Whether to use speculative path for the next burst.
    public var shouldSpeculate: Bool { rollingAcceptance >= fallbackThreshold }
}

// MARK: - Debug helpers (fp16 scalar dump + L2 norm)

private func fp16BitsToFloat(_ bits: UInt16) -> Float {
    // Use Swift's built-in Float16. Avoids the manual subnormal shift loop
    // that underflowed UInt32 `e` on subnormals (e.g., 0x8212 → e=0, mant≠0).
    Float(Float16(bitPattern: bits))
}

private func hiddenL2Norm(_ a: MLMultiArray) -> Float {
    let n = a.count
    let p = a.dataPointer.bindMemory(to: UInt16.self, capacity: n)
    var sumSq: Float = 0
    for i in 0..<n {
        let v = fp16BitsToFloat(p[i])
        sumSq += v * v
    }
    return sumSq.squareRoot()
}
