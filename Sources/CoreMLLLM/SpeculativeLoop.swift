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
//  See docs/EAGLE3_DEPLOY.md for the full contract.
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

    /// Commit `tokens` to the running KV cache. After this call, subsequent
    /// decode steps MUST see position advanced by `tokens.count`.
    func commitAccepted(_ tokens: [Int32]) throws
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

    /// Emits proposals + target-argmax comparison for the first N bursts after
    /// instantiation. Off after the counter decays.
    public var debugBurstsRemaining: Int = 3

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

        // 1. Fetch multi-layer hidden states at fusionLayers from the target's
        //    last decode step.
        let hs = try target.lastHiddenMulti(at: fusionLayers)
        guard hs.count == fusionLayers.count else {
            throw SpeculativeError.verifyFailed(
                "target returned \(hs.count) hiddens, expected \(fusionLayers.count)")
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

        // 3. Draft K tokens autoregressively.
        var hPrev: MLMultiArray = hFused
        var eNext: MLMultiArray = try tokenEmbed(tTokNext)
        var proposals: [Int32] = []
        proposals.reserveCapacity(K)

        for _ in 0..<K {
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
            hPrev = hOut
            eNext = try tokenEmbed(pred)
        }

        // 4. Run target verify on [tTokNext, proposals[0..K-2]]. Target's argmax
        //    at each of those K positions is what it "would" emit next.
        var verifyTokens = [tTokNext]
        verifyTokens.append(contentsOf: proposals.dropLast())
        let targetArgmax = try target.verifyCandidates(verifyTokens, K: K)
        guard targetArgmax.count == K else {
            throw SpeculativeError.verifyFailed(
                "verify returned \(targetArgmax.count), expected \(K)")
        }

        if debugBurstsRemaining > 0 {
            debugBurstsRemaining -= 1
            print("[SpecDbg] tTokNext=\(tTokNext)")
            print("[SpecDbg]   verifyTokens = \(verifyTokens)")
            print("[SpecDbg]   proposals    = \(proposals)")
            print("[SpecDbg]   targetArgmax = \(targetArgmax)")
            let matches = (0..<K).map { proposals[$0] == targetArgmax[$0] }
            print("[SpecDbg]   matches      = \(matches)")
        }

        // 5. Accept prefix up to first disagreement. tTokNext is always accepted
        //    (it is target's own pick from the previous step's argmax).
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
        // If all K matched, no bonus token is appended here; the next burst
        // will re-derive multi-layer hiddens from the extended prefix.

        // 6. Commit accepted tokens to target's running state.
        try target.commitAccepted(accepted)

        // 7. Update rolling acceptance for adaptive fallback.
        let rate = Double(matched) / Double(K)
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance

        return accepted
    }

    /// Whether to use speculative path for the next burst.
    public var shouldSpeculate: Bool { rollingAcceptance >= fallbackThreshold }
}
