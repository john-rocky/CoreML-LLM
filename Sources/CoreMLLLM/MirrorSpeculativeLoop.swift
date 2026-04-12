//
//  MirrorSpeculativeLoop.swift
//  CoreMLLLM
//
//  Mirror Speculative Decoding (Apple ML Research, 2026) — running the
//  EAGLE-3 draft on A19 Pro GPU tensor cores while target verify stays on
//  ANE. Scaffold for Approach B of docs/UNEXPLORED_APPROACHES.md.
//
//  v1 (this file): serial-within-burst. Draft runs on GPU, verify runs on
//    ANE, but a single burst waits on the draft to complete before verify
//    begins because verify consumes the proposals. The wall-clock win vs
//    the ANE-only EAGLE-3 comes from:
//      (a) Draft kernel executes on GPU tensor cores (compute-bound
//          Gemma4-style decoder layer + SwiGLU + lm_head at ~7.5 TFLOPS),
//          which is typically faster per burst than ANE's matmul path
//          at the small batch sizes a draft uses.
//      (b) ANE is not blocked by draft, so verify can start the moment
//          draft returns with no ANE queue contention.
//    Expected lift over serial EAGLE-3: ~+15-20% on iPhone 17 Pro.
//
//  v2 (future): cross-burst pipelining. Dispatch burst N+1's fusion+draft
//    on GPU while burst N's verify is still running on ANE. Needs an actor
//    to own the running KV state and a queue of speculation results. This
//    is where the Apple paper's +30% lands. Not implemented here.
//
//  Bidirectional Mirror (target emitting streaming corrections while draft
//  works) is v3 / research territory.
//
//  Integration: reuses the SpeculativeTarget protocol from SpeculativeLoop.swift
//  so ChunkedEngine only needs a single conformance to support both v0 (pure
//  ANE EAGLE-3) and v1 (GPU draft + ANE verify).
//

import CoreML
import Foundation

public final class MirrorSpeculativeLoop {
    // MARK: - Assets

    /// Fusion: stays on ANE (tiny graph, bandwidth fine either place).
    public let fusion: MLModel

    /// Draft on GPU tensor cores (compute-bound; A19 Pro+ / M5 for Neural Accelerators).
    public let draftGPU: MLModel

    public let K: Int
    public let fusionLayers: [Int]
    public let embedScale: Float

    private(set) public var rollingAcceptance: Double = 1.0
    private let rollingAlpha: Double = 0.05
    public var fallbackThreshold: Double = 0.30

    // MARK: - Init

    public init(
        fusionURL: URL,
        draftGPUURL: URL,
        K: Int = 3,
        fusionLayers: [Int],
        embedScale: Float,
        fusionConfiguration: MLModelConfiguration? = nil,
        draftGPUConfiguration: MLModelConfiguration? = nil
    ) throws {
        let fusionCfg = fusionConfiguration ?? {
            let c = MLModelConfiguration()
            c.computeUnits = .cpuAndNeuralEngine
            return c
        }()
        let draftCfg = draftGPUConfiguration ?? {
            let c = MLModelConfiguration()
            c.computeUnits = .cpuAndGPU   // A19 Pro+ neural accelerators (MPP, Xcode 26.1+)
            return c
        }()
        guard FileManager.default.fileExists(atPath: fusionURL.path),
              FileManager.default.fileExists(atPath: draftGPUURL.path) else {
            throw SpeculativeError.assetMissing("fusion or draftGPU mlpackage")
        }
        self.fusion = try MLModel(contentsOf: fusionURL, configuration: fusionCfg)
        self.draftGPU = try MLModel(contentsOf: draftGPUURL, configuration: draftCfg)
        self.K = K
        self.fusionLayers = fusionLayers
        self.embedScale = embedScale
    }

    // MARK: - Burst

    /// One decoding burst. Returns accepted tokens (length 1..K+1).
    ///
    /// Parameters:
    ///   - target:      the target model interface (ChunkedEngine conforming to SpeculativeTarget).
    ///   - tTokNext:    target's own argmax for the next position, taken from the previous step's argmax output.
    ///   - tokenEmbed:  closure producing `(1, 1, H) fp16` = embed(token) * embedScale.
    public func drawBurst(
        target: SpeculativeTarget,
        tTokNext: Int32,
        tokenEmbed: (Int32) throws -> MLMultiArray
    ) async throws -> [Int32] {

        // 1. Multi-layer hidden states at fusionLayers from the last decode step.
        let hs = try target.lastHiddenMulti(at: fusionLayers)
        guard hs.count == fusionLayers.count else {
            throw SpeculativeError.verifyFailed(
                "target returned \(hs.count) hiddens, expected \(fusionLayers.count)")
        }

        // 2. Fuse them via the fusion mlpackage (ANE).
        var fusionInputs: [String: Any] = [:]
        let names = ["h_low", "h_mid", "h_high"]
        for (i, name) in names.enumerated() where i < hs.count { fusionInputs[name] = hs[i] }
        let fusionIn = try MLDictionaryFeatureProvider(dictionary: fusionInputs)
        let fusionOut = try await fusion.prediction(from: fusionIn)
        guard let hFused = fusionOut.featureValue(for: "h_fused")?.multiArrayValue else {
            throw SpeculativeError.verifyFailed("fusion missing h_fused output")
        }

        // 3. Draft K tokens on GPU. Serial within burst — step k depends on
        //    step k-1's hidden. Parallelism with verify is not possible
        //    inside a single burst because verify consumes the proposals.
        //    Cross-burst pipelining is v2.
        var hPrev: MLMultiArray = hFused
        var eNext: MLMultiArray = try tokenEmbed(tTokNext)
        var proposals: [Int32] = []
        proposals.reserveCapacity(K)
        for _ in 0..<K {
            let draftIn = try MLDictionaryFeatureProvider(dictionary: [
                "h_prev": hPrev,
                "e_next": eNext
            ])
            let draftOut = try await draftGPU.prediction(from: draftIn)
            guard
                let hOut = draftOut.featureValue(for: "h_out")?.multiArrayValue,
                let tok = draftOut.featureValue(for: "token")?.multiArrayValue
            else {
                throw SpeculativeError.verifyFailed("draftGPU missing outputs")
            }
            let pred: Int32 = tok.dataPointer
                .bindMemory(to: Int32.self, capacity: 1)
                .pointee
            proposals.append(pred)
            hPrev = hOut
            eNext = try tokenEmbed(pred)
        }

        // 4. Run target verify on [tTokNext, proposals[0..K-2]] (ANE).
        var verifyTokens = [tTokNext]
        verifyTokens.append(contentsOf: proposals.dropLast())
        let targetArgmax = try target.verifyCandidates(verifyTokens, K: K)
        guard targetArgmax.count == K else {
            throw SpeculativeError.verifyFailed(
                "verify returned \(targetArgmax.count), expected \(K)")
        }

        // 5. Greedy accept up to first disagreement.
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
        // All-matched case: no bonus token here; next burst re-derives hiddens.

        // 6. Commit to target's running state.
        try target.commitAccepted(accepted)

        // 7. Rolling acceptance for adaptive fallback.
        let rate = Double(matched) / Double(K)
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance

        return accepted
    }

    /// Whether to use the speculative path for the next burst.
    public var shouldSpeculate: Bool { rollingAcceptance >= fallbackThreshold }
}
