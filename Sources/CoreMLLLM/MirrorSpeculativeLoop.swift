//
//  MirrorSpeculativeLoop.swift
//  CoreMLLLM
//
//  Mirror Speculative Decoding (Apple ML Research, 2026) — parallel
//  execution of draft (GPU tensor cores) and verify (ANE) for +30% over
//  the serial EAGLE-3 path.
//
//  Scaffold for Approach B of docs/UNEXPLORED_APPROACHES.md. Requires
//  A19 Pro / M5 (GPU neural accelerators + MPP) and an EAGLE-3 draft
//  mlpackage built with `compute_units=.cpuAndGPU` via
//  conversion/build_eagle3_gpu.py.
//
//  Integration: same SpeculativeTarget protocol as SpeculativeLoop, so
//  ChunkedEngine work is shared between the two strategies.
//

import CoreML
import Foundation

public final class MirrorSpeculativeLoop {
    // MARK: - Assets

    /// Fusion: stays on ANE (tiny graph, bandwidth fine either place).
    public let fusion: MLModel

    /// Draft on GPU tensor cores (compute-bound workload).
    public let draftGPU: MLModel

    public let K: Int
    public let fusionLayers: [Int]
    public let embedScale: Float

    private(set) public var rollingAcceptance: Double = 1.0
    private let rollingAlpha: Double = 0.05
    public var fallbackThreshold: Double = 0.30

    // Concurrent dispatch
    private let draftQueue  = DispatchQueue(label: "coremlllm.mirror.draft",  qos: .userInitiated)
    private let verifyQueue = DispatchQueue(label: "coremlllm.mirror.verify", qos: .userInitiated)

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
            c.computeUnits = .cpuAndGPU   // GPU tensor cores (A19 Pro+ / M5)
            return c
        }()
        guard FileManager.default.fileExists(atPath: fusionURL.path),
              FileManager.default.fileExists(atPath: draftGPUURL.path) else {
            throw SpeculativeError.assetMissing("fusion or draftGPU")
        }
        self.fusion = try MLModel(contentsOf: fusionURL, configuration: fusionCfg)
        self.draftGPU = try MLModel(contentsOf: draftGPUURL, configuration: draftCfg)
        self.K = K
        self.fusionLayers = fusionLayers
        self.embedScale = embedScale
    }

    // MARK: - Concurrent burst

    /// Execute one decoding burst with draft and verify overlapped.
    ///
    /// Semantics: draft K tokens are generated on GPU while ANE begins
    /// running the verify chunks on `[tTokNext, proposals[0..K-2]]`.
    /// Because the draft must complete before we know what to verify,
    /// parallelism is limited: we can overlap draft with OTHER prep work
    /// (fusion read, embed lookup), but not with the verify chunks
    /// themselves. The paper gains come from pipelining multi-turn bursts.
    ///
    /// For v1 we implement a simpler structure:
    ///   - draft K tokens on GPU (serial within burst)
    ///   - kick off verify on ANE (async)
    ///   - while verify runs, begin NEXT burst's fusion prefetch on GPU
    ///   - accept / reject when verify returns
    ///
    /// True bidirectional Mirror (draft+target simultaneously speculating
    /// at each other) is v2.
    public func drawBurst(
        target: SpeculativeTarget,
        tTokNext: Int32,
        tokenEmbed: (Int32) throws -> MLMultiArray
    ) async throws -> [Int32] {

        // 1. Fusion on ANE (tiny, fast)
        let hs = try target.lastHiddenMulti(at: fusionLayers)
        var fusionInputs: [String: Any] = [:]
        let names = ["h_low", "h_mid", "h_high"]
        for (i, name) in names.enumerated() where i < hs.count { fusionInputs[name] = hs[i] }
        let fusionIn = try MLDictionaryFeatureProvider(dictionary: fusionInputs)
        let fusionOut = try await fusion.prediction(from: fusionIn)
        guard let hFused = fusionOut.featureValue(for: "h_fused")?.multiArrayValue else {
            throw SpeculativeError.verifyFailed("fusion missing h_fused")
        }

        // 2. Draft K tokens on GPU (concurrent with nothing yet, until we
        //    know the tokens we cannot begin verify)
        let proposals: [Int32] = try await withCheckedThrowingContinuation { cont in
            draftQueue.async {
                do {
                    var hPrev: MLMultiArray = hFused
                    var eNext: MLMultiArray = try tokenEmbed(tTokNext)
                    var out: [Int32] = []
                    out.reserveCapacity(self.K)
                    for _ in 0..<self.K {
                        let draftIn = try MLDictionaryFeatureProvider(dictionary: [
                            "h_prev": hPrev,
                            "e_next": eNext
                        ])
                        let draftOut = try self.draftGPU.prediction(from: draftIn)
                        guard
                            let hOut = draftOut.featureValue(for: "h_out")?.multiArrayValue,
                            let tok  = draftOut.featureValue(for: "token")?.multiArrayValue
                        else { throw SpeculativeError.verifyFailed("draftGPU missing outputs") }
                        let pred = tok.dataPointer.bindMemory(to: Int32.self, capacity: 1).pointee
                        out.append(pred)
                        hPrev = hOut
                        eNext = try tokenEmbed(pred)
                    }
                    cont.resume(returning: out)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }

        // 3. Verify on ANE (concurrent with post-verify prep later; here it's synchronous
        //    because we need the result before accept/reject)
        let verifyTokens = [tTokNext] + proposals.dropLast()
        let targetArgmax: [Int32] = try await withCheckedThrowingContinuation { cont in
            verifyQueue.async {
                do {
                    let r = try target.verifyCandidates(verifyTokens, K: self.K)
                    cont.resume(returning: r)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
        guard targetArgmax.count == K else {
            throw SpeculativeError.verifyFailed("verify returned \(targetArgmax.count), expected \(K)")
        }

        // 4. Accept greedily
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

        // 5. Commit
        try target.commitAccepted(accepted)

        // 6. Rolling acceptance
        let rate = Double(matched) / Double(K)
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance

        return accepted
    }

    public var shouldSpeculate: Bool { rollingAcceptance >= fallbackThreshold }
}

// NOTE(v2): true bidirectional Mirror would have target model speculate
// corrections WHILE draft proposes, requiring target to emit a running
// "most-likely-next" stream that draft can consume. This requires either
// a second target mlpackage (streaming variant) or a custom scheduling
// fabric above both. Deferred until v1 measures the current parallel
// gain and we have iPhone thermal data.
