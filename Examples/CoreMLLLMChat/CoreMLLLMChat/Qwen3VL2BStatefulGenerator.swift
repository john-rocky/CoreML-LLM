// Qwen3-VL 2B text-only generator — stateful (MLState + slice_update)
// path for Phase 1. Runs alongside the existing v1.4.0
// Qwen3VL2BGenerator (which keeps the vision + batched-prefill path).
//
// Artifacts on disk:
//   Documents/Models/qwen3-vl-2b-stateful/qwen3_vl_2b_stateful_chunks/
//     embed_weight.bin
//     chunk_0..chunk_N.mlpackage / .mlmodelc  (MLState-based)
//     chunk_head.mlpackage / .mlmodelc
//
// Per chunk inputs (matches conversion/build_qwen3_vl_2b_stateful_chunks.py):
//   hidden_in   (1, 1, 2048) fp16
//   cos, sin    (1, 1, 128)  fp16
//   causal_mask (1, 1, 1, max_seq) fp16 — -1e4 for slots > current_pos
//   current_pos (1,) int32
//   state       kv_cache_0 — managed by Core ML, Swift calls makeState()
//
// Text prefill runs through this same decode path one token at a time
// (simpler; batched prefill is a later commit via multifunction).

import Accelerate
import CoreML
import Foundation


@Observable
final class Qwen3VL2BStatefulGenerator {
    struct Config {
        let maxSeq: Int
        let vocab: Int
        let hiddenSize: Int
        let numLayers: Int
        let numKVHeads: Int
        let headDim: Int
        let numBodyChunks: Int
        let layersPerChunk: Int
        let ropeTheta: Float
        let computeUnits: MLComputeUnits

        static let defaultFourChunk = Config(
            maxSeq: 2048, vocab: 151936,
            hiddenSize: 2048, numLayers: 28,
            numKVHeads: 8, headDim: 128,
            numBodyChunks: 4, layersPerChunk: 7,
            ropeTheta: 5_000_000,
            computeUnits: .cpuAndNeuralEngine)
    }

    var status = "Idle"
    var running = false
    var outputText = ""
    var stats = ""

    private var cfg = Config.defaultFourChunk

    // Models + per-generate state handles (one per chunk).
    private var bodyChunks: [MLModel] = []
    private var headChunk: MLModel?

    // Embed sidecar (mmap'd fp16 vocab x hidden).
    private var embedMmapBase: UnsafeMutableRawPointer?
    private var embedMmapPtr: UnsafePointer<UInt16>?
    private var embedMmapLen: Int = 0
    private var embedMmapFD: Int32 = -1

    // Reusable per-step buffers.
    private var reusableHidden: MLMultiArray!
    private var reusableCos: MLMultiArray!
    private var reusableSin: MLMultiArray!
    private var reusableMask: MLMultiArray!
    private var reusablePos: MLMultiArray!
    private var fvHidden: MLFeatureValue!
    private var fvCos: MLFeatureValue!
    private var fvSin: MLFeatureValue!
    private var fvMask: MLFeatureValue!
    private var fvPos: MLFeatureValue!

    private var cosTable: [Float] = []
    private var sinTable: [Float] = []

    init(cfg: Config = .defaultFourChunk) {
        self.cfg = cfg
        cosTable = buildRope(isCos: true)
        sinTable = buildRope(isCos: false)
        allocBuffers()
    }

    deinit { releaseEmbedMmap() }

    // MARK: - Buffer allocation

    private func allocBuffers() {
        reusableHidden = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.hiddenSize)], dataType: .float16)
        reusableCos = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        reusableSin = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        reusableMask = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.maxSeq)], dataType: .float16)
        reusablePos = try! MLMultiArray(shape: [1], dataType: .int32)
        fvHidden = MLFeatureValue(multiArray: reusableHidden)
        fvCos = MLFeatureValue(multiArray: reusableCos)
        fvSin = MLFeatureValue(multiArray: reusableSin)
        fvMask = MLFeatureValue(multiArray: reusableMask)
        fvPos = MLFeatureValue(multiArray: reusablePos)
    }

    // MARK: - RoPE (text-only 1D, matches existing Qwen3VL2BGenerator)

    private func buildRope(isCos: Bool) -> [Float] {
        let d = cfg.headDim
        let half = d / 2
        var out = [Float](repeating: 0, count: cfg.maxSeq * d)
        for p in 0..<cfg.maxSeq {
            for i in 0..<half {
                let theta = powf(cfg.ropeTheta, Float(2 * i) / Float(d))
                let a = Float(p) / theta
                let v = isCos ? cosf(a) : sinf(a)
                out[p * d + i] = v
                out[p * d + i + half] = v
            }
        }
        return out
    }

    private func fillCosSin(forPosition pos: Int) {
        let d = cfg.headDim
        let clamped = min(max(pos, 0), cfg.maxSeq - 1)
        let cosDst = reusableCos.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sinDst = reusableSin.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<d {
            cosDst[i] = Float16(cosTable[clamped * d + i]).bitPattern
            sinDst[i] = Float16(sinTable[clamped * d + i]).bitPattern
        }
    }

    private func fillCausalMask(forPosition pos: Int) {
        let dst = reusableMask.dataPointer.assumingMemoryBound(to: UInt16.self)
        // fp16(0.0) = 0x0000; fp16(-1e4) = 0xF0FF? Actually -10000 in fp16
        // is approximated; use Float16(-1e4).bitPattern.
        let neg1e4 = Float16(-10_000.0).bitPattern
        let p = min(max(pos, 0), cfg.maxSeq - 1)
        for i in 0..<cfg.maxSeq {
            dst[i] = (i <= p) ? 0 : neg1e4
        }
    }

    private func setCurrentPos(_ pos: Int) {
        let p = reusablePos.dataPointer.assumingMemoryBound(to: Int32.self)
        p[0] = Int32(pos)
    }

    // MARK: - Embed

    private func mmapEmbedWeight(url: URL) throws {
        releaseEmbedMmap()
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "Qwen3VL2BStateful", code: 30,
                userInfo: [NSLocalizedDescriptionKey:
                    "failed to open embed_weight.bin at \(url.path)"])
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw NSError(domain: "Qwen3VL2BStateful", code: 31,
                userInfo: [NSLocalizedDescriptionKey: "fstat failed"])
        }
        let size = Int(st.st_size)
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "Qwen3VL2BStateful", code: 32,
                userInfo: [NSLocalizedDescriptionKey: "mmap failed"])
        }
        embedMmapBase = base
        embedMmapLen = size
        embedMmapFD = fd
        embedMmapPtr = UnsafePointer(base.assumingMemoryBound(to: UInt16.self))
        madvise(base, size, MADV_RANDOM)
    }

    private func releaseEmbedMmap() {
        if let base = embedMmapBase, embedMmapLen > 0 { munmap(base, embedMmapLen) }
        if embedMmapFD >= 0 { close(embedMmapFD) }
        embedMmapBase = nil; embedMmapPtr = nil
        embedMmapLen = 0; embedMmapFD = -1
    }

    private func embedLookup(token: Int32) {
        guard let ptr = embedMmapPtr else { return }
        let src = ptr + Int(token) * cfg.hiddenSize
        let dst = reusableHidden.dataPointer.assumingMemoryBound(to: UInt16.self)
        memcpy(dst, src, cfg.hiddenSize * 2)
    }

    // MARK: - Resolve model directory

    var modelFolderOverride: URL?

    private func resolveURLs() -> (body: [URL], head: URL, embed: URL)? {
        let subdir = "qwen3_vl_2b_stateful_chunks"
        let fm = FileManager.default

        func resolveOne(_ dir: URL, _ base: String) -> URL? {
            let mlc = dir.appendingPathComponent("\(base).mlmodelc")
            if fm.fileExists(atPath: mlc.path) { return mlc }
            let pkg = dir.appendingPathComponent("\(base).mlpackage")
            if fm.fileExists(atPath: pkg.path) {
                return try? MLModel.compileModel(at: pkg)
            }
            return nil
        }

        func resolve(_ base: URL) -> (body: [URL], head: URL, embed: URL)? {
            let dir = base.appendingPathComponent(subdir)
            let embed = dir.appendingPathComponent("embed_weight.bin")
            guard fm.fileExists(atPath: embed.path) else { return nil }
            var bodies: [URL] = []
            for ci in 0..<cfg.numBodyChunks {
                guard let u = resolveOne(dir, "chunk_\(ci)") else { return nil }
                bodies.append(u)
            }
            guard let h = resolveOne(dir, "chunk_head") else { return nil }
            return (bodies, h, embed)
        }

        if let folder = modelFolderOverride, let r = resolve(folder) { return r }
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                  appropriateFor: nil, create: false),
           let r = resolve(docs) { return r }
        let defaultFolder = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                         appropriateFor: nil, create: false)
        return defaultFolder.flatMap { resolve($0.appendingPathComponent("Models/qwen3-vl-2b-stateful")) }
    }

    // MARK: - Load

    func load() throws {
        guard let r = resolveURLs() else {
            throw NSError(domain: "Qwen3VL2BStateful", code: 40,
                userInfo: [NSLocalizedDescriptionKey:
                    "qwen3_vl_2b_stateful_chunks/{embed_weight.bin, chunk_0..N, chunk_head} "
                    + "not found in Documents/ or Documents/Models/qwen3-vl-2b-stateful/"])
        }
        try mmapEmbedWeight(url: r.embed)
        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits
        bodyChunks = try r.body.map { try MLModel(contentsOf: $0, configuration: mcfg) }
        headChunk = try MLModel(contentsOf: r.head, configuration: mcfg)
        status = "Loaded: \(bodyChunks.count) chunks + head, units=\(cfg.computeUnits.rawValue)"
    }

    // MARK: - Step

    /// Build a feature provider that returns the 5 chunk inputs by name.
    /// `fvHiddenSrc` supplies "hidden_in" (either embed output for chunk 0
    /// or the previous chunk's "hidden" MLMultiArray for chunk i>0).
    private final class StatefulBodyProvider: NSObject, MLFeatureProvider {
        let fvHiddenIn: MLFeatureValue
        let fvCos: MLFeatureValue
        let fvSin: MLFeatureValue
        let fvMask: MLFeatureValue
        let fvPos: MLFeatureValue
        let featureNames: Set<String> = [
            "hidden_in", "cos", "sin", "causal_mask", "current_pos"
        ]
        init(hiddenIn: MLFeatureValue, cos: MLFeatureValue, sin: MLFeatureValue,
             mask: MLFeatureValue, pos: MLFeatureValue) {
            self.fvHiddenIn = hiddenIn; self.fvCos = cos; self.fvSin = sin
            self.fvMask = mask; self.fvPos = pos
            super.init()
        }
        func featureValue(for n: String) -> MLFeatureValue? {
            switch n {
            case "hidden_in":   return fvHiddenIn
            case "cos":         return fvCos
            case "sin":         return fvSin
            case "causal_mask": return fvMask
            case "current_pos": return fvPos
            default: return nil
            }
        }
    }

    private func stepPredict(token: Int32, position: Int,
                              states: [MLState]) async throws -> Int32 {
        embedLookup(token: token)
        fillCosSin(forPosition: position)
        fillCausalMask(forPosition: position)
        setCurrentPos(position)

        var hiddenFV = fvHidden!
        let opts = MLPredictionOptions()
        for (ci, chunk) in bodyChunks.enumerated() {
            let prov = StatefulBodyProvider(hiddenIn: hiddenFV,
                                             cos: fvCos, sin: fvSin,
                                             mask: fvMask, pos: fvPos)
            let out = try await chunk.prediction(
                from: prov, using: states[ci], options: opts)
            guard let fv = out.featureValue(for: "hidden") else {
                throw NSError(domain: "Qwen3VL2BStateful", code: 50,
                    userInfo: [NSLocalizedDescriptionKey:
                        "chunk_\(ci) did not emit 'hidden'"])
            }
            hiddenFV = fv
        }
        // Head: one-input feature provider
        let head = headChunk!
        let headProv = try MLDictionaryFeatureProvider(
            dictionary: ["hidden_in": hiddenFV])
        let out = try await head.prediction(from: headProv, options: opts)
        guard let fv = out.featureValue(for: "next_token"),
              let arr = fv.multiArrayValue
        else {
            throw NSError(domain: "Qwen3VL2BStateful", code: 51,
                userInfo: [NSLocalizedDescriptionKey: "head did not emit 'next_token'"])
        }
        return arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    // MARK: - Generate

    func generate(inputIds: [Int32], maxNewTokens: Int = 64) async throws -> [Int32] {
        guard !bodyChunks.isEmpty, headChunk != nil else {
            throw NSError(domain: "Qwen3VL2BStateful", code: 60,
                userInfo: [NSLocalizedDescriptionKey: "not loaded"])
        }
        let states = bodyChunks.map { $0.makeState() }

        var position = 0
        var lastToken: Int32 = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        var prefillEnd: CFAbsoluteTime = t0

        // Recurrent prefill (T=1 per step — batched prefill via multifunction later).
        for (i, tok) in inputIds.enumerated() {
            _ = try await stepPredict(token: tok, position: position, states: states)
            lastToken = tok
            position += 1
            if i == inputIds.count - 1 {
                prefillEnd = CFAbsoluteTimeGetCurrent()
            }
        }

        // Decode: the last prefill step already produced next_token. To
        // keep stepPredict uniform we redo the last-prefill-step here so
        // we grab the token id into decoded[]. Small waste on a long
        // prompt; fixed later when batched prefill lands.
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var decoded: [Int32] = []
        for _ in 0..<maxNewTokens {
            let next = try await stepPredict(token: lastToken, position: position,
                                             states: states)
            decoded.append(next)
            lastToken = next
            position += 1
            if position >= cfg.maxSeq { break }
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        let prefillMs = (prefillEnd - t0) * 1000
        let decodeMs = (t1 - decodeStart) * 1000
        let decodeTokPerS = Double(decoded.count) / ((t1 - decodeStart))
        stats = String(format:
            "prefill %d tok in %.1fms (%.1f tok/s) | decode %d tok in %.1fms (%.1f tok/s)",
            inputIds.count, prefillMs,
            Double(inputIds.count) / max(prefillEnd - t0, 1e-3),
            decoded.count, decodeMs, decodeTokPerS)
        return decoded
    }
}
