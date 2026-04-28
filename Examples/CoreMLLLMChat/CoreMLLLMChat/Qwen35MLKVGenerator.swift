// Qwen3.5 MLKV decode generator — KV cache via Core ML MLState +
// slice_update; SSM (Gated DeltaNet) state through classic input/output.
//
// Disk layout (under Documents/ or Documents/Models/<model>/):
//   qwen3_5_(0_8b|2b)_decode_chunks_mlkv/
//     embed_weight.bin    raw fp16 (vocab × hidden) — Swift mmaps
//     chunk_a..chunk_d.mlpackage / .mlmodelc
//
// Per chunk inputs (matches build_qwen35_decode_chunks_mlkv.py):
//   hidden_in     (1, 1, hidden) fp16
//   cos, sin      (1, 1, rotary_dim) fp16
//   causal_mask   (1, 1, 1, max_seq) fp16   -1e4 for slots > current_pos
//   current_pos   (1,) int32
//   conv_state_<i> + rec_state_<i> for each linear_attention layer i
//                                       in this chunk's range
//   state         kv_cache  (managed by Core ML, makeState() per chunk)
//
// chunk_a..c output:
//   hidden                       (1, 1, hidden) fp16
//   new_conv_state_<i>, new_rec_state_<i> for each lin layer in chunk
//
// chunk_d output:
//   next_token                    (1, 1) int32   in-graph TopK[k=1]
//   new_conv_state_<i>, new_rec_state_<i> for each lin layer in chunk
//
// Recurrent prefill: each prompt token runs through the same decode
// step (no separate batched prefill mlpackage); state accumulates.

import Accelerate
import CoreML
import Foundation


@Observable
final class Qwen35MLKVGenerator {
    struct Config {
        let maxSeq: Int
        let vocab: Int
        let hiddenSize: Int
        let numLayers: Int
        let numKVHeads: Int
        let headDim: Int
        let rotaryDim: Int          // head_dim * 0.25
        let numChunks: Int
        let computeUnits: MLComputeUnits

        static let default0_8B = Config(
            maxSeq: 2048, vocab: 248320,
            hiddenSize: 1024, numLayers: 24,
            numKVHeads: 2, headDim: 256,
            rotaryDim: 64,
            numChunks: 4,
            computeUnits: .cpuAndNeuralEngine)

        static let default2B = Config(
            maxSeq: 2048, vocab: 248320,
            hiddenSize: 2048, numLayers: 24,
            numKVHeads: 2, headDim: 256,
            rotaryDim: 64,
            numChunks: 4,
            computeUnits: .cpuAndNeuralEngine)
    }

    var status = "Idle"
    var running = false
    var generatedIds: [Int32] = []
    var prefillMs: Double = 0
    var decodeMsAvg: Double = 0
    var tokensPerSecond: Double = 0

    @ObservationIgnored private var cfg: Config
    @ObservationIgnored private var bodyChunks: [MLModel] = []
    @ObservationIgnored private var states: [MLState] = []
    @ObservationIgnored private var chunkLinIndices: [[Int]] = []  // per chunk, abs layer ids of SSM layers
    @ObservationIgnored private var chunkBoundaries: [(Int, Int)] = []

    @ObservationIgnored private var modelFolderOverride: URL?

    // Reusable per-step buffers (avoid alloc churn).
    @ObservationIgnored private var reusableHidden: MLMultiArray!
    @ObservationIgnored private var reusableCos: MLMultiArray!
    @ObservationIgnored private var reusableSin: MLMultiArray!
    @ObservationIgnored private var reusableMask: MLMultiArray!
    @ObservationIgnored private var reusableCurPos: MLMultiArray!
    // SSM state per layer — kept on Swift side, copied to/from Core ML each step.
    // Key = absolute layer id; value = pair (conv, rec).
    @ObservationIgnored private var ssmConv: [Int: MLMultiArray] = [:]
    @ObservationIgnored private var ssmRec: [Int: MLMultiArray] = [:]
    @ObservationIgnored private var fvHidden: MLFeatureValue!
    @ObservationIgnored private var fvCos: MLFeatureValue!
    @ObservationIgnored private var fvSin: MLFeatureValue!
    @ObservationIgnored private var fvMask: MLFeatureValue!
    @ObservationIgnored private var fvCurPos: MLFeatureValue!

    // RoPE precomputed tables.
    @ObservationIgnored private var cosTable: [Float] = []
    @ObservationIgnored private var sinTable: [Float] = []

    // Sampling parameters set by generate(); used in stepPredict's
    // sampleFromTopK call. Defaults break greedy emoji-loops.
    @ObservationIgnored private var samplingTemperature: Float = 0.7
    @ObservationIgnored private var samplingTopP: Float = 0.95
    @ObservationIgnored private var samplingRepPenalty: Float = 1.1
    /// Sliding window of recently generated tokens — used by the
    /// repetition penalty branch. 64-token window matches HF default.
    @ObservationIgnored private var recentTokens: [Int32] = []
    private let recentWindow = 64

    // mmap'd embed sidecar.
    @ObservationIgnored private var embedMmapPtr: UnsafePointer<UInt16>?
    @ObservationIgnored private var embedMmapBase: UnsafeMutableRawPointer?
    @ObservationIgnored private var embedMmapLen: Int = 0
    @ObservationIgnored private var embedMmapFD: Int32 = -1

    init(cfg: Config) {
        self.cfg = cfg
        self.cosTable = buildRope(isCos: true)
        self.sinTable = buildRope(isCos: false)
    }

    // MARK: Public API

    func setModelFolder(_ url: URL) {
        modelFolderOverride = url
    }

    func setComputeUnits(_ units: MLComputeUnits) {
        cfg = Config(
            maxSeq: cfg.maxSeq, vocab: cfg.vocab,
            hiddenSize: cfg.hiddenSize, numLayers: cfg.numLayers,
            numKVHeads: cfg.numKVHeads, headDim: cfg.headDim,
            rotaryDim: cfg.rotaryDim, numChunks: cfg.numChunks,
            computeUnits: units)
        bodyChunks = []
        states = []
        releaseEmbedMmap()
        status = "Idle (units changed, will reload)"
    }

    func load() throws {
        guard let r = resolveURLs() else {
            throw NSError(domain: "Qwen35MLKV", code: 40,
                userInfo: [NSLocalizedDescriptionKey:
                    "qwen3_5_*_decode_chunks_mlkv/{embed_weight.bin, chunk_a..d} not found"])
        }
        try mmapEmbedWeight(url: r.embed)
        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits
        bodyChunks = try r.body.map { try MLModel(contentsOf: $0, configuration: mcfg) }

        // Per-chunk SSM layer index lists.
        chunkBoundaries = chunkRanges(numLayers: cfg.numLayers, numChunks: cfg.numChunks)
        chunkLinIndices = chunkBoundaries.map { (s, e) in
            (s..<e).filter { $0 % 4 != 3 }
        }

        // Allocate per-step reusable buffers + SSM state buffers.
        try allocReusable()
        try allocSSMStates()

        // Fresh MLState per chunk — kv_cache zero-initialized by Core ML.
        states = bodyChunks.map { $0.makeState() }
        status = "Loaded MLKV bundle (\(unitsName(cfg.computeUnits)))"
    }

    /// Decode with sampling.
    /// chunk_d emits top-K (default 40) indices + values; this function
    /// applies temperature → repetition penalty → top-p → multinomial
    /// sample over the candidate set.
    /// - temperature 0: greedy (just take top-1 of the K candidates).
    /// - topK: ignored — fixed by the chunk_d build (K=40 in shipping).
    /// - repetitionPenalty: > 1 reduces logits of tokens in the recent
    ///   window (default last 64 tokens).
    /// - topP: nucleus filter (0.0..1.0). Default 0.95.
    @discardableResult
    func generate(inputIds: [Int32], maxNewTokens: Int = 64,
                  temperature: Float = 0.7,
                  topK: Int = 40,
                  topP: Float = 0.95,
                  repetitionPenalty: Float = 1.1,
                  eosTokenIds: Set<Int32> = [248044, 248045, 248046],
                  onToken: ((Int32) -> Void)? = nil) async throws -> [Int32]
    {
        self.samplingTemperature = temperature
        self.samplingTopP = topP
        self.samplingRepPenalty = repetitionPenalty
        self.recentTokens = []
        if bodyChunks.isEmpty { try load() }
        running = true
        defer { running = false }

        // Reset state for fresh generation.
        states = bodyChunks.map { $0.makeState() }
        try allocSSMStates()  // re-zero SSM state per layer
        generatedIds = []

        let opts = MLPredictionOptions()

        // Recurrent prefill.
        let t0Pre = Date()
        var nextToken: Int32 = -1
        for (t, tok) in inputIds.enumerated() {
            nextToken = try await stepPredict(token: tok, position: t, opts: opts)
        }
        prefillMs = Date().timeIntervalSince(t0Pre) * 1000.0

        // Decode.
        let S = inputIds.count
        let t0Dec = Date()
        var stepCount = 0
        var produced: [Int32] = []
        for step in 0..<maxNewTokens {
            let pos = S + step
            if pos >= cfg.maxSeq { break }
            produced.append(nextToken)
            generatedIds = produced
            stepCount += 1
            onToken?(nextToken)
            if eosTokenIds.contains(nextToken) { break }
            nextToken = try await stepPredict(token: nextToken, position: pos, opts: opts)
        }
        let dt = Date().timeIntervalSince(t0Dec)
        decodeMsAvg = stepCount > 0 ? (dt * 1000.0 / Double(stepCount)) : 0
        tokensPerSecond = stepCount > 0 ? Double(stepCount) / dt : 0
        status = String(format: "Done — %.1f tok/s decode, %.0f ms prefill",
                         tokensPerSecond, prefillMs)
        return produced
    }

    // MARK: Per-step

    private func stepPredict(token: Int32, position: Int,
                              opts: MLPredictionOptions) async throws -> Int32
    {
        // Embed lookup: copy one fp16 row from mmap'd embed_weight.bin
        // into reusableHidden.
        embedLookup(token: token)
        fillCosSin(forPosition: position)
        fillCausalMask(forPosition: position)
        setCurrentPos(Int32(position))

        var hiddenFV: MLFeatureValue = fvHidden!
        for ci in 0..<bodyChunks.count {
            let chunk = bodyChunks[ci]
            let st = states[ci]
            let isLast = ci == bodyChunks.count - 1
            let prov = try buildFeatureProvider(
                hiddenFV: hiddenFV, chunkIdx: ci)
            let out = try await chunk.prediction(from: prov, using: st, options: opts)

            // Update SSM state for layers in this chunk.
            for absI in chunkLinIndices[ci] {
                if let v = out.featureValue(for: "new_conv_state_\(absI)")?.multiArrayValue {
                    ssmConv[absI] = v
                }
                if let v = out.featureValue(for: "new_rec_state_\(absI)")?.multiArrayValue {
                    ssmRec[absI] = v
                }
            }
            if isLast {
                // chunk_d emits top_indices (1, 1, K) int32 + top_values
                // (1, 1, K) fp16 — Swift samples over the candidate set.
                guard let idxArr = out.featureValue(for: "top_indices")?.multiArrayValue,
                      let valArr = out.featureValue(for: "top_values")?.multiArrayValue else {
                    throw NSError(domain: "Qwen35MLKV", code: 50,
                        userInfo: [NSLocalizedDescriptionKey:
                            "chunk_d missing top_indices/top_values output"])
                }
                return sampleFromTopK(topIndices: idxArr, topValues: valArr)
            } else if let h = out.featureValue(for: "hidden") {
                hiddenFV = h
            }
        }
        return -1  // unreachable
    }

    /// Sample one token from chunk_d's top-K candidates.
    /// Pipeline: read top_indices/top_values into Swift arrays → apply
    /// repetition penalty (lower logits of tokens in `recentTokens`) →
    /// scale by `1/temperature` → softmax → top-p (nucleus) filter →
    /// multinomial sample. Updates `recentTokens` sliding window.
    private func sampleFromTopK(topIndices: MLMultiArray,
                                  topValues: MLMultiArray) -> Int32 {
        let K = topIndices.count
        let idxPtr = topIndices.dataPointer.assumingMemoryBound(to: Int32.self)
        let valPtr = topValues.dataPointer.assumingMemoryBound(to: UInt16.self)

        var indices = [Int32](repeating: 0, count: K)
        var logits  = [Float](repeating: 0, count: K)
        for i in 0..<K {
            indices[i] = idxPtr[i]
            let bits = valPtr[i]
            let h = withUnsafeBytes(of: bits) { $0.load(as: Float16.self) }
            logits[i] = Float(h)
        }

        // 1. Repetition penalty.
        if samplingRepPenalty > 1.0 && !recentTokens.isEmpty {
            let recent = Set(recentTokens)
            for i in 0..<K {
                if recent.contains(indices[i]) {
                    if logits[i] > 0 { logits[i] /= samplingRepPenalty }
                    else             { logits[i] *= samplingRepPenalty }
                }
            }
        }

        // 2. Temperature 0 = greedy over the (possibly rep-penalized) top-K.
        if samplingTemperature < 0.001 {
            var bestI = 0
            for i in 1..<K where logits[i] > logits[bestI] { bestI = i }
            updateRecent(indices[bestI])
            return indices[bestI]
        }

        // 3. Temperature scaling + softmax.
        let invT = 1.0 / samplingTemperature
        let maxLogit = logits.max() ?? 0
        var probs = logits.map { Float(exp(Double(($0 - maxLogit) * invT))) }
        let sumP = probs.reduce(0, +)
        if sumP > 0 { for i in 0..<K { probs[i] /= sumP } }

        // 4. Top-p (nucleus): sort by prob desc, keep until cumulative ≥ topP.
        let order = (0..<K).sorted { probs[$0] > probs[$1] }
        var cum: Float = 0
        var keep = Set<Int>()
        for idx in order {
            keep.insert(idx)
            cum += probs[idx]
            if cum >= samplingTopP { break }
        }
        var filtered = [Float](repeating: 0, count: K)
        for i in keep { filtered[i] = probs[i] }
        let fSum = filtered.reduce(0, +)
        if fSum > 0 { for i in 0..<K { filtered[i] /= fSum } }

        // 5. Multinomial sample.
        let r = Float.random(in: 0..<1)
        var cumP: Float = 0
        for i in 0..<K {
            cumP += filtered[i]
            if r < cumP {
                updateRecent(indices[i])
                return indices[i]
            }
        }
        updateRecent(indices[order[0]])
        return indices[order[0]]
    }

    private func updateRecent(_ tok: Int32) {
        recentTokens.append(tok)
        if recentTokens.count > recentWindow {
            recentTokens.removeFirst(recentTokens.count - recentWindow)
        }
    }

    private func buildFeatureProvider(hiddenFV: MLFeatureValue,
                                       chunkIdx: Int) throws -> MLFeatureProvider
    {
        var dict: [String: MLFeatureValue] = [
            "hidden_in": hiddenFV,
            "cos": fvCos,
            "sin": fvSin,
            "causal_mask": fvMask,
            "current_pos": fvCurPos,
        ]
        for absI in chunkLinIndices[chunkIdx] {
            guard let conv = ssmConv[absI], let rec = ssmRec[absI] else {
                throw NSError(domain: "Qwen35MLKV", code: 51,
                    userInfo: [NSLocalizedDescriptionKey:
                        "missing SSM state for layer \(absI)"])
            }
            dict["conv_state_\(absI)"] = MLFeatureValue(multiArray: conv)
            dict["rec_state_\(absI)"]  = MLFeatureValue(multiArray: rec)
        }
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    // MARK: Buffer helpers

    private func allocReusable() throws {
        reusableHidden = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.hiddenSize)], dataType: .float16)
        reusableCos = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.rotaryDim)], dataType: .float16)
        reusableSin = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.rotaryDim)], dataType: .float16)
        reusableMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.maxSeq)], dataType: .float16)
        reusableCurPos = try MLMultiArray(shape: [1], dataType: .int32)
        fvHidden = MLFeatureValue(multiArray: reusableHidden)
        fvCos = MLFeatureValue(multiArray: reusableCos)
        fvSin = MLFeatureValue(multiArray: reusableSin)
        fvMask = MLFeatureValue(multiArray: reusableMask)
        fvCurPos = MLFeatureValue(multiArray: reusableCurPos)
    }

    private func allocSSMStates() throws {
        // SSM shapes (Qwen3.5):
        //   conv_state per layer: (1, conv_dim, K=4)
        //     conv_dim = key_dim*2 + value_dim
        //     key_dim = linear_key_head_dim * linear_num_key_heads
        //     value_dim = linear_value_head_dim * linear_num_value_heads
        //   rec_state per layer: (1, num_v, Dk, Dv)
        //
        // For both 0.8B and 2B (same SSM dims):
        //   key_dim = 128 * 16 = 2048;  value_dim = 128 * 16 = 2048
        //   conv_dim = 2048*2 + 2048 = 6144 (0.8B) / wait 0.8B differs
        //
        // Read from a representative chunk's input descriptions instead of
        // hardcoding — chunk_a is loaded already and exposes the shapes.
        let descs = bodyChunks[0].modelDescription.inputDescriptionsByName
        var convShape: [NSNumber]?
        var recShape: [NSNumber]?
        for (name, d) in descs {
            if name.hasPrefix("conv_state_"),
               let s = d.multiArrayConstraint?.shape, convShape == nil {
                convShape = s
            }
            if name.hasPrefix("rec_state_"),
               let s = d.multiArrayConstraint?.shape, recShape == nil {
                recShape = s
            }
        }
        guard let cs = convShape, let rs = recShape else {
            throw NSError(domain: "Qwen35MLKV", code: 52,
                userInfo: [NSLocalizedDescriptionKey:
                    "couldn't read SSM state shapes from chunk_a"])
        }
        ssmConv.removeAll(keepingCapacity: true)
        ssmRec.removeAll(keepingCapacity: true)
        for i in 0..<cfg.numLayers where i % 4 != 3 {
            let conv = try MLMultiArray(shape: cs, dataType: .float16)
            let rec  = try MLMultiArray(shape: rs, dataType: .float16)
            // Zero-init.
            memset(conv.dataPointer, 0, conv.count * 2)
            memset(rec.dataPointer, 0, rec.count * 2)
            ssmConv[i] = conv
            ssmRec[i] = rec
        }
    }

    private func embedLookup(token: Int32) {
        guard let ptr = embedMmapPtr else { return }
        let H = cfg.hiddenSize
        let src = ptr + Int(token) * H
        let dst = reusableHidden.dataPointer.assumingMemoryBound(to: UInt16.self)
        memcpy(dst, src, H * 2)
    }

    private func fillCosSin(forPosition pos: Int) {
        let rd = cfg.rotaryDim
        let row = pos * rd
        let cosDst = reusableCos.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sinDst = reusableSin.dataPointer.assumingMemoryBound(to: UInt16.self)
        // Convert Float → fp16 bits via Float16 (Swift's native fp16 repr).
        for i in 0..<rd {
            cosDst[i] = float16Bits(cosTable[row + i])
            sinDst[i] = float16Bits(sinTable[row + i])
        }
    }

    private func fillCausalMask(forPosition pos: Int) {
        // Build (1, 1, 1, max_seq) fp16 row: 0 for j<=pos, -1e4 otherwise.
        let max_seq = cfg.maxSeq
        let dst = reusableMask.dataPointer.assumingMemoryBound(to: UInt16.self)
        let zero = float16Bits(0)
        let negInf = float16Bits(-1e4)
        for j in 0..<max_seq {
            dst[j] = j <= pos ? zero : negInf
        }
    }

    private func setCurrentPos(_ pos: Int32) {
        let p = reusableCurPos.dataPointer.assumingMemoryBound(to: Int32.self)
        p[0] = pos
    }

    private func float16Bits(_ x: Float) -> UInt16 {
        // Swift's _Float16 is the IEEE 754 binary16 type on Apple silicon.
        let h = Float16(x)
        return withUnsafeBytes(of: h) { $0.load(as: UInt16.self) }
    }

    // MARK: RoPE table (theta=1e7, partial_rotary_factor=0.25)

    private func buildRope(isCos: Bool) -> [Float] {
        let rd = cfg.rotaryDim
        let half = rd / 2
        let base: Float = 10_000_000.0
        var out = [Float](repeating: 0, count: cfg.maxSeq * rd)
        for p in 0..<cfg.maxSeq {
            for i in 0..<half {
                let theta = powf(base, Float(-2 * i) / Float(rd))
                let a = Float(p) * theta
                let v = isCos ? cosf(a) : sinf(a)
                out[p * rd + i]        = v
                out[p * rd + i + half] = v
            }
        }
        return out
    }

    // MARK: Disk resolution

    private func resolveURLs() -> (body: [URL], embed: URL)? {
        let fm = FileManager.default
        let candidates = ["qwen3_5_0_8b_decode_chunks_mlkv",
                          "qwen3_5_2b_decode_chunks_mlkv"]
        let names = ["chunk_a", "chunk_b", "chunk_c", "chunk_d"]
        func resolveAt(_ base: URL) -> (body: [URL], embed: URL)? {
            for sub in candidates {
                let dir = base.appendingPathComponent(sub)
                let embed = dir.appendingPathComponent("embed_weight.bin")
                guard fm.fileExists(atPath: embed.path) else { continue }
                var urls: [URL] = []
                var ok = true
                for n in names {
                    let pkg = dir.appendingPathComponent("\(n).mlpackage")
                    let mlc = dir.appendingPathComponent("\(n).mlmodelc")
                    if fm.fileExists(atPath: mlc.path) {
                        urls.append(mlc)
                    } else if fm.fileExists(atPath: pkg.path) {
                        guard let compiled = try? MLModel.compileModel(at: pkg) else {
                            ok = false; break
                        }
                        urls.append(compiled)
                    } else {
                        ok = false; break
                    }
                }
                if ok && urls.count == names.count { return (urls, embed) }
            }
            return nil
        }
        if let folder = modelFolderOverride, let r = resolveAt(folder) { return r }
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                   appropriateFor: nil, create: false),
           let r = resolveAt(docs) { return r }
        return nil
    }

    // MARK: Embed sidecar mmap

    private func mmapEmbedWeight(url: URL) throws {
        releaseEmbedMmap()
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "Qwen35MLKV", code: 60,
                userInfo: [NSLocalizedDescriptionKey: "open failed: \(url.path)"])
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw NSError(domain: "Qwen35MLKV", code: 61,
                userInfo: [NSLocalizedDescriptionKey: "fstat failed"])
        }
        let size = Int(st.st_size)
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "Qwen35MLKV", code: 62,
                userInfo: [NSLocalizedDescriptionKey: "mmap failed"])
        }
        embedMmapBase = base
        embedMmapLen = size
        embedMmapFD = fd
        embedMmapPtr = UnsafePointer(base.assumingMemoryBound(to: UInt16.self))
        madvise(base, size, MADV_RANDOM)
    }

    private func releaseEmbedMmap() {
        if let base = embedMmapBase, embedMmapLen > 0 {
            munmap(base, embedMmapLen)
        }
        if embedMmapFD >= 0 { close(embedMmapFD) }
        embedMmapBase = nil
        embedMmapPtr = nil
        embedMmapLen = 0
        embedMmapFD = -1
    }

    deinit { releaseEmbedMmap() }

    // MARK: helpers

    private func chunkRanges(numLayers: Int, numChunks: Int) -> [(Int, Int)] {
        let per = numLayers / numChunks
        return (0..<numChunks).map { i in (i * per, (i + 1) * per) }
    }

    private func unitsName(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU+GPU"
        case .cpuAndNeuralEngine: return "CPU+ANE"
        case .all: return "ALL"
        @unknown default: return "?"
        }
    }
}
