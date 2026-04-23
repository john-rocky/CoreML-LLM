// End-to-end Qwen3-VL 4B (text-only) generation on device.
//
// Layout on disk (under the model folder, e.g. Documents/Models/qwen3-vl-4b/):
//   qwen3_vl_4b_decode_chunks/
//     embed_weight.bin            (raw fp16, Swift mmap'd)
//     chunk_0.mlpackage..chunk_5  (6 layers each)
//     chunk_head.mlpackage        (final_norm + lm_head)
//
// Per-step decode:
//   embed lookup (mmap, 1 row memcpy) → chunk_0 → ... → chunk_5 → chunk_head → logits
//
// Architecture (Qwen3-VL 4B text backbone):
//   hidden=2560, layers=36 (6 chunks × 6), num_attention_heads=32,
//   num_key_value_heads=8 (GQA 4:1), head_dim=128, vocab=151936,
//   tie_word_embeddings=True, rope_theta=5_000_000.
//   For text-only inputs the mRoPE [24,20,20] interleave collapses to
//   standard 1D RoPE over the full head_dim.
//
// State: per-layer KV caches, (1, num_kv_heads=8, max_seq=2048, head_dim=128) × 2.
// Total state per chunk: 6 layers × 2 × 2 × 8 × 2048 × 128 × 2 bytes = 100 MB.

import Accelerate
import CoreML
import Foundation


/// Adapter that maps body-chunk output feature names (`new_k_X`,
/// `new_v_X`) to the next call's input names (`k_X`, `v_X`) without
/// allocating fresh MLFeatureValue wrappers per step. Forwards
/// `hidden_in` from the previous step's predecessor (chunk[i-1] output
/// or the embed lookup buffer for chunk[0]).
private final class VL4BBodyFeatures: NSObject, MLFeatureProvider {
    private let hiddenSource: MLFeatureProvider  // carries "hidden" feature
    private let fvPos: MLFeatureValue
    private let fvCos: MLFeatureValue
    private let fvSin: MLFeatureValue
    private let prevOut: MLFeatureProvider?
    private let initialKVFVs: [String: MLFeatureValue]?
    private let kvRename: [String: String]   // k_X → new_k_X (and v_X → new_v_X)
    let featureNames: Set<String>

    init(hiddenSource: MLFeatureProvider,
         pos: MLFeatureValue, cos: MLFeatureValue, sin: MLFeatureValue,
         prevOut: MLFeatureProvider?,
         initialKVFVs: [String: MLFeatureValue]?,
         kvInputNames: [String], kvOutputNames: [String]) {
        self.hiddenSource = hiddenSource
        self.fvPos = pos; self.fvCos = cos; self.fvSin = sin
        self.prevOut = prevOut
        self.initialKVFVs = initialKVFVs
        var rename: [String: String] = [:]
        rename.reserveCapacity(kvInputNames.count)
        for (a, b) in zip(kvInputNames, kvOutputNames) { rename[a] = b }
        self.kvRename = rename
        var names: Set<String> = ["hidden_in", "position", "cos", "sin"]
        for n in kvInputNames { names.insert(n) }
        self.featureNames = names
        super.init()
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "hidden_in": return hiddenSource.featureValue(for: "hidden")
        case "position":  return fvPos
        case "cos":       return fvCos
        case "sin":       return fvSin
        default:
            if let prev = prevOut, let outName = kvRename[featureName] {
                return prev.featureValue(for: outName)
            }
            return initialKVFVs?[featureName]
        }
    }
}

/// Trivial provider that wraps the embed-lookup MLMultiArray as
/// `hidden`, so chunk[0] reads it through the same path as chunks[i>0]
/// reading from the previous chunk's output.
private final class VL4BHiddenProvider: NSObject, MLFeatureProvider {
    let fvHidden: MLFeatureValue
    let featureNames: Set<String> = ["hidden"]
    init(fvHidden: MLFeatureValue) {
        self.fvHidden = fvHidden
        super.init()
    }
    func featureValue(for featureName: String) -> MLFeatureValue? {
        featureName == "hidden" ? fvHidden : nil
    }
}

@Observable
final class Qwen3VL4BGenerator {
    struct Config {
        let maxSeq: Int          // 2048
        let vocab: Int           // 151936
        let hiddenSize: Int      // 2560
        let numLayers: Int       // 36
        let numKVHeads: Int      // 8
        let headDim: Int         // 128
        let numBodyChunks: Int   // 6
        let layersPerChunk: Int  // 6
        let ropeTheta: Float     // 5_000_000
        let decodeUnits: MLComputeUnits

        static let `default` = Config(
            maxSeq: 2048, vocab: 151936, hiddenSize: 2560, numLayers: 36,
            numKVHeads: 8, headDim: 128, numBodyChunks: 6, layersPerChunk: 6,
            ropeTheta: 5_000_000.0,
            decodeUnits: .cpuAndNeuralEngine)
    }

    var status = "Idle"
    var running = false
    var generatedIds: [Int32] = []
    var prefillMs: Double = 0
    var decodeMsAvg: Double = 0
    var tokensPerSecond: Double = 0
    var firstStepDebug: [(Int32, Float)] = []

    private var cfg: Config

    /// Body chunk models in order chunk_0..chunk_5.
    @ObservationIgnored private var bodyChunks: [MLModel] = []
    @ObservationIgnored private var headChunk: MLModel?

    /// mmap'd embed_weight.bin
    @ObservationIgnored private var embedMmapPtr: UnsafePointer<UInt16>?
    @ObservationIgnored private var embedMmapBase: UnsafeMutableRawPointer?
    @ObservationIgnored private var embedMmapLen: Int = 0
    @ObservationIgnored private var embedMmapFD: Int32 = -1

    // Reusable per-step buffers
    @ObservationIgnored private var reusableHidden: MLMultiArray!  // (1,1,hidden) fp16
    @ObservationIgnored private var reusablePos: MLMultiArray!     // (1,) fp32
    @ObservationIgnored private var reusableCos: MLMultiArray!     // (1,1,head_dim) fp16
    @ObservationIgnored private var reusableSin: MLMultiArray!     // (1,1,head_dim) fp16
    @ObservationIgnored private var fvHidden: MLFeatureValue!
    @ObservationIgnored private var fvPos: MLFeatureValue!
    @ObservationIgnored private var fvCos: MLFeatureValue!
    @ObservationIgnored private var fvSin: MLFeatureValue!
    @ObservationIgnored private var hiddenProvider: VL4BHiddenProvider!

    /// Per-chunk KV-cache name slices (k_X / v_X / new_k_X / new_v_X).
    /// Outer index = chunk; inner array = 12 names per chunk
    /// (6 layers × 2: k + v, both inputs and outputs).
    @ObservationIgnored private var kvInputNames: [[String]] = []
    @ObservationIgnored private var kvOutputNames: [[String]] = []

    /// Pre-built RoPE tables (max_seq, head_dim) — fp32 source, sliced
    /// per step into the reusable fp16 buffers. Standard 1D RoPE over
    /// full head_dim (mRoPE collapses for text-only).
    @ObservationIgnored private var cosTable: [Float] = []
    @ObservationIgnored private var sinTable: [Float] = []

    /// Reusable Float buffer for logits — avoids 600 KB heap alloc per step.
    @ObservationIgnored private var logitsBuffer: [Float] = []

    init(cfg: Config = .default) {
        self.cfg = cfg
        self.cosTable = buildRope(isCos: true)
        self.sinTable = buildRope(isCos: false)
        self.logitsBuffer = [Float](repeating: 0, count: cfg.vocab)
        // Per-chunk KV input/output name slices
        var inSlices: [[String]] = []
        var outSlices: [[String]] = []
        for chunkIdx in 0..<cfg.numBodyChunks {
            let start = chunkIdx * cfg.layersPerChunk
            let end = start + cfg.layersPerChunk
            var ins: [String] = []; ins.reserveCapacity((end - start) * 2)
            var outs: [String] = []; outs.reserveCapacity((end - start) * 2)
            for i in start..<end {
                ins.append("k_\(i)"); ins.append("v_\(i)")
                outs.append("new_k_\(i)"); outs.append("new_v_\(i)")
            }
            inSlices.append(ins); outSlices.append(outs)
        }
        self.kvInputNames = inSlices
        self.kvOutputNames = outSlices
        // Pre-allocate per-step input arrays + their feature wrappers.
        self.reusableHidden = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.hiddenSize)], dataType: .float16)
        self.reusablePos = try! MLMultiArray(shape: [1], dataType: .float32)
        self.reusableCos = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        self.reusableSin = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        self.fvHidden = MLFeatureValue(multiArray: reusableHidden)
        self.fvPos = MLFeatureValue(multiArray: reusablePos)
        self.fvCos = MLFeatureValue(multiArray: reusableCos)
        self.fvSin = MLFeatureValue(multiArray: reusableSin)
        self.hiddenProvider = VL4BHiddenProvider(fvHidden: fvHidden)
    }

    /// 1D RoPE table builder. For text-only Qwen3-VL the mRoPE
    /// interleave collapses to standard full-head_dim RoPE: for each
    /// position p, freqs[i] = p / theta^(2i / head_dim) for i in
    /// [0, head_dim/2), then duplicated to full head_dim
    /// (cos = cat([cos(freqs), cos(freqs)])).
    private func buildRope(isCos: Bool) -> [Float] {
        let d = cfg.headDim
        let half = d / 2
        var out = [Float](repeating: 0, count: cfg.maxSeq * d)
        for p in 0..<cfg.maxSeq {
            for i in 0..<half {
                let theta = powf(cfg.ropeTheta, Float(2 * i) / Float(d))
                let a = Float(p) / theta
                let v = isCos ? cosf(a) : sinf(a)
                out[p * d + i]        = v
                out[p * d + i + half] = v
            }
        }
        return out
    }

    var modelFolderOverride: URL?

    /// Locate VL4B chunks + embed sidecar. Returns
    /// (bodyURLs, headURL, embedURL) or nil if any piece is missing.
    /// Honors both `.mlpackage` (HF download) and `.mlmodelc`
    /// (devicectl sideload).
    private func resolveDecodeURLs()
        -> (body: [URL], head: URL, embed: URL)? {
        let subdir = "qwen3_vl_4b_decode_chunks"
        let embedBinName = "embed_weight.bin"
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
            let embedURL = dir.appendingPathComponent(embedBinName)
            guard fm.fileExists(atPath: embedURL.path) else { return nil }
            var bodies: [URL] = []
            for ci in 0..<cfg.numBodyChunks {
                guard let u = resolveOne(dir, "chunk_\(ci)") else { return nil }
                bodies.append(u)
            }
            guard let head = resolveOne(dir, "chunk_head") else { return nil }
            return (bodies, head, embedURL)
        }

        if let folder = modelFolderOverride, let r = resolve(folder) { return r }
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                  appropriateFor: nil, create: false),
           let r = resolve(docs) { return r }
        return nil
    }

    /// mmap the embed_weight.bin file read-only. Clean pages don't
    /// inflate phys_footprint — the 778 MB embed table lives in
    /// virtual memory only and only touched rows page in.
    private func mmapEmbedWeight(url: URL) throws {
        releaseEmbedMmap()
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "Qwen3VL4BGenerator", code: 30,
                userInfo: [NSLocalizedDescriptionKey:
                    "failed to open embed_weight.bin at \(url.path)"])
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw NSError(domain: "Qwen3VL4BGenerator", code: 31,
                userInfo: [NSLocalizedDescriptionKey: "fstat failed"])
        }
        let size = Int(st.st_size)
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "Qwen3VL4BGenerator", code: 32,
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

    /// Copy one row of the fp16 embed table into `reusableHidden`.
    /// Equivalent to `F.embedding(input_token, embed_w)` in PyTorch.
    private func embedLookup(token: Int32) {
        guard let ptr = embedMmapPtr else { return }
        let hiddenSize = cfg.hiddenSize
        let src = ptr + Int(token) * hiddenSize
        let dst = reusableHidden.dataPointer.assumingMemoryBound(to: UInt16.self)
        memcpy(dst, src, hiddenSize * 2)
    }

    deinit {
        releaseEmbedMmap()
    }

    // MARK: - Loading

    func setComputeUnits(_ units: MLComputeUnits) {
        cfg = Config(
            maxSeq: cfg.maxSeq, vocab: cfg.vocab,
            hiddenSize: cfg.hiddenSize, numLayers: cfg.numLayers,
            numKVHeads: cfg.numKVHeads, headDim: cfg.headDim,
            numBodyChunks: cfg.numBodyChunks, layersPerChunk: cfg.layersPerChunk,
            ropeTheta: cfg.ropeTheta, decodeUnits: units)
        bodyChunks = []
        headChunk = nil
        releaseEmbedMmap()
        status = "Idle (units changed, will reload on next run)"
    }

    func load() throws {
        let dCfg = MLModelConfiguration(); dCfg.computeUnits = cfg.decodeUnits
        guard let resolved = resolveDecodeURLs() else {
            throw NSError(domain: "Qwen3VL4BGenerator", code: 40,
                userInfo: [NSLocalizedDescriptionKey:
                    "qwen3_vl_4b_decode_chunks/{embed_weight.bin, chunk_0..5, chunk_head} not found"])
        }
        try mmapEmbedWeight(url: resolved.embed)
        bodyChunks = try resolved.body.map {
            try MLModel(contentsOf: $0, configuration: dCfg)
        }
        headChunk = try MLModel(contentsOf: resolved.head, configuration: dCfg)
        status = "Loaded VL4B (\(cfg.numBodyChunks) body + head) on \(unitsName(cfg.decodeUnits))"
        // Diagnostic: per-chunk compute plan audit. Same pattern as
        // Qwen35Generator — async, doesn't block prediction.
        for (idx, url) in resolved.body.enumerated() {
            auditComputePlan(url: url, requestedUnits: cfg.decodeUnits,
                              variant: "VL4B-body#\(idx)")
        }
        auditComputePlan(url: resolved.head, requestedUnits: cfg.decodeUnits,
                          variant: "VL4B-head")
    }

    private func unitsName(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly: return "CPU"
        case .cpuAndNeuralEngine: return "CPU+ANE"
        case .cpuAndGPU: return "CPU+GPU"
        case .all: return "All"
        @unknown default: return "?"
        }
    }

    private func auditComputePlan(url: URL, requestedUnits: MLComputeUnits, variant: String) {
        let cfg = MLModelConfiguration(); cfg.computeUnits = requestedUnits
        Task.detached(priority: .utility) {
            guard #available(iOS 17.0, *) else { return }
            do {
                let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
                guard case .program(let program) = plan.modelStructure else { return }
                var total = 0, ane = 0, gpu = 0, cpu = 0, other = 0
                for (_, fn) in program.functions {
                    Self.walkOps(fn.block, plan: plan,
                                  total: &total, ane: &ane, gpu: &gpu,
                                  cpu: &cpu, other: &other)
                }
                let compute = max(1, total)
                print(String(format:
                    "[Qwen3VL4B] compute plan (\(variant)): total=%d ANE=%d (%.1f%%) GPU=%d (%.1f%%) CPU=%d (%.1f%%)",
                    total, ane, 100.0*Double(ane)/Double(compute),
                    gpu, 100.0*Double(gpu)/Double(compute),
                    cpu, 100.0*Double(cpu)/Double(compute)))
            } catch {
                print("[Qwen3VL4B] compute plan audit failed: \(error)")
            }
        }
    }

    @available(iOS 17.0, *)
    private static func walkOps(_ block: MLModelStructure.Program.Block,
                                 plan: MLComputePlan,
                                 total: inout Int, ane: inout Int,
                                 gpu: inout Int, cpu: inout Int,
                                 other: inout Int) {
        let constOps: Set<String> = [
            "const", "constexpr_lut_to_dense", "constexpr_affine_dequantize",
            "constexpr_blockwise_shift_scale", "constexpr_sparse_to_dense",
            "constexpr_cast",
        ]
        for op in block.operations {
            if constOps.contains(op.operatorName) {
                for inner in op.blocks {
                    walkOps(inner, plan: plan,
                            total: &total, ane: &ane, gpu: &gpu, cpu: &cpu, other: &other)
                }
                continue
            }
            total += 1
            switch plan.deviceUsage(for: op)?.preferred {
            case .cpu:          cpu += 1
            case .gpu:          gpu += 1
            case .neuralEngine: ane += 1
            default:            other += 1
            }
            for inner in op.blocks {
                walkOps(inner, plan: plan,
                        total: &total, ane: &ane, gpu: &gpu, cpu: &cpu, other: &other)
            }
        }
    }

    /// Build zero-init KV cache for a chunk's layer range.
    /// MLMultiArray(shape:dataType:) is NOT guaranteed to zero-init —
    /// must explicitly memset. Same lesson as Qwen3.5 generators.
    private func makeZeroKVStates(start: Int, end: Int) throws
        -> [String: MLMultiArray] {
        var dict: [String: MLMultiArray] = [:]
        let shape: [NSNumber] = [
            1,
            NSNumber(value: cfg.numKVHeads),
            NSNumber(value: cfg.maxSeq),
            NSNumber(value: cfg.headDim),
        ]
        for i in start..<end {
            let k = try MLMultiArray(shape: shape, dataType: .float16)
            let v = try MLMultiArray(shape: shape, dataType: .float16)
            memset(k.dataPointer, 0, k.count * 2)
            memset(v.dataPointer, 0, v.count * 2)
            dict["k_\(i)"] = k
            dict["v_\(i)"] = v
        }
        return dict
    }

    private func writeDecodeScalars(token: Int32, position: Int) {
        // input_token isn't fed to any chunk (embed lookup happens in
        // Swift), but `position` and the RoPE tables are.
        reusablePos.dataPointer.assumingMemoryBound(to: Float.self)[0] = Float(position)
        let d = cfg.headDim
        let cp = reusableCos.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sp = reusableSin.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<d {
            cp[i] = Float16(cosTable[position * d + i]).bitPattern
            sp[i] = Float16(sinTable[position * d + i]).bitPattern
        }
    }

    /// Holds per-call mutable scratch threaded through the decode loop.
    /// One last-output slot per body chunk; the head is stateless.
    private final class CallState {
        var lastBodyOuts: [MLFeatureProvider?] = []
        var initialKVFVs: [[String: MLFeatureValue]?] = []
    }

    private func stepPredict(token: Int32, position: Int,
                              state: CallState) async throws -> MLFeatureProvider {
        embedLookup(token: token)
        writeDecodeScalars(token: token, position: position)
        var lastStepOut: MLFeatureProvider = hiddenProvider
        for ci in 0..<bodyChunks.count {
            let features = VL4BBodyFeatures(
                hiddenSource: lastStepOut,
                pos: fvPos, cos: fvCos, sin: fvSin,
                prevOut: state.lastBodyOuts[ci],
                initialKVFVs: state.initialKVFVs[ci],
                kvInputNames: kvInputNames[ci],
                kvOutputNames: kvOutputNames[ci])
            let out = try await bodyChunks[ci].prediction(from: features)
            state.lastBodyOuts[ci] = out
            state.initialKVFVs[ci] = nil
            lastStepOut = out
        }
        // Head: takes the final hidden, returns logits.
        guard let head = headChunk else {
            throw NSError(domain: "Qwen3VL4BGenerator", code: 50,
                userInfo: [NSLocalizedDescriptionKey: "head chunk not loaded"])
        }
        let headIn = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_in": lastStepOut.featureValue(for: "hidden")!,
        ])
        return try await head.prediction(from: headIn)
    }

    /// Generate up to `maxNewTokens` tokens starting from `inputIds`.
    /// Greedy by default (temperature=0). Recurrent prefill replays the
    /// prompt through the same per-step path so we don't need a
    /// separate prefill mlpackage.
    @discardableResult
    func generate(inputIds: [Int32], maxNewTokens: Int = 64,
                   temperature: Float = 0.0,
                   eosTokenIds: Set<Int32> = [],
                   onToken: ((Int32) -> Void)? = nil) async throws -> [Int32] {
        running = true
        defer { running = false }
        generatedIds.removeAll()

        if bodyChunks.isEmpty || headChunk == nil { try load() }

        let S = inputIds.count
        guard S > 0, S <= cfg.maxSeq - 1 else {
            throw NSError(domain: "Qwen3VL4BGenerator", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "input length \(S) must be in (0, \(cfg.maxSeq - 1)]"])
        }

        // Per-chunk zero KV cache (initial state)
        let callState = CallState()
        callState.lastBodyOuts = [MLFeatureProvider?](
            repeating: nil, count: cfg.numBodyChunks)
        var initial: [[String: MLFeatureValue]?] = []
        initial.reserveCapacity(cfg.numBodyChunks)
        for ci in 0..<cfg.numBodyChunks {
            let start = ci * cfg.layersPerChunk
            let end = start + cfg.layersPerChunk
            let states = try makeZeroKVStates(start: start, end: end)
            var fvs: [String: MLFeatureValue] = [:]
            fvs.reserveCapacity(states.count)
            for (k, v) in states { fvs[k] = MLFeatureValue(multiArray: v) }
            initial.append(fvs)
        }
        callState.initialKVFVs = initial

        // --- Recurrent prefill ---
        status = "Prefill (recurrent via decode)..."
        let prefillStart = Date()
        var lastNextToken: Int32 = 0
        for (t, tok) in inputIds.enumerated() {
            let out = try await stepPredict(token: tok, position: t, state: callState)
            if t == S - 1 {
                lastNextToken = readNextToken(from: out)
            }
            if (t & 0x7) == 0 { status = "Prefill \(t + 1)/\(S)..." }
        }
        prefillMs = Date().timeIntervalSince(prefillStart) * 1000

        // Head chunk computes argmax in-graph and emits `next_token`
        // (int32) directly — saves a 600 KB vocab-fp32 transfer per
        // step plus Swift-side argmax. 1st token is the prefill's
        // final output.
        var nextToken = lastNextToken
        firstStepDebug = []
        generatedIds.append(nextToken)
        onToken?(nextToken)

        // --- Decode loop ---
        status = "Decoding..."
        let decodeStart = Date()
        var position = S
        var decodedSteps = 0
        for step in 0..<(maxNewTokens - 1) {
            if position >= cfg.maxSeq { break }
            let out = try await stepPredict(token: nextToken, position: position,
                                             state: callState)
            nextToken = readNextToken(from: out)
            generatedIds.append(nextToken)
            onToken?(nextToken)
            position += 1
            decodedSteps += 1
            if (step & 0x7) == 0 {
                status = "Decoding \(step + 2)/\(maxNewTokens)..."
            }
            if eosTokenIds.contains(nextToken) { break }
        }

        let totalDecodeMs = Date().timeIntervalSince(decodeStart) * 1000
        decodeMsAvg = totalDecodeMs / Double(max(decodedSteps, 1))
        tokensPerSecond = Double(generatedIds.count) / ((prefillMs + totalDecodeMs) / 1000.0)
        status = String(format: "Done: %d tokens, prefill=%.0fms, decode=%.1fms/tok, %.1f tok/s",
                         generatedIds.count, prefillMs, decodeMsAvg, tokensPerSecond)
        return generatedIds
    }

    // MARK: - next-token extraction

    /// Read the head chunk's in-graph argmax output. The converter
    /// embeds an argmax in the graph so per-step ANE→Swift transfer
    /// drops from ~600 KB (fp32 logits over 151K vocab) to 4 bytes
    /// (one int32). Falls back to Swift-side fp32 argmax if the
    /// output is a legacy fp32/fp16 logits tensor — lets a v1 bundle
    /// (fp32 logits) still work with this generator.
    private func readNextToken(from out: MLFeatureProvider) -> Int32 {
        if let ntArr = out.featureValue(for: "next_token")?.multiArrayValue {
            // MLMultiArrayDataType only exposes .int32 (+ fp/double);
            // the head chunk is built to return int32 directly.
            if ntArr.dataType == .int32 {
                return ntArr.dataPointer
                    .assumingMemoryBound(to: Int32.self)[0]
            }
        }
        if let logits = out.featureValue(for: "logits")?.multiArrayValue {
            return fastArgmax(logits, vocab: cfg.vocab)
        }
        return 0
    }

    /// Single-pass fp32 argmax over (1, 1, vocab). Logits emerge as
    /// fp32 from the head chunk's logits output.
    private func fastArgmax(_ arr: MLMultiArray, vocab: Int) -> Int32 {
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            var maxV: Float = 0
            var idx: vDSP_Length = 0
            vDSP_maxvi(p, 1, &maxV, &idx, vDSP_Length(vocab))
            return Int32(idx)
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self)
            var bestV: Float16 = -.infinity
            var bestIdx: Int = 0
            for v in 0..<vocab {
                let x = p[v]
                if x > bestV { bestV = x; bestIdx = v }
            }
            return Int32(bestIdx)
        default:
            var bestV: Float = -.infinity
            var bestIdx: Int = 0
            for v in 0..<vocab {
                let x = arr[[0, 0, NSNumber(value: v)] as [NSNumber]].floatValue
                if x > bestV { bestV = x; bestIdx = v }
            }
            return Int32(bestIdx)
        }
    }
}
