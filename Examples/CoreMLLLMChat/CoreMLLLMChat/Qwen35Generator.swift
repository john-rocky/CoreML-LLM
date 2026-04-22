// End-to-end Qwen3.5-0.8B generation on device.
//
// Uses the stateful decode mlpackage as BOTH prefill and decode: the prompt
// is replayed token-by-token through the decode step to build up state,
// then generation continues from there. This avoids the monolithic stateful
// prefill mlpackage (which fails iPhone ANE compile budget and has untested
// iPhone GPU behavior), and keeps the entire generation path on a single
// model — the decode model is validated: top-3 = 100% vs fp32 oracle on
// both Mac M4 and iPhone A18.
//
// Required bundle resources:
//   qwen3_5_0_8b_decode_fp16_mseq128.mlmodelc (or .mlpackage in Documents/)
//
// Dimensions (Qwen3.5-0.8B):
//   hidden=1024  num_layers=24 (18 linear + 6 full)  vocab=248320
//   linear state per layer: conv (1,6144,4) + rec (1,16,128,128)
//   full   state per layer: k_cache + v_cache (1,2,128,256)

import Accelerate
import CoreML
import Foundation

/// Adapter that maps the decode model's output feature names
/// (`new_state_X_Y`) to the next call's input names (`state_X_Y`) without
/// allocating 48 fresh MLFeatureValue wrappers per step. The 4 non-state
/// inputs (token/pos/cos/sin) are supplied from pre-built MLFeatureValues.
/// Initial-call case: when `prevOut` is nil, state inputs come from
/// zero-initialized MLMultiArrays wrapped in MLFeatureValues.
private final class Qwen35DecodeFeatures: NSObject, MLFeatureProvider {
    private let fvInpTok: MLFeatureValue
    private let fvInpPos: MLFeatureValue
    private let fvCos: MLFeatureValue
    private let fvSin: MLFeatureValue
    /// Either the output of the previous decode call (maps new_state_X_Y)
    /// OR nil for the first call (then `initialStateFVs` is consulted).
    private let prevOut: MLFeatureProvider?
    private let initialStateFVs: [String: MLFeatureValue]?
    private let stateRename: [String: String]  // state_X_Y -> new_state_X_Y
    let featureNames: Set<String>

    init(tok: MLFeatureValue, pos: MLFeatureValue,
         cos: MLFeatureValue, sin: MLFeatureValue,
         prevOut: MLFeatureProvider?,
         initialStateFVs: [String: MLFeatureValue]?,
         stateInputNames: [String], stateOutputNames: [String]) {
        self.fvInpTok = tok; self.fvInpPos = pos
        self.fvCos = cos; self.fvSin = sin
        self.prevOut = prevOut
        self.initialStateFVs = initialStateFVs
        var rename: [String: String] = [:]
        rename.reserveCapacity(stateInputNames.count)
        for (a, b) in zip(stateInputNames, stateOutputNames) { rename[a] = b }
        self.stateRename = rename
        var names: Set<String> = ["input_token", "position", "cos", "sin"]
        for n in stateInputNames { names.insert(n) }
        self.featureNames = names
        super.init()
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_token": return fvInpTok
        case "position":    return fvInpPos
        case "cos":         return fvCos
        case "sin":         return fvSin
        default:
            if let prev = prevOut, let outName = stateRename[featureName] {
                return prev.featureValue(for: outName)
            }
            return initialStateFVs?[featureName]
        }
    }
}

@Observable
final class Qwen35Generator {
    struct Config {
        let seqLen: Int           // prefill fixed seq length (64)
        let maxSeq: Int           // decode + prefill max length (128)
        let vocab: Int            // 248320
        let numLayers: Int        // 24
        let rotaryDim: Int        // head_dim * 0.25 = 64
        /// Compute units for prefill. The monolithic stateful prefill does
        /// not fit in iPhone ANE's compile budget (E5 compile fails). GPU
        /// is the fast and accurate path for prefill; it incurs Metal heap
        /// only during the single prefill call.
        let prefillUnits: MLComputeUnits
        /// Compute units for decode. ANE is validated (top-3 = 100% vs
        /// fp32 oracle, 22 tok/s on iPhone 17 Pro, zero Metal heap).
        let decodeUnits: MLComputeUnits
        /// ANE-first by default — zero sustained Metal heap, 20 tok/s on
        /// iPhone 17 Pro. First load triggers ~4 min on-device E5 compile
        /// (status reflects "Compiling decode model..."); cached after that.
        /// Switch to `.cpuAndGPU` for bit-exact output (22 tok/s, ~3 GB
        /// Metal) or `.cpuOnly` for no accelerator usage.
        static let `default` = Config(seqLen: 64, maxSeq: 128, vocab: 248320,
                                      numLayers: 24, rotaryDim: 64,
                                      prefillUnits: .cpuAndNeuralEngine,
                                      decodeUnits: .cpuAndNeuralEngine)
    }

    var status = "Idle"
    var running = false
    var generatedIds: [Int32] = []
    var prefillMs: Double = 0
    var decodeMsAvg: Double = 0
    var tokensPerSecond: Double = 0
    /// Debug: top-5 (id, logit) at the first decode position — useful to
    /// diagnose degenerate distributions.
    var firstStepDebug: [(Int32, Float)] = []

    /// Per-phase timing profile — decode-loop breakdown aggregated over
    /// all decode calls in the last generation. Each value is the mean
    /// milliseconds per decode step. Use this to identify which phase
    /// dominates the per-token latency.
    struct PhaseProfile {
        var inputsBuild: Double = 0   // makeDecodeInputs (allocs + dict)
        var predict: Double = 0       // decode.prediction
        var stateCopy: Double = 0     // featureValue reads + dict writes
        var logitRead: Double = 0     // copyLogits / fastArgmax / sampling
        var total: Double = 0         // wall-clock per step
        var count: Int = 0            // number of samples
    }
    var decodeProfile = PhaseProfile()

    private var decode: MLModel?
    /// True when the loaded decode model emits `next_token` directly
    /// (in-graph argmax). Skips the 248K-logit CPU transfer per step.
    @ObservationIgnored private var decodeHasInGraphArgmax = false
    private var cfg: Config

    /// Reset per-phase profile accumulators. Call between comparable
    /// measurement runs (e.g., before running the same prompt on a
    /// different compute backend to compare).
    func resetProfile() {
        decodeProfile = PhaseProfile()
    }

    /// Swap compute units at runtime. Since this Generator uses the decode
    /// model for both prefill and decode, only `decode` units take effect;
    /// the `prefill` parameter is accepted for API compat and ignored.
    func setComputeUnits(prefill: MLComputeUnits, decode: MLComputeUnits) {
        cfg = Config(seqLen: cfg.seqLen, maxSeq: cfg.maxSeq, vocab: cfg.vocab,
                      numLayers: cfg.numLayers, rotaryDim: cfg.rotaryDim,
                      prefillUnits: prefill, decodeUnits: decode)
        self.decode = nil
        // Backend changed — previous profile samples are not comparable.
        decodeProfile = PhaseProfile()
        status = "Idle (units changed, will reload on next run)"
    }

    // RoPE cos/sin tables (max_seq, rotary_dim) — Qwen3.5 text: theta=1e7, partial=0.25
    // @ObservationIgnored + explicit init avoids the @Observable-vs-`lazy`
    // macro conflict: Observation-tracked lazy properties would need init
    // accessors that can't reach the backing storage.
    @ObservationIgnored private var cosTable: [Float] = []
    @ObservationIgnored private var sinTable: [Float] = []
    /// Reusable Float buffer for logits — avoids 1MB heap alloc per step.
    @ObservationIgnored private var logitsBuffer: [Float] = []
    /// Pre-computed state tensor names so we don't re-build strings via
    /// string interpolation 48 times per decode step.
    @ObservationIgnored private var stateInputNames: [String] = []    // "state_0_a", "state_0_b", ...
    @ObservationIgnored private var stateOutputNames: [String] = []   // "new_state_0_a", "new_state_0_b", ...

    /// Reusable MLMultiArrays for the 4 scalar/small inputs that change
    /// every decode step. Allocated once, data pointer written per step.
    /// Saves 4 MLMultiArray allocations per step + 4 MLFeatureValue wrappings.
    @ObservationIgnored private var reusableInpTok: MLMultiArray!
    @ObservationIgnored private var reusableInpPos: MLMultiArray!
    @ObservationIgnored private var reusableCos: MLMultiArray!
    @ObservationIgnored private var reusableSin: MLMultiArray!
    /// Pre-wrapped MLFeatureValues for the 4 reusable inputs.
    @ObservationIgnored private var fvInpTok: MLFeatureValue!
    @ObservationIgnored private var fvInpPos: MLFeatureValue!
    @ObservationIgnored private var fvCos: MLFeatureValue!
    @ObservationIgnored private var fvSin: MLFeatureValue!

    init(cfg: Config = .default) {
        self.cfg = cfg
        self.cosTable = buildRope(isCos: true)
        self.sinTable = buildRope(isCos: false)
        self.logitsBuffer = [Float](repeating: 0, count: cfg.vocab)
        var inNames: [String] = []; inNames.reserveCapacity(cfg.numLayers * 2)
        var outNames: [String] = []; outNames.reserveCapacity(cfg.numLayers * 2)
        for i in 0..<cfg.numLayers {
            inNames.append("state_\(i)_a"); inNames.append("state_\(i)_b")
            outNames.append("new_state_\(i)_a"); outNames.append("new_state_\(i)_b")
        }
        self.stateInputNames = inNames
        self.stateOutputNames = outNames
        // Pre-allocate the 4 per-step input arrays + their feature wrappers.
        // Data pointer is rewritten each decode step; no alloc churn.
        self.reusableInpTok = try! MLMultiArray(shape: [1, 1], dataType: .int32)
        self.reusableInpPos = try! MLMultiArray(shape: [1], dataType: .float32)
        self.reusableCos = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.rotaryDim)], dataType: .float16)
        self.reusableSin = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.rotaryDim)], dataType: .float16)
        self.fvInpTok = MLFeatureValue(multiArray: reusableInpTok)
        self.fvInpPos = MLFeatureValue(multiArray: reusableInpPos)
        self.fvCos = MLFeatureValue(multiArray: reusableCos)
        self.fvSin = MLFeatureValue(multiArray: reusableSin)
    }

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

    private func isLinearAttn(_ i: Int) -> Bool { i % 4 != 3 }

    // MARK: - Loading

    /// Optional override folder (e.g., `Documents/Models/qwen3.5-0.8b/`)
    /// to look inside for the mlpackage. When set, this is tried before
    /// the Documents/ top-level or the app bundle.
    var modelFolderOverride: URL?

    /// Resolution order (first hit wins):
    ///   1. `modelFolderOverride/<base>.mlpackage`  (on-device compile)
    ///   2. `<Documents>/<base>.mlmodelc`           (pre-compiled, hot-swap)
    ///   3. `<Documents>/<base>.mlpackage`          (on-device compile)
    ///   4. app bundle `<base>.mlmodelc`
    private func resolveModelURL(_ base: String) throws -> URL {
        if let folder = modelFolderOverride {
            let pkg = folder.appendingPathComponent("\(base).mlpackage")
            if FileManager.default.fileExists(atPath: pkg.path) {
                return try MLModel.compileModel(at: pkg)
            }
            let mlc = folder.appendingPathComponent("\(base).mlmodelc")
            if FileManager.default.fileExists(atPath: mlc.path) {
                return mlc
            }
        }
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let docMLC = docs.appendingPathComponent("\(base).mlmodelc")
        if FileManager.default.fileExists(atPath: docMLC.path) {
            return docMLC
        }
        let docPkg = docs.appendingPathComponent("\(base).mlpackage")
        if FileManager.default.fileExists(atPath: docPkg.path) {
            return try MLModel.compileModel(at: docPkg)
        }
        if let bundled = Bundle.main.url(forResource: base, withExtension: "mlmodelc") {
            return bundled
        }
        throw NSError(domain: "Qwen35Generator", code: 10,
            userInfo: [NSLocalizedDescriptionKey:
                "\(base) not found in \(modelFolderOverride?.path ?? "-") / Documents / app bundle"])
    }

    /// Deprecated — retained for API compat. `generate(...)` now calls
    /// `loadDecodeOnly()` directly. Kept so external callers don't break.
    func load() throws {
        try loadDecodeOnly()
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

    // MARK: - Generation entry

    /// Generate up to `maxNewTokens` tokens starting from `inputIds`.
    /// Uses the decode model as a recurrent-prefill: each prompt token is
    /// fed through the decode step one by one, building state incrementally.
    /// This avoids the monolithic stateful prefill mlpackage entirely, which
    /// keeps the whole generation on a single model — no cross-backend
    /// handoff, no Metal heap during prefill, and the ANE E5 compile happens
    /// exactly once for the decode graph.
    ///
    /// `temperature` = 0.0 enables greedy argmax. Any positive value enables
    /// top-K sampling (K=40 default) which is important for ANE since argmax
    /// fragility on the 248K vocab + base-model greedy loops produce
    /// repetitive output. `topP` (nucleus) and `repetitionPenalty` further
    /// suppress loops.
    @discardableResult
    func generate(inputIds: [Int32], maxNewTokens: Int = 32,
                   temperature: Float = 0.7, topK: Int = 40,
                   repetitionPenalty: Float = 1.1,
                   eosTokenIds: Set<Int32> = [],
                   onToken: ((Int32) -> Void)? = nil) async throws -> [Int32] {
        running = true
        defer { running = false }
        generatedIds.removeAll()

        if decode == nil { try loadDecodeOnly() }
        guard let decode else { return [] }

        let S = inputIds.count
        guard S > 0, S <= cfg.maxSeq - 1 else {
            throw NSError(domain: "Qwen35Generator", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "input length \(S) must be in (0, \(cfg.maxSeq - 1)]"])
        }

        // Zero-init state tensors (24 layers × 2 states each)
        let states: [String: MLMultiArray] = try makeZeroStates()
        var initialStateFVs: [String: MLFeatureValue]? = {
            var d: [String: MLFeatureValue] = [:]
            d.reserveCapacity(states.count)
            for (k, v) in states { d[k] = MLFeatureValue(multiArray: v) }
            return d
        }()
        /// Last predict output — used as the state source for the next call.
        /// Avoids 48 MLFeatureValue wraps per step by delegating feature
        /// lookup to the output provider (which already holds MLFeatureValues).
        var lastOutput: MLFeatureProvider? = nil

        // --- 1. Recurrent prefill (feed prompt through decode step by step) ---
        status = "Prefill (recurrent via decode)..."
        let prefillStart = Date()
        var lastLogits: MLMultiArray?
        var lastNextToken: Int32 = 0
        for (t, tok) in inputIds.enumerated() {
            writeDecodeScalars(token: tok, position: t)
            let features = Qwen35DecodeFeatures(
                tok: fvInpTok, pos: fvInpPos, cos: fvCos, sin: fvSin,
                prevOut: lastOutput, initialStateFVs: initialStateFVs,
                stateInputNames: stateInputNames,
                stateOutputNames: stateOutputNames)
            let dOut = try await decode.prediction(from: features)
            lastOutput = dOut
            initialStateFVs = nil  // zeros only needed on first call
            if t == S - 1 {
                if decodeHasInGraphArgmax {
                    if let ntArr = dOut.featureValue(for: "next_token")?.multiArrayValue {
                        lastNextToken = ntArr.dataPointer
                            .assumingMemoryBound(to: Int32.self)[0]
                    }
                } else {
                    lastLogits = dOut.featureValue(for: "logits")?.multiArrayValue
                }
            }
            if (t & 0x7) == 0 {
                status = "Prefill \(t + 1)/\(S)..."
            }
        }
        prefillMs = Date().timeIntervalSince(prefillStart) * 1000
        // First new token: either from in-graph argmax (int32 direct) or
        // by sampling/argmax over logits in Swift.
        var rng = SystemRandomNumberGenerator()
        let recentHistory: Int = 64
        var nextToken: Int32
        if decodeHasInGraphArgmax {
            nextToken = lastNextToken
            firstStepDebug = []  // debug panel N/A when logits aren't read
        } else {
            guard let pLogits = lastLogits else {
                throw NSError(domain: "Qwen35Generator", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "recurrent prefill: no final logits"])
            }
            copyLogitsInto(&logitsBuffer, arr: pLogits, position: 0, vocab: cfg.vocab)
            let debugSorted = (0..<cfg.vocab).sorted { logitsBuffer[$0] > logitsBuffer[$1] }
            firstStepDebug = debugSorted.prefix(5).map { (Int32($0), logitsBuffer[$0]) }
            nextToken = samplePosition(pLogits, position: 0, vocab: cfg.vocab,
                                         priorTokens: inputIds,
                                         temperature: temperature, topK: topK,
                                         repetitionPenalty: repetitionPenalty,
                                         recentN: recentHistory, rng: &rng)
        }
        generatedIds.append(nextToken)
        onToken?(nextToken)

        // --- 2. Decode loop (with per-phase profiling) ---
        status = "Decoding..."
        var decodeTotal = 0.0
        // Carry over the previous call's profile so short responses still
        // contribute data points. `resetProfile()` explicitly zeroes.
        var sumInputs = decodeProfile.inputsBuild * Double(decodeProfile.count)
        var sumPredict = decodeProfile.predict * Double(decodeProfile.count)
        var sumStateCopy = decodeProfile.stateCopy * Double(decodeProfile.count)
        var sumLogit = decodeProfile.logitRead * Double(decodeProfile.count)
        var profileCount = decodeProfile.count
        let decodeStart = Date()
        var position = S
        for step in 0..<(maxNewTokens - 1) {
            if position >= cfg.maxSeq { break }
            let stepStart = CFAbsoluteTimeGetCurrent()

            let tIn0 = CFAbsoluteTimeGetCurrent()
            writeDecodeScalars(token: nextToken, position: position)
            let features = Qwen35DecodeFeatures(
                tok: fvInpTok, pos: fvInpPos, cos: fvCos, sin: fvSin,
                prevOut: lastOutput, initialStateFVs: nil,
                stateInputNames: stateInputNames,
                stateOutputNames: stateOutputNames)
            let tIn1 = CFAbsoluteTimeGetCurrent()

            let dOut = try await decode.prediction(from: features)
            let tPred = CFAbsoluteTimeGetCurrent()

            // State plumbing: just retain the output — its new_state_X_Y
            // features are mapped on-the-fly in Qwen35DecodeFeatures.
            lastOutput = dOut
            let tState = CFAbsoluteTimeGetCurrent()

            let stepEnd = tState  // logit read happens after; split below
            decodeTotal += (stepEnd - stepStart) * 1000

            if decodeHasInGraphArgmax {
                guard let ntArr = dOut.featureValue(for: "next_token")?.multiArrayValue
                else { break }
                nextToken = ntArr.dataPointer.assumingMemoryBound(to: Int32.self)[0]
            } else {
                guard let dLogits = dOut.featureValue(for: "logits")?.multiArrayValue
                else { break }
                let priorTokens = inputIds + generatedIds
                nextToken = samplePosition(dLogits, position: 0, vocab: cfg.vocab,
                                             priorTokens: priorTokens,
                                             temperature: temperature, topK: topK,
                                             repetitionPenalty: repetitionPenalty,
                                             recentN: recentHistory, rng: &rng)
            }
            let tLogit = CFAbsoluteTimeGetCurrent()

            // Accumulate per-phase timings for every step. Previously we
            // skipped step 0-1 as warmup, but that loses signal on short
            // responses (EOS firing before step 2). The first-call cold
            // path adds ~1-2ms noise at most, negligible in aggregate.
            sumInputs    += (tIn1 - tIn0) * 1000
            sumPredict   += (tPred - tIn1) * 1000
            sumStateCopy += (tState - tPred) * 1000
            sumLogit     += (tLogit - tState) * 1000
            profileCount += 1

            generatedIds.append(nextToken)
            onToken?(nextToken)
            position += 1
            if (step & 0x7) == 0 {
                status = "Decoding \(step + 2)/\(maxNewTokens)..."
            }
            if eosTokenIds.contains(nextToken) { break }
        }

        let totalDecodeMs = Date().timeIntervalSince(decodeStart) * 1000
        let decodedCount = max(generatedIds.count - 1, 1)
        decodeMsAvg = totalDecodeMs / Double(decodedCount)
        tokensPerSecond = Double(generatedIds.count) / ((prefillMs + totalDecodeMs) / 1000.0)
        if profileCount > 0 {
            let n = Double(profileCount)
            decodeProfile = PhaseProfile(
                inputsBuild: sumInputs / n,
                predict: sumPredict / n,
                stateCopy: sumStateCopy / n,
                logitRead: sumLogit / n,
                total: (sumInputs + sumPredict + sumStateCopy + sumLogit) / n,
                count: profileCount
            )
        }
        status = String(format: "Done: %d tokens, prefill=%.0fms, decode=%.1fms/tok, %.1f tok/s",
                         generatedIds.count, prefillMs, decodeMsAvg, tokensPerSecond)
        return generatedIds
    }

    /// Load only the decode model. Preference order:
    ///   1. argmax-in-graph fp16 (`next_token` int32 output) — not shipped
    ///   2. INT8 palettized (754 MB, same parity as fp16, preferred default)
    ///   3. fp16 (1.4 GB, ground-truth precision)
    func loadDecodeOnly() throws {
        let dCfg = MLModelConfiguration(); dCfg.computeUnits = cfg.decodeUnits
        let loadedURL: URL
        let variant: String
        // 2B INT8 preferred when present (opt-in via Documents/ push);
        // falls back to 0.8B variants otherwise.
        if let url = try? resolveModelURL("qwen3_5_2b_decode_int8_mseq128") {
            decode = try MLModel(contentsOf: url, configuration: dCfg)
            decodeHasInGraphArgmax = false
            loadedURL = url; variant = "2B-int8"
        } else if let url = try? resolveModelURL("qwen3_5_0_8b_decode_argmax_fp16_mseq128") {
            decode = try MLModel(contentsOf: url, configuration: dCfg)
            decodeHasInGraphArgmax = true
            loadedURL = url; variant = "0.8B-argmax-fp16"
        } else if let url = try? resolveModelURL("qwen3_5_0_8b_decode_int8_mseq128") {
            decode = try MLModel(contentsOf: url, configuration: dCfg)
            decodeHasInGraphArgmax = false
            loadedURL = url; variant = "0.8B-int8"
        } else {
            let dURL = try resolveModelURL("qwen3_5_0_8b_decode_fp16_mseq128")
            decode = try MLModel(contentsOf: dURL, configuration: dCfg)
            decodeHasInGraphArgmax = false
            loadedURL = dURL; variant = "0.8B-fp16"
        }
        status = "Loaded decode (\(variant)) on \(unitsName(cfg.decodeUnits))"
        // Diagnostic: print ANE op-placement percentage so we can verify
        // the chosen compute units are actually being honored. Prior bug:
        // default silently ran on GPU despite being set to ANE, because a
        // debug override wasn't reverted. Logging here catches that.
        auditComputePlan(url: loadedURL, requestedUnits: cfg.decodeUnits, variant: variant)
    }

    private func auditComputePlan(url: URL, requestedUnits: MLComputeUnits, variant: String) {
        let cfg = MLModelConfiguration(); cfg.computeUnits = requestedUnits
        Task.detached(priority: .utility) {
            guard #available(iOS 17.0, *) else { return }
            do {
                let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
                guard case .program(let program) = plan.modelStructure else {
                    print("[Qwen35] compute plan: structure is not a program")
                    return
                }
                var total = 0, ane = 0, gpu = 0, cpu = 0, other = 0
                for (_, fn) in program.functions {
                    Self.walkOps(fn.block, plan: plan,
                                  total: &total, ane: &ane, gpu: &gpu,
                                  cpu: &cpu, other: &other)
                }
                let compute = max(1, total)
                print(String(format:
                    "[Qwen35] compute plan (\(variant), requested=\(Self.unitsLabel(requestedUnits))): total=%d ANE=%d (%.1f%%) GPU=%d (%.1f%%) CPU=%d (%.1f%%)",
                    total, ane, 100.0*Double(ane)/Double(compute),
                    gpu, 100.0*Double(gpu)/Double(compute),
                    cpu, 100.0*Double(cpu)/Double(compute)))
            } catch {
                print("[Qwen35] compute plan audit failed: \(error)")
            }
        }
    }

    private static func unitsLabel(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "GPU"
        case .cpuAndNeuralEngine: return "ANE"
        case .all: return "All"
        @unknown default: return "?"
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
                for inner in op.blocks { walkOps(inner, plan: plan,
                    total: &total, ane: &ane, gpu: &gpu, cpu: &cpu, other: &other) }
                continue
            }
            total += 1
            // MLComputeDevice is an enum in iOS 18+; the `is SomeDevice`
            // type check we used first doesn't match — all ops fell into
            // `other`. Use pattern matching on the enum cases.
            switch plan.deviceUsage(for: op)?.preferred {
            case .cpu:          cpu += 1
            case .gpu:          gpu += 1
            case .neuralEngine: ane += 1
            default:            other += 1
            }
            for inner in op.blocks {
                walkOps(inner, plan: plan,
                        total: &total, ane: &ane, gpu: &gpu,
                        cpu: &cpu, other: &other)
            }
        }
    }

    /// Build zero-initialized state tensors matching the decode inputs.
    /// MLMultiArray(shape:dataType:) does NOT zero-init on iOS — we must
    /// explicitly memset the backing buffer. Using uninit memory as initial
    /// state produces garbage logits (seen in the wild as "!!!!" degenerate
    /// output with sampling).
    private func makeZeroStates() throws -> [String: MLMultiArray] {
        var dict: [String: MLMultiArray] = [:]
        let linearConvShape: [NSNumber] = [1, 6144, 4]
        let linearRecShape:  [NSNumber] = [1, 16, 128, 128]
        let fullKvShape:     [NSNumber] = [1, 2, NSNumber(value: cfg.maxSeq), 256]
        for i in 0..<cfg.numLayers {
            let isLinear = (i % 4 != 3)
            let shapeA = isLinear ? linearConvShape : fullKvShape
            let shapeB = isLinear ? linearRecShape  : fullKvShape
            let a = try MLMultiArray(shape: shapeA, dataType: .float16)
            let b = try MLMultiArray(shape: shapeB, dataType: .float16)
            // Explicitly zero the fp16 buffer
            memset(a.dataPointer, 0, a.count * 2)
            memset(b.dataPointer, 0, b.count * 2)
            dict["state_\(i)_a"] = a
            dict["state_\(i)_b"] = b
        }
        return dict
    }

    // MARK: - Input builders

    /// Write per-step scalar inputs (token, position) and RoPE tables into
    /// the pre-allocated reusable MLMultiArrays. Called once per decode
    /// step — state plumbing is handled by Qwen35DecodeFeatures without
    /// touching MLMultiArrays at all.
    private func writeDecodeScalars(token: Int32, position: Int) {
        reusableInpTok.dataPointer.assumingMemoryBound(to: Int32.self)[0] = token
        reusableInpPos.dataPointer.assumingMemoryBound(to: Float.self)[0] = Float(position)
        let rd = cfg.rotaryDim
        let cp = reusableCos.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sp = reusableSin.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<rd {
            cp[i] = Float16(cosTable[position * rd + i]).bitPattern
            sp[i] = Float16(sinTable[position * rd + i]).bitPattern
        }
    }

    /// Legacy: kept for API surface compat (was used before the custom
    /// MLFeatureProvider). Unused in the hot path now.
    private func makeDecodeInputs(token: Int32, position: Int,
                                   states: [String: MLMultiArray]
                                   ) throws -> MLDictionaryFeatureProvider {
        // Scalar inputs: write into the single reusable MLMultiArray
        reusableInpTok.dataPointer.assumingMemoryBound(to: Int32.self)[0] = token
        reusableInpPos.dataPointer.assumingMemoryBound(to: Float.self)[0] = Float(position)
        // cos/sin: memcpy the UInt16 bits from the pre-built float table
        let rd = cfg.rotaryDim
        let cp = reusableCos.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sp = reusableSin.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<rd {
            cp[i] = Float16(cosTable[position * rd + i]).bitPattern
            sp[i] = Float16(sinTable[position * rd + i]).bitPattern
        }

        var feat: [String: MLFeatureValue] = [
            "input_token": fvInpTok,
            "position":    fvInpPos,
            "cos":         fvCos,
            "sin":         fvSin,
        ]
        feat.reserveCapacity(4 + states.count)
        for (k, v) in states {
            feat[k] = MLFeatureValue(multiArray: v)
        }
        return try MLDictionaryFeatureProvider(dictionary: feat)
    }

    // MARK: - Argmax over (1, S, V) or (1, 1, V)

    /// Direct fp16 argmax with stride-safe single-pass scan.
    /// Key perf trick: compare as Float16 directly (native on Apple Silicon
    /// — no Float32 conversion in the hot loop). Compiler unrolls / auto-
    /// vectorizes the tight loop with SIMD fp16 compare on A-series chips.
    private func fastArgmax(_ arr: MLMultiArray, position: Int, vocab: Int) -> Int32 {
        let strides = arr.strides.map(\.intValue)
        let reportedV = strides.last ?? 0
        let reportedP = strides.count >= 2 ? strides[strides.count - 2] : 0
        let vStride = (reportedV >= 1 && reportedP >= vocab) ? reportedV : 1
        let pStride = (reportedV >= 1 && reportedP >= vocab) ? reportedP : vocab
        let offset = position * pStride
        switch arr.dataType {
        case .float16 where vStride == 1:
            // Hottest path: stride-1 fp16 compare. Inner loop is a single
            // load + compare on Apple Silicon, auto-vectorized to NEON fp16.
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self).advanced(by: offset)
            var bestV: Float16 = -.infinity
            var bestIdx: Int = 0
            for v in 0..<vocab {
                let x = p[v]
                if x > bestV { bestV = x; bestIdx = v }
            }
            return Int32(bestIdx)
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self)
            var bestV: Float16 = -.infinity
            var bestIdx: Int = 0
            for v in 0..<vocab {
                let x = p[offset + v * vStride]
                if x > bestV { bestV = x; bestIdx = v }
            }
            return Int32(bestIdx)
        case .float32 where vStride == 1:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: offset)
            var maxV: Float = 0
            var idx: vDSP_Length = 0
            vDSP_maxvi(p, 1, &maxV, &idx, vDSP_Length(vocab))
            return Int32(idx)
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            var bestV: Float = -.infinity
            var bestIdx: Int = 0
            for v in 0..<vocab {
                let x = p[offset + v * vStride]
                if x > bestV { bestV = x; bestIdx = v }
            }
            return Int32(bestIdx)
        default:
            var bestV: Float = -.infinity
            var bestIdx: Int = 0
            for v in 0..<vocab {
                let x = arr[[0, NSNumber(value: position), NSNumber(value: v)] as [NSNumber]].floatValue
                if x > bestV { bestV = x; bestIdx = v }
            }
            return Int32(bestIdx)
        }
    }

    /// In-place logit copy into a caller-provided Float buffer (reused).
    private func copyLogitsInto(_ out: inout [Float], arr: MLMultiArray,
                                 position: Int, vocab: Int) {
        let strides = arr.strides.map(\.intValue)
        let reportedV = strides.last ?? 0
        let reportedP = strides.count >= 2 ? strides[strides.count - 2] : 0
        let vStride = (reportedV >= 1 && reportedP >= vocab) ? reportedV : 1
        let pStride = (reportedV >= 1 && reportedP >= vocab) ? reportedP : vocab
        let offset = position * pStride
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for v in 0..<vocab {
                let x = p[offset + v * vStride]
                out[v] = x.isFinite ? x : 0
            }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            for v in 0..<vocab {
                let x = Float(Float16(bitPattern: p[offset + v * vStride]))
                out[v] = x.isFinite ? x : 0
            }
        default:
            for v in 0..<vocab {
                out[v] = arr[[0, NSNumber(value: position), NSNumber(value: v)] as [NSNumber]]
                    .floatValue
            }
        }
    }

    /// Logit copy. Uses reported strides if they look sane; else falls back
    /// to contiguous row-major layout (vStride=1, pStride=vocab), which is
    /// the actual layout for the decode model's (1,1,V) output on all
    /// tested backends. The `.strides` property of Core ML outputs can
    /// return zero-strides for ANE-dispatched tensors — trusting it blindly
    /// collapses all indices to one value, producing "!!!!" degenerate
    /// output (token id 0 on Qwen tokenizer).
    private func copyLogits(_ arr: MLMultiArray, position: Int, vocab: Int) -> [Float] {
        var out = [Float](repeating: 0, count: vocab)
        let strides = arr.strides.map(\.intValue)
        let reportedV = strides.last ?? 0
        let reportedP = strides.count >= 2 ? strides[strides.count - 2] : 0
        // Sanity: last stride ≥ 1, position stride ≥ vocab (so v*vS < pS).
        let vStride: Int
        let pStride: Int
        if reportedV >= 1 && reportedP >= vocab {
            vStride = reportedV
            pStride = reportedP
        } else {
            vStride = 1
            pStride = vocab
        }
        let offset = position * pStride
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for v in 0..<vocab {
                let x = p[offset + v * vStride]
                out[v] = x.isFinite ? x : 0
            }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            for v in 0..<vocab {
                let x = Float(Float16(bitPattern: p[offset + v * vStride]))
                out[v] = x.isFinite ? x : 0
            }
        default:
            for v in 0..<vocab {
                let x = arr[[0, NSNumber(value: position), NSNumber(value: v)] as [NSNumber]]
                    .floatValue
                out[v] = x.isFinite ? x : 0
            }
        }
        return out
    }

    /// Token sampling with temperature + top-K + repetition penalty.
    /// Falls back to argmax when temperature = 0.
    private func samplePosition(_ arr: MLMultiArray, position: Int, vocab: Int,
                                 priorTokens: [Int32],
                                 temperature: Float, topK: Int,
                                 repetitionPenalty: Float,
                                 recentN: Int,
                                 rng: inout SystemRandomNumberGenerator) -> Int32 {
        // Fast path: argmax directly on the MLMultiArray without allocating
        // a 248K Float buffer. Single pass, stride-safe fp16 read, no
        // intermediate NaN-filtered copy.
        //  - No rep_penalty → plain fastArgmax.
        //  - With rep_penalty in greedy mode → single-pass argmax that
        //    also tracks the best "non-recent" token, so if the global
        //    argmax is in the recent window we can fall back to the
        //    non-recent best without a second scan or a 1MB Float buffer.
        if temperature <= 0 {
            if repetitionPenalty <= 1.0 {
                return fastArgmax(arr, position: position, vocab: vocab)
            }
            let start = max(0, priorTokens.count - recentN)
            var recent = Set<Int32>()
            recent.reserveCapacity(priorTokens.count - start)
            for i in start..<priorTokens.count { recent.insert(priorTokens[i]) }
            return fastArgmaxAvoidingRecent(arr, position: position,
                                              vocab: vocab, recent: recent)
        }
        // Sampling / rep-penalty path reuses the pre-allocated logitsBuffer
        // to avoid 1 MB heap allocation every decode step.
        copyLogitsInto(&logitsBuffer, arr: arr, position: position, vocab: vocab)
        var logits = logitsBuffer
        // Repetition penalty on recent tokens (small set, scalar loop is fine)
        if repetitionPenalty > 1.0 {
            let start = max(0, priorTokens.count - recentN)
            var seen = Set<Int32>()
            for i in start..<priorTokens.count { seen.insert(priorTokens[i]) }
            for t in seen {
                let idx = Int(t)
                if idx < 0 || idx >= vocab { continue }
                if logits[idx] > 0 { logits[idx] /= repetitionPenalty }
                else                 { logits[idx] *= repetitionPenalty }
            }
        }
        // Greedy-with-rep-penalty: pick argmax on adjusted logits
        // without entering the softmax sampling path. This is what loop-
        // breaking mode wants: deterministic but not locked-in.
        if temperature <= 0 {
            var best = 0; var bv: Float = -.infinity
            for v in 0..<vocab { if logits[v] > bv { bv = logits[v]; best = v } }
            return Int32(best)
        }
        // Temperature scaling via SIMD
        if temperature != 1.0 {
            var inv: Float = 1.0 / temperature
            vDSP_vsmul(logits, 1, &inv, &logits, 1, vDSP_Length(vocab))
        }
        // Top-K via partial nth_element style: find Kth-largest threshold,
        // then scan once to collect indices ≥ threshold. Much faster than
        // O(V*K) insert-sort on 248K vocab.
        let k = max(1, min(topK, vocab))
        // Approach: sort a copy of logits descending and pick threshold.
        // For K=40 on V=248K this is still expensive. Use quickselect via
        // partitioning a mutable copy.
        var sortedCopy = logits
        // Swift stdlib sort is intro-sort, O(N log N). For 248K this is
        // ~15ms — acceptable and correct.
        sortedCopy.sort(by: >)
        let threshold = sortedCopy[k - 1]
        // Collect top-K indices (handle ties by order)
        var topIdx = [Int](); topIdx.reserveCapacity(k)
        var topVal = [Float](); topVal.reserveCapacity(k)
        for v in 0..<vocab {
            if logits[v] >= threshold {
                topIdx.append(v); topVal.append(logits[v])
                if topIdx.count >= k { break }
            }
        }
        // Softmax over top-K (numerically stable)
        let maxV = topVal.max() ?? 0
        var exps = [Float](repeating: 0, count: topIdx.count)
        var sum: Float = 0
        for i in 0..<topIdx.count {
            let e = expf(topVal[i] - maxV)
            exps[i] = e; sum += e
        }
        // Draw
        let r = Float.random(in: 0..<1, using: &rng) * sum
        var acc: Float = 0
        for i in 0..<topIdx.count {
            acc += exps[i]
            if r < acc { return Int32(topIdx[i]) }
        }
        return Int32(topIdx[topIdx.count - 1])
    }

    /// Greedy argmax that rejects the best token if it's in `recent`,
    /// returning the next-best non-recent token instead. Single pass
    /// over the 248K vocab in fp16 without any intermediate buffer —
    /// same cost as `fastArgmax` plus a Set lookup per iteration.
    ///
    /// This is the loop-breaking path used when rep_penalty > 1.0 on
    /// greedy decode: INT8 quantization noise compounding the SSM state
    /// can pin the argmax onto a repeating token; rejecting it once per
    /// step is enough to escape.
    private func fastArgmaxAvoidingRecent(_ arr: MLMultiArray, position: Int,
                                           vocab: Int, recent: Set<Int32>) -> Int32 {
        let strides = arr.strides.map(\.intValue)
        let reportedV = strides.last ?? 0
        let reportedP = strides.count >= 2 ? strides[strides.count - 2] : 0
        let vStride = (reportedV >= 1 && reportedP >= vocab) ? reportedV : 1
        let pStride = (reportedV >= 1 && reportedP >= vocab) ? reportedP : vocab
        let offset = position * pStride
        var bestIdx = 0
        var bestV: Float16 = -.infinity
        var bestAltIdx = 0
        var bestAltV: Float16 = -.infinity
        if arr.dataType == .float16, vStride == 1 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self).advanced(by: offset)
            for v in 0..<vocab {
                let x = p[v]
                if x > bestV { bestV = x; bestIdx = v }
                if !recent.contains(Int32(v)) && x > bestAltV {
                    bestAltV = x; bestAltIdx = v
                }
            }
        } else {
            // Fallback: stride-aware or fp32 via general path
            let logits = copyLogits(arr, position: position, vocab: vocab)
            var bestF: Float = -.infinity
            var bestAltF: Float = -.infinity
            for v in 0..<vocab {
                let x = logits[v]
                if x > bestF { bestF = x; bestIdx = v }
                if !recent.contains(Int32(v)) && x > bestAltF {
                    bestAltF = x; bestAltIdx = v
                }
            }
        }
        return recent.contains(Int32(bestIdx)) ? Int32(bestAltIdx) : Int32(bestIdx)
    }

    private func argmaxAtPosition(_ arr: MLMultiArray, position: Int, vocab: Int) -> Int32 {
        // Stride-safe fallback via copyLogits
        let logits = copyLogits(arr, position: position, vocab: vocab)
        var best: Int32 = 0
        var bv: Float = -.infinity
        for v in 0..<vocab {
            if logits[v] > bv { bv = logits[v]; best = Int32(v) }
        }
        return best
    }

    private func argmaxAtPositionLegacy(_ arr: MLMultiArray, position: Int, vocab: Int) -> Int32 {
        let base = position * vocab
        var best: Int32 = 0
        if arr.dataType == .float32 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            var bv: Float = -.infinity
            for v in 0..<vocab {
                let x = p[base + v]
                if x > bv { bv = x; best = Int32(v) }
            }
        } else if arr.dataType == .float16 {
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            var bv: Float = -.infinity
            for v in 0..<vocab {
                let x = Float(Float16(bitPattern: p[base + v]))
                if x > bv { bv = x; best = Int32(v) }
            }
        }
        return best
    }
}
