import Foundation

/// Per-step sampling helper. Converts top-K logits into a committed token
/// id, applying temperature, top-K, top-P, min-P, and repetition penalty.
///
/// The implementation is intentionally allocation-free in the hot path
/// (fixed-size buffers sized to `options.topK ?? 64`) so adding sampling
/// does not regress the greedy token rate.
///
/// **Input contract.** The caller supplies top-K `(tokenId, logit)` pairs
/// sorted by descending logit. When the runtime can only provide the
/// argmax (the common case today), the caller passes `topK == 1`, and
/// the sampler trivially forwards it.
struct Sampler {
    struct Candidate {
        var tokenId: Int32
        var logit: Float
    }

    let options: GenerationOptions
    /// Capped ring of recent decode tokens, used for repetition penalty.
    private var recentTokens: [Int32] = []
    /// Deterministic RNG when `options.seed` is set.
    private var rng: any RandomNumberGenerator

    init(options: GenerationOptions) {
        self.options = options
        if let seed = options.seed {
            self.rng = SeededRNG(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }

    mutating func sample(from candidates: [Candidate]) -> Int32 {
        guard !candidates.isEmpty else { return 0 }
        // Fast-path: greedy (no-op) — bit-exact with the previous API.
        let t = options.temperature.isNaN ? 0 : options.temperature
        let greedy = t <= 0 && options.topK == 1
        if greedy || candidates.count == 1 {
            let chosen = candidates[0].tokenId
            recordRecent(chosen)
            return chosen
        }

        // Copy to a mutable working set.
        var work = candidates

        // Apply repetition penalty by depressing logits of recently-
        // emitted tokens. The canonical HF formula divides positive
        // logits by `penalty` and multiplies negative logits by it,
        // pushing both toward zero probability.
        if options.repetitionPenalty != 1.0 {
            let penalty = Float(options.repetitionPenalty)
            for i in 0..<work.count {
                if recentTokens.contains(work[i].tokenId) {
                    let l = work[i].logit
                    work[i].logit = l > 0 ? l / penalty : l * penalty
                }
            }
            // Re-sort — penalty can reorder the top-K.
            work.sort { $0.logit > $1.logit }
        }

        // Top-K truncation (0 = keep all).
        if options.topK > 0 && options.topK < work.count {
            work.removeLast(work.count - options.topK)
        }

        // Temperature scaling + softmax.
        let temp = max(Float(t), 1e-4)  // avoid div-by-zero
        var maxL = work[0].logit
        for c in work where c.logit > maxL { maxL = c.logit }
        var probs = [Float](repeating: 0, count: work.count)
        var sum: Float = 0
        for i in 0..<work.count {
            let p = expf((work[i].logit - maxL) / temp)
            probs[i] = p
            sum += p
        }
        if sum > 0 {
            for i in 0..<probs.count { probs[i] /= sum }
        } else {
            // Degenerate — fall back to greedy.
            let chosen = work[0].tokenId
            recordRecent(chosen)
            return chosen
        }

        // min-P: drop any probability below `minP * max_prob`.
        if options.minP > 0 {
            let maxP = probs[0]
            let thresh = Float(options.minP) * maxP
            var kept: [(Int, Float)] = []
            for i in 0..<probs.count where probs[i] >= thresh {
                kept.append((i, probs[i]))
            }
            if !kept.isEmpty {
                var renorm: Float = 0
                for (_, p) in kept { renorm += p }
                if renorm > 0 {
                    var newProbs = [Float](repeating: 0, count: work.count)
                    for (i, p) in kept { newProbs[i] = p / renorm }
                    probs = newProbs
                }
            }
        }

        // top-P (nucleus): walk top→bottom until cumulative ≥ p.
        if options.topP < 1.0 {
            var cutoff = probs.count
            var cum: Float = 0
            for i in 0..<probs.count {
                cum += probs[i]
                if cum >= Float(options.topP) {
                    cutoff = i + 1
                    break
                }
            }
            if cutoff < probs.count {
                for i in cutoff..<probs.count { probs[i] = 0 }
                let renorm = probs.reduce(0, +)
                if renorm > 0 {
                    for i in 0..<probs.count { probs[i] /= renorm }
                }
            }
        }

        // Multinomial draw.
        let r = Float.random(in: 0..<1, using: &rng)
        var cum: Float = 0
        var chosenIdx = probs.count - 1
        for i in 0..<probs.count {
            cum += probs[i]
            if r < cum { chosenIdx = i; break }
        }
        let chosen = work[chosenIdx].tokenId
        recordRecent(chosen)
        return chosen
    }

    private mutating func recordRecent(_ tokenId: Int32) {
        recentTokens.append(tokenId)
        if recentTokens.count > options.repetitionWindow {
            recentTokens.removeFirst(recentTokens.count - options.repetitionWindow)
        }
    }
}

/// Tiny deterministic RNG (splitmix64) so that a fixed seed produces
/// bit-exact token streams across runs. System RNG is used when seed
/// is nil.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
