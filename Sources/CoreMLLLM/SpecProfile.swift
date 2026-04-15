//
//  SpecProfile.swift
//  CoreMLLLM
//
//  Per-burst timing logs for the speculative-decoding engines.
//
//  Activated by setting `SPECULATIVE_PROFILE=1` in the environment, or
//  the `SPECULATIVE_PROFILE` UserDefaults bool. Disabled by default so
//  baseline runs see no overhead beyond the `isEnabled` check (one env
//  lookup cached at first read).
//
//  Log format (single line per burst, fields tab-separated for easy
//  awk-parsing) — the MTP and CrossVocab engines and the DrafterUnion
//  all emit through `SpecProfile.logBurst` with engine tag {"mtp",
//  "cv", "union"} so a single grep across an iPhone session log gives
//  comparable distributions:
//
//      [SpecProfile cv #042] draft=14.3ms verify=53.2ms commit=0.4ms
//          accepted=2/2 emitted=3 rolling=0.62
//
//      [SpecProfile union #042 src=cv] draft_total=14.6ms (cv=14.3
//          pl3=0.05 pl2=0.05) verify=53.2ms commit=0.4ms
//          accepted=2/2 emitted=3
//
//  Bootstrap and fallback emit their own single-line entries with the
//  same tag-prefix discipline so the consumer can distinguish.
//

import Foundation

enum SpecProfile {

    /// Cached at first access. Re-read is cheap but env access is not
    /// guaranteed thread-safe across processes; cache once.
    static let isEnabled: Bool = {
        if ProcessInfo.processInfo.environment["SPECULATIVE_PROFILE"] != nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: "SPECULATIVE_PROFILE")
    }()

    /// Chat-CV residual investigation (docs/PHASE_B_CHAT_CV_RESIDUAL.md).
    /// Additive to `SPECULATIVE_PROFILE`; prints pre-propose / post-propose /
    /// post-commit CV state per burst so we can disentangle rolling-gate
    /// closure, bootstrap replay effects, and mid-burst state drift.
    /// Strictly env-gated — zero overhead when unset.
    static let isUnionDebugCV: Bool = {
        ProcessInfo.processInfo.environment["UNION_DEBUG_CV"] != nil
    }()

    /// Structured per-burst CV-state log for UNION_DEBUG_CV. All fields on a
    /// single tab-friendly line so awk / python can parse directly.
    static func logUnionDebugCV(cycle: Int,
                                source: String,
                                rollingCV: Double,
                                rollingPL3: Double,
                                rollingPL2: Double,
                                cvProposed: Bool,
                                cvPosBefore: Int,
                                cvPosAfterPropose: Int,
                                cvPosAfterRewind: Int,
                                cvPosAfterCommit: Int,
                                enginePosBefore: Int,
                                enginePosAfterCommit: Int,
                                matchCount: Int,
                                compareLen: Int) {
        guard isUnionDebugCV else { return }
        print(String(format:
            "[UnionDebugCV #%04d src=%@] rCV=%.3f rPL3=%.3f rPL2=%.3f "
          + "cvProposed=%d cvPos[before=%d afterPropose=%d afterRewind=%d afterCommit=%d] "
          + "enginePos[before=%d afterCommit=%d] match=%d/%d",
            cycle, source, rollingCV, rollingPL3, rollingPL2,
            cvProposed ? 1 : 0,
            cvPosBefore, cvPosAfterPropose, cvPosAfterRewind, cvPosAfterCommit,
            enginePosBefore, enginePosAfterCommit,
            matchCount, compareLen))
    }

    /// Time a throwing block. Returns (result, elapsed milliseconds).
    /// Always runs the block — caller decides whether to log based on
    /// `isEnabled`. Kept un-gated because the timing itself is cheap
    /// (one CFAbsoluteTimeGetCurrent pair).
    @inline(__always)
    static func time<T>(_ block: () throws -> T) rethrows -> (T, Double) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let r = try block()
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        return (r, ms)
    }

    /// Per-burst log line for the simple MTP / CrossVocab path.
    static func logBurst(engine: String,
                         cycle: Int,
                         draftMs: Double,
                         verifyMs: Double,
                         commitMs: Double,
                         accepted: Int,
                         compareLen: Int,
                         emitted: Int,
                         rolling: Double) {
        guard isEnabled else { return }
        print(String(format:
            "[SpecProfile %@ #%04d] draft=%.2fms verify=%.2fms commit=%.2fms "
          + "accepted=%d/%d emitted=%d rolling=%.3f",
            engine, cycle, draftMs, verifyMs, commitMs,
            accepted, compareLen, emitted, rolling))
    }

    /// Per-burst log line for the DrafterUnion path (multiple drafters
    /// run, only one is selected). `perSourceMs` keys: "cv", "pl3", "pl2".
    static func logUnionBurst(cycle: Int,
                              source: String,
                              perSourceMs: [String: Double],
                              verifyMs: Double,
                              commitMs: Double,
                              accepted: Int,
                              compareLen: Int,
                              emitted: Int) {
        guard isEnabled else { return }
        let total = perSourceMs.values.reduce(0, +)
        let cv = perSourceMs["cv"] ?? 0
        let p3 = perSourceMs["pl3"] ?? 0
        let p2 = perSourceMs["pl2"] ?? 0
        print(String(format:
            "[SpecProfile union #%04d src=%@] draft_total=%.2fms (cv=%.2f pl3=%.3f pl2=%.3f) "
          + "verify=%.2fms commit=%.2fms accepted=%d/%d emitted=%d",
            cycle, source, total, cv, p3, p2,
            verifyMs, commitMs, accepted, compareLen, emitted))
    }

    /// One-shot bootstrap log. Bootstrap costs (especially the
    /// per-token Qwen replay) dominate TTFT for the cross-vocab path
    /// so isolating the number is high-signal.
    static func logBootstrap(engine: String,
                             replayCount: Int,
                             replayMs: Double,
                             targetStepMs: Double) {
        guard isEnabled else { return }
        print(String(format:
            "[SpecProfile %@ bootstrap] replay=%d (%.2fms) target_step=%.2fms",
            engine, replayCount, replayMs, targetStepMs))
    }

    /// Single-step fallback (drafter couldn't propose / disabled).
    static func logFallback(engine: String,
                            cycle: Int,
                            targetStepMs: Double) {
        guard isEnabled else { return }
        print(String(format:
            "[SpecProfile %@ #%04d fallback] target_step=%.2fms",
            engine, cycle, targetStepMs))
    }
}
