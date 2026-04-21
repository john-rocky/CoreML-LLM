import Accelerate
import CoreML
import Foundation

/// Memory-mapped INT8 quantized embedding lookup.
///
/// Reads int8 embeddings from disk without loading the entire table into RAM.
/// Each token's embedding is stored as `int8[dim]` with a per-token float16
/// scale: `float16_out = int8_val * (scale_fp16 / 127.0) * embedScale`.
///
/// Uses Accelerate (vDSP + vImage) for vectorized INT8 → FP16 conversion
/// instead of scalar loops (~4-6x faster per lookup).
final class EmbeddingLookup {
    private let data: Data  // memory-mapped (vocabSize * dim, int8)
    private let scales: Data  // memory-mapped (vocabSize, float16)
    private let vocabSize: Int
    private let dim: Int
    private let scale: Float

    // Preallocated buffers for vectorized dequantization
    private var f32Buffer: [Float]
    private var f16Buffer: [UInt16]

    // FP16 LRU cache of recently dequantized embeddings (T1, see
    // docs/LITERT_PERF_ADOPTIONS.md). Chat workloads reuse a small
    // working set of common subword tokens; bypassing the vDSP/vImage
    // dequant on a hit moves that work off the P-core. Each entry is
    // dim * 2 bytes (e.g. 1536 * 2 = 3 KiB), so the default 256-entry
    // cache costs ~768 KiB. The unscaled and PLE variants own their own
    // caches because their post-scale values differ.
    private let cacheCapacity: Int
    private var lookupCache: EmbeddingFP16LRU
    private var lookupUnscaledCache: EmbeddingFP16LRU
    private var lookupRawCache: EmbeddingFP16LRU
    private(set) var cacheHits: Int = 0
    private(set) var cacheMisses: Int = 0

    init(dataURL: URL, scalesURL: URL, vocabSize: Int, dim: Int,
         scale: Float = 1.0, cacheCapacity: Int = 256) throws {
        self.data = try Data(contentsOf: dataURL, options: .mappedIfSafe)
        self.scales = try Data(contentsOf: scalesURL, options: .mappedIfSafe)
        self.vocabSize = vocabSize
        self.dim = dim
        self.scale = scale
        self.f32Buffer = [Float](repeating: 0, count: dim)
        self.f16Buffer = [UInt16](repeating: 0, count: dim)
        self.cacheCapacity = max(0, cacheCapacity)
        self.lookupCache = EmbeddingFP16LRU(capacity: self.cacheCapacity, dim: dim)
        self.lookupUnscaledCache = EmbeddingFP16LRU(capacity: self.cacheCapacity, dim: dim)
        self.lookupRawCache = EmbeddingFP16LRU(capacity: self.cacheCapacity, dim: dim)
    }

    /// Look up embedding for a single token. Returns float16 MLMultiArray.
    func lookup(_ tokenID: Int, shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        if let cached = lookupCache.get(tokenID) {
            memcpy(dstPtr, cached, dim * MemoryLayout<UInt16>.stride)
            cacheHits += 1
            return result
        }
        cacheMisses += 1
        dequantize(tokenID: tokenID, scaleMul: scale, dst: dstPtr)
        lookupCache.put(tokenID, src: dstPtr)
        return result
    }

    /// Look up embedding WITHOUT the global embedScale factor.
    /// Used by MTP drafter which expects unscaled embeddings.
    func lookupUnscaled(_ tokenID: Int, shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        if let cached = lookupUnscaledCache.get(tokenID) {
            memcpy(dstPtr, cached, dim * MemoryLayout<UInt16>.stride)
            cacheHits += 1
            return result
        }
        cacheMisses += 1
        dequantize(tokenID: tokenID, scaleMul: 1.0, dst: dstPtr)
        lookupUnscaledCache.put(tokenID, src: dstPtr)
        return result
    }

    /// Look up and return as raw float16 bit array (for PLE computation).
    /// The returned buffer is the instance's f16 scratch and is only valid
    /// until the next `lookupRaw` call on this instance.
    func lookupRaw(_ tokenID: Int) -> [UInt16] {
        if let cached = lookupRawCache.get(tokenID) {
            _ = f16Buffer.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, cached, dim * MemoryLayout<UInt16>.stride)
            }
            cacheHits += 1
            return f16Buffer
        }
        cacheMisses += 1
        f16Buffer.withUnsafeMutableBufferPointer { dst in
            dequantize(tokenID: tokenID, scaleMul: scale, dst: dst.baseAddress!)
        }
        f16Buffer.withUnsafeBufferPointer { src in
            lookupRawCache.put(tokenID, src: src.baseAddress!)
        }
        return f16Buffer
    }

    /// Dequantize one token row (int8 + per-token fp16 scale) into the
    /// caller-provided fp16 buffer. `scaleMul` lets callers fold the
    /// global embedScale in (or omit it for the unscaled drafter path).
    private func dequantize(tokenID: Int, scaleMul: Float,
                            dst: UnsafeMutablePointer<UInt16>) {
        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self)
                .advanced(by: tokenID * dim)
            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScale = fp16ToF32(scalePtr[tokenID]) / 127.0 * scaleMul
                vDSP.convertElements(of: UnsafeBufferPointer(start: int8Ptr, count: dim),
                                     to: &f32Buffer)
                vDSP.multiply(rowScale, f32Buffer, result: &f32Buffer)
                f32Buffer.withUnsafeBufferPointer { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                                               height: 1, width: UInt(dim), rowBytes: dim * 4)
                    var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dst),
                                               height: 1, width: UInt(dim), rowBytes: dim * 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
            }
        }
    }
}

/// Tiny FP16-row LRU keyed by token id. Storage is one contiguous
/// `capacity * dim * 2` byte arena plus a slot dictionary, so a hit is
/// one dict lookup + one memcpy with no allocations. Eviction picks
/// the least-recently-used entry by access counter (linear scan, fine
/// at capacity ≈ 256).
final class EmbeddingFP16LRU {
    private struct Slot { var tokenID: Int; var lastUsed: UInt64 }
    private let capacity: Int
    private let dim: Int
    private var arena: [UInt16]
    private var slots: [Slot]
    private var index: [Int: Int]
    private var clock: UInt64 = 0

    init(capacity: Int, dim: Int) {
        self.capacity = capacity
        self.dim = dim
        self.arena = [UInt16](repeating: 0, count: max(capacity, 0) * dim)
        self.slots = []
        self.slots.reserveCapacity(capacity)
        self.index = [:]
        self.index.reserveCapacity(capacity)
    }

    /// Returns a pointer into the arena (read-only) or nil on miss. The
    /// pointer remains valid until the next mutating `put` call.
    func get(_ tokenID: Int) -> UnsafePointer<UInt16>? {
        guard capacity > 0, let slotIdx = index[tokenID] else { return nil }
        clock &+= 1
        slots[slotIdx].lastUsed = clock
        return arena.withUnsafeBufferPointer { buf in
            UnsafePointer(buf.baseAddress!.advanced(by: slotIdx * dim))
        }
    }

    /// Insert (or refresh) `tokenID`'s row from `src`. Evicts the LRU slot
    /// when full. `src` must be a `dim`-long fp16 buffer.
    func put(_ tokenID: Int, src: UnsafePointer<UInt16>) {
        guard capacity > 0 else { return }
        clock &+= 1
        if let slotIdx = index[tokenID] {
            _ = arena.withUnsafeMutableBufferPointer { buf in
                memcpy(buf.baseAddress!.advanced(by: slotIdx * dim),
                       src, dim * MemoryLayout<UInt16>.stride)
            }
            slots[slotIdx].lastUsed = clock
            return
        }
        let slotIdx: Int
        if slots.count < capacity {
            slotIdx = slots.count
            slots.append(Slot(tokenID: tokenID, lastUsed: clock))
        } else {
            var victim = 0
            var oldest = slots[0].lastUsed
            for i in 1..<slots.count where slots[i].lastUsed < oldest {
                victim = i; oldest = slots[i].lastUsed
            }
            index.removeValue(forKey: slots[victim].tokenID)
            slotIdx = victim
            slots[slotIdx] = Slot(tokenID: tokenID, lastUsed: clock)
        }
        index[tokenID] = slotIdx
        _ = arena.withUnsafeMutableBufferPointer { buf in
            memcpy(buf.baseAddress!.advanced(by: slotIdx * dim),
                   src, dim * MemoryLayout<UInt16>.stride)
        }
    }
}

// MARK: - Float16 conversion (used across the package)

func fp16ToF32(_ bits: UInt16) -> Float {
    let sign: UInt32 = UInt32(bits >> 15) << 31
    let exp = UInt32((bits >> 10) & 0x1F)
    let frac = UInt32(bits & 0x3FF)
    if exp == 0 { return frac == 0 ? Float(bitPattern: sign) : 0 }
    if exp == 31 { return Float.infinity }
    return Float(bitPattern: sign | ((exp + 112) << 23) | (frac << 13))
}

func f32ToFp16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    let exp = Int((bits >> 23) & 0xFF) - 127 + 15
    let frac = UInt16((bits >> 13) & 0x3FF)
    if exp <= 0 { return sign }
    if exp >= 31 { return sign | 0x7C00 }
    return sign | UInt16(exp) << 10 | frac
}
