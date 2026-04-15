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

    init(dataURL: URL, scalesURL: URL, vocabSize: Int, dim: Int, scale: Float = 1.0) throws {
        self.data = try Data(contentsOf: dataURL, options: .mappedIfSafe)
        self.scales = try Data(contentsOf: scalesURL, options: .mappedIfSafe)
        self.vocabSize = vocabSize
        self.dim = dim
        self.scale = scale
        self.f32Buffer = [Float](repeating: 0, count: dim)
        self.f16Buffer = [UInt16](repeating: 0, count: dim)
    }

    /// Look up embedding for a single token. Returns float16 MLMultiArray.
    func lookup(_ tokenID: Int, shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self)
                .advanced(by: tokenID * dim)
            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScale = fp16ToF32(scalePtr[tokenID]) / 127.0 * scale
                // Vectorized: int8 → float32 → scale → float16
                vDSP.convertElements(of: UnsafeBufferPointer(start: int8Ptr, count: dim),
                                     to: &f32Buffer)
                vDSP.multiply(rowScale, f32Buffer, result: &f32Buffer)
                f32Buffer.withUnsafeBufferPointer { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                                               height: 1, width: UInt(dim), rowBytes: dim * 4)
                    var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr),
                                               height: 1, width: UInt(dim), rowBytes: dim * 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
            }
        }
        return result
    }

    /// Look up embedding for `tokenID` directly into a caller-owned UInt16
    /// destination buffer. Avoids allocating a fresh `MLMultiArray` per call —
    /// used by the warm-path decode loop which reuses a scratch input buffer.
    func lookupInto(_ tokenID: Int, dst: UnsafeMutablePointer<UInt16>) {
        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self)
                .advanced(by: tokenID * dim)
            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScale = fp16ToF32(scalePtr[tokenID]) / 127.0 * scale
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

    /// Look up embedding WITHOUT the global embedScale factor.
    /// Used by MTP drafter which expects unscaled embeddings.
    func lookupUnscaled(_ tokenID: Int, shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self)
                .advanced(by: tokenID * dim)
            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScale = fp16ToF32(scalePtr[tokenID]) / 127.0
                vDSP.convertElements(of: UnsafeBufferPointer(start: int8Ptr, count: dim),
                                     to: &f32Buffer)
                vDSP.multiply(rowScale, f32Buffer, result: &f32Buffer)
                f32Buffer.withUnsafeBufferPointer { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                                               height: 1, width: UInt(dim), rowBytes: dim * 4)
                    var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr),
                                               height: 1, width: UInt(dim), rowBytes: dim * 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
            }
        }
        return result
    }

    /// Look up and return as raw float16 bit array (for PLE computation).
    func lookupRaw(_ tokenID: Int) -> [UInt16] {
        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self)
                .advanced(by: tokenID * dim)
            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScale = fp16ToF32(scalePtr[tokenID]) / 127.0 * scale
                vDSP.convertElements(of: UnsafeBufferPointer(start: int8Ptr, count: dim),
                                     to: &f32Buffer)
                vDSP.multiply(rowScale, f32Buffer, result: &f32Buffer)
                f32Buffer.withUnsafeBufferPointer { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                                               height: 1, width: UInt(dim), rowBytes: dim * 4)
                    f16Buffer.withUnsafeMutableBufferPointer { dst in
                        var dstBuf = vImage_Buffer(data: dst.baseAddress!,
                                                   height: 1, width: UInt(dim), rowBytes: dim * 2)
                        vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                    }
                }
            }
        }
        return f16Buffer
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
