// Qwen3-VL 2B vision encoder wrapper.
//
// Loads `vision.mlmodelc` / `vision.mlpackage` and produces the
// DeepStack-aware merger output expected by chunk_0_vision. Input is a
// single `CGImage`; output is (hidden, [ds_0, ds_1, ds_2]) each shaped
// (196, 2048) fp16 on ANE-resident memory.
//
// Fixed-grid configuration:
//   - image_size = 448 (square, already aligned to patch * merge)
//   - patches    = 28 × 28 = 784 at patch_size=16
//   - merger     = spatial_merge=2 → 196 vision tokens
//
// Preprocess: resize → normalize with CLIP-style mean/std (Qwen3-VL uses
// the same constants as Qwen2-VL) → duplicate the frame along the
// temporal axis (T=2) → emit (C=3, T=2, H=448, W=448) fp16 as the single
// `pixel_values` input.

import CoreML
import CoreGraphics
import Foundation
import Accelerate

struct Qwen3VL2BVisionFeatures {
    /// Pooled vision tokens. Shape (196, 2048) fp16.
    let hidden: MLMultiArray
    /// Three DeepStack tensors injected at text layers 0/1/2, each
    /// shape (196, 2048) fp16.
    let deepstack: [MLMultiArray]
    /// Number of image tokens (= merger output rows, 196 at 448×448).
    var count: Int { hidden.shape[0].intValue }
}

@Observable
final class Qwen3VL2BVisionEncoder {
    struct Config {
        let imageSize: Int       // 448
        let computeUnits: MLComputeUnits
        static let `default` = Config(imageSize: 448,
                                      computeUnits: .cpuAndNeuralEngine)
    }

    var status = "Idle"
    @ObservationIgnored private let cfg: Config
    @ObservationIgnored private var model: MLModel?

    // Pre-allocated input buffer reused across encode() calls.
    @ObservationIgnored private var pixelBuffer: MLMultiArray!
    @ObservationIgnored private var pixelFV: MLFeatureValue!

    // CLIP / Qwen3-VL normalization constants.
    private static let imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    private static let imageStd:  [Float] = [0.26862954, 0.26130258, 0.27577711]

    init(cfg: Config = .default) {
        self.cfg = cfg
        let s = NSNumber(value: cfg.imageSize)
        self.pixelBuffer = try! MLMultiArray(
            shape: [3, 2, s, s], dataType: .float16)
        self.pixelFV = MLFeatureValue(multiArray: pixelBuffer)
    }

    func load(modelURL: URL) throws {
        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: mcfg)
        status = "Loaded vision encoder"
    }

    /// Resolve `vision.mlmodelc` or `vision.mlpackage` under
    /// `<folder>/qwen3_vl_2b_vision/` (compiling the package on demand).
    static func resolveModel(folder: URL) -> URL? {
        let dir = folder.appendingPathComponent("qwen3_vl_2b_vision")
        let fm = FileManager.default
        let mlc = dir.appendingPathComponent("vision.mlmodelc")
        if fm.fileExists(atPath: mlc.path) { return mlc }
        let pkg = dir.appendingPathComponent("vision.mlpackage")
        if fm.fileExists(atPath: pkg.path) {
            return try? MLModel.compileModel(at: pkg)
        }
        return nil
    }

    /// Preprocess + encode a single image. Returns the 196-token merger
    /// hidden + DeepStack slices.
    func encode(_ cgImage: CGImage) async throws -> Qwen3VL2BVisionFeatures {
        guard let model else {
            throw NSError(domain: "Qwen3VL2BVisionEncoder", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Vision encoder not loaded"])
        }
        preprocess(cgImage)
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["pixel_values": pixelFV!])
        let out = try await model.prediction(from: input)
        guard let hidden = out.featureValue(for: "hidden")?.multiArrayValue,
              let d0 = out.featureValue(for: "deepstack_0")?.multiArrayValue,
              let d1 = out.featureValue(for: "deepstack_1")?.multiArrayValue,
              let d2 = out.featureValue(for: "deepstack_2")?.multiArrayValue else {
            throw NSError(domain: "Qwen3VL2BVisionEncoder", code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "Vision encoder missing expected outputs"])
        }
        return Qwen3VL2BVisionFeatures(
            hidden: hidden, deepstack: [d0, d1, d2])
    }

    /// CGImage → (3, 2, 448, 448) fp16 in `pixelBuffer`.
    /// Fills T=0 and T=1 with the same normalized frame (temporal patch=2
    /// for single-image inputs).
    private func preprocess(_ cgImage: CGImage) {
        let size = cfg.imageSize
        let count = size * size
        // 1) Resize to imageSize × imageSize via CGContext (device RGB,
        //    premultipliedLast).
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = size * 4
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: size, height: size,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace, bitmapInfo: bitmapInfo) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        // 2) Split channels + normalize to (3, H, W) fp32 then cast to fp16
        //    into pixelBuffer at both T=0 and T=1.
        let dst = pixelBuffer.dataPointer
            .assumingMemoryBound(to: UInt16.self)
        var plane = [Float](repeating: 0, count: count)
        let stridePerChannel = 2 * count                   // T=2
        let stridePerTemporal = count
        for c in 0..<3 {
            let mean = Self.imageMean[c]
            let std  = Self.imageStd[c]
            let invStd = 1.0 / std
            // Deinterleave: channel c of each pixel at offset c.
            for i in 0..<count {
                let v = Float(rgba[i * 4 + c]) / 255.0
                plane[i] = (v - mean) * invStd
            }
            // Cast fp32 → fp16 for both T=0 and T=1.
            let baseT0 = c * stridePerChannel
            let baseT1 = c * stridePerChannel + stridePerTemporal
            var srcBuf = plane  // mutable copy for vImage
            srcBuf.withUnsafeMutableBufferPointer { sp in
                var srcBuffer = vImage_Buffer(
                    data: sp.baseAddress,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size)
                var dst0 = vImage_Buffer(
                    data: dst.advanced(by: baseT0),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size)
                var dst1 = vImage_Buffer(
                    data: dst.advanced(by: baseT1),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size)
                _ = vImageConvert_PlanarFtoPlanar16F(&srcBuffer, &dst0, 0)
                _ = vImageConvert_PlanarFtoPlanar16F(&srcBuffer, &dst1, 0)
            }
        }
    }
}
