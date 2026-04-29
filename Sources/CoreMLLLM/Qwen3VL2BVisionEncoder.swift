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

public struct Qwen3VL2BVisionFeatures {
    /// Pooled vision tokens. Shape (196, 2048) fp16.
    public let hidden: MLMultiArray
    /// Three DeepStack tensors injected at text layers 0/1/2, each
    /// shape (196, 2048) fp16.
    public let deepstack: [MLMultiArray]
    /// Number of image tokens (= merger output rows, 196 at 448×448).
    public var count: Int { hidden.shape[0].intValue }
}

@Observable
public final class Qwen3VL2BVisionEncoder {
    public struct Config {
        let imageSize: Int       // 448
        let computeUnits: MLComputeUnits
        public static let `default` = Config(imageSize: 448,
                                      computeUnits: .cpuAndNeuralEngine)
    }

    public var status = "Idle"
    @ObservationIgnored private let cfg: Config
    @ObservationIgnored private var model: MLModel?

    // Pre-allocated input buffer reused across encode() calls.
    @ObservationIgnored private var pixelBuffer: MLMultiArray!
    @ObservationIgnored private var pixelFV: MLFeatureValue!

    // Qwen3-VL normalization constants. The Qwen3-VL image processor
    // uses mean=(0.5, 0.5, 0.5) / std=(0.5, 0.5, 0.5) — i.e. rescale
    // to [-1, 1] — **not** the CLIP values [0.48145, 0.45783, 0.40821]
    // / [0.26863, 0.26130, 0.27578] that Qwen2-VL and most vision
    // LMs inherit. A previous revision of this file used the CLIP
    // values by inheritance from a Gemma/LLaVA-style pipeline, which
    // offset every pixel by a per-channel bias and shrunk the std by
    // ~2×, dropping on-device vision-feature cosine parity vs HF from
    // 0.96 to 0.47 — the model saw "digital corruption, horizontal
    // streaks" instead of the image content.
    private static let imageMean: [Float] = [0.5, 0.5, 0.5]
    private static let imageStd:  [Float] = [0.5, 0.5, 0.5]

    public init(cfg: Config = .default) {
        self.cfg = cfg
        // Pre-patchified layout: (num_patches, patch_flat) = (784, 1536).
        // num_patches = (grid_h * grid_w) = 28 * 28 with grid_t = 1
        // patch_flat  = C * T_p * P * P   = 3 * 2 * 16 * 16
        // Doing the HF patchify permutation on CPU (~1 ms per image)
        // keeps the in-graph reshape rank 5, which A18 Pro ANE
        // compiles cleanly. The earlier raw (3, 2, 448, 448) layout
        // forced a rank-10 permute inside the model that faulted
        // iPhone's ANE compiler with EXC_BAD_ACCESS at MLModel init
        // (Mac Studio ANE handled it; A18 did not).
        let numPatches = (cfg.imageSize / 16) * (cfg.imageSize / 16)
        let patchFlat = 3 * 2 * 16 * 16
        self.pixelBuffer = try! MLMultiArray(
            shape: [NSNumber(value: numPatches), NSNumber(value: patchFlat)],
            dataType: .float16)
        self.pixelFV = MLFeatureValue(multiArray: pixelBuffer)
    }

    public func load(modelURL: URL) throws {
        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: mcfg)
        status = "Loaded vision encoder"
    }

    /// Resolve `vision.mlmodelc` or `vision.mlpackage` under
    /// `<folder>/qwen3_vl_2b_vision/` (compiling the package on demand).
    public static func resolveModel(folder: URL) -> URL? {
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
    public func encode(_ cgImage: CGImage) async throws -> Qwen3VL2BVisionFeatures {
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

    /// CGImage → pre-patchified (784, 1536) fp16 in `pixelBuffer`,
    /// matching HF's `Qwen2VLImageProcessor._preprocess` layout
    /// exactly so the Core ML graph's `patch_embed` sees the same
    /// patches HF would.
    ///
    /// The HF processor iterates merged 2×2 blocks as the outer axis
    /// and packs `(C, T_p, P_h, P_w)` into each row's 1536 elements,
    /// via the permutation `(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)` over the
    /// 10-axis view `(B, grid_t, T_p, C, gh_m, m, P, gw_m, m, P)`.
    /// We apply the inverse-order scan in Swift directly rather than
    /// materializing the 10-axis intermediate.
    private func preprocess(_ cgImage: CGImage) {
        let size = cfg.imageSize                     // 448
        let P = 16                                    // patch_size
        let merge = 2                                 // spatial_merge_size
        let gridH = size / P                          // 28
        let gridW = size / P                          // 28
        let ghM = gridH / merge                       // 14
        let gwM = gridW / merge                       // 14
        let count = size * size
        let patchFlat = 3 * 2 * P * P                 // 1536

        // 1) Decode + resize to size × size RGBA (device RGB).
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = size * 4
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        var rgba = [UInt8](repeating: 0, count: count * 4)
        rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: size, height: size,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace, bitmapInfo: bitmapInfo) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        // 2) Normalize per channel into a (C, H, W) fp32 plane.
        //    Qwen3-VL: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5).
        var plane = [Float](repeating: 0, count: 3 * count)
        for c in 0..<3 {
            let mean = Self.imageMean[c]
            let invStd = 1.0 / Self.imageStd[c]
            let base = c * count
            for i in 0..<count {
                plane[base + i] = (Float(rgba[i * 4 + c]) / 255.0 - mean) * invStd
            }
        }
        // 3) Patchify + cast to fp16 directly into pixelBuffer in HF
        //    processor order. For merged-block (i, j) in
        //    [0, ghM) × [0, gwM) and inner-offset (mi, mj) in
        //    [0, merge) × [0, merge) the token index is:
        //        row = (i * gwM + j) * merge*merge + mi * merge + mj
        //    The token's 1536 elements pack (C, T_p, P_h, P_w) in
        //    row-major order; T_p=2 duplicates the frame for still
        //    images. Source pixel coords:
        //        h = (i * merge + mi) * P + ph
        //        w = (j * merge + mj) * P + pw
        precondition(pixelBuffer.shape[0].intValue == ghM * gwM * merge * merge
                     && pixelBuffer.shape[1].intValue == patchFlat,
                     "pixel_values must be (num_patches, patch_flat)")
        let dst = pixelBuffer.dataPointer
            .assumingMemoryBound(to: UInt16.self)
        let patchPixels = P * P
        for i in 0..<ghM {
            for j in 0..<gwM {
                for mi in 0..<merge {
                    for mj in 0..<merge {
                        let row = ((i * gwM + j) * merge + mi) * merge + mj
                        let rowBase = row * patchFlat
                        // (C, T_p, P, P) inside the patch row.
                        let hBase = (i * merge + mi) * P
                        let wBase = (j * merge + mj) * P
                        for c in 0..<3 {
                            let planeBase = c * count
                            let outCBase = rowBase + c * 2 * patchPixels
                            for tp in 0..<2 {
                                let outBase = outCBase + tp * patchPixels
                                for ph in 0..<P {
                                    let srcRow = (hBase + ph) * size + wBase
                                    var k = 0
                                    while k < P {
                                        let v = Float16(plane[planeBase + srcRow + k]).bitPattern
                                        dst[outBase + ph * P + k] = v
                                        k += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if ProcessInfo.processInfo.environment["VL2B_VISION_DEBUG"] != nil {
            print("[VL2BVisionEncoder] patchified \(ghM * gwM * merge * merge) patches × \(patchFlat) values")
        }
    }
}
