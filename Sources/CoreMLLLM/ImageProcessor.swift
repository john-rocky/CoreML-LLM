import CoreML
import CoreGraphics
import Foundation

/// Processes images for Gemma 4 multimodal vision encoder.
///
/// Matches HuggingFace Gemma3nImageProcessor:
///   1. Aspect-ratio-preserving resize to max 645,120 pixels (2520 × 16²)
///   2. Each side rounded down to a multiple of 48 (pooling_kernel × patch_size)
///   3. Patch extraction: 16×16, channels-last, /255 normalization
///   4. Meshgrid position IDs (x, y) = (px, py) matching HF indexing="xy"
///   5. Padding positions marked with -1
public enum ImageProcessor {

    /// Process an image through the vision encoder CoreML model.
    ///
    /// Returns image features MLMultiArray (1, 280, hidden_size).
    public static func process(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        let ps = 16
        let total = 2520
        let pd = ps * ps * 3  // 768 per patch

        // 1. Aspect-ratio-preserving resize (each side multiple of 48).
        let origH = Double(image.height)
        let origW = Double(image.width)
        let targetPx = Double(total * ps * ps)  // 645_120
        let factor = sqrt(targetPx / (origH * origW))
        let sideMult = 48
        var tH = Int(floor(factor * origH / Double(sideMult))) * sideMult
        var tW = Int(floor(factor * origW / Double(sideMult))) * sideMult
        if tH < sideMult { tH = sideMult }
        if tW < sideMult { tW = sideMult }
        let Hp = tH / ps
        let Wp = tW / ps
        let realPatches = Hp * Wp

        // 2. Draw into (tW, tH) RGBA canvas with bicubic interpolation.
        var pixels = [UInt8](repeating: 0, count: tW * tH * 4)
        let bitmap = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let ctx = CGContext(data: &pixels, width: tW, height: tH, bitsPerComponent: 8,
                            bytesPerRow: tW * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bitmap.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: tW, height: tH))

        // 3. Build pixel_values (1, 2520, 768) fp32 and pixel_position_ids (1, 2520, 2) int32.
        let pv = try MLMultiArray(shape: [1, NSNumber(value: total), NSNumber(value: pd)],
                                  dataType: .float32)
        let pid = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .int32)
        let pvp = pv.dataPointer.bindMemory(to: Float.self, capacity: total * pd)
        let pidp = pid.dataPointer.bindMemory(to: Int32.self, capacity: total * 2)
        memset(pvp, 0, total * pd * MemoryLayout<Float>.stride)

        var pi = 0
        for py in 0..<Hp {
            for px in 0..<Wp {
                var o = pi * pd
                for dy in 0..<ps {
                    for dx in 0..<ps {
                        let srcIdx = ((py * ps + dy) * tW + (px * ps + dx)) * 4
                        pvp[o]   = Float(pixels[srcIdx])   / 255
                        pvp[o+1] = Float(pixels[srcIdx+1]) / 255
                        pvp[o+2] = Float(pixels[srcIdx+2]) / 255
                        o += 3
                    }
                }
                pidp[pi * 2]     = Int32(px)
                pidp[pi * 2 + 1] = Int32(py)
                pi += 1
            }
        }
        for i in realPatches..<total {
            pidp[i * 2]     = -1
            pidp[i * 2 + 1] = -1
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pv),
            "pixel_position_ids": MLFeatureValue(multiArray: pid),
        ])
        guard let features = try visionModel.prediction(from: input)
                .featureValue(for: "image_features")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        return features
    }

    /// Process a single video frame through the video-grade vision encoder.
    ///
    /// Matches the `video_processor` path in Gemma 4 (max_soft_tokens=70 →
    /// 64 real tokens on a square input). The frame is letter-cropped to
    /// 384×384 and tiled into a 24×24 patch grid (576 real + 54 padding =
    /// 630 total), which is the fixed input shape the CoreML-converted
    /// video encoder expects.
    ///
    /// Returns `(1, 64, hidden_size)` — same layout the per-frame feature
    /// slot expects, so no Swift-side pooling is needed.
    public static func processVideoFrame(
        _ image: CGImage,
        with videoVisionModel: MLModel
    ) throws -> MLMultiArray {
        let ps = 16
        let side = 384              // 24 · 16
        let Hp = side / ps          // 24
        let Wp = side / ps          // 24
        let realPatches = Hp * Wp   // 576
        let total = 630
        let pd = ps * ps * 3        // 768 per patch

        // 1. Draw into a 384×384 RGBA canvas. Caller is expected to have
        //    already center-cropped to square; if not, the bicubic resize
        //    below will squash the aspect ratio — that matches what HF's
        //    video processor does for unpadded square inputs.
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        let bitmap = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let ctx = CGContext(data: &pixels, width: side, height: side,
                            bitsPerComponent: 8, bytesPerRow: side * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bitmap.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))

        // 2. Build pixel_values (1, 630, 768) fp32 and
        //    pixel_position_ids (1, 630, 2) int32. Padding patches (indices
        //    576..629) get position id (-1, -1) to match the HF convention.
        let pv = try MLMultiArray(shape: [1, NSNumber(value: total), NSNumber(value: pd)],
                                  dataType: .float32)
        let pid = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .int32)
        let pvp = pv.dataPointer.bindMemory(to: Float.self, capacity: total * pd)
        let pidp = pid.dataPointer.bindMemory(to: Int32.self, capacity: total * 2)
        memset(pvp, 0, total * pd * MemoryLayout<Float>.stride)

        var pi = 0
        for py in 0..<Hp {
            for px in 0..<Wp {
                var o = pi * pd
                for dy in 0..<ps {
                    for dx in 0..<ps {
                        let srcIdx = ((py * ps + dy) * side + (px * ps + dx)) * 4
                        pvp[o]   = Float(pixels[srcIdx])   / 255
                        pvp[o+1] = Float(pixels[srcIdx+1]) / 255
                        pvp[o+2] = Float(pixels[srcIdx+2]) / 255
                        o += 3
                    }
                }
                pidp[pi * 2]     = Int32(px)
                pidp[pi * 2 + 1] = Int32(py)
                pi += 1
            }
        }
        for i in realPatches..<total {
            pidp[i * 2]     = -1
            pidp[i * 2 + 1] = -1
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pv),
            "pixel_position_ids": MLFeatureValue(multiArray: pid),
        ])
        guard let features = try videoVisionModel.prediction(from: input)
                .featureValue(for: "image_features")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        return features
    }

    /// Process an image through the ANE-targeted still-image vision encoder.
    ///
    /// Matches `conversion/models/gemma4_vision.py::convert_still_image_vision_ane_to_coreml`:
    /// the model expects a fixed 48×48 square grid (2304 patches, 256
    /// soft tokens after k=3 avg pool) and fp16 pixel values. We force-
    /// resize to 768×768, so aspect ratio is squashed — acceptable for
    /// natural photos, less so for extreme ratios. For now the legacy
    /// GPU encoder is used when aspect ratio matters; this path is
    /// opted into automatically when `vision.ane.*` is present on disk.
    ///
    /// Returns image features MLMultiArray (1, 256, hidden_size) fp16.
    public static func processANE(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        let ps = 16
        let Hp = 48
        let Wp = 48
        let total = Hp * Wp   // 2304
        let pd = ps * ps * 3  // 768

        // 1. Rasterize into a 768×768 RGBA canvas.
        let side = Hp * ps    // 768
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        let bitmap = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let ctx = CGContext(data: &pixels, width: side, height: side,
                            bitsPerComponent: 8, bytesPerRow: side * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bitmap.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))

        // 2. Build pixel_values (1, 2304, 768) fp16 and pixel_position_ids (1, 2304, 2) int32.
        //    No padding — every slot holds a valid patch, so the HF pooler's
        //    masked_fill + one_hot fold into compile-time constants.
        let pv = try MLMultiArray(shape: [1, NSNumber(value: total), NSNumber(value: pd)],
                                  dataType: .float16)
        let pid = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .int32)
        let pvp = pv.dataPointer.bindMemory(to: UInt16.self, capacity: total * pd)
        let pidp = pid.dataPointer.bindMemory(to: Int32.self, capacity: total * 2)

        var pi = 0
        for py in 0..<Hp {
            for px in 0..<Wp {
                var o = pi * pd
                for dy in 0..<ps {
                    for dx in 0..<ps {
                        let srcIdx = ((py * ps + dy) * side + (px * ps + dx)) * 4
                        let r = Float16(Float(pixels[srcIdx])     / 255)
                        let g = Float16(Float(pixels[srcIdx + 1]) / 255)
                        let b = Float16(Float(pixels[srcIdx + 2]) / 255)
                        pvp[o]     = r.bitPattern
                        pvp[o + 1] = g.bitPattern
                        pvp[o + 2] = b.bitPattern
                        o += 3
                    }
                }
                pidp[pi * 2]     = Int32(px)
                pidp[pi * 2 + 1] = Int32(py)
                pi += 1
            }
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pv),
            "pixel_position_ids": MLFeatureValue(multiArray: pid),
        ])
        guard let features = try visionModel.prediction(from: input)
                .featureValue(for: "image_features")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        return features
    }

    /// Extract a single image feature token from the vision output.
    public static func sliceFeature(_ features: MLMultiArray, at index: Int,
                                     hiddenSize: Int) -> MLMultiArray {
        let r = try! MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)], dataType: .float16)
        let s = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let d = r.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
        memcpy(d, s.advanced(by: index * hiddenSize), hiddenSize * MemoryLayout<UInt16>.stride)
        return r
    }
}
