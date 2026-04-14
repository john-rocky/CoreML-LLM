import AVFoundation
import CoreGraphics
import CoreML
import Foundation

/// Loads a video file and produces the inputs the Gemma 4 E2B pipeline needs:
/// a small set of RGB frames sampled at a target fps, and (optionally) the
/// audio track resampled to mono 16 kHz float PCM for the Conformer encoder.
///
/// The Gemma 4 video chat template is a sequence of per-frame blocks:
///
///     MM:SS <|image><|image|>×256<image|>   MM:SS <|image>…<image|>
///
/// joined by single spaces, optionally followed by a `<|audio>…<audio|>`
/// block. This processor only extracts pixels and PCM — prompt assembly and
/// feature concatenation live in `CoreMLLLM`.
public enum VideoProcessor {

    public struct Frame: Sendable {
        public let image: CGImage
        public let timestampSeconds: Double
    }

    public struct Options: Sendable {
        /// Target frames per second for frame sampling.
        public var fps: Double
        /// Upper bound on extracted frames (protects against long videos
        /// exploding the prompt — see MULTIMODAL.md).
        public var maxFrames: Int
        /// Also extract the audio track as mono 16 kHz float PCM.
        public var includeAudio: Bool
        /// Center-crop each frame to square before handing it to the vision
        /// encoder. The encoder always emits 280 soft tokens but only the
        /// first N are "real", where N depends on the aspect ratio. Our
        /// feature-injection path assumes 256 tokens per frame (matching a
        /// square input), so the default here is true.
        public var centerCropSquare: Bool
        /// Soft tokens per frame to inject into the prompt. Gemma 4's
        /// `video_processor` uses `max_soft_tokens=70` (≈ 64 real per square
        /// frame) while `image_processor` uses 280/256. The still-image
        /// encoder shipped here produces 256 tokens per square input; set
        /// `poolTokensPerFrame=64` to 2×2-average-pool each frame's 16×16
        /// token grid down to 8×8 so the LLM sees a video-grade token count
        /// rather than 256 "still-image-res" tokens per frame.
        public var tokensPerFrame: Int

        public init(fps: Double = 1.0, maxFrames: Int = 8,
                    includeAudio: Bool = false,
                    centerCropSquare: Bool = true,
                    tokensPerFrame: Int = 64) {
            self.fps = fps
            self.maxFrames = maxFrames
            self.includeAudio = includeAudio
            self.centerCropSquare = centerCropSquare
            self.tokensPerFrame = tokensPerFrame
        }
    }

    // MARK: - Frames

    /// Sample up to `options.maxFrames` frames from `url` at `options.fps`.
    /// Returned frames carry the wall-clock offset from the start of the clip.
    public static func extractFrames(
        from url: URL,
        options: Options
    ) async throws -> [Frame] {
        let asset = AVURLAsset(url: url)
        let durationSec = try await asset.load(.duration).seconds
        guard durationSec.isFinite, durationSec > 0 else { return [] }

        let step = max(1.0 / max(options.fps, 0.01), 1.0 / 120.0)
        let maxByDuration = Int(floor(durationSec / step)) + 1
        let count = max(1, min(options.maxFrames, maxByDuration))

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        // We're sampling sparsely; allow the generator to snap to the nearest
        // keyframe within half the sampling interval to avoid expensive seeks.
        let tol = CMTime(seconds: step / 2, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tol
        generator.requestedTimeToleranceAfter = tol

        var frames: [Frame] = []
        frames.reserveCapacity(count)
        for i in 0..<count {
            let t = min(Double(i) * step, max(0, durationSec - 0.01))
            let cmTime = CMTime(seconds: t, preferredTimescale: 600)
            let result = try await generator.image(at: cmTime)
            let img = options.centerCropSquare ? centerSquareCrop(result.image) : result.image
            frames.append(Frame(image: img, timestampSeconds: t))
        }
        return frames
    }

    /// Crop a CGImage to the largest centered square.
    static func centerSquareCrop(_ image: CGImage) -> CGImage {
        let w = image.width, h = image.height
        if w == h { return image }
        let side = min(w, h)
        let x = (w - side) / 2
        let y = (h - side) / 2
        return image.cropping(to: CGRect(x: x, y: y, width: side, height: side)) ?? image
    }

    // MARK: - Audio

    /// Extract the first audio track as mono float32 PCM at 16 kHz.
    /// Returns nil if the asset has no audio track.
    public static func extractAudioPCM16k(from url: URL) async throws -> [Float]? {
        let asset = AVURLAsset(url: url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let track = audioTracks.first else { return nil }

        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { return nil }
        reader.add(output)
        guard reader.startReading() else {
            if let err = reader.error { throw err }
            return nil
        }

        var samples: [Float] = []
        while let buffer = output.copyNextSampleBuffer() {
            defer { CMSampleBufferInvalidate(buffer) }
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            var length = 0
            var dataPtr: UnsafeMutablePointer<Int8>?
            let status = CMBlockBufferGetDataPointer(
                block, atOffset: 0, lengthAtOffsetOut: nil,
                totalLengthOut: &length, dataPointerOut: &dataPtr)
            guard status == noErr, let p = dataPtr, length > 0 else { continue }
            let count = length / MemoryLayout<Float>.size
            p.withMemoryRebound(to: Float.self, capacity: count) { fp in
                samples.append(contentsOf: UnsafeBufferPointer(start: fp, count: count))
            }
        }
        if reader.status == .failed, let err = reader.error { throw err }
        return samples
    }

    // MARK: - Helpers for prompt building

    /// `MM:SS` string used by the Gemma 4 video chat template.
    public static func timestampLabel(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.rounded()))
        return String(format: "%02d:%02d", s / 60, s % 60)
    }
}
