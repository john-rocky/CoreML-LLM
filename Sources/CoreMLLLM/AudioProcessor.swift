import Accelerate
import CoreML
import Foundation

/// Computes mel spectrograms for the Gemma 4 audio encoder.
///
/// Matches HuggingFace Gemma4AudioFeatureExtractor:
///   1. Semicausal pad (prepend frameLength/2 zeros)
///   2. Frame extraction: size = 321, step = 160, then drop last sample → 320
///   3. Apply periodic Hann window (320)
///   4. Zero-pad to 512 and RFFT
///   5. Magnitude spectrum (|FFT|, NOT power)
///   6. Mel filterbank matmul (257 × 128)
///   7. log(mel + 0.001)
public enum AudioProcessor {

    // MARK: - Constants (matching HF Gemma4AudioFeatureExtractor)

    static let sampleRate = 16000
    static let frameLength = 320
    static let hopLength = 160
    static let fftLength = 512
    static let numMelBins = 128
    static let numFFTBins = fftLength / 2 + 1  // 257
    static let melFloor: Float = 0.001

    // MARK: - Public API

    /// Process raw audio through the audio encoder CoreML model.
    ///
    /// - Parameters:
    ///   - samples: Raw PCM audio samples (Float, mono, 16kHz)
    ///   - audioModel: Compiled audio CoreML model
    ///   - melFilterbank: Mel filterbank matrix (257 × 128), loaded from mel_filterbank.bin
    ///   - targetFrames: Number of mel frames the model expects (e.g. 200)
    /// - Returns: Audio features MLMultiArray (1, numTokens, 1536)
    public static func process(
        _ samples: [Float],
        with audioModel: MLModel,
        melFilterbank: [Float],
        targetFrames: Int
    ) throws -> MLMultiArray {
        let mel = computeMelSpectrogram(
            samples, melFilterbank: melFilterbank, targetFrames: targetFrames)
        // Dump mel for verification
        let melPtr = mel.dataPointer.bindMemory(to: Float16.self, capacity: mel.count)
        let m0 = Float(melPtr[0]), m1 = Float(melPtr[1]), m2 = Float(melPtr[2])
        let m3 = Float(melPtr[3]), m4 = Float(melPtr[4])
        print("[AudioProc] mel[0,:5] = [\(m0), \(m1), \(m2), \(m3), \(m4)]")
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: mel),
        ])
        guard let features = try audioModel.prediction(from: input)
                .featureValue(for: "audio_features")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        return features
    }

    /// Compute mel spectrogram from raw audio.
    ///
    /// - Parameters:
    ///   - samples: Raw PCM audio at 16kHz
    ///   - melFilterbank: (257 × 128) mel filterbank, row-major float32
    ///   - targetFrames: Pad/truncate mel spectrogram to this many frames
    /// - Returns: MLMultiArray (1, targetFrames, 128) float16
    public static func computeMelSpectrogram(
        _ samples: [Float],
        melFilterbank: [Float],
        targetFrames: Int
    ) -> MLMultiArray {
        // Semicausal pad: prepend frameLength/2 zeros
        let padLeft = frameLength / 2
        var padded = [Float](repeating: 0, count: padLeft + samples.count)
        padded.replaceSubrange(padLeft..<padLeft + samples.count, with: samples)

        // Frame extraction: size=321 (frameLength+1), step=hopLength
        let unfoldSize = frameLength + 1
        let numFrames = max(0, (padded.count - unfoldSize) / hopLength + 1)

        // Hann window (periodic): w[n] = 0.5 - 0.5 * cos(2π * n / N)
        var window = [Float](repeating: 0, count: frameLength)
        for n in 0..<frameLength {
            window[n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(frameLength))
        }

        // FFT setup
        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Output: mel spectrogram (numFrames × numMelBins)
        let actualFrames = min(numFrames, targetFrames)
        var melSpec = [Float](repeating: 0, count: targetFrames * numMelBins)

        // Buffers for FFT (reused per frame)
        var realp = [Float](repeating: 0, count: fftLength / 2)
        var imagp = [Float](repeating: 0, count: fftLength / 2)
        var windowed = [Float](repeating: 0, count: fftLength)
        var magnitude = [Float](repeating: 0, count: numFFTBins)

        for f in 0..<actualFrames {
            let start = f * hopLength

            // Extract frame: take 321 samples, drop last → 320
            let frameSlice = Array(padded[start..<min(start + unfoldSize, padded.count)])
            let frame: [Float]
            if frameSlice.count >= unfoldSize {
                frame = Array(frameSlice[..<frameLength])
            } else {
                var tmp = frameSlice
                tmp.append(contentsOf: [Float](repeating: 0, count: unfoldSize - tmp.count))
                frame = Array(tmp[..<frameLength])
            }

            // Apply Hann window
            vDSP_vmul(frame, 1, window, 1, &windowed, 1, vDSP_Length(frameLength))
            // Zero-pad 320→512 (already zeroed from init)
            for i in frameLength..<fftLength { windowed[i] = 0 }

            // Pack real data into split complex (interleaved → split)
            windowed.withUnsafeBufferPointer { buf in
                buf.baseAddress!.withMemoryRebound(
                    to: DSPComplex.self, capacity: fftLength / 2
                ) { complexPtr in
                    realp.withUnsafeMutableBufferPointer { rp in
                        imagp.withUnsafeMutableBufferPointer { ip in
                            var split = DSPSplitComplex(realp: rp.baseAddress!,
                                                        imagp: ip.baseAddress!)
                            vDSP_ctoz(complexPtr, 2, &split, 1,
                                       vDSP_Length(fftLength / 2))
                        }
                    }
                }
            }

            // Forward real FFT (in-place)
            realp.withUnsafeMutableBufferPointer { rp in
                imagp.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!,
                                                imagp: ip.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n,
                                   FFTDirection(kFFTDirection_Forward))
                }
            }

            // vDSP FFT applies a 2x scale factor; correct it
            var half: Float = 0.5
            vDSP_vsmul(realp, 1, &half, &realp, 1, vDSP_Length(fftLength / 2))
            vDSP_vsmul(imagp, 1, &half, &imagp, 1, vDSP_Length(fftLength / 2))

            // Magnitude: |FFT| for 257 bins
            // Bin 0 (DC): realp[0] only (imagp[0] holds Nyquist in packed format)
            magnitude[0] = abs(realp[0])
            // Bins 1..255: sqrt(realp[k]^2 + imagp[k]^2)
            for k in 1..<(fftLength / 2) {
                magnitude[k] = sqrt(realp[k] * realp[k] + imagp[k] * imagp[k])
            }
            // Bin 256 (Nyquist): imagp[0] in packed format
            magnitude[fftLength / 2] = abs(imagp[0])

            // Mel filterbank matmul: mel = magnitude(1×257) @ filterbank(257×128)
            let melOffset = f * numMelBins
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        Int32(numFFTBins), Int32(numMelBins),
                        1.0, melFilterbank, Int32(numMelBins),
                        magnitude, 1,
                        0.0, &melSpec[melOffset], 1)

            // log(mel + mel_floor)
            for i in 0..<numMelBins {
                melSpec[melOffset + i] = log(melSpec[melOffset + i] + melFloor)
            }
        }

        // Zero-fill remaining frames (if audio was shorter than target)
        // Already zeroed from init — log(0 + 0.001) = log(0.001) ≈ -6.9
        if actualFrames < targetFrames {
            let logFloor = log(melFloor)
            for i in (actualFrames * numMelBins)..<(targetFrames * numMelBins) {
                melSpec[i] = logFloor
            }
        }

        // Convert to MLMultiArray (1, targetFrames, 128) float16
        let result = try! MLMultiArray(
            shape: [1, NSNumber(value: targetFrames), NSNumber(value: numMelBins)],
            dataType: .float16)
        let dst = result.dataPointer.bindMemory(to: UInt16.self,
                                                 capacity: targetFrames * numMelBins)
        for i in 0..<(targetFrames * numMelBins) {
            dst[i] = f32ToFp16(melSpec[i])
        }
        return result
    }

    /// Load mel filterbank matrix from binary file.
    /// File format: 257 × 128 float32, row-major.
    public static func loadMelFilterbank(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let count = numFFTBins * numMelBins  // 257 * 128 = 32896
        var result = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            for i in 0..<min(count, data.count / MemoryLayout<Float>.stride) {
                result[i] = src[i]
            }
        }
        return result
    }

    /// Extract a single audio feature token from the audio model output.
    public static func sliceFeature(_ features: MLMultiArray, at index: Int,
                                     hiddenSize: Int) -> MLMultiArray {
        let r = try! MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)],
                                  dataType: .float16)
        let s = features.dataPointer.bindMemory(to: UInt16.self,
                                                 capacity: features.count)
        let d = r.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
        memcpy(d, s.advanced(by: index * hiddenSize),
               hiddenSize * MemoryLayout<UInt16>.stride)
        return r
    }

    // MARK: - Float16 conversion

    private static func f32ToFp16(_ value: Float) -> UInt16 {
        var f = value
        var h: UInt16 = 0
        withUnsafeMutablePointer(to: &f) { fPtr in
            withUnsafeMutablePointer(to: &h) { hPtr in
                var src = vImage_Buffer(data: fPtr, height: 1, width: 1,
                                        rowBytes: MemoryLayout<Float>.stride)
                var dst = vImage_Buffer(data: hPtr, height: 1, width: 1,
                                        rowBytes: MemoryLayout<UInt16>.stride)
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
            }
        }
        return h
    }
}
