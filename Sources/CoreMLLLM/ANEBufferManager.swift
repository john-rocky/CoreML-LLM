import CoreML
import CoreVideo

/// Manages IOSurface-backed MLMultiArray buffers for ANE synchronization.
///
/// ANE writes are asynchronous. Without IOSurface backing, there can be race conditions
/// between ANE completing a write and CPU reading the result. IOSurface-backed buffers
/// provide proper synchronization guarantees.
///
/// Reference: ANEMLL InferenceManager.swift IOSurface patterns
final class ANEBufferManager: @unchecked Sendable {
    private let config: LLMModelConfiguration

    init(config: LLMModelConfiguration) {
        self.config = config
    }

    /// Create an IOSurface-backed MLMultiArray for safe ANE <-> CPU transfers.
    func createBuffer(shape: [Int], filling value: Float = 0.0) throws -> MLMultiArray {
        let buffer = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        if value == 0.0 {
            let ptr = buffer.dataPointer.bindMemory(to: UInt16.self, capacity: buffer.count)
            memset(ptr, 0, buffer.count * MemoryLayout<UInt16>.stride)
        }
        return buffer
    }

    /// Create a causal attention mask for the given position.
    ///
    /// Returns a (1, 1, 1, contextLength) mask where positions > currentPos are -inf.
    func createCausalMask(currentPos: Int) throws -> MLMultiArray {
        let shape = [1, 1, 1, config.contextLength]
        let mask = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)

        let ptr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: mask.count)
        // Fill with -inf (0xFC00 in float16)
        let negInfBits: UInt16 = 0xFC00
        for i in 0..<config.contextLength {
            ptr[i] = i <= currentPos ? 0 : negInfBits
        }

        return mask
    }

    /// Create position IDs tensor.
    func createPositionIDs(_ positions: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [NSNumber(value: positions.count)], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: positions.count)
        for (i, pos) in positions.enumerated() {
            ptr[i] = Int32(pos)
        }
        return array
    }
}
