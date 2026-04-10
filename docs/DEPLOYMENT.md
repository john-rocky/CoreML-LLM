# CoreML Model Deployment Notes

Hard-won lessons from deploying CoreML models to iPhone.

## .mlmodelc vs .mlpackage

| Format | What | Where it runs |
|---|---|---|
| `.mlpackage` | Source format from `ct.convert()` | Mac only (needs compile) |
| `.mlmodelc` | Compiled format | iPhone + Mac |

### Creating .mlmodelc for iPhone

**The correct way** (produces cross-platform .mlmodelc):
```python
import coremltools as ct
import shutil

# 1. Load mlpackage on macOS — triggers internal compilation
model = ct.models.MLModel("chunk1.mlpackage")

# 2. Get compiled path WHILE model object is alive
compiled = model.get_compiled_model_path()

# 3. Copy immediately (temp path gets deleted when model is GC'd)
shutil.copytree(compiled, "chunk1.mlmodelc")
```

**What does NOT work:**
- `xcrun coremlcompiler compile` — produces Mac-specific binary. The `model.mil` file is binary protobuf, not text MIL. iPhone's CoreML runtime expects text MIL starting with `program(X.Y)`.
- `str(mil_program)` — produces debug display format (`main[CoreML8]`), not the serialization format (`program(1.3)`).

### .mlmodelc directory structure

A valid .mlmodelc for iPhone contains:
```
chunk1.mlmodelc/
  model.mil          ← TEXT MIL program (starts with "program(1.3)\n")
  coremldata.bin     ← Small metadata (~910 bytes)
  metadata.json      ← Model metadata (~8 KB)
  weights/
    weight.bin        ← INT4 palettized weights
  analytics/
    coremldata.bin    ← Analytics metadata (~250 bytes)
```

### Key insight: model.mil format

| Source | First bytes | Format | iPhone? |
|---|---|---|---|
| coremltools MLModel.get_compiled_model_path() | `program(1.3)` | Text MIL | YES |
| xcrun coremlcompiler | `0x0809...` | Binary protobuf | NO |
| str(MIL Program) | `main[CoreML8]` | Debug display | NO |

## ANE Compilation on iPhone

First-time model load on a new device triggers ANE compilation:
- Takes 30s-2min per chunk
- Result is cached by iOS (subsequent loads are fast)
- If .mlpackage is used instead of .mlmodelc, `MLModel(contentsOf:)` compiles automatically

## IOSurface-backed KV Cache

```swift
var pixelBuffer: CVPixelBuffer?
CVPixelBufferCreate(kCFAllocatorDefault, width, height,
    kCVPixelFormatType_OneComponent16Half,
    [kCVPixelBufferIOSurfacePropertiesKey: [:],
     kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
    &pixelBuffer)
let kv = try MLMultiArray(pixelBuffer: pixelBuffer!, shape: shape)
```

Measured improvement: ~5% decode speed (28 → 29.4 tok/s).

**WARNING**: Preallocating masks/buffers and reusing across predictions can cause ANE race conditions (predict regressed from 31.8ms to 33.2ms). Only preallocate KV cache, not per-step temporaries.

## Embedding Lookup Optimization

Vectorized INT8→FP16 with Accelerate (2.0ms → 0.4ms per token):
```swift
vDSP.convertElements(of: int8Buffer, to: &f32Buffer)
vDSP.multiply(rowScale, f32Buffer, result: &f32Buffer)
vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
```

## SDPA Fusion — Does NOT Work on ANE

Tested approaches:
1. Pre-scale Q by sqrt(d) → fp16 overflow
2. Pre-scale Q and K by d^(1/4) → CoreML produces wrong results
3. `is_causal=True` flag → same wrong results
4. `attn_mask` parameter → same wrong results

Root cause: coremltools' MIL SDPA op has no `scale` parameter (hardcoded to 1/sqrt(d)). The decomposition into individual ops has different precision behavior from manual attention. **Every successful ANE LLM project uses manual `matmul→add→softmax→matmul`.**

## Memory Measurement

```swift
// phys_footprint = iOS jetsam basis (the REAL memory usage)
var info = task_vm_info_data_t()
var count = mach_msg_type_number_t(...)
task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ...)
let realMemory = info.phys_footprint  // ~1 GB for Gemma 4 E2B
```

Xcode's memory gauge underreports when CoreML loads INT4 palettized weights.

## Audio Tower Conversion (WIP)

Blockers:
1. `torch.Tensor.unfold()` not supported by coremltools → rewrite with explicit slice
2. NumPy 2.4+ breaks `_int` cast → downgrade to `numpy<2.4`
3. Dynamic masking → monkey-patch `create_bidirectional_mask = None`

See [project_audio_conversion.md](../.claude/projects/.../memory/project_audio_conversion.md) for full plan.
