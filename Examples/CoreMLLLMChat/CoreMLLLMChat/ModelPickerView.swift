import CoreML
import SwiftUI
import CoreMLLLM

struct ModelPickerView: View {
    let downloader = ModelDownloader.shared
    let onModelReady: (URL) -> Void

    var body: some View {
        NavigationStack {
            List {
                Section("Available Models") {
                    ForEach(downloader.availableModels) { model in
                        let _ = downloader.refreshTrigger
                        let isThisModel = downloader.downloadingModelId == model.id
                        ModelRow(
                            model: model,
                            isDownloaded: downloader.isDownloaded(model),
                            hasFiles: downloader.hasFiles(model),
                            isDownloading: downloader.isDownloading && isThisModel,
                            isPaused: downloader.isPaused && isThisModel,
                            progress: downloader.progress,
                            onDownload: { downloadAndLoad(model) },
                            onLoad: {
                                if let url = downloader.localModelURL(for: model) {
                                    onModelReady(url)
                                }
                            },
                            onPause: { downloader.pause() },
                            onResume: { downloadAndLoad(model) },
                            onCancel: { downloader.cancelDownload() },
                            onDelete: {
                                do {
                                    try downloader.delete(model)
                                    downloader.status = "Deleted \(model.name)"
                                } catch {
                                    downloader.status = "Delete failed: \(error.localizedDescription)"
                                }
                            }
                        )
                    }
                }

                if downloader.isDownloading {
                    Section {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(downloader.status)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if !downloader.isPaused {
                                ProgressView(value: downloader.progress)
                            }
                            Text(String(format: "%.0f%%", downloader.progress * 100))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                } else if !downloader.status.isEmpty {
                    Section {
                        Text(downloader.status)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Qwen3.5 is now a first-class entry in "Available Models"
                // above — select "Qwen3.5 0.8B (ANE)" to download and then
                // chat via the regular ChatView. The standalone research
                // screens below are preserved in source for direct
                // debugging but not shown in the picker.
                //
                // Section("Qwen3.5-0.8B (ANE) — research") {
                //     NavigationLink { Qwen35ChatView() } label: {
                //         Label("Qwen3.5 Chat", systemImage: "bubble.left.and.bubble.right")
                //     }
                //     NavigationLink { Qwen35BenchmarkView() } label: {
                //         Label("Prefill benchmark", systemImage: "stopwatch")
                //     }
                //     NavigationLink { Qwen35DecodeBenchmarkView() } label: {
                //         Label("Decode benchmark", systemImage: "speedometer")
                //     }
                //     NavigationLink { Qwen35GeneratorView() } label: {
                //         Label("End-to-end (token IDs)", systemImage: "text.bubble")
                //     }
                // }

                Section("Qwen3-VL 2B — Phase 1") {
                    NavigationLink { GateZeroBenchmarkView() } label: {
                        Label("Gate Zero (MLState stub)", systemImage: "bolt.shield")
                    }
                    NavigationLink { Qwen3VL2BStatefulGeneratorView() } label: {
                        Label("Stateful 64-token smoke test", systemImage: "bolt.fill")
                    }
                }

                Section {
                    Toggle(isOn: Binding(
                        get: { UserDefaults.standard.object(
                            forKey: ModelDownloader.includeMultimodalKey) as? Bool ?? true },
                        set: { UserDefaults.standard.set(
                            $0, forKey: ModelDownloader.includeMultimodalKey) }
                    )) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Include multimodal (vision / audio / video)")
                            Text("Default ON. Saves ~990 MB when off — text-only install. " +
                                 "Applies to Gemma 4 E2B variants. Re-download if changed " +
                                 "after install.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Download Options")
                }

                Section("MLState Probe (research)") {
                    Button {
                        runMLStateProbe()
                    } label: {
                        Label("(1) T=288 prefill compile", systemImage: "stethoscope")
                    }
                    Button {
                        runStateBridgeProbe()
                    } label: {
                        Label("(2) State buffer bridge", systemImage: "arrow.left.arrow.right")
                    }
                    Text(downloader.status)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(nil)
                }

                Section("Troubleshooting") {
                    Button(role: .destructive) {
                        do {
                            try downloader.resetAllModels()
                            downloader.status = "Cleared all model files"
                        } catch {
                            downloader.status = "Reset failed: \(error.localizedDescription)"
                        }
                    } label: {
                        Label("Clear all cached models", systemImage: "exclamationmark.triangle")
                    }
                }
            }
            .navigationTitle("Models")
        }
    }

    private func downloadAndLoad(_ model: ModelDownloader.ModelInfo) {
        Task {
            do {
                let url = try await downloader.download(model)
                onModelReady(url)
            } catch is CancellationError {
                // User cancelled
            } catch {
                downloader.status = "Error: \(error.localizedDescription)"
            }
        }
    }

    /// Stage 8 MLState multimodal feasibility probe.
    /// Loads `Documents/mlstate_probe/chunk_1.mlmodelc` (a T=288 single-
    /// function stateful prefill chunk built by
    /// `conversion/probe_stateful_singlefunc_prefill.py` and pushed via
    /// devicectl) with `.cpuAndNeuralEngine`. The Stage 3 finding was
    /// that **multifunction** T>1 stateful is rejected by iPhone ANE 18
    /// (`ANECCompile FAILED 11`); single-function T>1 stateful was never
    /// tried. This probe answers that question in one button press.
    private func runMLStateProbe() {
        downloader.status = "Probe: loading T=288 stateful prefill chunk..."
        Task {
            do {
                let docs = FileManager.default.urls(
                    for: .documentDirectory, in: .userDomainMask).first!
                let url = docs
                    .appendingPathComponent("mlstate_probe")
                    .appendingPathComponent("chunk_1.mlmodelc")
                guard FileManager.default.fileExists(atPath: url.path) else {
                    await MainActor.run {
                        downloader.status = "Probe: file missing — push via " +
                            "`xcrun devicectl device copy to ... " +
                            "Documents/mlstate_probe/chunk_1.mlmodelc`"
                    }
                    return
                }
                let cfg = MLModelConfiguration()
                cfg.computeUnits = .cpuAndNeuralEngine
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try MLModel(contentsOf: url, configuration: cfg)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                await MainActor.run {
                    downloader.status = String(format:
                        "Probe: PASS — T=288 single-function stateful " +
                        "prefill compiled on this device in %.0f ms. " +
                        "Stateful multimodal is feasible.", dt)
                }
            } catch {
                await MainActor.run {
                    downloader.status = "Probe: FAIL — \(error.localizedDescription). " +
                        "Stateful multimodal blocked at single-function " +
                        "T>1 (same wall as multifunction)."
                }
            }
        }
    }

    /// Stage 8 follow-up probe: state-buffer bridging via memcpy.
    /// Verifies the API + memory model that the multimodal architecture
    /// needs for the prefill-model → decode-model hand-off:
    ///   1. .withMultiArray(for:) gives a CPU-readable/writable view
    ///   2. dataPointer is stable within the closure
    ///   3. memcpy between two MLState buffers (nested closures) works
    /// Tests against TWO STATES OF THE SAME MODEL (same shape by
    /// construction). Cross-model bridging is the identical operation
    /// when the two models declare matching StateType shape.
    private func runStateBridgeProbe() {
        downloader.status = "Probe 2: state-buffer bridging..."
        Task {
            do {
                let docs = FileManager.default.urls(
                    for: .documentDirectory, in: .userDomainMask).first!
                let probeURL = docs
                    .appendingPathComponent("mlstate_probe")
                    .appendingPathComponent("chunk_1.mlmodelc")
                guard FileManager.default.fileExists(atPath: probeURL.path) else {
                    await MainActor.run {
                        downloader.status = "Probe 2: probe mlmodelc missing"
                    }
                    return
                }
                let cfg = MLModelConfiguration()
                cfg.computeUnits = .cpuAndNeuralEngine
                let model = try MLModel(contentsOf: probeURL, configuration: cfg)
                // Two states of the same model — same StateType shape,
                // tests the .withMultiArray closure + memcpy mechanism.
                let prefillState = model.makeState()
                let decodeState = model.makeState()

                // CoreML 9: state.withMultiArray(for:) { closure } gives a
                // mutable view into the buffer; pointer is only valid
                // inside the closure. Write a known pattern through the
                // prefill state buffer, memcpy to the decode state buffer
                // inside nested closures, read back, verify match.
                let stateName = "kv_cache_sliding"
                var summary = ""
                prefillState.withMultiArray(for: stateName) { src in
                    let srcCount = src.count
                    summary += "  prefill[\(stateName)]: shape=\(src.shape.map { $0.intValue }), count=\(srcCount)\n"
                    let srcPtr = src.dataPointer.bindMemory(
                        to: UInt16.self, capacity: srcCount)
                    // Write a counter pattern so we can distinguish from a
                    // zero-init decode state.
                    for i in 0..<srcCount {
                        srcPtr[i] = UInt16(i & 0xFFFF)
                    }
                    let firstSrc = srcPtr[0]
                    let lastSrc = srcPtr[srcCount - 1]
                    summary += "  pattern wrote: first=0x\(String(format: "%04X", firstSrc)), last=0x\(String(format: "%04X", lastSrc))\n"
                    decodeState.withMultiArray(for: stateName) { dst in
                        let dstCount = dst.count
                        summary += "  decode[\(stateName)]: shape=\(dst.shape.map { $0.intValue }), count=\(dstCount)\n"
                        guard dstCount == srcCount else {
                            summary += "  SHAPE MISMATCH — bridging not directly viable\n"
                            return
                        }
                        let dstPtr = dst.dataPointer.bindMemory(
                            to: UInt16.self, capacity: dstCount)
                        memcpy(dstPtr, srcPtr,
                               srcCount * MemoryLayout<UInt16>.stride)
                        let firstDst = dstPtr[0]
                        let lastDst = dstPtr[srcCount - 1]
                        let ok = firstDst == firstSrc && lastDst == lastSrc
                        summary += "  bridged readback: first=0x\(String(format: "%04X", firstDst)), last=0x\(String(format: "%04X", lastDst))  \(ok ? "MATCH ✓" : "MISMATCH ✗")\n"
                    }
                }
                await MainActor.run {
                    downloader.status = "Probe 2 result:\n\(summary)"
                }
            } catch {
                await MainActor.run {
                    downloader.status = "Probe 2: FAIL — \(error.localizedDescription)"
                }
            }
        }
    }
}

struct ModelRow: View {
    let model: ModelDownloader.ModelInfo
    let isDownloaded: Bool
    let hasFiles: Bool
    let isDownloading: Bool
    let isPaused: Bool
    let progress: Double
    let onDownload: () -> Void
    let onLoad: () -> Void
    let onPause: () -> Void
    let onResume: () -> Void
    let onCancel: () -> Void
    let onDelete: () -> Void

    @State private var showDeleteConfirm = false

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)
                HStack(spacing: 4) {
                    Text(model.size)
                    if hasFiles && !isDownloaded {
                        Text("(incomplete)")
                            .foregroundStyle(.orange)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if isDownloaded {
                HStack(spacing: 12) {
                    Button("Load") { onLoad() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    Button(role: .destructive) { showDeleteConfirm = true } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
            } else if isDownloading {
                HStack(spacing: 8) {
                    if isPaused {
                        Button("Resume") { onResume() }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                    } else {
                        Button { onPause() } label: {
                            Image(systemName: "pause.fill")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    Button(role: .destructive) { onCancel() } label: {
                        Image(systemName: "xmark")
                    }
                    .controlSize(.small)
                }
            } else {
                HStack(spacing: 8) {
                    Button("Download") { onDownload() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(ModelDownloader.shared.isDownloading)
                    if hasFiles {
                        Button(role: .destructive) { onDelete() } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                    }
                }
            }
        }
        .padding(.vertical, 4)
        .confirmationDialog("Delete \(model.name)?", isPresented: $showDeleteConfirm) {
            Button("Delete", role: .destructive) { onDelete() }
        } message: {
            Text("Downloaded model files will be removed.")
        }
    }
}
