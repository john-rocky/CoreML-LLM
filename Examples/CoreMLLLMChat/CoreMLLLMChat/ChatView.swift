import SwiftUI
import PhotosUI
import CoreML
import CoreMLLLM

enum ComputeChoice: String, CaseIterable, Identifiable {
    case ane = "ANE"
    case gpu = "GPU"
    case all = "All"
    var id: String { rawValue }
    var mlValue: MLComputeUnits {
        switch self {
        case .ane: return .cpuAndNeuralEngine
        case .gpu: return .cpuAndGPU
        case .all: return .all
        }
    }
}

struct ChatView: View {
    @State private var runner = LLMRunner()
    @State private var computeChoice: ComputeChoice = .ane
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var showModelPicker = false
    @State private var streamingText = ""
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var selectedImage: CGImage?
    @State private var selectedImageData: Data?

    // Video picker (Gemma 4 video path)
    @State private var selectedVideoItem: PhotosPickerItem?
    @State private var selectedVideoURL: URL?
    @State private var selectedVideoLabel: String?
    @State private var videoFrames: Int = 6
    @State private var videoIncludeAudio: Bool = false

    // Audio recording
    @State private var audioRecorder = AudioRecorder()

    // Battery benchmark state
    @State private var benchmarkRunning = false
    @State private var benchmarkStatus: String = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !runner.isLoaded {
                    statusBar
                    Button {
                        showModelPicker = true
                    } label: {
                        HStack {
                            Image(systemName: "arrow.down.circle.fill")
                            Text("Get Model").fontWeight(.semibold)
                        }
                        .padding(.horizontal, 24).padding(.vertical, 12)
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                    }
                    .padding(.top, 12)
                }

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(messages) { message in
                                MessageBubble(message: message)
                            }
                            if !streamingText.isEmpty {
                                MessageBubble(message: ChatMessage(role: .assistant, content: streamingText))
                                    .id("streaming")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: streamingText) {
                        withAnimation { proxy.scrollTo("streaming", anchor: .bottom) }
                    }
                }

                // tok/s stays visible after generation finishes so the
                // final speed can still be read off the screen.
                if runner.isLoaded && (runner.isGenerating || runner.tokensPerSecond > 0) {
                    HStack(spacing: 6) {
                        if runner.isGenerating {
                            ProgressView().scaleEffect(0.8)
                        }
                        Text(String(format: "%.1f tok/s", runner.tokensPerSecond))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        if runner.mtpAcceptanceRate > 0 {
                            Text(String(format: "acc0=%.0f%%", runner.mtpAcceptanceRate * 100))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        if runner.crossVocabAcceptanceRate > 0 {
                            Text(String(format: "xv=%.0f%%", runner.crossVocabAcceptanceRate * 100))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        if !runner.isGenerating {
                            Text("(last)")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .padding(.vertical, 4)
                }

                // Image preview
                if let imageData = selectedImageData, let uiImage = UIImage(data: imageData) {
                    HStack {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 60)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        Button { clearImage() } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.top, 4)
                }

                // Video preview
                if let label = selectedVideoLabel {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "video.fill")
                                .foregroundStyle(.blue)
                            Text(label)
                                .font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button { clearVideo() } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                        }
                        HStack(spacing: 12) {
                            Stepper("frames: \(videoFrames)",
                                    value: $videoFrames, in: 1...24)
                                .font(.caption)
                                .fixedSize()
                            if runner.hasAudio {
                                Toggle("audio", isOn: $videoIncludeAudio)
                                    .toggleStyle(.switch)
                                    .font(.caption)
                                    .fixedSize()
                            }
                            Spacer()
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 4)
                }

                // Audio preview
                if audioRecorder.recordedSamples != nil || audioRecorder.isRecording {
                    HStack {
                        Image(systemName: "waveform")
                            .foregroundStyle(.purple)
                        if audioRecorder.isRecording {
                            Text(String(format: "Recording... %.1fs", audioRecorder.duration))
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text(String(format: "Audio ready (%.1fs)",
                                        Double(audioRecorder.recordedSamples?.count ?? 0) / 16000.0))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if !audioRecorder.isRecording {
                            Button { audioRecorder.clear() } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.top, 4)
                }

                if benchmarkRunning || !benchmarkStatus.isEmpty {
                    Text(benchmarkStatus)
                        .font(.caption.monospaced())
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(Color.orange.opacity(0.15))
                }

                Divider()
                inputBar
            }
            .navigationTitle(runner.isLoaded ? runner.modelName : "CoreML LLM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // Show "Switch" only when a model is already loaded.
                // The "Get Model" entry point lives in the big in-view button
                // (toolbar topBarLeading + inline title has a SwiftUI iOS 18
                // hit-test bug that swallows taps).
                if runner.isLoaded {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("Switch") { showModelPicker = true }
                            .disabled(runner.isGenerating || benchmarkRunning)
                    }
                }
                if runner.isLoaded {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Mem") {
                            messages.append(ChatMessage(role: .system, content: runner.memoryReport()))
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("ANE?") { verifyANE() }
                            .disabled(runner.isGenerating || benchmarkRunning)
                    }
                }
                if runner.isLoaded {
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu("Bench") {
                            Button("5 min")  { startBenchmark(minutes: 5) }
                            Button("10 min") { startBenchmark(minutes: 10) }
                            Button("30 min") { startBenchmark(minutes: 30) }
                            Divider()
                            Section("Compare ANE vs GPU") {
                                Button("2 min each")  { startABBenchmark(minutesPerSide: 2) }
                                Button("5 min each")  { startABBenchmark(minutesPerSide: 5) }
                                Button("10 min each") { startABBenchmark(minutesPerSide: 10) }
                            }
                        }
                        .disabled(runner.isGenerating || benchmarkRunning)
                    }
                }
                if runner.isLoaded {
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu {
                            Picker("Compute", selection: $computeChoice) {
                                ForEach(ComputeChoice.allCases) { c in
                                    Text(c.rawValue).tag(c)
                                }
                            }
                        } label: {
                            Text(computeChoice.rawValue)
                        }
                        .disabled(runner.isGenerating || benchmarkRunning)
                        .onChange(of: computeChoice) { _, new in
                            reloadWithCompute(new)
                        }
                    }
                }
                if runner.hasAudio {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Test") { runAudioTest() }
                            .disabled(runner.isGenerating)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Clear") {
                        messages.removeAll()
                        streamingText = ""
                        clearImage()
                        runner.resetConversation()
                    }
                    .disabled(runner.isGenerating)
                }
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView { modelURL in
                    showModelPicker = false
                    loadModel(from: modelURL.deletingLastPathComponent())
                }
            }
            .onChange(of: selectedPhoto) {
                loadPhoto()
            }
            .onChange(of: selectedVideoItem) {
                loadVideo()
            }
        }
    }

    private var statusBar: some View {
        HStack {
            Image(systemName: runner.isLoaded ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(runner.isLoaded ? .green : .secondary)
            Text(runner.loadingStatus)
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal).padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial)
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            // Image picker (only for multimodal models)
            if runner.hasVision {
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Image(systemName: "photo")
                        .font(.title3)
                }
                .disabled(runner.isGenerating)
            }

            // Video picker (Gemma 4 video path)
            if runner.hasVision {
                PhotosPicker(selection: $selectedVideoItem, matching: .videos) {
                    Image(systemName: "video")
                        .font(.title3)
                }
                .disabled(runner.isGenerating)
            }

            // Mic button (only for audio-capable models)
            if runner.hasAudio {
                Button { toggleRecording() } label: {
                    Image(systemName: audioRecorder.isRecording ? "stop.circle.fill" : "mic")
                        .font(.title3)
                        .foregroundStyle(audioRecorder.isRecording ? .red : .accentColor)
                }
                .disabled(runner.isGenerating)
            }

            TextField("Message", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .disabled(!runner.isLoaded || runner.isGenerating)

            Button { sendMessage() } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled((inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        && audioRecorder.recordedSamples == nil
                        && selectedVideoURL == nil)
                       || !runner.isLoaded || runner.isGenerating)
        }
        .padding()
    }

    private func loadModel(from folderURL: URL) {
        // Always honor the user's selection from ModelPickerView. The previous
        // `Bundle.main.url(forResource: "gemma4-e2b")` fallback silently
        // overrode E4B selections when a bundled E2B existed in the Xcode
        // project — breaking multi-model switching.
        let folder = folderURL
        let modelURL = folder.appendingPathComponent("model.mlpackage")
        messages.append(ChatMessage(role: .system, content: "Loading \(folder.lastPathComponent)..."))
        // Detached so the synchronous MLModel(contentsOf:) calls inside
        // loadChunked can't block the main actor / UI thread.
        Task.detached(priority: .userInitiated) {
            do {
                try await runner.loadModel(from: modelURL)
                await MainActor.run {
                    var caps = [String]()
                    if runner.hasVision { caps.append("Image") }
                    if runner.hasAudio { caps.append("Audio") }
                    let capsStr = caps.isEmpty ? "" : " " + caps.joined(separator: " + ") + " enabled."
                    messages.append(ChatMessage(role: .system, content: "Model loaded!" + capsStr))
                }
            } catch {
                await MainActor.run {
                    messages.append(ChatMessage(role: .system, content: "Failed: \(error.localizedDescription)"))
                }
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let audio = audioRecorder.recordedSamples
        let videoURL = selectedVideoURL
        print("[ChatView] sendMessage: text='\(text.prefix(30))', audio=\(audio != nil ? "\(audio!.count) samples" : "nil"), video=\(videoURL?.lastPathComponent ?? "nil")")

        guard !text.isEmpty || audio != nil || videoURL != nil else { return }

        let attachedImageData = selectedImageData
        let content: String
        if videoURL != nil && text.isEmpty {
            content = "[Video]"
        } else if videoURL != nil {
            content = "[Video] " + text
        } else if audio != nil && text.isEmpty {
            content = "[Audio]"
        } else if audio != nil {
            content = "[Audio] " + text
        } else {
            content = text
        }
        let userMessage = ChatMessage(role: .user, content: content,
                                       imageData: attachedImageData)
        let userMessageId = userMessage.id
        messages.append(userMessage)
        inputText = ""
        streamingText = ""

        let image = selectedImage
        let frames = videoFrames
        let includeAudio = videoIncludeAudio
        clearImage()
        audioRecorder.clear()

        Task {
            do {
                let stream: AsyncStream<String>
                if let videoURL {
                    let opts = VideoProcessor.Options(
                        fps: 1.0, maxFrames: frames,
                        includeAudio: includeAudio)
                    // Surface the same frames the model is about to see in the
                    // chat bubble. Extraction is fast (~50 ms × N at 1 fps) so
                    // we do it inline before kicking off inference, keeping the
                    // thumbnail row in sync with the prompt the encoder gets.
                    let extracted = try? await VideoProcessor.extractFrames(
                        from: videoURL, options: opts)
                    if let extracted, !extracted.isEmpty {
                        let thumbs = await Self.buildThumbnails(extracted)
                        await MainActor.run {
                            if let idx = messages.firstIndex(where: { $0.id == userMessageId }) {
                                messages[idx].videoFrames = thumbs
                            }
                        }
                    }
                    stream = try await runner.generate(
                        messages: messages, videoURL: videoURL, videoOptions: opts)
                } else {
                    stream = try await runner.generate(
                        messages: messages, image: image, audio: audio)
                }
                for await token in stream {
                    streamingText += token
                }
                if !streamingText.isEmpty {
                    messages.append(ChatMessage(role: .assistant, content: streamingText))
                    streamingText = ""
                }
                if videoURL != nil { await MainActor.run { clearVideo() } }
            } catch {
                messages.append(ChatMessage(role: .system, content: "Error: \(error.localizedDescription)"))
            }
        }
    }

    private func loadVideo() {
        guard let item = selectedVideoItem else { return }
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    await MainActor.run {
                        messages.append(ChatMessage(role: .system,
                            content: "Video load failed (no data)."))
                    }
                    return
                }
                let ext = "mov"
                let tmp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("picked-\(UUID().uuidString).\(ext)")
                try data.write(to: tmp)
                let mb = Double(data.count) / 1_048_576.0
                await MainActor.run {
                    selectedVideoURL = tmp
                    selectedVideoLabel = String(format: "Video ready (%.1f MB)", mb)
                }
            } catch {
                await MainActor.run {
                    messages.append(ChatMessage(role: .system,
                        content: "Video load error: \(error.localizedDescription)"))
                }
            }
        }
    }

    private func clearVideo() {
        if let url = selectedVideoURL { try? FileManager.default.removeItem(at: url) }
        selectedVideoURL = nil
        selectedVideoLabel = nil
        selectedVideoItem = nil
    }

    /// Downscale each frame to ~96 px on the long edge and JPEG-encode.
    /// Runs off the main actor; output is small enough (~3–6 KB / thumb)
    /// that storing it in `ChatMessage` keeps the bubble lightweight.
    private static func buildThumbnails(
        _ frames: [VideoProcessor.Frame]
    ) async -> [(Data, Double)] {
        await Task.detached(priority: .userInitiated) {
            let target: CGFloat = 96
            return frames.compactMap { frame -> (Data, Double)? in
                let w = CGFloat(frame.image.width)
                let h = CGFloat(frame.image.height)
                let scale = max(w, h) > target ? target / max(w, h) : 1
                let tw = max(1, Int(w * scale))
                let th = max(1, Int(h * scale))
                guard let ctx = CGContext(
                    data: nil, width: tw, height: th, bitsPerComponent: 8,
                    bytesPerRow: tw * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue).rawValue
                ) else { return nil }
                ctx.interpolationQuality = .medium
                ctx.draw(frame.image, in: CGRect(x: 0, y: 0, width: tw, height: th))
                guard let cg = ctx.makeImage(),
                      let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.7)
                else { return nil }
                return (data, frame.timestampSeconds)
            }
        }.value
    }

    private func toggleRecording() {
        if audioRecorder.isRecording {
            audioRecorder.stop()
        } else {
            audioRecorder.maxDuration = runner.maxAudioDuration
            do {
                try audioRecorder.start()
            } catch {
                messages.append(ChatMessage(role: .system,
                    content: "Mic error: \(error.localizedDescription)"))
            }
        }
    }

    private func startBenchmark(minutes: Int) {
        // Warn if plugged in — drain can't be measured accurately while charging.
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state = UIDevice.current.batteryState
        if state == .charging || state == .full {
            messages.append(ChatMessage(role: .system, content: "[Benchmark] Device is charging — unplug for accurate SoC drain measurement."))
        }

        benchmarkRunning = true
        benchmarkStatus = "Benchmark starting… (\(minutes) min)"
        messages.append(ChatMessage(role: .system, content: "[Benchmark] Starting \(minutes)-minute sustained generation. Unplug, airplane mode recommended. Screen will stay on."))

        // Keep the screen awake during the benchmark so the OS doesn't
        // auto-lock and park the app in the background.
        UIApplication.shared.isIdleTimerDisabled = true

        Task {
            defer { UIApplication.shared.isIdleTimerDisabled = false }
            do {
                let result = try await runner.runBenchmark(
                    duration: TimeInterval(minutes * 60)
                ) { prog in
                    let batNow = prog.batteryNow >= 0 ? Int(prog.batteryNow * 100) : -1
                    let batStart = prog.batteryStart >= 0 ? Int(prog.batteryStart * 100) : -1
                    benchmarkStatus = String(
                        format: "[Bench] %ds / round %d  %d tok  avg %.1f tok/s  SoC %d→%d%%  %@",
                        Int(prog.elapsed),
                        prog.round,
                        prog.totalTokens,
                        prog.avgTokPerSec,
                        batStart,
                        batNow,
                        LLMRunner.thermalString(prog.thermal) as NSString
                    )
                }

                benchmarkRunning = false
                let bs = result.batteryStart >= 0 ? Int(result.batteryStart * 100) : -1
                let be = result.batteryEnd >= 0 ? Int(result.batteryEnd * 100) : -1
                let abortNote = result.abortedThermal
                    ? "\nAborted       : YES (thermal .serious — protecting battery)"
                    : ""
                let logLines = result.batteryLog.map { entry in
                    "  \(String(format: "%5.0f", entry.0))s → \(Int(entry.1 * 100))%"
                }.joined(separator: "\n")
                let summary = """
                [Benchmark RESULT]
                Duration      : \(Int(result.duration))s (\(String(format: "%.1f", result.duration / 60.0)) min)
                Rounds        : \(result.rounds)
                Total tokens  : \(result.totalTokens)
                Avg tok/s     : \(String(format: "%.2f", result.avgTokPerSec))
                Battery       : \(bs)% → \(be)%  (Δ \(String(format: "%.2f", result.drainedPercent))%)
                Drain rate    : \(String(format: "%.3f", result.drainedPerMinute))%/min
                Tokens/%SoC   : \(String(format: "%.0f", result.tokensPerPercent))
                Thermal       : \(LLMRunner.thermalString(result.thermalStart)) → \(LLMRunner.thermalString(result.thermalEnd))\(abortNote)
                Battery log:
                \(logLines)
                """
                print(summary)
                benchmarkStatus = "Benchmark done. See chat for result."
                messages.append(ChatMessage(role: .system, content: summary))
            } catch {
                benchmarkRunning = false
                benchmarkStatus = ""
                messages.append(ChatMessage(role: .system, content: "[Benchmark] Failed: \(error.localizedDescription)"))
            }
        }
    }

    private func reloadWithCompute(_ choice: ComputeChoice) {
        guard let folder = runner.modelFolderURL else { return }
        let modelURL = folder.appendingPathComponent("model.mlpackage")
        messages.append(ChatMessage(role: .system,
            content: "Reloading model on \(LLMRunner.computeUnitsString(choice.mlValue))…"))
        Task.detached(priority: .userInitiated) {
            do {
                try await runner.loadModel(from: modelURL, computeUnits: choice.mlValue)
                await MainActor.run {
                    messages.append(ChatMessage(role: .system,
                        content: "Loaded on \(LLMRunner.computeUnitsString(choice.mlValue))."))
                }
            } catch {
                await MainActor.run {
                    messages.append(ChatMessage(role: .system,
                        content: "Reload failed: \(error.localizedDescription)"))
                }
            }
        }
    }

    private func startABBenchmark(minutesPerSide: Int) {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state = UIDevice.current.batteryState
        if state == .charging || state == .full {
            messages.append(ChatMessage(role: .system, content: "[A/B] Device is charging — unplug for accurate SoC drain measurement."))
        }
        benchmarkRunning = true
        benchmarkStatus = "A/B starting… (\(minutesPerSide) min per side)"
        messages.append(ChatMessage(role: .system,
            content: "[A/B] Running ANE then GPU for \(minutesPerSide) min each. Reload between sides takes ~1 min; 60s cool-down in between."))
        UIApplication.shared.isIdleTimerDisabled = true

        Task {
            defer { UIApplication.shared.isIdleTimerDisabled = false }
            do {
                let ab = try await runner.runABBenchmark(
                    durationPerSide: TimeInterval(minutesPerSide * 60),
                    onPhase: { phase in
                        benchmarkStatus = phase
                    },
                    onProgress: { prog in
                        let batNow = prog.batteryNow >= 0 ? Int(prog.batteryNow * 100) : -1
                        benchmarkStatus = String(
                            format: "[A/B] %ds  %d tok  avg %.1f tok/s  SoC %d%%  %@",
                            Int(prog.elapsed), prog.totalTokens, prog.avgTokPerSec,
                            batNow,
                            LLMRunner.thermalString(prog.thermal) as NSString)
                    }
                )
                benchmarkRunning = false
                benchmarkStatus = "A/B done. See chat for result."

                var out = ["[A/B RESULT] \(minutesPerSide) min per side"]
                for e in ab.entries {
                    let label = LLMRunner.computeUnitsString(e.units)
                    if let err = e.error {
                        out.append("\(label): \(err)")
                        continue
                    }
                    guard let r = e.result else { continue }
                    let bs = r.batteryStart >= 0 ? Int(r.batteryStart * 100) : -1
                    let be = r.batteryEnd   >= 0 ? Int(r.batteryEnd   * 100) : -1
                    out.append("""
                    \(label):
                      tok/s avg   : \(String(format: "%.2f", r.avgTokPerSec))
                      tokens      : \(r.totalTokens)  (rounds \(r.rounds))
                      battery     : \(bs)% → \(be)%  (Δ \(String(format: "%.2f", r.drainedPercent))%)
                      drain/min   : \(String(format: "%.3f", r.drainedPerMinute))%/min
                      tokens/%SoC : \(String(format: "%.0f", r.tokensPerPercent))
                      thermal     : \(LLMRunner.thermalString(r.thermalStart)) → \(LLMRunner.thermalString(r.thermalEnd))\(r.abortedThermal ? "  (aborted .serious)" : "")
                    """)
                }
                // Head-to-head delta on tok/s and drain if both sides finished.
                let done = ab.entries.compactMap { e -> (MLComputeUnits, LLMRunner.BenchmarkResult)? in
                    if let r = e.result { return (e.units, r) } else { return nil }
                }
                if done.count >= 2 {
                    let (u0, r0) = done[0]
                    let (u1, r1) = done[1]
                    let fasterLabel = r0.avgTokPerSec >= r1.avgTokPerSec
                        ? LLMRunner.computeUnitsString(u0)
                        : LLMRunner.computeUnitsString(u1)
                    let speedRatio = r0.avgTokPerSec > 0 && r1.avgTokPerSec > 0
                        ? max(r0.avgTokPerSec, r1.avgTokPerSec) / min(r0.avgTokPerSec, r1.avgTokPerSec)
                        : 1.0
                    let coolerLabel: String
                    if r0.drainedPerMinute > 0 && r1.drainedPerMinute > 0 {
                        coolerLabel = r0.drainedPerMinute <= r1.drainedPerMinute
                            ? LLMRunner.computeUnitsString(u0)
                            : LLMRunner.computeUnitsString(u1)
                    } else {
                        coolerLabel = "n/a (charging or unmeasured)"
                    }
                    out.append("""
                    Summary:
                      faster       : \(fasterLabel)  (×\(String(format: "%.2f", speedRatio)))
                      lower drain  : \(coolerLabel)
                    """)
                }
                messages.append(ChatMessage(role: .system, content: out.joined(separator: "\n\n")))
            } catch {
                benchmarkRunning = false
                benchmarkStatus = ""
                messages.append(ChatMessage(role: .system, content: "[A/B] Failed: \(error.localizedDescription)"))
            }
        }
    }

    private func verifyANE() {
        messages.append(ChatMessage(role: .system, content: "Checking MLComputePlan device placement..."))
        Task.detached(priority: .userInitiated) {
            if #available(iOS 17.0, *) {
                let report = await runner.verifyANEPlacement()
                print(report)
                await MainActor.run {
                    messages.append(ChatMessage(role: .system, content: report))
                }
            } else {
                await MainActor.run {
                    messages.append(ChatMessage(role: .system, content: "MLComputePlan requires iOS 17+."))
                }
            }
        }
    }

    private func loadPhoto() {
        guard let item = selectedPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self) {
                selectedImageData = data
                if let uiImage = UIImage(data: data) {
                    selectedImage = uiImage.cgImage
                }
            }
        }
    }

    private func clearImage() {
        selectedPhoto = nil
        selectedImage = nil
        selectedImageData = nil
    }

    /// Load test_audio.pcm from Documents and run through audio pipeline.
    /// Compare with HF reference to verify on-device accuracy.
    private func runAudioTest() {
        messages.append(ChatMessage(role: .system, content: "Running audio test (5.8s C-major chord)..."))
        Task {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let pcmURL = docs.appendingPathComponent("test_audio.pcm")
            guard let data = try? Data(contentsOf: pcmURL) else {
                messages.append(ChatMessage(role: .system, content: "test_audio.pcm not found in Documents"))
                return
            }
            let count = data.count / MemoryLayout<Float>.stride
            var samples = [Float](repeating: 0, count: count)
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.assumingMemoryBound(to: Float.self)
                for i in 0..<count { samples[i] = src[i] }
            }
            messages.append(ChatMessage(role: .system, content: "Loaded \(count) samples (\(String(format: "%.1f", Double(count)/16000))s)"))

            do {
                let stream = try await runner.generate(
                    messages: [ChatMessage(role: .user, content: "What do you hear in this audio?")],
                    audio: samples)
                var response = ""
                for await token in stream { response += token }
                if !response.isEmpty {
                    messages.append(ChatMessage(role: .assistant, content: response))
                    // HF reference for this exact audio:
                    // "The audio appears to contain a melodic sound, possibly a musical instrument or vocalization."
                    messages.append(ChatMessage(role: .system,
                        content: "HF reference: \"The audio appears to contain a melodic sound, possibly a musical instrument or vocalization.\""))
                }
            } catch {
                messages.append(ChatMessage(role: .system, content: "Error: \(error.localizedDescription)"))
            }
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.role == .user ? "You" : message.role == .assistant ? "Assistant" : "System")
                    .font(.caption2).foregroundStyle(.secondary)
                if let data = message.imageData, let uiImage = UIImage(data: data) {
                    Image(uiImage: uiImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 200, maxHeight: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                if let frames = message.videoFrames, !frames.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        LazyHStack(alignment: .top, spacing: 6) {
                            ForEach(Array(frames.enumerated()), id: \.offset) { _, item in
                                let (data, ts) = item
                                if let uiImage = UIImage(data: data) {
                                    VStack(spacing: 2) {
                                        Image(uiImage: uiImage)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 64, height: 64)
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                        Text(Self.timestampLabel(ts))
                                            .font(.caption2.monospacedDigit())
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                        .padding(.vertical, 2)
                    }
                    .frame(maxWidth: 280)
                }
                Text(message.content)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(backgroundColor)
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
            if message.role != .user { Spacer(minLength: 60) }
        }
    }

    private var backgroundColor: Color {
        switch message.role {
        case .user: .blue
        case .assistant: Color(.systemGray5)
        case .system: Color.orange.opacity(0.2)
        }
    }

    private static func timestampLabel(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

#Preview { ChatView() }
