import SwiftUI
import PhotosUI
import CoreMLLLM

struct ChatView: View {
    @State private var runner = LLMRunner()
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var showModelPicker = false
    /// Per-token streaming text lives on its own @Observable so that only
    /// the streaming bubble (and nothing else in ChatView's body) is
    /// invalidated per generated token. See `StreamingBuffer` below.
    @State private var streaming = StreamingBuffer()
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

                // `.defaultScrollAnchor(.bottom)` keeps the viewport pinned
                // to the bottom as content grows, without a per-token
                // `withAnimation { scrollTo }`. Each generated token used
                // to open a Core Animation transaction; at 31 tok/s those
                // overlapped and kept the CPU busy building implicit
                // animations no one asked for.
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            MessageBubble(message: message)
                        }
                        StreamingBubble(buffer: streaming)
                    }
                    .padding()
                }
                .defaultScrollAnchor(.bottom)

                // The HUD reads `runner.tokensPerSecond` etc., which change
                // every token. Keeping it in a nested view scopes those
                // observations so that ChatView's body is not re-evaluated
                // per token just to redraw the tok/s counter.
                TokHUD(runner: runner)

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
                        }
                        .disabled(runner.isGenerating || benchmarkRunning)
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
                        streaming.text = ""
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
        streaming.text = ""

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
                    streaming.text += token
                }
                if !streaming.text.isEmpty {
                    messages.append(ChatMessage(role: .assistant, content: streaming.text))
                    streaming.text = ""
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

/// Streaming-only text buffer. Kept as a reference type so that mutating
/// `text` from the decode loop does **not** invalidate ChatView's body —
/// only views that actually read `buffer.text` (i.e. `StreamingBubble`)
/// re-render per token. The previous `@State var streamingText: String`
/// forced the whole ChatView tree to be re-evaluated at the decode rate
/// (~31 Hz on Gemma 4 E2B), which showed up as sustained CPU load during
/// long responses.
@Observable
final class StreamingBuffer {
    var text: String = ""
}

/// The assistant-side bubble shown while tokens are streaming in. Isolated
/// into its own view so that per-token mutations of `buffer.text` only
/// invalidate this subtree, not the parent ChatView.
private struct StreamingBubble: View {
    let buffer: StreamingBuffer

    var body: some View {
        if !buffer.text.isEmpty {
            MessageBubble(
                message: ChatMessage(role: .assistant, content: buffer.text)
            )
            .id("streaming")
        }
    }
}

/// tok/s counter + speculative acceptance rates. Split out so that the
/// @Observable reads on `runner.tokensPerSecond` / `isGenerating` /
/// acceptance-rate properties only invalidate this HUD, not ChatView's
/// toolbar, previews, or message list.
private struct TokHUD: View {
    let runner: LLMRunner

    var body: some View {
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
