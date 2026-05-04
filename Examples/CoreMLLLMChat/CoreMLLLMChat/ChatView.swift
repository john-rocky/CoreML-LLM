import SwiftUI
import PhotosUI
import CoreMLLLM

struct ChatView: View {
    @State private var runner = LLMRunner()
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    /// Folder name (e.g. "qwen3-vl-2b-fashion") of the currently loaded
    /// model, used to gate domain-specific UX like the Fashion auto-prompt.
    @State private var activeModelFolder: String?
    @State private var showModelPicker = false
    /// Per-token streaming text lives on its own @Observable so that only
    /// the streaming bubble (and nothing else in ChatView's body) is
    /// invalidated per generated token. See `StreamingBuffer` below.
    @State private var streaming = StreamingBuffer()
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var selectedImage: CGImage?
    @State private var selectedImageData: Data?
    /// Tracks whether the currently-selected image has already been shown
    /// in a user bubble. Image persists across turns (so the generator
    /// can reuse its KV cache), but we only render the thumbnail in the
    /// first message that introduces it — subsequent turns are text-only
    /// in the chat scroll, while the image stays implicitly attached.
    @State private var imageDisplayedInChat: Bool = false

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

                // Scroll strategy:
                // - ScrollViewReader + scrollTo (no `withAnimation`) keeps
                //   Core Animation transactions out of the per-token path.
                //   The old `withAnimation { scrollTo }` opened overlapping
                //   CA transactions at 31 tok/s; plain scrollTo just sets
                //   the content offset, which is cheap.
                // - `.defaultScrollAnchor(.bottom)` was tried but bottom-
                //   aligns short content (empty chat → first bubble appears
                //   at the bottom) and leaves dead space when content
                //   shrinks (streaming ends). Explicit scrollTo to a
                //   sentinel avoids both.
                // - Per-token scrollTo is triggered from *inside*
                //   StreamingBubble, so the onChange observation stays
                //   scoped to that subtree. ChatView's body is still not
                //   re-evaluated per token.
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(messages) { message in
                                MessageBubble(message: message)
                            }
                            StreamingBubble(buffer: streaming, scrollProxy: proxy)
                            // Zero-height sentinel that is always present so
                            // scrollTo has a stable target whether or not
                            // the streaming bubble is currently visible.
                            // Placed *below* LazyVStack's bottom padding
                            // (which we drop — see padding call below) so
                            // that scrollTo(anchor: .bottom) lands exactly
                            // at content end; otherwise the bottom padding
                            // sits below the sentinel and shows as empty
                            // space at max scroll.
                            Color.clear
                                .frame(height: 1)
                                .id("bottom-anchor")
                        }
                        // Deliberately no `.padding(.bottom)` — bottom padding
                        // below the sentinel would be visible as dead space
                        // after scrollTo(anchor: .bottom).
                        .padding(.horizontal)
                        .padding(.top)
                        .contentShape(Rectangle())
                        .simultaneousGesture(TapGesture().onEnded {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder),
                                to: nil, from: nil, for: nil)
                        })
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onChange(of: messages.count) { _, _ in
                        // Dispatch to the next runloop tick so LazyVStack has
                        // laid out the freshly appended MessageBubble before
                        // we compute the scroll target. Without this, the
                        // scrollTo can run against the *previous* content
                        // size and overshoot once the new bubble appears.
                        Task { @MainActor in
                            proxy.scrollTo("bottom-anchor", anchor: .bottom)
                        }
                    }
                }

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
        activeModelFolder = folder.lastPathComponent
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

        let attachedImageData = imageDisplayedInChat ? nil : selectedImageData
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
        if attachedImageData != nil { imageDisplayedInChat = true }
        inputText = ""
        streaming.text = ""

        let image = selectedImage
        let frames = videoFrames
        let includeAudio = videoIncludeAudio
        // Image is intentionally NOT cleared after send: it remains
        // attached to the session so follow-up turns can reuse the
        // generator's KV cache (image at a fixed sequence offset across
        // turns). The user clears it explicitly via the X on the
        // preview, picking a new image, or the Clear toolbar button.
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
                // Picking a new image resets the "shown" flag so the
                // next user message displays the new thumbnail; the
                // generator's vision fingerprint will mismatch on the
                // next generate, forcing a fresh KV state.
                imageDisplayedInChat = false
                // Auto-prompt when the active model is the Fashion FT
                // (Gemma 4 or Qwen3-VL) and the input is empty. Editable
                // before send. No effect on other models.
                await MainActor.run {
                    let isFashion = activeModelFolder == "gemma4-e2b-fashion"
                        || activeModelFolder == "qwen3-vl-2b-fashion"
                    if isFashion,
                       inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        inputText = Self.fashionAutoPrompt
                    }
                }
            }
        }
    }

    /// Trained user prompt for the Fashion LoRAs. Mirrors
    /// `vision/scripts/prepare_train.py:PROMPT` byte-exactly so on-device
    /// inputs match the training distribution and trip the structured-JSON
    /// output path. v3 schema: 5 axes (color/silhouette/material/design/
    /// item_type) + coordinate_silhouette (I/A/Y/off + style_score).
    private static let fashionAutoPrompt =
        "画像のコーディネートを MB ドレス/カジュアル理論で採点し JSON で出力してください。"
        + "items: [{category, description, "
        + "scores:{color, silhouette, material, design, item_type}, item_dress_score}], "
        + "overall_dress_ratio, "
        + "coordinate_silhouette:{type, style_score, rationale}, "
        + "target_ratio, verdict, advice"

    private func clearImage() {
        selectedPhoto = nil
        selectedImage = nil
        selectedImageData = nil
        imageDisplayedInChat = false
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
///
/// The per-token `onChange` → `scrollTo` lives here (not in ChatView) so
/// that observing `buffer.text` does not pull ChatView into the per-token
/// invalidation set. `scrollTo` without `withAnimation` is a plain content-
/// offset set — no CA transaction is created per token.
private struct StreamingBubble: View {
    let buffer: StreamingBuffer
    let scrollProxy: ScrollViewProxy

    var body: some View {
        if !buffer.text.isEmpty {
            MessageBubble(
                message: ChatMessage(role: .assistant, content: buffer.text)
            )
            .id("streaming")
            .onChange(of: buffer.text) { _, _ in
                scrollProxy.scrollTo("bottom-anchor", anchor: .bottom)
            }
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
                if message.role == .assistant,
                   let report = FashionReport.parse(from: message.content) {
                    FashionReportCard(report: report, raw: message.content)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text(message.content)
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(backgroundColor)
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
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
// MARK: - Fashion report rendering
//
// Assistant output from the Gemma 4 E2B Fashion LoRA is a JSON document
// following MB dress/casual theory. When we can parse it, render as a
// structured card instead of a raw text bubble. All fields are optional
// because on-device int4 decoding occasionally drifts from the trained
// schema — we render whatever we got and fall back to raw text if the
// parse returns nothing useful.

struct FashionReport {
    struct Scores {
        let color: Double?
        let silhouette: Double?
        let material: Double?
        let design: Double?
        /// v3 5th axis: garment-type baseline (independent of color/cut/etc).
        let item_type: Double?
    }
    struct Item {
        let category: String?
        let description: String?
        let scores: Scores?
        let item_dress_score: Double?
    }

    /// v3 outfit-level style axis: I/A/Y silhouette classification.
    struct CoordinateSilhouette {
        let type: String?      // "I" | "A" | "Y" | "off"
        let style_score: Double?
        let rationale: String?
    }

    let items: [Item]
    let overall_dress_ratio: Double?
    /// Legacy v2 field; kept for parser tolerance against old model output.
    /// Not rendered in v3 card UI.
    let tpo_assumption: String?
    let target_ratio: Double?
    let coordinate_silhouette: CoordinateSilhouette?
    let verdict: String?
    let advice: String?

    /// Non-empty if any of the top-level schema fields survived parsing.
    var isRenderable: Bool {
        !items.isEmpty
            || overall_dress_ratio != nil
            || coordinate_silhouette != nil
            || verdict != nil
            || advice != nil
    }

    // ---- Parsing ----
    //
    // The on-device int4 decoder drifts from the trained schema in many
    // ways: aliased keys (`attributes` for `scores`, `dress_score` for
    // `item_dress_score`, `overall_ratio` for `overall_dress_ratio`),
    // positional arrays where dicts were expected (`scores: [c,s,m,d]`),
    // and occasionally several malformed JSON blobs concatenated. We use
    // a tolerant parse that walks JSONSerialization output via dictionary
    // lookups + alias tables, then merges results across every balanced
    // top-level `{...}` span found in the text.

    private static let topRatioKeys = [
        "overall_dress_ratio", "overall_ratio", "ratio",
    ]
    private static let topTPOKeys = ["tpo_assumption", "tp_assumption", "durationType", "target_tpo", "tpo"]
    private static let topTargetKeys = ["target_ratio"]
    private static let topVerdictKeys = ["verdict"]
    private static let topAdviceKeys = ["advice", "comment"]
    private static let topCoordSilhouetteKeys = [
        "coordinate_silhouette", "outfit_silhouette", "style_silhouette",
    ]
    private static let itemDescriptionKeys = ["description", "item", "name"]
    private static let itemDressScoreKeys = ["item_dress_score", "dress_score", "score_break"]
    private static let scoresContainerKeys = ["scores", "attributes", "score"]
    /// v3 positional fallback: 5 axes (item_type added 5th).
    private static let scoresPositionalAxes = ["color", "silhouette", "material", "design", "item_type"]
    private static let itemTypeKeys = ["item_type", "type", "garment_type", "category_score"]
    private static let silhouetteTypeKeys = ["type", "silhouette_type", "shape"]
    private static let silhouetteStyleScoreKeys = ["style_score", "score", "completion"]
    private static let silhouetteRationaleKeys = ["rationale", "reason", "note"]

    static func parse(from text: String) -> FashionReport? {
        let cleaned = stripNoise(text)
        guard !cleaned.isEmpty else { return nil }

        var items: [Item] = []
        var ratio: Double?
        var tpo: String?
        var target: Double?
        var verdict: String?
        var advice: String?
        var silhouette: CoordinateSilhouette?

        // Walk every balanced {...} span. Multi-blob outputs are merged so
        // a fragmented response (items in one blob, overall_ratio in another)
        // still produces a coherent card.
        for span in balancedJSONSpans(in: cleaned) {
            guard let data = span.data(using: .utf8),
                  let raw = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
                  let dict = raw as? [String: Any]
            else { continue }

            if let arr = dict["items"] as? [Any] {
                for case let entry as [String: Any] in arr {
                    items.append(buildItem(from: entry))
                }
            }
            if ratio == nil, let v = lookupDouble(topRatioKeys, in: dict) { ratio = v }
            if tpo == nil, let v = lookupString(topTPOKeys, in: dict) { tpo = v }
            if target == nil, let v = lookupDouble(topTargetKeys, in: dict) { target = v }
            if verdict == nil, let v = lookupString(topVerdictKeys, in: dict) { verdict = v }
            if advice == nil, let v = lookupString(topAdviceKeys, in: dict) { advice = v }
            if silhouette == nil {
                let sValue = topCoordSilhouetteKeys.compactMap { dict[$0] }.first
                if let sDict = sValue as? [String: Any] {
                    silhouette = CoordinateSilhouette(
                        type: lookupString(silhouetteTypeKeys, in: sDict),
                        style_score: lookupDouble(silhouetteStyleScoreKeys, in: sDict),
                        rationale: lookupString(silhouetteRationaleKeys, in: sDict))
                } else if let sStr = sValue as? String {
                    // Tolerate flat string form: "coordinate_silhouette":"I"
                    silhouette = CoordinateSilhouette(
                        type: sStr, style_score: nil, rationale: nil)
                }
            }
        }

        let report = FashionReport(
            items: items,
            overall_dress_ratio: ratio,
            tpo_assumption: tpo,
            target_ratio: target,
            coordinate_silhouette: silhouette,
            verdict: verdict,
            advice: advice)
        return report.isRenderable ? report : nil
    }

    private static func stripNoise(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<channel|>", with: "")
            .replacingOccurrences(of: "<|channel|>", with: "")
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Return every top-level balanced `{...}` substring (in source order),
    /// skipping `{`/`}` characters inside JSON string literals. When the
    /// scan ends with the outermost object still open (model's response
    /// got truncated mid-stream by max_new_tokens), append a synthesised
    /// "best-effort close" so the partial JSON can still be salvaged
    /// by JSONSerialization. The salvage truncates back to the last
    /// completed top-level field and closes any open structures with
    /// matching `]` / `"` / `}` tokens.
    private static func balancedJSONSpans(in text: String) -> [String] {
        var spans: [String] = []
        var depth = 0
        var inString = false
        var escape = false
        var startIdx: String.Index?
        var bracketStack: [Character] = []   // tracks { and [ in order, for salvage
        for idx in text.indices {
            let ch = text[idx]
            if escape {
                escape = false
                continue
            }
            if ch == "\\" {
                escape = true
                continue
            }
            if ch == "\"" {
                inString.toggle()
                continue
            }
            if inString { continue }
            if ch == "{" {
                if depth == 0 { startIdx = idx; bracketStack.removeAll() }
                depth += 1
                bracketStack.append("{")
            } else if ch == "[" {
                bracketStack.append("[")
            } else if ch == "]" {
                if bracketStack.last == "[" { bracketStack.removeLast() }
            } else if ch == "}" {
                depth -= 1
                if bracketStack.last == "{" { bracketStack.removeLast() }
                if depth == 0, let s = startIdx {
                    spans.append(String(text[s...idx]))
                    startIdx = nil
                    bracketStack.removeAll()
                } else if depth < 0 {
                    depth = 0
                    startIdx = nil
                    bracketStack.removeAll()
                }
            }
        }

        // Salvage path: outermost object never closed. Truncate back to
        // the last "completed top-level field boundary" (last comma at
        // depth 1 outside strings) and append the brackets needed to
        // close every still-open structure. JSONSerialization is
        // permissive enough to parse the result.
        if let s = startIdx, !bracketStack.isEmpty {
            let raw = String(text[s..<text.endIndex])
            if let salvaged = trySalvageOpenJSON(raw) {
                spans.append(salvaged)
            }
        }
        return spans
    }

    private static func trySalvageOpenJSON(_ raw: String) -> String? {
        // Walk from the start, tracking depth + open-bracket stack +
        // record positions of each comma at top-of-object level (depth==1
        // wrt the outermost {). The last such comma is the safe truncation
        // boundary — everything after it may be a half-written field.
        var depth = 0
        var inString = false
        var escape = false
        var stack: [Character] = []
        var lastSafeBoundary: String.Index? = nil
        for idx in raw.indices {
            let ch = raw[idx]
            if escape { escape = false; continue }
            if ch == "\\" { escape = true; continue }
            if ch == "\"" { inString.toggle(); continue }
            if inString { continue }
            switch ch {
            case "{":
                depth += 1
                stack.append("{")
            case "[":
                stack.append("[")
            case "]":
                if stack.last == "[" { stack.removeLast() }
            case "}":
                depth -= 1
                if stack.last == "{" { stack.removeLast() }
            case ",":
                // Boundary at outermost object level: depth == 1 AND the
                // top of stack is the outer `{` (no array open between).
                if depth == 1 && stack.last == "{" {
                    lastSafeBoundary = idx
                }
            default:
                break
            }
        }
        guard let boundary = lastSafeBoundary else {
            // No top-level field completed yet; can't salvage.
            return nil
        }
        // Truncate to the last safe boundary, then close all still-open
        // structures with matching brackets. We don't try to salvage
        // strings — if the truncation lands inside one, the closing
        // sequence won't help, JSONSerialization will reject and we
        // skip the span.
        var truncated = String(raw[..<boundary])
        // Close from innermost to outermost.
        // Re-walk the truncated text to compute the final stack state.
        var depth2 = 0
        var stack2: [Character] = []
        var inString2 = false
        var escape2 = false
        for ch in truncated {
            if escape2 { escape2 = false; continue }
            if ch == "\\" { escape2 = true; continue }
            if ch == "\"" { inString2.toggle(); continue }
            if inString2 { continue }
            switch ch {
            case "{": depth2 += 1; stack2.append("{")
            case "[": stack2.append("[")
            case "]": if stack2.last == "[" { stack2.removeLast() }
            case "}": depth2 -= 1; if stack2.last == "{" { stack2.removeLast() }
            default: break
            }
        }
        if inString2 {
            // The truncation point landed inside a string literal; close it.
            truncated.append("\"")
        }
        for open in stack2.reversed() {
            truncated.append(open == "{" ? "}" : "]")
        }
        return truncated
    }

    private static func buildItem(from dict: [String: Any]) -> Item {
        let category = dict["category"] as? String
        let description = lookupString(itemDescriptionKeys, in: dict)
        let scoresValue = scoresContainerKeys.compactMap { dict[$0] }.first
        let scores = buildScores(from: scoresValue)
        var dressScore = lookupDouble(itemDressScoreKeys, in: dict)
        // int8 drift fallback: model occasionally nests item_dress_score
        // inside the scores object instead of at item level. Pull it back.
        if dressScore == nil, let nested = scoresValue as? [String: Any] {
            dressScore = lookupDouble(itemDressScoreKeys, in: nested)
        }
        return Item(category: category, description: description,
                    scores: scores, item_dress_score: dressScore)
    }

    private static func buildScores(from value: Any?) -> Scores? {
        if let dict = value as? [String: Any] {
            // Object form. Pick known axes; ignore unknown keys. v3 adds
            // a 5th axis `item_type` (with aliases for drift tolerance).
            return Scores(
                color: dict["color"] as? Double ?? (dict["color"] as? NSNumber)?.doubleValue,
                silhouette: dict["silhouette"] as? Double ?? (dict["silhouette"] as? NSNumber)?.doubleValue,
                material: dict["material"] as? Double ?? (dict["material"] as? NSNumber)?.doubleValue,
                design: dict["design"] as? Double ?? (dict["design"] as? NSNumber)?.doubleValue,
                item_type: lookupDouble(itemTypeKeys, in: dict))
        }
        if let arr = value as? [Any] {
            // Positional form: [color, silhouette, material, design, item_type].
            // Tolerate any combination of Double / Int / NSNumber. Truncate to 5.
            let nums = arr.prefix(scoresPositionalAxes.count).map { v -> Double? in
                (v as? Double) ?? (v as? NSNumber)?.doubleValue
            }
            guard nums.contains(where: { $0 != nil }) else { return nil }
            let pad = Array(repeating: Double?.none, count: max(0, 5 - nums.count))
            let padded = Array(nums) + pad
            return Scores(
                color: padded[0],
                silhouette: padded[1],
                material: padded[2],
                design: padded[3],
                item_type: padded.count > 4 ? padded[4] : nil)
        }
        return nil
    }

    private static func lookupDouble(_ keys: [String], in dict: [String: Any]) -> Double? {
        for k in keys {
            if let v = dict[k] as? Double { return v }
            if let v = dict[k] as? NSNumber { return v.doubleValue }
            if let s = dict[k] as? String, let v = Double(s) { return v }
        }
        return nil
    }

    private static func lookupString(_ keys: [String], in dict: [String: Any]) -> String? {
        for k in keys {
            if let v = dict[k] as? String, !v.isEmpty { return v }
        }
        return nil
    }
}

struct FashionReportCard: View {
    let report: FashionReport
    /// Raw model output kept so the user can copy/inspect the JSON when needed.
    let raw: String
    @State private var showingRaw = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let ratio = report.overall_dress_ratio {
                DressRatioGauge(ratio: ratio, target: report.target_ratio)
            }

            // v3 silhouette chip ("Iライン (0.85)" etc). Replaces the v2 TPO
            // chip — TPO was dropped from the schema in favor of pin-70.
            if let silhouette = report.coordinate_silhouette {
                HStack(spacing: 8) {
                    SilhouetteChip(silhouette: silhouette)
                    if let rationale = silhouette.rationale, !rationale.isEmpty {
                        Text(rationale)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
            }

            if !report.items.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Items")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(Array(report.items.enumerated()), id: \.offset) { _, item in
                        FashionItemRow(item: item)
                    }
                }
            }

            if let verdict = report.verdict, !verdict.isEmpty {
                VerdictBlock(text: verdict)
            }

            if let advice = report.advice, !advice.isEmpty {
                AdviceCard(text: advice)
            }

            Button(showingRaw ? "Hide JSON" : "Show JSON") {
                showingRaw.toggle()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            if showingRaw {
                ScrollView(.horizontal, showsIndicators: false) {
                    Text(raw)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding(14)
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

private struct DressRatioGauge: View {
    let ratio: Double
    let target: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Dress ratio")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.2f", clamped))
                    .font(.title2.monospacedDigit().weight(.semibold))
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.systemGray5))
                    Capsule()
                        .fill(LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .leading, endPoint: .trailing))
                        .frame(width: proxy.size.width * clamped)
                    if let target {
                        let x = proxy.size.width * min(max(target, 0), 1)
                        Rectangle()
                            .fill(Color.orange)
                            .frame(width: 2)
                            .offset(x: x - 1)
                    }
                }
            }
            .frame(height: 10)
            HStack {
                Text("Casual").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text("Dress").font(.caption2).foregroundStyle(.secondary)
            }
            // MB rule of thumb caption. The model emits TPO-specific targets
            // (smart_casual=0.70, weekend=0.60, business=0.95 …) but MB's
            // public framing is the 7:3 town-wear answer; reinforcing it in
            // the card keeps the demo tied to his vocabulary regardless of
            // which TPO the model picks.
            HStack(spacing: 6) {
                Text("MB理論")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.orange)
                Text("街着の正解は 7:3 (0.70)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 2)
        }
    }

    private var clamped: Double { min(max(ratio, 0), 1) }
}

private struct Chip: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.caption)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(Color(.systemGray5))
            .clipShape(Capsule())
    }
}

/// v3 outfit-level style chip. Shows "Iライン (0.85)" for I/A/Y types or
/// "シルエット off" for off. Orange tint mirrors the MB理論 7:3 caption
/// under DressRatioGauge so silhouette + dress are visually paired.
private struct SilhouetteChip: View {
    let silhouette: FashionReport.CoordinateSilhouette

    var body: some View {
        let label = displayLabel
        let isOff = (silhouette.type?.lowercased() == "off")
        Text(label)
            .font(.caption.weight(.semibold))
            .foregroundStyle(isOff ? Color.secondary : Color.orange)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(
                (isOff ? Color(.systemGray5) : Color.orange.opacity(0.15))
            )
            .clipShape(Capsule())
    }

    private var displayLabel: String {
        let t = (silhouette.type ?? "").uppercased()
        let scoreSuffix = silhouette.style_score
            .map { String(format: " (%.2f)", min(max($0, 0), 1)) } ?? ""
        switch t {
        case "I": return "Iライン" + scoreSuffix
        case "A": return "Aライン" + scoreSuffix
        case "Y": return "Yライン" + scoreSuffix
        case "OFF", "":
            return "シルエット off" + scoreSuffix
        default:
            return t + "ライン" + scoreSuffix
        }
    }
}

private struct FashionItemRow: View {
    let item: FashionReport.Item

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(item.category ?? "item")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if let score = item.item_dress_score {
                    Text(String(format: "%.2f", min(max(score, 0), 1)))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if let desc = item.description, !desc.isEmpty {
                Text(desc)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let scores = item.scores {
                VStack(spacing: 3) {
                    ScoreBar(label: "color", value: scores.color)
                    ScoreBar(label: "silhouette", value: scores.silhouette)
                    ScoreBar(label: "material", value: scores.material)
                    ScoreBar(label: "design", value: scores.design)
                    ScoreBar(label: "item_type", value: scores.item_type)
                }
            }
        }
        .padding(10)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

private struct ScoreBar: View {
    let label: String
    let value: Double?

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 64, alignment: .leading)
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.systemGray5))
                    if let v = value {
                        Capsule()
                            .fill(Color.blue.opacity(0.7))
                            .frame(width: proxy.size.width * min(max(v, 0), 1))
                    }
                }
            }
            .frame(height: 5)
            Text(value.map { String(format: "%.2f", min(max($0, 0), 1)) } ?? "—")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 32, alignment: .trailing)
        }
    }
}

private struct VerdictBlock: View {
    let text: String
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Rectangle()
                .fill(Color.accentColor)
                .frame(width: 3)
            Text(text)
                .font(.subheadline.italic())
                .foregroundStyle(.primary)
        }
    }
}

private struct AdviceCard: View {
    let text: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Advice")
                .font(.caption).foregroundStyle(.secondary)
            Text(text)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(Color.accentColor.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
