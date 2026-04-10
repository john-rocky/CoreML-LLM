import SwiftUI
import PhotosUI

struct ChatView: View {
    @State private var runner = LLMRunner()
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var showModelPicker = false
    @State private var streamingText = ""
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var selectedImage: CGImage?
    @State private var selectedImageData: Data?

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
                        && audioRecorder.recordedSamples == nil)
                       || !runner.isLoaded || runner.isGenerating)
        }
        .padding()
    }

    private func loadModel(from folderURL: URL) {
        // Use bundled model if available (for testing without HF download)
        let bundled = Bundle.main.url(forResource: "gemma4-e2b", withExtension: nil)
        let folder = bundled ?? folderURL
        let modelURL = folder.appendingPathComponent("model.mlpackage")
        messages.append(ChatMessage(role: .system, content: "Loading model..."))
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
        var text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let audio = audioRecorder.recordedSamples
        print("[ChatView] sendMessage: text='\(text.prefix(30))', audio=\(audio != nil ? "\(audio!.count) samples" : "nil"), isRecording=\(audioRecorder.isRecording)")

        // Allow audio-only sends with a default prompt
        if text.isEmpty && audio != nil {
            text = "What do you hear in this audio?"
        }
        guard !text.isEmpty else { return }

        let attachedImageData = selectedImageData
        var content = text
        if audio != nil { content = "[Audio] " + text }
        messages.append(ChatMessage(role: .user, content: content, imageData: attachedImageData))
        inputText = ""
        streamingText = ""

        let image = selectedImage
        clearImage()
        audioRecorder.clear()

        Task {
            do {
                print("[ChatView] calling generate with audio=\(audio != nil ? "\(audio!.count)" : "nil")")
                let stream = try await runner.generate(messages: messages, image: image,
                                                        audio: audio)
                for await token in stream {
                    streamingText += token
                }
                if !streamingText.isEmpty {
                    messages.append(ChatMessage(role: .assistant, content: streamingText))
                    streamingText = ""
                }
            } catch {
                messages.append(ChatMessage(role: .system, content: "Error: \(error.localizedDescription)"))
            }
        }
    }

    private func toggleRecording() {
        if audioRecorder.isRecording {
            audioRecorder.stop()
        } else {
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
}

#Preview { ChatView() }
