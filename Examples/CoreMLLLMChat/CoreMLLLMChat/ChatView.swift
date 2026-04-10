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

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !runner.isLoaded {
                    statusBar
                    // Big fallback button in case the toolbar button has
                    // SwiftUI hit-test issues. Always shown when no model.
                    Button {
                        print("[UI] Big Get Model tapped")
                        showModelPicker = true
                    } label: {
                        HStack {
                            Image(systemName: "arrow.down.circle.fill")
                            Text("Get Model")
                                .fontWeight(.semibold)
                        }
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
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
                        Button("Switch") {
                            print("[UI] Switch tapped")
                            showModelPicker = true
                        }
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

            TextField("Message", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .disabled(!runner.isLoaded || runner.isGenerating)

            Button { sendMessage() } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                       || !runner.isLoaded || runner.isGenerating)
        }
        .padding()
    }

    private func loadModel(from folderURL: URL) {
        let modelURL = folderURL.appendingPathComponent("model.mlpackage")
        messages.append(ChatMessage(role: .system, content: "Loading model..."))
        // Detached so the synchronous MLModel(contentsOf:) calls inside
        // loadChunked can't block the main actor / UI thread.
        Task.detached(priority: .userInitiated) {
            do {
                try await runner.loadModel(from: modelURL)
                await MainActor.run {
                    messages.append(ChatMessage(role: .system, content: "Model loaded! " + (runner.hasVision ? "Image input enabled." : "")))
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
        guard !text.isEmpty else { return }

        messages.append(ChatMessage(role: .user, content: text))
        inputText = ""
        streamingText = ""

        let image = selectedImage
        clearImage()

        Task {
            do {
                let stream = try await runner.generate(messages: messages, image: image)
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
