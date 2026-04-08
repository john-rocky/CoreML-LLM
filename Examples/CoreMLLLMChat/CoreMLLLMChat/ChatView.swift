import SwiftUI

struct ChatView: View {
    @State private var runner = LLMRunner()
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var showModelPicker = false
    @State private var modelURL: URL?
    @State private var streamingText = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                if !runner.isLoaded {
                    statusBar
                }

                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(messages) { message in
                                MessageBubble(message: message)
                            }

                            // Streaming response
                            if !streamingText.isEmpty {
                                MessageBubble(message: ChatMessage(role: .assistant, content: streamingText))
                                    .id("streaming")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: streamingText) {
                        withAnimation {
                            proxy.scrollTo("streaming", anchor: .bottom)
                        }
                    }
                }

                // Performance indicator
                if runner.isGenerating {
                    HStack {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text(String(format: "%.1f tok/s", runner.tokensPerSecond))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                Divider()

                // Input bar
                inputBar
            }
            .navigationTitle("CoreML LLM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Load Model") {
                        showModelPicker = true
                    }
                    .disabled(runner.isGenerating)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Clear") {
                        messages.removeAll()
                        streamingText = ""
                        runner.resetConversation()
                    }
                    .disabled(runner.isGenerating)
                }
            }
            .fileImporter(
                isPresented: $showModelPicker,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    loadModel(from: url)
                }
            }
        }
    }

    // MARK: - Subviews

    private var statusBar: some View {
        HStack {
            Image(systemName: runner.isLoaded ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(runner.isLoaded ? .green : .secondary)
            Text(runner.loadingStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial)
    }

    private var inputBar: some View {
        HStack(spacing: 12) {
            TextField("Message", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .disabled(!runner.isLoaded || runner.isGenerating)

            Button {
                sendMessage()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                       || !runner.isLoaded
                       || runner.isGenerating)
        }
        .padding()
    }

    // MARK: - Actions

    private func loadModel(from folderURL: URL) {
        // Look for model.mlpackage in the selected folder
        let modelURL = folderURL.appendingPathComponent("model.mlpackage")

        Task {
            do {
                try await runner.loadModel(from: modelURL)
            } catch {
                messages.append(ChatMessage(role: .system, content: "Failed to load: \(error.localizedDescription)"))
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        messages.append(ChatMessage(role: .user, content: text))
        inputText = ""
        streamingText = ""

        Task {
            do {
                let stream = try await runner.generate(messages: messages)
                for await token in stream {
                    streamingText += token
                }
                // Move streaming text to messages
                if !streamingText.isEmpty {
                    messages.append(ChatMessage(role: .assistant, content: streamingText))
                    streamingText = ""
                }
            } catch {
                messages.append(ChatMessage(role: .system, content: "Error: \(error.localizedDescription)"))
            }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.role == .user ? "You" : "Assistant")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Text(message.content)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(backgroundColor)
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }

            if message.role != .user { Spacer(minLength: 60) }
        }
    }

    private var backgroundColor: Color {
        switch message.role {
        case .user: return .blue
        case .assistant: return Color(.systemGray5)
        case .system: return Color(.systemOrange).opacity(0.2)
        }
    }
}

#Preview {
    ChatView()
}
