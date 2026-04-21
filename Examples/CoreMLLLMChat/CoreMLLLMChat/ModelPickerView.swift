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

                Section("Research") {
                    NavigationLink {
                        Qwen35BenchmarkView()
                    } label: {
                        Label("Qwen3.5-0.8B prefill benchmark", systemImage: "stopwatch")
                    }
                    NavigationLink {
                        Qwen35DecodeBenchmarkView()
                    } label: {
                        Label("Qwen3.5-0.8B decode benchmark", systemImage: "speedometer")
                    }
                    NavigationLink {
                        Qwen35GeneratorView()
                    } label: {
                        Label("Qwen3.5-0.8B end-to-end generate", systemImage: "text.bubble")
                    }
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
