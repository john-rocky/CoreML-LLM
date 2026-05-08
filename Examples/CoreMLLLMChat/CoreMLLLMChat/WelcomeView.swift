import SwiftUI
import CoreMLLLM

/// First-launch screen: explain the app and trigger the model download.
/// Shown only when the on-disk Fashion model is missing; subsequent
/// launches go straight to `MainView` after `AppState.bootstrap`.
struct WelcomeView: View {
    @EnvironmentObject var state: AppState
    @State private var showsAdvanced = false

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 60)

            VStack(spacing: 14) {
                Image(systemName: "tshirt.fill")
                    .font(.system(size: 64, weight: .light))
                    .foregroundStyle(Color.orange.gradient)

                Text("MB Fashion")
                    .font(.system(size: 36, weight: .semibold, design: .default))

                Text("あなたのコーデを MB 理論で採点します")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            Spacer()

            VStack(spacing: 12) {
                if state.downloader.isDownloading {
                    downloadProgress
                } else {
                    downloadCTA
                }
            }
            .padding(.horizontal, 32)

            Spacer(minLength: 60)
        }
        .padding()
    }

    private var downloadCTA: some View {
        VStack(spacing: 10) {
            Button {
                state.startDownloadAndLoad()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle.fill")
                    Text("モデルを入手")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color.orange)
                .foregroundStyle(.white)
                .clipShape(Capsule())
            }
            Text("約 2.9 GB · Wi-Fi 推奨")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var downloadProgress: some View {
        VStack(spacing: 10) {
            ProgressView(value: state.downloader.progress)
                .progressViewStyle(.linear)
                .tint(.orange)

            HStack {
                Text(state.downloader.status.isEmpty
                     ? "ダウンロード中..."
                     : state.downloader.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(state.downloader.progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// Plain spinner + caption shown during model warm-up after the bytes
/// are on disk (compile-on-first-run, mmap, ANE plan). Distinct from
/// `WelcomeView.downloadProgress` because the latter has a percentage.
struct LoadingView: View {
    let message: String

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.4)
                .tint(.orange)
            Text(message.isEmpty ? "準備中..." : message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

/// Catch-all error screen with a "Back" affordance. Sparse on purpose —
/// the user shouldn't be stuck reading a stack trace.
struct ErrorView: View {
    let message: String
    let onBack: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40))
                .foregroundStyle(.orange)

            Text(message)
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.primary)
                .padding(.horizontal, 24)

            Button("戻る") { onBack() }
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 32).padding(.vertical, 12)
                .background(Color.orange)
                .clipShape(Capsule())
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}
