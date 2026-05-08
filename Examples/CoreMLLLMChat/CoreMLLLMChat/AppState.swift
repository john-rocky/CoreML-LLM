import SwiftUI
import CoreMLLLM
import UIKit

/// Single source of truth for MBFashion's screen flow + inference state.
///
/// `phase` drives `RootView`'s switch. The runner / downloader live here
/// so individual screens stay declarative — they read from `@EnvironmentObject`
/// and call methods on `AppState` instead of holding their own runners.
@MainActor
final class AppState: ObservableObject {

    enum Phase {
        case welcome
        case modelLoading
        case ready
        case analyzing
        case results
        case error(String)
    }

    @Published var phase: Phase = .welcome
    @Published var loadingMessage: String = ""
    /// Generation progress in 0...1. Driven by token count vs `maxTokens`,
    /// capped at `progressCap` while streaming and snapped to 1.0 on finish.
    @Published var analyzeProgress: Double = 0
    @Published var selectedImage: UIImage?
    @Published var lastReport: FashionReport?
    @Published var lastRawText: String = ""

    /// Bumped on every phase transition so SwiftUI can drive the
    /// `.animation(_:value:)` modifier on the dispatcher. Phase enums
    /// don't conform to Equatable cleanly because of the associated value
    /// on `.error`, so a monotonic int avoids the conformance hassle.
    @Published var phaseTag: Int = 0

    let runner = LLMRunner()
    let downloader = ModelDownloader.shared

    /// HF identifier of the model we ship as MBFashion. The picker is
    /// gone — MBFashion is single-model, so we hardcode the entry here.
    private let modelInfo = ModelDownloader.ModelInfo.qwen3vl2bFashion

    /// Hard cap on output tokens. The Fashion LoRA emits 700–1000 tokens
    /// for a typical outfit at 24 tok/s; 2000 leaves headroom for the
    /// occasional verbose advice paragraph without blowing past the
    /// 30–40s analyze budget.
    let maxTokens = 2000
    /// Don't push the bar past this while streaming — last 5% is reserved
    /// for the "finalising / parsing" beat at the end so the bar visibly
    /// completes when the result actually lands.
    let progressCap = 0.95

    private var generationTask: Task<Void, Never>?

    /// Decide the initial phase based on whether the model is already
    /// downloaded. Called once on app launch from `RootView.task`.
    func bootstrap() async {
        if downloader.localModelURL(for: modelInfo) != nil {
            await loadModel()
        } else {
            transition(to: .welcome)
        }
    }

    /// Kick off the model download and load. Called from `WelcomeView`
    /// when the user taps the download button.
    ///
    /// Stays in `.welcome` phase while bytes are coming down — WelcomeView
    /// renders its own progress UI from `state.downloader.progress`. We
    /// only transition to `.modelLoading` for the post-download warm-up
    /// (compile-on-first-run + ANE plan), where there's nothing
    /// quantitative to show.
    func startDownloadAndLoad() {
        Task {
            do {
                _ = try await downloader.download(modelInfo)
                await loadModel()
            } catch {
                transition(to: .error("ダウンロードに失敗しました\n\(error.localizedDescription)"))
            }
        }
    }

    /// Resolve the model URL on disk and hand it to LLMRunner. Detached
    /// because the synchronous MLModel(contentsOf:) calls inside loadChunked
    /// would otherwise block the main actor and freeze the UI.
    func loadModel() async {
        guard let modelURL = downloader.localModelURL(for: modelInfo) else {
            transition(to: .error("モデルが見つかりませんでした"))
            return
        }
        transition(to: .modelLoading)
        loadingMessage = "モデルを読み込み中..."
        await Task.detached(priority: .userInitiated) { [runner] in
            do {
                try await runner.loadModel(from: modelURL)
                await MainActor.run { [weak self] in
                    self?.transition(to: .ready)
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.transition(to: .error("モデルの読み込みに失敗しました\n\(error.localizedDescription)"))
                }
            }
        }.value
    }

    /// Begin inference for the picked photo. Drives the AnalyzingView
    /// progress bar from the token stream, then hands off to ResultView.
    func analyze(image: UIImage) {
        guard let cg = image.cgImage else {
            transition(to: .error("写真の読み込みに失敗しました"))
            return
        }
        // Reset cross-call state so the previous image's KV cache doesn't
        // leak into this turn (the runner's vision fingerprint mismatch
        // would force a refresh anyway, but resetting is cheap and keeps
        // the AppState transitions tidy).
        runner.resetConversation()
        selectedImage = image
        lastReport = nil
        lastRawText = ""
        analyzeProgress = 0
        transition(to: .analyzing)

        let prompt = ChatMessage(role: .user, content: Self.fashionAutoPrompt)

        generationTask?.cancel()
        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                let stream = try await self.runner.generate(messages: [prompt], image: cg)
                var buffer = ""
                var tokenCount = 0
                for await chunk in stream {
                    if Task.isCancelled { return }
                    buffer += chunk
                    tokenCount += 1
                    let raw = Double(tokenCount) / Double(self.maxTokens)
                    self.analyzeProgress = min(raw, self.progressCap)
                }
                self.analyzeProgress = 1.0
                self.lastRawText = buffer
                self.lastReport = FashionReport.parse(from: buffer)
                if self.lastReport == nil {
                    self.transition(to: .error("分析に失敗しました\nもう一度お試しください"))
                } else {
                    // Tiny pause so the bar visibly hits 100% before the
                    // result card animates in — without it the analyzing
                    // screen looks like it skipped the final beat.
                    try? await Task.sleep(nanoseconds: 250_000_000)
                    self.transition(to: .results)
                }
            } catch {
                self.transition(to: .error("分析中にエラーが発生しました\n\(error.localizedDescription)"))
            }
        }
    }

    /// Cancel in-flight inference and return to Main. Wired to the
    /// "もう一枚撮る" button on the result screen.
    func resetToMain() {
        generationTask?.cancel()
        generationTask = nil
        analyzeProgress = 0
        lastReport = nil
        lastRawText = ""
        selectedImage = nil
        transition(to: .ready)
    }

    private func transition(to next: Phase) {
        phase = next
        phaseTag &+= 1
    }

    /// Trained user prompt for the Fashion LoRA. Must mirror
    /// `vision/scripts/prepare_train.py:PROMPT` byte-for-byte so the
    /// inference distribution matches the training distribution and the
    /// structured-JSON output path triggers reliably.
    static let fashionAutoPrompt =
        "画像のコーディネートを MB ドレス/カジュアル理論で採点し JSON で出力してください。"
        + "items: [{category, description, "
        + "scores:{color, silhouette, material, design, item_type}, item_dress_score}], "
        + "overall_dress_ratio, "
        + "coordinate_silhouette:{type, style_score, rationale}, "
        + "target_ratio, verdict, advice"
}
