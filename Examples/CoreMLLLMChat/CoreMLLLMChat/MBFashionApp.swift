import SwiftUI
import CoreMLLLM

@main
struct MBFashionApp: App {
    @UIApplicationDelegateAdaptor(MBFashionAppDelegate.self) var delegate
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(state)
                .preferredColorScheme(.light)
        }
    }
}

final class MBFashionAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        // Re-attach the background download session so model downloads
        // resume correctly when the app is woken in the background.
        ModelDownloader.shared.backgroundCompletionHandler = completionHandler
    }
}

/// Top-level dispatcher: pick the screen for the current `AppState.phase`.
/// Each phase owns its own view; transitions happen via state mutation.
struct RootView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ZStack {
            switch state.phase {
            case .welcome:
                WelcomeView()
                    .transition(.opacity)
            case .modelLoading:
                LoadingView(message: state.loadingMessage)
                    .transition(.opacity)
            case .ready:
                MainView()
                    .transition(.opacity)
            case .analyzing:
                AnalyzingView()
                    .transition(.opacity)
            case .results:
                ResultView()
                    .transition(.opacity)
            case .error(let message):
                ErrorView(message: message) {
                    state.resetToMain()
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: state.phaseTag)
        .task {
            await state.bootstrap()
        }
    }
}
