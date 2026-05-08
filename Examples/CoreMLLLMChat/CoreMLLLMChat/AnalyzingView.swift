import SwiftUI

/// Mid-inference screen: keep the user's photo prominent while a thin
/// bar communicates "still working." We deliberately hide token counts
/// and tok/s — MBFashion is supposed to feel like a polished consumer
/// app, not a model debugger.
struct AnalyzingView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 16)

            if let image = state.selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .frame(maxHeight: 460)
                    .clipShape(RoundedRectangle(cornerRadius: 24))
                    .shadow(color: .black.opacity(0.06), radius: 12, y: 4)
                    .padding(.horizontal, 24)
            }

            VStack(spacing: 10) {
                ProgressView(value: state.analyzeProgress)
                    .progressViewStyle(.linear)
                    .tint(.orange)

                Text("あなたのコーデを採点中...")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 32)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}
