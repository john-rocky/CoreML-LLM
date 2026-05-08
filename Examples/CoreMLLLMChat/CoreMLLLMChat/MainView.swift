import SwiftUI
import PhotosUI
import CoreMLLLM

/// Idle / pre-pick screen: hero CTA that opens the photo picker.
/// Once a photo is chosen, the picker closure transitions AppState
/// straight to `.analyzing` — there is no intermediate confirmation.
struct MainView: View {
    @EnvironmentObject var state: AppState
    @State private var pickedItem: PhotosPickerItem?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.top, 24)
                .padding(.bottom, 8)

            Spacer()

            PhotosPicker(selection: $pickedItem, matching: .images) {
                photoCTA
            }
            .padding(.horizontal, 32)

            Spacer()

            footer
                .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .onChange(of: pickedItem) { _, newItem in
            guard let newItem else { return }
            Task { await loadAndAnalyze(item: newItem) }
        }
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("MB Fashion")
                .font(.system(size: 28, weight: .semibold))
            Text("写真を選んで採点")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var photoCTA: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(Color.orange.opacity(0.12))
                    .frame(width: 180, height: 180)
                Image(systemName: "camera.fill")
                    .font(.system(size: 64, weight: .regular))
                    .foregroundStyle(Color.orange)
            }

            Text("コーデを撮る")
                .font(.headline)
                .foregroundStyle(.primary)
            Text("カメラロールから1枚選択")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var footer: some View {
        Text("MB理論 街着の正解は 7:3 (0.70)")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    /// Pull bytes off the PhotosPicker item, decode to UIImage, then
    /// hand to AppState. We intentionally do not show a transitional
    /// "loading photo" UI — typical decode is < 100 ms.
    private func loadAndAnalyze(item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                state.phase = .error("写真の読み込みに失敗しました")
                state.phaseTag &+= 1
                return
            }
            state.analyze(image: image)
            // Reset the picker selection so the user can re-pick the same
            // photo from the result screen if they want to re-analyze.
            await MainActor.run { pickedItem = nil }
        } catch {
            state.phase = .error("写真の読み込みに失敗しました\n\(error.localizedDescription)")
            state.phaseTag &+= 1
        }
    }
}
