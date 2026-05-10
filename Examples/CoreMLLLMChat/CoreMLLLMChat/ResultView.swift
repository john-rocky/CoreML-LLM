import SwiftUI
import PhotosUI
import CoreMLLLM

/// Final screen: shows the parsed FashionReport in a large card layout
/// plus a thumbnail of the analysed photo. The "もう一枚撮る" CTA pops
/// the picker directly without a detour through `MainView`.
struct ResultView: View {
    @EnvironmentObject var state: AppState
    @State private var pickedItem: PhotosPickerItem?
    @State private var showsRawJSON = false
    @State private var gaugeAppeared = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                photoHeader
                if let report = state.lastReport {
                    reportCard(report: report)
                } else {
                    Text("分析データがありません")
                        .foregroundStyle(.secondary)
                }
                debugSection
                anotherCTA
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .onAppear {
            withAnimation(.easeOut(duration: 0.8)) {
                gaugeAppeared = true
            }
        }
        .onDisappear {
            gaugeAppeared = false
        }
        .onChange(of: pickedItem) { _, newItem in
            guard let newItem else { return }
            Task { await loadAndAnalyze(item: newItem) }
        }
    }

    private var photoHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            if let image = state.selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 80, height: 80)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("MB Fashion")
                    .font(.headline)
                Text("採点結果")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.top, 16)
    }

    /// Hero result card — overrides the package's default DressRatioGauge
    /// height so the gauge fills the screen. Animates the bar from 0 to
    /// the actual ratio on appear.
    private func reportCard(report: FashionReport) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            if let ratio = report.overall_dress_ratio {
                DressRatioGauge(
                    ratio: gaugeAppeared ? ratio : 0,
                    target: report.target_ratio,
                    barHeight: 16)
                    .padding(.bottom, 4)
            }

            // SilhouetteChip is hidden when the silhouette is "off". The
            // value comes from `displayCoordinateSilhouette` which uses
            // the v3 model's explicit field when present, or derives
            // I/A/Y/off from per-item silhouette scores when not (v2
            // model). Hidden on "off" because that's typically a
            // low-confidence partial photo or a genuine no-contrast
            // outfit — showing a wrong "off" pill is worse than nothing.
            if let silhouette = report.displayCoordinateSilhouette,
               let type = silhouette.type,
               type.lowercased() != "off"
            {
                HStack(spacing: 8) {
                    SilhouetteChip(silhouette: silhouette)
                    if let rationale = silhouette.rationale, !rationale.isEmpty {
                        Text(rationale)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !report.items.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Items")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ForEach(Array(report.items.enumerated()), id: \.offset) { _, item in
                        FashionItemRow(item: item)
                    }
                }
            }

            if let verdict = report.verdict, !verdict.isEmpty {
                VerdictBlock(text: verdict)
            }

            // v0.1: advice is collapsed by default. Quality is uneven
            // (token-level drift in free text + 2B-model semantic ceiling
            // for nuanced advisory voice). Show on demand for testers
            // willing to read in beta context, hide from primary view.
            if let advice = report.advice, !advice.isEmpty {
                AdviceDisclosure(text: advice)
            }

            v01Footer
        }
        .padding(20)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
    }

    private var v01Footer: some View {
        HStack {
            Spacer()
            Text("MB Fashion v0.1 (early preview)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .padding(.top, 4)
    }

    /// Debug-only collapsible JSON pane. Hidden behind a small button so
    /// it doesn't clutter the consumer flow but is reachable when MB-san
    /// or the user wants to inspect raw model output during testing.
    private var debugSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                showsRawJSON.toggle()
            } label: {
                Label(showsRawJSON ? "JSON を隠す" : "JSON を表示 (debug)",
                      systemImage: showsRawJSON ? "chevron.up" : "chevron.down")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if showsRawJSON {
                ScrollView(.horizontal, showsIndicators: true) {
                    Text(state.lastRawText.isEmpty ? "(empty)" : state.lastRawText)
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.primary)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .background(Color(.systemGray6))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private var anotherCTA: some View {
        PhotosPicker(selection: $pickedItem, matching: .images) {
            HStack(spacing: 6) {
                Image(systemName: "camera.fill")
                Text("もう一枚 撮る")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Color.orange)
            .foregroundStyle(.white)
            .clipShape(Capsule())
        }
    }

    private func loadAndAnalyze(item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                return
            }
            state.analyze(image: image)
            await MainActor.run { pickedItem = nil }
        } catch {
            // Silently swallow — the user can retry.
        }
    }
}

/// v0.1: collapsed-by-default advice disclosure. Tappable header expands
/// to reveal the AdviceCard. Frame the advice as preview/beta so users
/// don't take quirky token-level drift in free text as authoritative.
private struct AdviceDisclosure: View {
    let text: String
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    expanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                    Text("AI コメント (β)")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                }
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)

            if expanded {
                AdviceCard(text: text)
            }
        }
    }
}
