import Foundation
#if canImport(UIKit)
import SwiftUI
#endif

// MARK: - Fashion report rendering
//
// Assistant output from the Qwen3-VL 2B / Gemma 4 E2B Fashion LoRA is a
// JSON document following MB dress/casual theory. When we can parse it,
// render as a structured card instead of a raw text bubble. All fields
// are optional because on-device int4 decoding occasionally drifts from
// the trained schema — render whatever we got and fall back to raw text
// if the parse returns nothing useful.
//
// Both the parser and the SwiftUI components live here so the MBFashion
// app and the dev chat target can share a single source of truth.

public struct FashionReport: Sendable {
    public struct Scores: Sendable {
        public let color: Double?
        public let silhouette: Double?
        public let material: Double?
        public let design: Double?
        /// v3 5th axis: garment-type baseline (independent of color/cut/etc).
        public let item_type: Double?

        public init(color: Double?, silhouette: Double?, material: Double?,
                    design: Double?, item_type: Double?) {
            self.color = color
            self.silhouette = silhouette
            self.material = material
            self.design = design
            self.item_type = item_type
        }
    }

    public struct Item: Sendable {
        public let category: String?
        public let description: String?
        public let scores: Scores?
        public let item_dress_score: Double?

        public init(category: String?, description: String?,
                    scores: Scores?, item_dress_score: Double?) {
            self.category = category
            self.description = description
            self.scores = scores
            self.item_dress_score = item_dress_score
        }
    }

    /// v3 outfit-level style axis: I/A/Y silhouette classification.
    public struct CoordinateSilhouette: Sendable {
        public let type: String?      // "I" | "A" | "Y" | "off"
        public let style_score: Double?
        public let rationale: String?

        public init(type: String?, style_score: Double?, rationale: String?) {
            self.type = type
            self.style_score = style_score
            self.rationale = rationale
        }
    }

    public let items: [Item]
    public let overall_dress_ratio: Double?
    /// Legacy v2 field; kept for parser tolerance against old model output.
    /// Not rendered in v3 card UI.
    public let tpo_assumption: String?
    public let target_ratio: Double?
    public let coordinate_silhouette: CoordinateSilhouette?
    public let verdict: String?
    public let advice: String?

    public init(items: [Item],
                overall_dress_ratio: Double?,
                tpo_assumption: String?,
                target_ratio: Double?,
                coordinate_silhouette: CoordinateSilhouette?,
                verdict: String?,
                advice: String?) {
        self.items = items
        self.overall_dress_ratio = overall_dress_ratio
        self.tpo_assumption = tpo_assumption
        self.target_ratio = target_ratio
        self.coordinate_silhouette = coordinate_silhouette
        self.verdict = verdict
        self.advice = advice
    }

    /// Non-empty if any of the top-level schema fields survived parsing.
    public var isRenderable: Bool {
        !items.isEmpty
            || overall_dress_ratio != nil
            || coordinate_silhouette != nil
            || verdict != nil
            || advice != nil
    }

    // ---- Parsing ----
    //
    // The on-device int4 decoder drifts from the trained schema in many
    // ways: aliased keys (`attributes` for `scores`, `dress_score` for
    // `item_dress_score`, `overall_ratio` for `overall_dress_ratio`),
    // positional arrays where dicts were expected (`scores: [c,s,m,d]`),
    // and occasionally several malformed JSON blobs concatenated. Use
    // a tolerant parse that walks JSONSerialization output via dictionary
    // lookups + alias tables, then merges results across every balanced
    // top-level `{...}` span found in the text.

    private static let topRatioKeys = [
        "overall_dress_ratio", "overall_ratio", "ratio",
    ]
    private static let topTPOKeys = ["tpo_assumption", "tp_assumption", "durationType", "target_tpo", "tpo"]
    private static let topTargetKeys = ["target_ratio"]
    private static let topVerdictKeys = ["verdict"]
    private static let topAdviceKeys = ["advice", "comment"]
    private static let topCoordSilhouetteKeys = [
        "coordinate_silhouette", "outfit_silhouette", "style_silhouette",
    ]
    private static let itemDescriptionKeys = ["description", "item", "name"]
    private static let itemDressScoreKeys = ["item_dress_score", "dress_score", "score_break"]
    private static let scoresContainerKeys = ["scores", "attributes", "score"]
    /// v3 positional fallback: 5 axes (item_type added 5th).
    private static let scoresPositionalAxes = ["color", "silhouette", "material", "design", "item_type"]
    private static let itemTypeKeys = ["item_type", "type", "garment_type", "category_score"]
    private static let silhouetteTypeKeys = ["type", "silhouette_type", "shape"]
    private static let silhouetteStyleScoreKeys = ["style_score", "score", "completion"]
    private static let silhouetteRationaleKeys = ["rationale", "reason", "note"]

    public static func parse(from text: String) -> FashionReport? {
        let cleaned = stripNoise(text)
        guard !cleaned.isEmpty else { return nil }

        var items: [Item] = []
        var ratio: Double?
        var tpo: String?
        var target: Double?
        var verdict: String?
        var advice: String?
        var silhouette: CoordinateSilhouette?

        // Walk every balanced {...} span. Multi-blob outputs are merged so
        // a fragmented response (items in one blob, overall_ratio in another)
        // still produces a coherent card.
        for span in balancedJSONSpans(in: cleaned) {
            guard let data = span.data(using: .utf8),
                  let raw = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
                  let dict = raw as? [String: Any]
            else { continue }

            if let arr = dict["items"] as? [Any] {
                for case let entry as [String: Any] in arr {
                    items.append(buildItem(from: entry))
                }
            }
            if ratio == nil, let v = lookupDouble(topRatioKeys, in: dict) { ratio = v }
            if tpo == nil, let v = lookupString(topTPOKeys, in: dict) { tpo = v }
            if target == nil, let v = lookupDouble(topTargetKeys, in: dict) { target = v }
            if verdict == nil, let v = lookupString(topVerdictKeys, in: dict) { verdict = v }
            if advice == nil, let v = lookupString(topAdviceKeys, in: dict) { advice = v }
            if silhouette == nil {
                let sValue = topCoordSilhouetteKeys.compactMap { dict[$0] }.first
                if let sDict = sValue as? [String: Any] {
                    silhouette = CoordinateSilhouette(
                        type: lookupString(silhouetteTypeKeys, in: sDict),
                        style_score: lookupDouble(silhouetteStyleScoreKeys, in: sDict),
                        rationale: lookupString(silhouetteRationaleKeys, in: sDict))
                } else if let sStr = sValue as? String {
                    // Tolerate flat string form: "coordinate_silhouette":"I"
                    silhouette = CoordinateSilhouette(
                        type: sStr, style_score: nil, rationale: nil)
                }
            }
        }

        let report = FashionReport(
            items: items,
            overall_dress_ratio: ratio,
            tpo_assumption: tpo,
            target_ratio: target,
            coordinate_silhouette: silhouette,
            verdict: verdict,
            advice: advice)
        return report.isRenderable ? report : nil
    }

    private static func stripNoise(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<channel|>", with: "")
            .replacingOccurrences(of: "<|channel|>", with: "")
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Return every top-level balanced `{...}` substring (in source order),
    /// skipping `{`/`}` characters inside JSON string literals. When the
    /// scan ends with the outermost object still open (model's response
    /// got truncated mid-stream by max_new_tokens), append a synthesised
    /// "best-effort close" so the partial JSON can still be salvaged
    /// by JSONSerialization. The salvage truncates back to the last
    /// completed top-level field and closes any open structures with
    /// matching `]` / `"` / `}` tokens.
    private static func balancedJSONSpans(in text: String) -> [String] {
        var spans: [String] = []
        var depth = 0
        var inString = false
        var escape = false
        var startIdx: String.Index?
        var bracketStack: [Character] = []   // tracks { and [ in order, for salvage
        for idx in text.indices {
            let ch = text[idx]
            if escape {
                escape = false
                continue
            }
            if ch == "\\" {
                escape = true
                continue
            }
            if ch == "\"" {
                inString.toggle()
                continue
            }
            if inString { continue }
            if ch == "{" {
                if depth == 0 { startIdx = idx; bracketStack.removeAll() }
                depth += 1
                bracketStack.append("{")
            } else if ch == "[" {
                bracketStack.append("[")
            } else if ch == "]" {
                if bracketStack.last == "[" { bracketStack.removeLast() }
            } else if ch == "}" {
                depth -= 1
                if bracketStack.last == "{" { bracketStack.removeLast() }
                if depth == 0, let s = startIdx {
                    spans.append(String(text[s...idx]))
                    startIdx = nil
                    bracketStack.removeAll()
                } else if depth < 0 {
                    depth = 0
                    startIdx = nil
                    bracketStack.removeAll()
                }
            }
        }

        // Salvage path: outermost object never closed. Truncate back to
        // the last "completed top-level field boundary" (last comma at
        // depth 1 outside strings) and append the brackets needed to
        // close every still-open structure. JSONSerialization is
        // permissive enough to parse the result.
        if let s = startIdx, !bracketStack.isEmpty {
            let raw = String(text[s..<text.endIndex])
            if let salvaged = trySalvageOpenJSON(raw) {
                spans.append(salvaged)
            }
        }
        return spans
    }

    private static func trySalvageOpenJSON(_ raw: String) -> String? {
        var depth = 0
        var inString = false
        var escape = false
        var stack: [Character] = []
        var lastSafeBoundary: String.Index? = nil
        for idx in raw.indices {
            let ch = raw[idx]
            if escape { escape = false; continue }
            if ch == "\\" { escape = true; continue }
            if ch == "\"" { inString.toggle(); continue }
            if inString { continue }
            switch ch {
            case "{":
                depth += 1
                stack.append("{")
            case "[":
                stack.append("[")
            case "]":
                if stack.last == "[" { stack.removeLast() }
            case "}":
                depth -= 1
                if stack.last == "{" { stack.removeLast() }
            case ",":
                if depth == 1 && stack.last == "{" {
                    lastSafeBoundary = idx
                }
            default:
                break
            }
        }
        guard let boundary = lastSafeBoundary else {
            return nil
        }
        var truncated = String(raw[..<boundary])
        var depth2 = 0
        var stack2: [Character] = []
        var inString2 = false
        var escape2 = false
        for ch in truncated {
            if escape2 { escape2 = false; continue }
            if ch == "\\" { escape2 = true; continue }
            if ch == "\"" { inString2.toggle(); continue }
            if inString2 { continue }
            switch ch {
            case "{": depth2 += 1; stack2.append("{")
            case "[": stack2.append("[")
            case "]": if stack2.last == "[" { stack2.removeLast() }
            case "}": depth2 -= 1; if stack2.last == "{" { stack2.removeLast() }
            default: break
            }
        }
        if inString2 {
            truncated.append("\"")
        }
        for open in stack2.reversed() {
            truncated.append(open == "{" ? "}" : "]")
        }
        return truncated
    }

    private static func buildItem(from dict: [String: Any]) -> Item {
        let category = dict["category"] as? String
        let description = lookupString(itemDescriptionKeys, in: dict)
        let scoresValue = scoresContainerKeys.compactMap { dict[$0] }.first
        let scores = buildScores(from: scoresValue)
        var dressScore = lookupDouble(itemDressScoreKeys, in: dict)
        // int8 drift fallback: model occasionally nests item_dress_score
        // inside the scores object instead of at item level. Pull it back.
        if dressScore == nil, let nested = scoresValue as? [String: Any] {
            dressScore = lookupDouble(itemDressScoreKeys, in: nested)
        }
        return Item(category: category, description: description,
                    scores: scores, item_dress_score: dressScore)
    }

    private static func buildScores(from value: Any?) -> Scores? {
        if let dict = value as? [String: Any] {
            return Scores(
                color: dict["color"] as? Double ?? (dict["color"] as? NSNumber)?.doubleValue,
                silhouette: dict["silhouette"] as? Double ?? (dict["silhouette"] as? NSNumber)?.doubleValue,
                material: dict["material"] as? Double ?? (dict["material"] as? NSNumber)?.doubleValue,
                design: dict["design"] as? Double ?? (dict["design"] as? NSNumber)?.doubleValue,
                item_type: lookupDouble(itemTypeKeys, in: dict))
        }
        if let arr = value as? [Any] {
            let nums = arr.prefix(scoresPositionalAxes.count).map { v -> Double? in
                (v as? Double) ?? (v as? NSNumber)?.doubleValue
            }
            guard nums.contains(where: { $0 != nil }) else { return nil }
            let pad = Array(repeating: Double?.none, count: max(0, 5 - nums.count))
            let padded = Array(nums) + pad
            return Scores(
                color: padded[0],
                silhouette: padded[1],
                material: padded[2],
                design: padded[3],
                item_type: padded.count > 4 ? padded[4] : nil)
        }
        return nil
    }

    private static func lookupDouble(_ keys: [String], in dict: [String: Any]) -> Double? {
        for k in keys {
            if let v = dict[k] as? Double { return v }
            if let v = dict[k] as? NSNumber { return v.doubleValue }
            if let s = dict[k] as? String, let v = Double(s) { return v }
        }
        return nil
    }

    private static func lookupString(_ keys: [String], in dict: [String: Any]) -> String? {
        for k in keys {
            if let v = dict[k] as? String, !v.isEmpty { return v }
        }
        return nil
    }
}

// MARK: - SwiftUI components
//
// Reused by both the dev chat target and the MBFashion TestFlight app.
// Sized for compact in-bubble use; MBFashion wraps these in a larger
// container with extra padding for full-screen presentation. iOS-only
// because `Color(.systemGray6)` and friends resolve to UIColor literals
// that have no NSColor counterpart.

#if canImport(UIKit)

public struct FashionReportCard: View {
    public let report: FashionReport
    /// Raw model output kept so the user can copy/inspect the JSON when needed.
    public let raw: String
    /// When false, the "Show JSON" button is hidden — useful for embedded
    /// uses where the host view provides its own debug toggle.
    public let showsRawToggle: Bool
    @State private var showingRaw = false

    public init(report: FashionReport, raw: String, showsRawToggle: Bool = true) {
        self.report = report
        self.raw = raw
        self.showsRawToggle = showsRawToggle
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let ratio = report.overall_dress_ratio {
                DressRatioGauge(ratio: ratio, target: report.target_ratio)
            }

            if let silhouette = report.coordinate_silhouette {
                HStack(spacing: 8) {
                    SilhouetteChip(silhouette: silhouette)
                    if let rationale = silhouette.rationale, !rationale.isEmpty {
                        Text(rationale)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
            }

            if !report.items.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Items")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(Array(report.items.enumerated()), id: \.offset) { _, item in
                        FashionItemRow(item: item)
                    }
                }
            }

            if let verdict = report.verdict, !verdict.isEmpty {
                VerdictBlock(text: verdict)
            }

            if let advice = report.advice, !advice.isEmpty {
                AdviceCard(text: advice)
            }

            if showsRawToggle {
                Button(showingRaw ? "Hide JSON" : "Show JSON") {
                    showingRaw.toggle()
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if showingRaw {
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(raw)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(8)
                            .background(Color(.systemGray6))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
        .padding(14)
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

public struct DressRatioGauge: View {
    public let ratio: Double
    public let target: Double?
    /// Vertical padding inside the gauge bar. MBFashion's hero card uses
    /// a taller bar than the in-bubble chat card.
    public let barHeight: CGFloat

    public init(ratio: Double, target: Double?, barHeight: CGFloat = 10) {
        self.ratio = ratio
        self.target = target
        self.barHeight = barHeight
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Dress ratio")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.2f", clamped))
                    .font(.title2.monospacedDigit().weight(.semibold))
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.systemGray5))
                    Capsule()
                        .fill(LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .leading, endPoint: .trailing))
                        .frame(width: proxy.size.width * clamped)
                    if let target {
                        let x = proxy.size.width * min(max(target, 0), 1)
                        Rectangle()
                            .fill(Color.orange)
                            .frame(width: 2)
                            .offset(x: x - 1)
                    }
                }
            }
            .frame(height: barHeight)
            HStack {
                Text("Casual").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text("Dress").font(.caption2).foregroundStyle(.secondary)
            }
            // MB rule of thumb caption. The model emits TPO-specific targets
            // (smart_casual=0.70, weekend=0.60, business=0.95 …) but MB's
            // public framing is the 7:3 town-wear answer; reinforcing it
            // here keeps the demo tied to his vocabulary.
            HStack(spacing: 6) {
                Text("MB理論")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.orange)
                Text("街着の正解は 7:3 (0.70)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 2)
        }
    }

    private var clamped: Double { min(max(ratio, 0), 1) }
}

/// v3 outfit-level style chip. Shows "Iライン (0.85)" for I/A/Y types or
/// "シルエット off" for off. Orange tint mirrors the MB理論 7:3 caption.
public struct SilhouetteChip: View {
    public let silhouette: FashionReport.CoordinateSilhouette

    public init(silhouette: FashionReport.CoordinateSilhouette) {
        self.silhouette = silhouette
    }

    public var body: some View {
        let label = displayLabel
        let isOff = (silhouette.type?.lowercased() == "off")
        Text(label)
            .font(.caption.weight(.semibold))
            .foregroundStyle(isOff ? Color.secondary : Color.orange)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(
                (isOff ? Color(.systemGray5) : Color.orange.opacity(0.15))
            )
            .clipShape(Capsule())
    }

    private var displayLabel: String {
        let t = (silhouette.type ?? "").uppercased()
        let scoreSuffix = silhouette.style_score
            .map { String(format: " (%.2f)", min(max($0, 0), 1)) } ?? ""
        switch t {
        case "I": return "Iライン" + scoreSuffix
        case "A": return "Aライン" + scoreSuffix
        case "Y": return "Yライン" + scoreSuffix
        case "OFF", "":
            return "シルエット off" + scoreSuffix
        default:
            return t + "ライン" + scoreSuffix
        }
    }
}

public struct FashionItemRow: View {
    public let item: FashionReport.Item

    public init(item: FashionReport.Item) {
        self.item = item
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(item.category ?? "item")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if let score = item.item_dress_score {
                    Text(String(format: "%.2f", min(max(score, 0), 1)))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if let desc = item.description, !desc.isEmpty {
                Text(desc)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let scores = item.scores {
                VStack(spacing: 3) {
                    ScoreBar(label: "color", value: scores.color)
                    ScoreBar(label: "silhouette", value: scores.silhouette)
                    ScoreBar(label: "material", value: scores.material)
                    ScoreBar(label: "design", value: scores.design)
                    // v3 5th axis. v2 model output omits item_type — hide
                    // the row in that case rather than render an empty
                    // "—" bar that looks broken.
                    if scores.item_type != nil {
                        ScoreBar(label: "item_type", value: scores.item_type)
                    }
                }
            }
        }
        .padding(10)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

public struct ScoreBar: View {
    public let label: String
    public let value: Double?

    public init(label: String, value: Double?) {
        self.label = label
        self.value = value
    }

    public var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 64, alignment: .leading)
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.systemGray5))
                    if let v = value {
                        Capsule()
                            .fill(Color.blue.opacity(0.7))
                            .frame(width: proxy.size.width * min(max(v, 0), 1))
                    }
                }
            }
            .frame(height: 5)
            Text(value.map { String(format: "%.2f", min(max($0, 0), 1)) } ?? "—")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 32, alignment: .trailing)
        }
    }
}

public struct VerdictBlock: View {
    public let text: String

    public init(text: String) {
        self.text = text
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Rectangle()
                .fill(Color.accentColor)
                .frame(width: 3)
            Text(text)
                .font(.subheadline.italic())
                .foregroundStyle(.primary)
        }
    }
}

public struct AdviceCard: View {
    public let text: String

    public init(text: String) {
        self.text = text
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Advice")
                .font(.caption).foregroundStyle(.secondary)
            Text(text)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(Color.accentColor.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

#endif
