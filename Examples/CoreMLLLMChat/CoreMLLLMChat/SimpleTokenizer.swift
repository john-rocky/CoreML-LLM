import Foundation

/// Minimal tokenizer that loads tokenizer.json from HuggingFace format.
/// For production use, prefer swift-transformers' AutoTokenizer.
final class SimpleTokenizer {
    private var vocab: [String: Int] = [:]
    private var reverseVocab: [Int: String] = [:]
    private var merges: [(String, String)] = []

    init?(modelPath: URL) {
        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: tokenizerURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        // Load vocabulary
        if let model = json["model"] as? [String: Any],
           let vocabDict = model["vocab"] as? [String: Int] {
            self.vocab = vocabDict
            for (token, id) in vocabDict {
                reverseVocab[id] = token
            }
        }

        // Load merges
        if let model = json["model"] as? [String: Any],
           let mergeList = model["merges"] as? [String] {
            self.merges = mergeList.compactMap { line in
                let parts = line.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }
        }
    }

    func encode(_ text: String) -> [Int] {
        // Simple character-level encoding with BPE merges
        // This is a minimal implementation — production should use swift-transformers
        var tokens: [Int] = []

        // Try to find whole words/subwords in vocab
        var remaining = text
        while !remaining.isEmpty {
            var found = false
            // Try longest match first
            for length in stride(from: min(remaining.count, 50), through: 1, by: -1) {
                let prefix = String(remaining.prefix(length))
                if let id = vocab[prefix] {
                    tokens.append(id)
                    remaining = String(remaining.dropFirst(length))
                    found = true
                    break
                }
            }
            if !found {
                // Skip unknown character
                remaining = String(remaining.dropFirst())
            }
        }

        return tokens
    }

    func decode(_ tokenIDs: [Int]) -> String {
        var result = ""
        for id in tokenIDs {
            if let token = reverseVocab[id] {
                // Handle BPE special characters
                let cleaned = token
                    .replacingOccurrences(of: "\u{0120}", with: " ")  // GPT-2 space
                    .replacingOccurrences(of: "\u{010A}", with: "\n") // GPT-2 newline
                    .replacingOccurrences(of: "\u{2581}", with: " ")  // SentencePiece space
                result += cleaned
            }
        }
        return result
    }
}
