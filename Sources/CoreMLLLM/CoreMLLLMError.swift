import Foundation

/// Errors thrown by CoreMLLLM.
public enum CoreMLLLMError: LocalizedError {
    case missingOutput(String)
    case emptyPrompt
    case modelNotFound(String)
    case configNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .missingOutput(let name):
            return "Missing model output: \(name)"
        case .emptyPrompt:
            return "Prompt cannot be empty"
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .configNotFound(let path):
            return "model_config.json not found at: \(path)"
        }
    }
}
