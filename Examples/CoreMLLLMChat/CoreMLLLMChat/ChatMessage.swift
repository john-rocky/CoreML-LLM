import Foundation
import UIKit

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var imageData: Data?   // JPEG thumbnail for display
    /// Per-frame thumbnails for video messages: `(jpegData, timestampSeconds)`.
    /// Surfaced in the chat bubble so the user can see the exact frames the
    /// model received (matches the 1 fps sampling the inference path uses).
    var videoFrames: [(Data, Double)]?
    let timestamp = Date()

    enum Role {
        case user
        case assistant
        case system
    }
}
