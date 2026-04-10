import Foundation
import UIKit

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var imageData: Data?   // JPEG thumbnail for display
    let timestamp = Date()

    enum Role {
        case user
        case assistant
        case system
    }
}
