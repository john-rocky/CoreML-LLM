import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            FunctionGemmaTab()
                .tabItem { Label("Function", systemImage: "function") }
            EmbeddingGemmaTab()
                .tabItem { Label("Embed", systemImage: "vector") }
        }
    }
}

#Preview { ContentView() }
