// accept-rate-bench is a macOS-only research tool. On iOS we compile an
// empty stub just to satisfy SwiftPM's executableTarget requirement that
// a `main` entry exist.
#if !os(macOS)
@main
struct AcceptRateBenchIOSStub {
    static func main() {}
}
#endif
