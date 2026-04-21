// union-bitexact is a macOS-only verifier. iOS build needs a `main` stub.
#if !os(macOS)
@main
struct UnionBitExactIOSStub {
    static func main() {}
}
#endif
