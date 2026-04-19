import XCTest
@testable import CoreMLLLM

/// Pure-Swift tests for the new LiteRT-LM-parity layer (sampling, stop
/// sequences, JSON detector, tool-call parser, chat template). These do
/// NOT load a CoreML model — they verify the surface logic in isolation
/// so the smoke test can focus on end-to-end integration.
final class GenerationAPITests: XCTestCase {

    // MARK: - Sampler

    func testGreedyIsPassthrough() {
        var sampler = Sampler(options: GenerationOptions(temperature: 0, topK: 1))
        let pick = sampler.sample(from: [
            .init(tokenId: 42, logit: 5.0),
            .init(tokenId: 7, logit: 3.0),
        ])
        XCTAssertEqual(pick, 42)
    }

    func testTemperatureSamplingIsDeterministicWithSeed() {
        let opts = GenerationOptions(temperature: 1.0, topK: 4, seed: 1234)
        var a = Sampler(options: opts)
        var b = Sampler(options: opts)
        let cands: [Sampler.Candidate] = [
            .init(tokenId: 1, logit: 2.0),
            .init(tokenId: 2, logit: 1.0),
            .init(tokenId: 3, logit: 0.5),
            .init(tokenId: 4, logit: 0.0),
        ]
        var seqA: [Int32] = []
        var seqB: [Int32] = []
        for _ in 0..<8 {
            seqA.append(a.sample(from: cands))
            seqB.append(b.sample(from: cands))
        }
        XCTAssertEqual(seqA, seqB, "identical seed must yield identical streams")
    }

    func testTopKTruncation() {
        // Only one candidate after top-K=1 should be chosen regardless of
        // temperature.
        var sampler = Sampler(options: GenerationOptions(temperature: 1.0, topK: 1, seed: 7))
        let pick = sampler.sample(from: [
            .init(tokenId: 99, logit: 3.0),
            .init(tokenId: 1, logit: 2.9),
            .init(tokenId: 2, logit: 2.8),
        ])
        XCTAssertEqual(pick, 99)
    }

    func testRepetitionPenaltyDepressesRecentTokens() {
        var sampler = Sampler(options: GenerationOptions(
            temperature: 0.01, topK: 2, repetitionPenalty: 5.0, seed: 42))
        let cands: [Sampler.Candidate] = [
            .init(tokenId: 10, logit: 2.0),
            .init(tokenId: 20, logit: 1.9),
        ]
        // First pick will almost certainly be 10 (highest logit).
        let first = sampler.sample(from: cands)
        XCTAssertEqual(first, 10)
        // Re-sample — the penalty should shift 10's logit below 20's,
        // flipping the pick.
        let second = sampler.sample(from: cands)
        XCTAssertEqual(second, 20)
    }

    // MARK: - StopSequenceMatcher

    func testStopSequenceFindsEarliest() {
        let m = StopSequenceMatcher(["</end>", "STOP"])
        let (safe, hit) = m.findStop(in: "hello STOP world</end>")
        XCTAssertEqual(safe, "hello ")
        XCTAssertEqual(hit, "STOP")
    }

    func testStopSequenceNoMatch() {
        let m = StopSequenceMatcher(["</end>"])
        let (safe, hit) = m.findStop(in: "hello world")
        XCTAssertEqual(safe, "hello world")
        XCTAssertNil(hit)
    }

    func testTailMightMatchTrueAndFalse() {
        let m = StopSequenceMatcher(["</tool_call>"])
        XCTAssertTrue(m.tailMightMatch("abc</to"))
        XCTAssertTrue(m.tailMightMatch("abc</tool_call"))
        XCTAssertFalse(m.tailMightMatch("abcxyz"))
    }

    // MARK: - JSONCompletionDetector

    func testJSONCompletionSimpleObject() {
        var d = JSONCompletionDetector()
        XCTAssertFalse(d.feed("not json yet"))
        XCTAssertFalse(d.feed(" {\"k\":"))
        XCTAssertFalse(d.feed(" 1"))
        XCTAssertTrue(d.feed("}trailing"))
    }

    func testJSONCompletionNestedAndStrings() {
        var d = JSONCompletionDetector()
        let chunks = ["{", "\"a\": [", "1, 2", "], ", "\"b\": \"a}b\"", "}"]
        var done = false
        for c in chunks where !done { done = d.feed(c) }
        XCTAssertTrue(done)
    }

    func testJSONCompletionEscapedQuote() {
        var d = JSONCompletionDetector()
        // Escaped quotes must not terminate the string prematurely.
        XCTAssertFalse(d.feed("{\"s\": \"he said \\\"hi\\\"\""))
        XCTAssertTrue(d.feed("}"))
    }

    // MARK: - ToolCallDetector

    func testToolCallBasic() {
        let body = "preamble <tool_call>{\"name\": \"weather\", " +
            "\"arguments\": {\"city\": \"Tokyo\"}}</tool_call> suffix"
        let parsed = ToolCallDetector.tryParse(body)
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?.invocation.name, "weather")
        XCTAssertEqual(parsed?.invocation.argumentsDict["city"] as? String, "Tokyo")
        XCTAssertEqual(parsed?.prefix, "preamble ")
    }

    func testToolCallIncompleteReturnsNil() {
        XCTAssertNil(ToolCallDetector.tryParse("<tool_call>{\"name\": \"x\"}"))
        XCTAssertNil(ToolCallDetector.tryParse("no tool call here"))
    }

    func testToolCallInvalidJSON() {
        // open+close but body is garbage — must not crash; returns nil.
        XCTAssertNil(ToolCallDetector.tryParse("<tool_call>not json</tool_call>"))
    }

    // MARK: - ChatTemplate

    func testGemma4Template() {
        let tpl = ChatTemplate(family: .gemma4)
        let out = tpl.render(messages: [
            .init(role: .system, content: "Be brief."),
            .init(role: .user, content: "Hi")
        ])
        XCTAssertTrue(out.hasPrefix("<bos>"))
        XCTAssertTrue(out.contains("Be brief.\n\nHi"),
                       "system prompt should be collapsed into first user turn")
        XCTAssertTrue(out.hasSuffix("<|turn>model\n"))
    }

    func testGemma4TemplateWithImage() {
        let tpl = ChatTemplate(family: .gemma4)
        let out = tpl.render(
            messages: [.init(role: .user, content: "what is this")],
            imageBlock: "<|image><|image|><image|>")
        XCTAssertTrue(out.contains("<|image><|image|><image|>\nwhat is this"))
    }

    func testQwenTemplate() {
        let tpl = ChatTemplate(family: .qwen)
        let out = tpl.render(messages: [
            .init(role: .system, content: "sys"),
            .init(role: .user, content: "u"),
            .init(role: .assistant, content: "a"),
        ])
        XCTAssertTrue(out.contains("<|im_start|>system\nsys"))
        XCTAssertTrue(out.contains("<|im_start|>assistant\na"))
        XCTAssertTrue(out.hasSuffix("<|im_start|>assistant\n"))
    }

    func testChatTemplateInjectTools() {
        let tpl = ChatTemplate(family: .gemma4)
        let tools = [ToolSpec(name: "echo", description: "repeats",
                               parameters: ["type": "object", "properties": [:]],
                               handler: { _ in "ok" })]
        let injected = tpl.injectTools(tools, into: [
            .init(role: .user, content: "hello")
        ])
        XCTAssertEqual(injected.first?.role, .system)
        XCTAssertTrue(injected.first?.content.contains("<tool_call>") ?? false)
    }

    // MARK: - GenerationOptions.needsSampling

    func testOptionsNeedsSamplingFlag() {
        XCTAssertFalse(GenerationOptions().needsSampling)
        XCTAssertFalse(GenerationOptions(temperature: 0, topK: 1).needsSampling)
        XCTAssertTrue(GenerationOptions(temperature: 0.7).needsSampling)
        XCTAssertTrue(GenerationOptions(topK: 50).needsSampling)
        XCTAssertTrue(GenerationOptions(topP: 0.9).needsSampling)
        XCTAssertTrue(GenerationOptions(repetitionPenalty: 1.1).needsSampling)
    }
}
