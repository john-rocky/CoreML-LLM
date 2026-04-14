//
//  Prompts.swift — bench corpus.
//
//  Four categories × a few prompts each. Categories chosen to expose where
//  each drafter shines or falls flat:
//  - code: expect Prompt Lookup / Suffix to hit on identifier repeats
//  - summary: expect Prompt Lookup to hit on input-quoting
//  - qa: mixed, tests when answer quotes prompt
//  - chat: novel text, should zero-out the string-matching drafters (sanity)
//
//  Keep prompts short so `maxTokens = 64–128` completes in reasonable time.
//

import Foundation

struct BenchPrompt: Codable {
    let id: String
    let category: String
    let text: String
}

let benchPrompts: [BenchPrompt] = [
    // -------- code --------
    .init(id: "code-complete-sum", category: "code",
          text: "Complete this Python function. Respond with just the function body, no explanation.\n```python\ndef sum_list(lst):\n    # Return the sum of all numbers in lst.\n    "),
    .init(id: "code-explain-fib", category: "code",
          text: "Explain briefly what this Python code does. Quote the code in your response.\n```python\ndef fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n```"),
    .init(id: "code-refactor-loop", category: "code",
          text: "Rewrite this loop using a list comprehension. Keep the variable names identical.\n```python\nresult = []\nfor x in numbers:\n    if x > 0:\n        result.append(x * 2)\n```"),

    // -------- summary --------
    .init(id: "sum-para-ane", category: "summary",
          text: "Summarize the following paragraph in one sentence that quotes the key phrase verbatim:\n\nThe Apple Neural Engine is a specialized hardware accelerator integrated into Apple silicon designed to speed up machine learning workloads. It first appeared in the A11 Bionic chip in 2017 and has been expanded with each generation."),
    .init(id: "sum-para-compiler", category: "summary",
          text: "Summarize the following paragraph. Reuse the phrase \"ahead-of-time compilation\" in your summary.\n\nSwift's ahead-of-time compilation produces native machine code at build time, trading compile speed for runtime efficiency. This differs from Python's just-in-time or interpreted execution and contributes to iOS apps starting more quickly than equivalent scripted alternatives."),

    // -------- qa --------
    .init(id: "qa-where-is-swift", category: "qa",
          text: "Based on this context, answer the question.\n\nContext: Swift was developed by Apple and first released in 2014 as a successor to Objective-C for iOS and macOS development.\n\nQuestion: In what year was Swift first released?"),
    .init(id: "qa-what-is-ane", category: "qa",
          text: "Based on this context, answer the question in one sentence that includes \"Neural Engine\".\n\nContext: The Apple Neural Engine runs machine learning workloads efficiently on iPhone and Mac. It's accessed via the Core ML framework.\n\nQuestion: What framework is used to access the Neural Engine?"),

    // -------- chat --------
    .init(id: "chat-define-transformer", category: "chat",
          text: "What is a transformer model in machine learning? Answer in two sentences."),
    .init(id: "chat-explain-rope", category: "chat",
          text: "Briefly explain rotary positional embeddings without quoting any specific paper."),
    .init(id: "chat-greeting", category: "chat",
          text: "Hi, I'm testing a speech model. Say hello and ask me what I want to do today."),
]
