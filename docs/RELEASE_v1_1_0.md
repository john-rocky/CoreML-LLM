# v1.1.0 — Qwen3.5 2B on iPhone ANE

## X / Twitter announcement drafts

**Long-form (technical):**

```
Shipped Qwen3.5-2B on iPhone 17 Pro Neural Engine.

· 2.04B hybrid SSM + attention (Gated DeltaNet)
· 4 INT8 transformer chunks, all ≥ 90% ANE placement
· 17 tok/s decode, 2048-token context
· ~200 MB phys_footprint during inference, 0 GB Metal heap
· 1 GB embed weight mmap'd as an fp16 sidecar — only touched rows
  page in, and those clean mmap pages don't count against the app's
  resident memory

First 2B-class hybrid SSM LLM on CoreML that we're aware of.

Model: https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML
Repo: https://github.com/john-rocky/CoreML-LLM
```

**Short / catchy:**

```
Qwen3.5 2B on iPhone Neural Engine @ 17 tok/s with only ~200 MB app
memory and zero Metal heap.

Trick: embed_tokens ships as a raw fp16 file and Swift mmaps it, so
CoreML never loads the 1 GB gather. All 4 transformer chunks sit on ANE.

🤗 https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML
```

**One-liner:**

```
Qwen3.5-2B on iPhone ANE: 17 tok/s, ~200 MB memory, 4-chunk INT8 +
mmap fp16 embed. 🤗 mlboydaisuke/qwen3.5-2B-CoreML
```

---

## Demo prompts

In recommended order — each exercises a different capability of the 2B
model within the 2048-token window.

### 1. Factual knowledge (ANE parity sanity)

```
What is the capital of France? Also list 5 famous landmarks there with a one-sentence description of each.
```

Expected: "Paris" + Eiffel Tower / Louvre / Notre-Dame / Arc de Triomphe / Montmartre-ish list. Within the context limit.

### 2. Multilingual coherence (no JP → EN codeswitch, the INT4 failure mode)

```
Explain the difference between the Kamakura shogunate and the Muromachi shogunate in Japan. Answer in Japanese, three bullet points.
```

Expected: response entirely in Japanese, bullet list covering shogun / shugo / economic base etc.

### 3. Reasoning (`<think>` mode)

```
Train A leaves station A at 9:00 travelling toward station B at 60 km/h. Train B leaves station B at 9:30 travelling toward station A at 90 km/h. The distance between the stations is 150 km. At what time do they meet? Show your reasoning.
```

Expected: `<think>` block with step-by-step algebra, final answer around 10:24.

### 4. Code generation

```
Write a Swift function that computes the longest common subsequence between two strings. Include a brief complexity analysis.
```

Expected: DP-table implementation, O(m·n) time and space note.

### 5. Long-form structured output

```
Give me a detailed recipe for Japanese gyoza. List the ingredients first, then the step-by-step procedure.
```

Expected: ingredients block followed by numbered steps. Runs into the several-hundred-token range — a good stress test for the 2048-token ceiling.

### 6. Multi-turn continuity

Turn 1:
```
Explain quantum entanglement in a way a 10-year-old could understand.
```

Turn 2 (follow-up in the same chat):
```
Use that analogy to describe what happens when you separate two entangled particles over a large distance.
```

Expected: the second turn builds on the metaphor from the first — confirms that recurrent prefill re-threads state cleanly across turns.

---

## Device spec (screenshots / promos)

| metric | value |
|---|---:|
| Model | Qwen3.5-2B (hybrid Gated DeltaNet + attention) |
| Bundle (HF) | 2.4 GB (4× INT8 chunks + 1 GB fp16 embed sidecar) |
| iPhone 17 Pro decode | **17 tok/s** |
| phys_footprint (inference) | **~200 MB** |
| Metal heap (sustained) | 0 GB |
| ANE placement (chunk_a/b/c/d) | 90.7% / 91.1% / 90.7% / 90.8% |
| Context window | 2048 tokens |
| First-load ANE E5 compile | ~15 min, cached after |
