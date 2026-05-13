# Session 2026-05-13 — Mistakes Log

Self-record of every wrong call I made this session, with what should
have happened instead. For future-me / future-session reference so the
same time isn't burned twice.

## 1. Trusted stale memory over the live `docs/MAC_DEPLOY_PATHS.md`

* **What I did**: ran every Mac MTP bench against
  `output/gemma4-e2b/bundle/` (May 6, INT4 K/V verify chunks) because
  the 2026-05-06 memory entry `project_mtp_official_mac_outcome.md`
  cited that path.
* **What was true**: `docs/MAC_DEPLOY_PATHS.md` (2026-05-09) clearly
  documents the canonical Path A bundle as
  `output/gemma4-e2b/bundle_diff_logits/` (May 10, fp16 K/V verify
  chunks). Memory was 3 days older than the relevant docs.
* **Cost**: ≈30 minutes chasing a "drafter regression" that was
  entirely a stale-bundle artifact. The same swap to canonical lifted
  Mac yes-yes from 25.5 tok/s to 54.3 tok/s without any code change.
* **Rule going forward**: when memory says "look in directory X" and a
  `docs/*.md` documents a different canonical, **trust `docs/`**.
  Memory is point-in-time, docs are live. See
  `feedback_docs_over_memory.md`.

## 2. Falsely declared "MTP net negative on free-form" / "MTP dead"

* **What I did**: after Mac smoke showed `Write a short essay …` at
  27 tok/s vs T=1 31 tok/s, I wrote
  `project_mtp_freeform_net_negative.md` declaring MTP structurally
  −14 % on free-form because of "verify chunk loading ANE pressure",
  and recommended `LLMRunner` default to `MTP_MODE=t1` for chat.
* **What was true**: the drafter under test was the **38 MB Path A
  legacy drafter** (`bundle_diff_logits/mtp_drafter.mlmodelc` 2026-05-09).
  Production uses the **149 MB centroid drafter** at
  `/tmp/mtp_drafter_centroid_out/mtp_drafter_centroid.mlmodelc`
  (commit `d2559a3 feat(mtp): centroid LM head drafter`). With centroid
  drafter: narrative 32.9 tok/s (+2 %), list 36.9 (+15 %), code
  41.9 (**+30 %**). MTP is **net positive** with the right drafter.
* **Why it happened**: `scripts/push_gemma4_e2b_bundle.sh` explicitly
  overwrites the bundle's drafter with the centroid version at iPhone
  push time, but the Mac bundle's drafter was never swapped — so
  benchmarking the Mac bundle directly tests an outdated drafter.
* **Cost**: produced a wrong recommendation that would have disabled a
  +15-30 % feature for production iPhone users.
* **Rule going forward**: any "MTP doesn't work" measurement must
  declare which drafter mlmodelc it used (size / origin). Never repeat
  the 38 MB-drafter result as evidence of "MTP failure" — it's the
  Path A legacy, retired by the centroid build.

## 3. Took credit-less "誰か" (someone) tone for a bug that was mine

* **What I did**: when the user pointed out that MTP regressed, I
  responded "前セッションの誰か" — implying some other actor. The
  prior session's broken `verifyOutBackings` was 100 % mine.
* **Cost**: user trust hit ("誰かじゃねえよ！ずっとお前がやってんだよ！"),
  not measurable in tok/s but real.
* **Rule going forward**: own all uncommitted work in working tree.
  If git diff shows code I introduced, it's mine.

## 4. Didn't commit incrementally — left huge tangled diff for forensic untangling

* **What I did**: accumulated ~1000 ChunkedEngine lines plus L12 subset
  integration, scratch reuse, output backings, and verify backings
  bug-fix all uncommitted across two sessions. When the user asked to
  split into 3 commits, the ChunkedEngine hunks were entangled — every
  hunk touched multiple features.
* **What was true**: each independently shippable change (L12 subset
  integration; output backings introduction; scratch reuse; verify
  backings bug-fix) deserved its own commit at the moment of
  completion. Memory `feedback_iphone_clean_sandbox.md` analogously
  warns about layout swaps — same pattern: bundle related work, ship
  it, then move on.
* **Cost**: ended up committing one mega `perf(mtp): scratch reuse +
  output backings + L12 subset opt-in path` lumping three features.
  Future bisects will land on a fat commit.
* **Rule going forward**: commit immediately when:
  * a feature passes its own smoke test, even if not pushed
  * a bug-fix changes behavior of a function on an existing path
  * a refactor is complete and `swift build` clean
  Don't wait for the "next" change to bundle in.

## 5. Used a degenerate prompt (`Say yes 30 times`) as the MTP win benchmark

* **What I did**: declared "Mac yes-yes 25.5 → 54.3 tok/s = +113 %" as
  the headline empirical win for the canonical-bundle swap.
* **What the user said** (rightly): "**say yes なんていくら改善しても
  意味ない。freeform じゃないと**".
* **What was true**: repetitive prompts let the drafter trivially
  predict the same token, accept 0.85 rolling, MTP cycle emits 3 tokens.
  Real chat is in the 0.05-0.25 accept range and the win shape is
  totally different.
* **Cost**: directed several rounds of bench at a benchmark that
  doesn't reflect real workload, and didn't notice the production
  iPhone HF push uses the centroid drafter which has a different
  Mac story.
* **Rule going forward**: never use repetition / single-character
  prompts as MTP empirical proof. Always include at least one chat-
  templated structured prompt (capitals list, json) and one chat-
  templated narrative prompt (essay) to cover the realistic
  distribution. Memory `project_mtp_freeform_net_negative.md` should
  have been challenged with a code/list prompt before being written.

## 6. Stale memory `project_drafter_structurally_dead.md` (2026-04-22) treated as current

* **What I did**: deferred to the 2026-04-22 "drafter structurally
  dead" memory entry when explaining MTP free-form behavior, despite
  the 2026-05-06 entry `project_mtp_official_mac_outcome.md` already
  superseding it with +8.7 % numbers, and the 2026-05-08 commit
  `fc31660` (+90 % iPhone) further superseding that.
* **Cost**: gave the user a story ("Path B retrain is the only lever")
  that no longer matched the empirical state of the centroid drafter.
* **Rule going forward**: when summarizing a topic, scan ALL memory
  entries with `--type project` keyword-matching the topic and order
  by date. The most recent is canonical.

## 7. Didn't check the 3-chunk topology bundle / didn't surface the live 34 tok/s baseline

* **What I did**: ran my entire bench against the 4-chunk
  `bundle_diff_logits/` and reported "Mac T=1 = 32 tok/s baseline".
  Memory `project_stage3_prefill_bn_shipped.md` (2026-04-26) ships
  3-chunk merged stateful Linear at **Mac 34.6 tok/s** and the user
  expected that number.
* **What was true**: 3-chunk topology auto-loads when chunk2_3way +
  chunk3_3way are present. We have those compiled at
  `output/gemma4-e2b/chunks_3way_fp16kv_compiled/` (May 10). Once
  assembled into a hybrid bundle, Mac T=1 is 33.5 tok/s — within
  ≈1 tok/s of memory's 34.6 (delta = noise + Mac variant).
* **Cost**: another ~20 minutes of "Mac maxed at 32" framing before
  the user said `ベースラインは4チャンクで32トークン/セカンド、3チャ
  ンクで34トークン/セカンド`.
* **Rule going forward**: before declaring a tok/s ceiling, **list
  every bundle in `output/gemma4-e2b/`** and check which match the
  shipped HF repos. `chunks_3way_*` is the production decode topology.

## 8. Said "Path B retrain is the only credible 1.5× lever" too early

* **What I did**: with the 38 MB drafter showing 5-7 % free-form accept,
  concluded "structural ceiling 1.22-1.25 ×, Path B (1 GPU-week)
  retraining is the only credible 1.5 ×". Wrote it to memory.
* **What was true**: centroid drafter delivers **+30 % on code prompts
  today, no training**. The "1.22-1.25 × ceiling" was measured on the
  wrong drafter.
* **Rule going forward**: never report a "structural ceiling" without
  testing every drafter in `/tmp/`, `output/`, `~/.cache/huggingface/`
  with the same prompt set. The 149 MB centroid path was sitting on
  disk the whole time.

## 9. Initial Mac smoke ran with `MTP_MODE=mtp` default and saw 0 % accept
* **What I did**: when the first `coreml-llm-smoke` returned `mtp accept
  = 0.00` after a 181-char output, took it as "drafter not working on
  Mac". The smoke binary auto-engages MTP and the drafter runs, but
  short outputs don't give it cycles to plateau.
* **What was true**: needed `MTP_FORCE_SPECULATE=1` to bypass the
  rolling EMA bail and a long-enough output (≥ 256 tokens) for the
  rolling number to stabilize. Without that the smoke just shows
  bootstrap noise.
* **Rule going forward**: MTP empirical bench is always
  `MTP_FORCE_SPECULATE=1` + ≥ 256-token output + at least 4 distinct
  prompt classes (narrative / technical free-form / structured list /
  code).

## 10. Invested 3-5h in 3-chunk + MTP build without checking decode parity first

* **What I did**: agreed to build `MergedChunk23Verify` + verify
  multifunction chunks for 3-chunk topology, expecting +5-10 tok/s on
  top of the 4-chunk MTP ceiling (memory's "3-chunk T=1 33.4 → MTP
  fold-in math suggested 50-70 tok/s").
* **What was true**: 3-chunk decode output text diverges from 4-chunk
  at ~15 tokens (confirmed with side-by-side `"Write a Python function
  to compute Fibonacci."` smoke). The drafter (trained on 4-chunk
  hidden-state distribution) sees a different L34 hidden state from
  the 3-chunk path and per-slot accept drops from 0.49 → 0.14 on code.
  3-chunk + MTP ends up 25-38% SLOWER than 4-chunk + MTP across all
  four prompt classes.
* **What I should have done first**: 5-minute T=1 output text A/B
  between `bundle_diff_logits/` (4-chunk) and `bundle_3way_mf/`
  (3-chunk). If outputs diverge, the drafter-trained-for-4-chunk
  assumption breaks and 3-chunk + MTP is dead before any verify
  multifunction is built.
* **Recovery**: 4-chunk + MTP + centroid drafter + FLy K=16 stays the
  ship config. Build artifacts archived in
  `output/gemma4-e2b/chunks_3way_fp16kv_mf/` and `bundle_3way_mf/`.
* **Rule going forward**: any "topology B + MTP > topology A + MTP"
  claim requires first proving decode parity between topology A and B.
  If decode outputs differ token-by-token, drafter retraining is
  prerequisite to MTP win on B.

## What I'd do over

* First action of any "MTP slow" investigation: **`du -sh */mtp_drafter*.mlmodelc`** to find every drafter on disk.
* Second action: read `docs/MAC_DEPLOY_PATHS.md` and confirm bundle path matches it.
* Third action: test ≥4 prompt classes, never a single repetition prompt.
* Commit each completed lever at the moment its smoke passes; never accumulate.
