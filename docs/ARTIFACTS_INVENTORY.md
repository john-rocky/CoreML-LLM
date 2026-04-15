# Local artifacts inventory

Reference for what lives under `~/Downloads/coreml-llm-artifacts/`
on dev machines. These are model bundles, backups, and helpers too
large to commit but needed for on-device testing and reproducing
the accept-rate bench. Update this file when you add or remove a
staging variant.

---

## Directory layout

```
~/Downloads/coreml-llm-artifacts/
├── backup-iphone-2k/            iPhone pull via xcrun devicectl
├── staging-2k-fast-prefill/
│   └── gemma4-e2b/               Gemma 2K + verify_qK + prefillN=512 + cross_vocab
├── staging-8k-baseline/
│   └── gemma4-e2b/               Gemma 8K decode chunks only
├── staging-8k-pr17/
│   └── gemma4-e2b/               Gemma 8K + rejected PR #17 ANE micro-opts
├── flash-8k-pr17/                Raw PR #17 reconverted chunks (8K)
├── hf-prefill-backup/            Snapshot of HF prefill/ dir before 2026-04-14 upload
├── hf-prefill-upload/            Staged metadata for the HF prefill/ upload
├── hf-verify/                    HF verify workspace
├── push-model.sh                 USB transfer helper (calls devicectl copy to)
└── SESSION_NOTES.md              Non-public work log (narrative / iPhone-specific notes)
```

Typical sizes: 2K bundles ≈ 3–4 GB, 8K bundles ≈ 3–4 GB, raw chunks
only ≈ 1 GB, HF snapshots ≈ 1 GB. Everything is APFS-clonefile
shared where possible; apparent disk use exaggerates actual use.

---

## Bundle details

### `backup-iphone-2k/`

Exact copy of `Documents/Models/gemma4-e2b/` pulled from an iPhone
with `xcrun devicectl device copy from`. Single-function chunks
(decode_q1 only, no `verify_qK`). Represents the ship-by-default 2K
model that HF distributes today.

| Key | Value |
|---|---|
| `chunk{1-4}.mlmodelc` ctx | 2048 |
| `verify_qK` function | ❌ absent |
| prefill chunks | ✅ present, prefillN=64 |
| `mtp_drafter.mlpackage` | ❌ absent |
| `cross_vocab/` | ❌ absent |
| `model_config.json` name | `gemma4-e2b-swa-2k` |

Use for: iPhone baseline reference, restoring a clean state after a
speculation experiment went sideways.

### `staging-2k-fast-prefill/gemma4-e2b/`

Multi-function 2K bundle with the faster prefill batch size.
Speculation-ready: verify chunks are live. Cross-vocab drafter is
installed so the DrafterUnion path can activate when
`crossVocabEnabled = true` is opted in.

| Key | Value |
|---|---|
| `chunk{1-4}.mlmodelc` ctx | 2048 |
| `verify_qK` function | ✅ present, K=3 |
| prefill chunks | ✅ present, prefillN=512 (8× faster than `backup-iphone-2k`'s 64) |
| `cross_vocab/qwen_drafter.mlmodelc` | ✅ installed (Qwen 2.5 0.5B, ctx=512 by its own spec) |
| `cross_vocab/qwen_gemma_vocab.bin` | ✅ 64 % Qwen→Gemma coverage |
| `mtp_drafter.mlpackage` | ⚠️ **OLD Path A drafter (2026-04-14 03:01)** — dead (acc=0 %); delete or overwrite before running MTP |
| `model_config.json` name | `gemma4-e2b-swa-ple` |

Use for: Phase B / MTP Path C / any 2K speculation experiment. Swap
the `mtp_drafter.mlpackage` for a freshly trained Path C artefact
before measuring MTP tok/s.

### `staging-8k-baseline/gemma4-e2b/`

8K target decode chunks without verify or prefill. Single-function
only. Useful as a raw 8K target but not as a speculation host.

| Key | Value |
|---|---|
| `chunk{1-4}.mlmodelc` ctx | 8192 |
| `verify_qK` | ❌ absent (decode_q1 only, not multi-function) |
| prefill chunks | ❌ absent |
| `mtp_drafter.mlpackage` | ❌ absent |
| `cross_vocab/` | ❌ absent |

Use for: 8K decode tok/s baseline. Speculation at 8K requires
reconverting these chunks as multi-function (decode_q1 + verify_qK,
K=3) plus a matching prefill set first.

### `staging-8k-pr17/gemma4-e2b/`

Same ctx=8192 structure as `staging-8k-baseline`, but the chunks are
the ones PR #17 reconverted with MLP tile / GQA broadcast / exp2
softmax. PR #17 was **rejected** (on-device 5.5× slower, see
`docs/PRIORITY_ROADMAP.md` Rejected table) — kept around as a
reference for the failed approach, not for use.

Do not push to iPhone unless specifically reproducing the PR #17
failure mode.

### `flash-8k-pr17/`

Raw PR #17 chunks only, outside the bundle hierarchy. Staging dir
before they were assembled into `staging-8k-pr17/`. Same rejection
status.

### `hf-prefill-backup/` / `hf-prefill-upload/` / `hf-verify/`

Tooling / state dirs from the 2026-04-14 HF prefill batch-size
upgrade (bumped the public prefill chunks from N=64 to N=512).
Retain as provenance for the HF change; not consumed by any test
directly.

### `push-model.sh`

Wrapper around `xcrun devicectl device copy to` that sets
`--remove-existing-content true` on the parent folder. That flag
**wipes the whole `Documents/Models/gemma4-e2b/` before copying** —
convenient for a clean full-bundle replace, destructive if another
session left important state on device (e.g. a fresh drafter not
yet backed up). For adding a single file without wiping, call
`devicectl` directly with a narrower `--source` and `--destination`
and **omit `--remove-existing-content`**.

### `SESSION_NOTES.md`

Non-public session log. Narrative / coordination state / iPhone
measurements that would be noisy or premature in the public repo.
Append new entries at the top; keep each under ~80 lines.

---

## Which bundle do I need?

| Experiment | Bundle |
|---|---|
| Clean 2K baseline decode | `backup-iphone-2k/` or any 2K bundle with flags off |
| DrafterUnion / CrossVocab speculation (2K) | `staging-2k-fast-prefill/gemma4-e2b/` |
| MTP Path C on 2K | `staging-2k-fast-prefill/gemma4-e2b/` + replace `mtp_drafter.mlpackage` |
| 8K decode without speculation | `staging-8k-baseline/gemma4-e2b/` |
| 8K speculation | ❌ none ready; reconvert first |
| Reproduce PR #17 regression | `staging-8k-pr17/gemma4-e2b/` |

---

## Restoring from a backup

If on-device state gets corrupted (partial push, wrong bundle,
speculation state wedged), push the verified iPhone snapshot back:

```bash
~/Downloads/coreml-llm-artifacts/push-model.sh backup-iphone-2k
```

This wipes `Documents/Models/gemma4-e2b/` and replaces it with the
2K single-function bundle that originally shipped with the app.
After that the baseline is 31 tok/s, no speculative paths active.
