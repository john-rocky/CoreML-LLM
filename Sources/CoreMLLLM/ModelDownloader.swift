import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Downloads and caches CoreML models with background URLSession and pause/resume support.
/// Uses up to 4 concurrent connections for faster HuggingFace downloads (~2x speedup).
@Observable
public final class ModelDownloader: NSObject {
    public static let shared = ModelDownloader()

    // MARK: - Observable State

    public var isDownloading = false
    public var isPaused = false
    public var progress: Double = 0
    public var status = ""
    public var availableModels: [ModelInfo] = ModelInfo.defaults
    public var refreshTrigger = 0
    public var downloadingModelId: String?

    // MARK: - Private

    private let fileManager = FileManager.default
    private var session: URLSession!
    private var currentModel: ModelInfo?
    private var destDir: URL?
    private var pendingFiles: [DownloadFile] = []
    private var totalBytesForAllFiles: Int64 = 0
    private var downloadContinuation: CheckedContinuation<URL, Error>?

    /// Set by the app delegate for background URL session events.
    public var backgroundCompletionHandler: (() -> Void)?
    private static let sessionIdentifier = "com.coreml-llm.model-download"

    // Parallel download state
    private let maxConcurrentDownloads = 4
    private var nextFileIndex = 0
    private var completedBytes: Int64 = 0
    private var activeDownloadTasks: [Int: URLSessionDownloadTask] = [:]
    private var activeTaskFileIndex: [Int: Int] = [:]
    private var activeTaskBytes: [Int: Int64] = [:]

    // A background URLSession can carry over tasks from a prior process.
    // We adopt those (or cancel orphans) once on init via getAllTasks; until
    // adoption completes we defer download/resume so we don't spawn fresh
    // tasks that would race with — and double-download — the survivors.
    private var tasksAdopted = false
    private var pendingAdoptionActions: [() -> Void] = []

    // MARK: - Types

    public struct ModelInfo: Identifiable, Sendable {
        public let id: String
        public let name: String
        public let size: String
        public let downloadURL: String
        public let folderName: String

        public init(id: String, name: String, size: String, downloadURL: String, folderName: String) {
            self.id = id
            self.name = name
            self.size = size
            self.downloadURL = downloadURL
            self.folderName = folderName
        }

        /// Gemma 4 E2B — multimodal (image + audio + video + text), 3.1 GB,
        /// ANE-optimized. Includes a native video vision encoder
        /// (`vision_video.mlmodelc`, 64 tokens/frame) so the Swift 2×2 pool
        /// no longer sits in the video path.
        public static let gemma4e2b = ModelInfo(
            id: "gemma4-e2b", name: "Gemma 4 E2B", size: "3.1 GB",
            // n1024 branch ships the N=1024 batched prefill that pairs with
            // the Swift SWA write fix (a878c44). Old clones still point at
            // `main` and keep downloading N=512, which is safe with the
            // unfixed Swift binary.
            downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/n1024",
            folderName: "gemma4-e2b")

        /// Qwen2.5 0.5B — text only, 309 MB.
        public static let qwen25_05b = ModelInfo(
            id: "qwen2.5-0.5b", name: "Qwen2.5 0.5B (Text)", size: "309 MB",
            downloadURL: "https://github.com/john-rocky/CoreML-LLM/releases/download/v0.1.0/qwen2.5-0.5b-coreml.zip",
            folderName: "qwen2.5-0.5b")

        /// Qwen3.5 0.8B — hybrid Gated-DeltaNet SSM + attention, text-only.
        /// Ships the INT8 palettized decode mlpackage (754 MB) — same
        /// semantic precision as fp16 (top-3 = 100% parity vs fp32 oracle),
        /// half the bundle size. Prefill is performed via the same model
        /// recurrently. Runs on CPU / GPU / ANE.
        public static let qwen35_08b = ModelInfo(
            id: "qwen3.5-0.8b", name: "Qwen3.5 0.8B (ANE)", size: "754 MB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML/resolve/main",
            folderName: "qwen3.5-0.8b")

        /// Qwen3.5 2B — same hybrid SSM/attention architecture as 0.8B,
        /// just hidden_size doubled (1024→2048) and intermediate
        /// (3072→6144). Shipped as 4 INT8 chunks (6 layers each, ~1.7 GB
        /// fp16-dequantized per chunk) matching the Gemma 4 E4B pattern
        /// that fits iPhone's single-mlprogram ANE compile budget.
        /// 2-chunk at 2 GB fp16/chunk failed ANE and fell to GPU; 4-chunk
        /// stays ANE-resident. Bigger = higher quality, slower vs 0.8B.
        public static let qwen35_2b = ModelInfo(
            id: "qwen3.5-2b", name: "Qwen3.5 2B (ANE)", size: "2.4 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML/resolve/main",
            folderName: "qwen3.5-2b")

        /// Qwen3-VL 2B — text backbone of the multimodal Qwen3-VL
        /// model. 36-layer plain GQA (head_dim=128, 8 KV heads,
        /// hidden=2560), shipped as 6 INT8 body chunks (6 layers each)
        /// + a tail (final_norm + lm_head) + raw fp16 embed sidecar
        /// that Swift mmaps. Vision tower is dropped in this Phase 1
        /// release; Phase 2 will add it via a separate vision_video
        /// mlpackage + DeepStack layer-tap injection.
        public static let qwen3vl_2b = ModelInfo(
            id: "qwen3-vl-2b", name: "Qwen3-VL 2B (text + vision, ANE)", size: "4.7 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3-vl-2b-coreml/resolve/main",
            folderName: "qwen3-vl-2b")

        /// Qwen3-VL 2B stateful (Phase 1) — MLState + slice_update KV,
        /// 4-chunk INT8 + fp16 embed sidecar. iPhone 17 Pro bench
        /// 24.4 tok/s decode / 264 MB phys_footprint — 6.4× memory drop
        /// vs v1.4.0's 1.7 GB recurrent build. Text-only first ship;
        /// sideload-only under Documents/Models/qwen3-vl-2b-stateful/
        /// via scripts/qwen3vl_stateful_push.sh.
        public static let qwen3vl_2b_stateful = ModelInfo(
            id: "qwen3-vl-2b-stateful", name: "Qwen3-VL 2B (stateful, Phase 1)",
            size: "2.3 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3-vl-2b-stateful-coreml/resolve/main",
            folderName: "qwen3-vl-2b-stateful")

        /// Gemma 4 E4B — 42 layers, hidden=2560, 2 KV heads, text-only decoder.
        /// INT4 palettized, ctx=2048. Baseline ~14 tok/s on iPhone 17 Pro.
        /// A local build (`conversion/build_gemma4_bundle.py --model gemma4-e4b`)
        /// + USB sideload to `Documents/Models/gemma4-e4b/` is also supported —
        /// the app treats the folder as "downloaded" once present.
        public static let gemma4e4b = ModelInfo(
            id: "gemma4-e4b", name: "Gemma 4 E4B", size: "5.5 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E4B-coreml/resolve/main",
            folderName: "gemma4-e4b")

        /// Gemma 4 E2B Fashion — MB dress/casual theory vision advisor.
        /// Local PEFT LoRA (rank=16, alpha=32) fine-tune on 598 Unsplash/Pexels
        /// outfit photos labelled by Claude Vision, merged into the E2B base
        /// and rebuilt via `build_gemma4_bundle.py --hf-dir <merged>`. Vision
        /// tower is the stock `vision_video.mlmodelc` (64 tok/frame) grafted
        /// from the production gemma4-e2b bundle — LoRA targets language_model
        /// only, so vision weights are bit-identical to the base. Sideload-only
        /// under `Documents/Models/gemma4-e2b-fashion/`; outputs a fixed JSON
        /// schema (items, overall_dress_ratio, tpo_assumption, verdict, advice).
        public static let gemma4e2bFashion = ModelInfo(
            id: "gemma4-e2b-fashion", name: "Gemma 4 E2B Fashion (MB)",
            size: "4.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-fashion")

        /// Gemma 4 E2B + EAGLE-3 speculative — same 4.6B E2B base but with
        /// decode chunks that emit `hidden_at_L{8,17,34}` taps plus three
        /// extra mlmodelc bundles (`eagle3_draft`, `eagle3_fusion`,
        /// `verify_chunk{1..4}`). CoreMLLLM auto-loads SpeculativeLoop when
        /// all three are present and the decode stream uses K=3 speculative
        /// bursts with T=1 fallback on any burst error. Sideload-only (no HF
        /// distribution yet); the app treats the folder as "downloaded"
        /// once present under `Documents/Models/gemma4-e2b-eagle3/`.
        public static let gemma4e2bEagle3 = ModelInfo(
            id: "gemma4-e2b-eagle3", name: "Gemma 4 E2B + EAGLE-3 (K=3)", size: "5.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-eagle3")

        /// Gemma 4 E2B + LookAhead K=8 probe bundle. Same base as gemma4e2b
        /// but the four chunks are multifunction (decode_q1 + verify_qK=8)
        /// and a `probe.marker` file asks LLMRunner to auto-enable
        /// `SPECULATIVE_PROFILE` so verify chunks actually load. Sideloaded
        /// to `Documents/Models/gemma4-e2b-lookahead-probe/` — keeps the
        /// production `gemma4-e2b/` bundle untouched so users can flip
        /// between the two from the model picker. See
        /// `docs/LOOKAHEAD_PROBE_RESULTS.md` for the workflow.
        public static let gemma4e2bLookaheadProbe = ModelInfo(
            id: "gemma4-e2b-lookahead-probe", name: "Gemma 4 E2B + LookAhead (K=8, probe)",
            size: "5.7 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-lookahead-probe")

        /// Gemma 4 E2B stateful — MLState + slice_update KV cache, mirrors
        /// the Qwen3-VL 2B v1.5.0 stateful pattern. Routed through
        /// `Gemma4StatefulGenerator` (Examples/CoreMLLLMChat). Lets us drop
        /// the explicit kv13/kv14 passthrough in chunk_2 → 3/4 in favor of
        /// CoreML-managed state buffers, plus enables cross-turn KV reuse
        /// for ~zero TTFT on prefix-extending prompts. Built by
        /// `conversion/build_gemma4_e2b_stateful_chunks.py`. Sideload-only
        /// to `Documents/Models/gemma4-e2b-stateful/gemma4_e2b_stateful_chunks/`.
        public static let gemma4e2bStateful = ModelInfo(
            id: "gemma4-e2b-stateful",
            name: "Gemma 4 E2B (stateful, MLState)", size: "4.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-stateful")

        /// Gemma 4 E2B stateful — Linear variant (cml9 PR #2577 native
        /// `linear` op activation-quant). Same layout as `gemma4e2bStateful`
        /// except every Conv2d(in, out, 1, ...) projection is replaced with
        /// `nn.Linear(in, out)` (weights reshaped from (out,in,1,1) to
        /// (out,in)). Mac chunk_1 + W4 latency probe (commit b5fef64+,
        /// `conversion/probe_e2e_linear_latency.py`) shows E2E parity (-1%
        /// vs Conv2d) at production scale, refuting MBA's 5-layer +21%
        /// gap as a synthetic-probe artifact. iPhone re-test gates the
        /// `ane_ops.Conv2dLinear` migration.
        public static let gemma4e2bStatefulLinear = ModelInfo(
            id: "gemma4-e2b-stateful-linear",
            name: "Gemma 4 E2B (stateful, Linear projections)", size: "4.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-stateful-linear")

        /// LM-head split A/B/C bundles (chunk3 lm_head Conv2d split into N
        /// parallel heads, per ANEMLL qwen_model.py:1006-1124). Same chunk1
        /// and chunk2 as production; only chunk3 differs. Sideloaded to
        /// `Documents/Models/gemma4-e2b-lmsplit-{baseline,8,16}/`. See
        /// `docs/SESSION_2026_04_25_LMSPLIT_DEPLOY.md`.
        public static let gemma4e2bLMSplitBaseline = ModelInfo(
            id: "gemma4-e2b-lmsplit-baseline",
            name: "Gemma 4 E2B (lm_split=1, baseline)",
            size: "1.1 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-lmsplit-baseline")

        public static let gemma4e2bLMSplit8 = ModelInfo(
            id: "gemma4-e2b-lmsplit8",
            name: "Gemma 4 E2B (lm_split=8)",
            size: "1.1 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-lmsplit8")

        public static let gemma4e2bLMSplit16 = ModelInfo(
            id: "gemma4-e2b-lmsplit16",
            name: "Gemma 4 E2B (lm_split=16, anemll-style)",
            size: "1.1 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-lmsplit16")

        /// Visible in the UI picker. EAGLE-3 / LookAhead probe variants are
        /// hidden unless `LLM_SHOW_EXPERIMENTAL=1` is set (or the
        /// UserDefaults key `showExperimentalModels` is true). Keeps the
        /// production picker clean while letting devs flip the flag for
        /// sideload testing.
        public static var defaults: [ModelInfo] {
            let experimental =
                ProcessInfo.processInfo.environment["LLM_SHOW_EXPERIMENTAL"] == "1"
                || UserDefaults.standard.bool(forKey: "showExperimentalModels")
            var list: [ModelInfo] = [gemma4e2b, gemma4e4b, gemma4e2bFashion, qwen25_05b, qwen35_08b, qwen35_2b, qwen3vl_2b, qwen3vl_2b_stateful]
            if experimental {
                list.insert(gemma4e2bEagle3, at: 2)  // after gemma4e4b
                list.insert(gemma4e2bLookaheadProbe, at: 3)  // after EAGLE-3
                list.insert(gemma4e2bStateful, at: 4)        // after LookAhead
                list.insert(gemma4e2bStatefulLinear, at: 5)  // Plan 3 A/B partner
                list.insert(gemma4e2bLMSplitBaseline, at: 6) // Stage 4 lm-split A/B/C
                list.insert(gemma4e2bLMSplit8, at: 7)
                list.insert(gemma4e2bLMSplit16, at: 8)
            }
            return list
        }
    }

    private struct DownloadFile: Codable {
        let remotePath: String
        let localPath: String
        let estimatedSize: Int64
    }

    private struct PersistedState: Codable {
        let modelId: String
        let totalBytes: Int64
        let files: [DownloadFile]
    }

    // MARK: - Init

    override init() {
        super.init()
        cleanGraveyard()
        restorePendingDownload()
        let config = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForResource = 7200
        config.httpMaximumConnectionsPerHost = maxConcurrentDownloads
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        session.getAllTasks { [weak self] tasks in
            DispatchQueue.main.async { self?.adoptExistingTasks(tasks) }
        }
    }

    /// Claim background tasks that survived a prior app process. Tasks whose
    /// `taskDescription` matches a file in the restored `pendingFiles` are
    /// reattached to the in-memory state; everything else is an orphan
    /// (different model, stale state) and gets cancelled. Without this,
    /// `resumeDownload` would create a second task for the same file and
    /// `completedBytes` would be counted twice.
    private func adoptExistingTasks(_ tasks: [URLSessionTask]) {
        var pathToIndex: [String: Int] = [:]
        for (i, f) in pendingFiles.enumerated() { pathToIndex[f.localPath] = i }
        for t in tasks {
            if let dl = t as? URLSessionDownloadTask,
               let desc = t.taskDescription,
               let idx = pathToIndex[desc] {
                activeDownloadTasks[t.taskIdentifier] = dl
                activeTaskFileIndex[t.taskIdentifier] = idx
            } else {
                t.cancel()
            }
        }
        tasksAdopted = true
        let actions = pendingAdoptionActions
        pendingAdoptionActions.removeAll()
        for a in actions { a() }
    }

    private func runAfterAdoption(_ action: @escaping () -> Void) {
        if tasksAdopted { action() } else { pendingAdoptionActions.append(action) }
    }

    // MARK: - Public

    public func isDownloaded(_ model: ModelInfo) -> Bool {
        if isDownloading && downloadingModelId == model.id { return false }
        return localModelURL(for: model) != nil
    }

    public func hasFiles(_ model: ModelInfo) -> Bool {
        fileManager.fileExists(atPath: modelsDirectory.appendingPathComponent(model.folderName).path)
    }

    public func localModelURL(for model: ModelInfo) -> URL? {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        // Qwen3.5 has its own mlpackage names (no `model.mlpackage`).
        // Check shipping variants in order. Any one of these marks the
        // folder as a Qwen3.5 model folder.
        //   1. 2B chunked (chunk_a + chunk_b must both exist)
        //   2. 0.8B INT8 (default shipping)
        //   3. 0.8B fp16 (ground-truth)
        //   4. 2B monolithic INT8 (Mac fallback; fails ANE budget on iPhone)
        let chunksDir = dir.appendingPathComponent("qwen3_5_2b_decode_chunks")
        func chunkExists(_ base: String) -> Bool {
            // HF-downloaded layout: chunk_X.mlpackage/Data/.../weight.bin
            let pkgWeights = chunksDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkgWeights.path) { return true }
            // Sideload layout: chunk_X.mlmodelc/weights/weight.bin
            let mlcWeights = chunksDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlcWeights.path)
        }
        // 4-chunk + embed-bin layout: all must be present to count as
        // downloaded. Missing any piece → fall through to monolithic /
        // 0.8B layouts below. Embed is a raw .bin, not an mlpackage.
        let embedBinURL = chunksDir.appendingPathComponent("embed_weight.bin")
        let embedPresent = fileManager.fileExists(atPath: embedBinURL.path)
        let requiredChunks = ["chunk_a", "chunk_b", "chunk_c", "chunk_d"]
        if embedPresent && requiredChunks.allSatisfy(chunkExists) {
            // Return chunksDir itself (a directory under the model folder)
            // so callers that do `url.deletingLastPathComponent()` land on
            // the model folder root — same convention 0.8B follows when
            // its mlpackage is returned directly. Qwen35Generator resolves
            // the actual chunk_*.{mlpackage,mlmodelc} + embed_weight.bin
            // from the folder.
            return chunksDir
        }
        for name in ["qwen3_5_0_8b_decode_int8_mseq128.mlpackage",
                     "qwen3_5_0_8b_decode_fp16_mseq128.mlpackage",
                     "qwen3_5_2b_decode_int8_mseq128.mlpackage"] {
            let pkg = dir.appendingPathComponent(name)
            if fileManager.fileExists(atPath: pkg.appendingPathComponent(
                "Data/com.apple.CoreML/weights/weight.bin").path) {
                return pkg
            }
        }

        // Qwen3-VL 2B stateful (Phase 1): chunk_0..N + chunk_head +
        // embed_weight.bin under qwen3_vl_2b_stateful_chunks/. N ≥ 2
        // (we ship 4, but any count ≥ 2 loads fine). All sideloaded.
        let vl2bStatefulDir = dir.appendingPathComponent("qwen3_vl_2b_stateful_chunks")
        func vl2bStatefulChunkExists(_ base: String) -> Bool {
            let pkgWeights = vl2bStatefulDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkgWeights.path) { return true }
            let mlcWeights = vl2bStatefulDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlcWeights.path)
        }
        let vl2bStatefulEmbed = vl2bStatefulDir.appendingPathComponent("embed_weight.bin")
        if fileManager.fileExists(atPath: vl2bStatefulEmbed.path)
            && vl2bStatefulChunkExists("chunk_0")
            && vl2bStatefulChunkExists("chunk_1")
            && vl2bStatefulChunkExists("chunk_head") {
            // Return the INNER chunks dir. ChatView strips one level
            // via .deletingLastPathComponent() before passing to its
            // local loadModel(), and LLMRunner does the same again, so
            // we need an extra layer of nesting in the URL we return.
            // Mirror the Qwen3.5 chunked + Qwen3-VL v1.4.0 convention.
            return vl2bStatefulDir
        }

        // Gemma 4 E2B stateful (MLState + slice_update): chunk_{1..4} +
        // embed_tokens_q8.bin + RoPE tables + tokenizer under
        // gemma4_e2b_stateful_chunks/. Two folder names share this
        // layout: gemma4-e2b-stateful (Conv2d) and
        // gemma4-e2b-stateful-linear (Plan 3 Linear A/B partner).
        let g4StatefulDir = dir.appendingPathComponent("gemma4_e2b_stateful_chunks")
        func g4StatefulChunkExists(_ base: String) -> Bool {
            let pkg = g4StatefulDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkg.path) { return true }
            let mlc = g4StatefulDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlc.path)
        }
        let g4StatefulEmbed = g4StatefulDir
            .appendingPathComponent("embed_tokens_q8.bin")
        if fileManager.fileExists(atPath: g4StatefulEmbed.path)
            && (1...4).allSatisfy({ g4StatefulChunkExists("chunk_\($0)") })
        {
            // Same nesting convention as Qwen3-VL stateful — return the
            // INNER chunks dir; LLMRunner strips one level and adds the
            // subdir back via its own resolver.
            return g4StatefulDir
        }

        // Qwen3-VL 2B: 4 body chunks + chunk_head + embed_weight.bin
        // under qwen3_vl_2b_decode_chunks/. All-or-nothing.
        let vl2bDir = dir.appendingPathComponent("qwen3_vl_2b_decode_chunks")
        func vl2bChunkExists(_ base: String) -> Bool {
            let pkgWeights = vl2bDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkgWeights.path) { return true }
            let mlcWeights = vl2bDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlcWeights.path)
        }
        let vl2bEmbedURL = vl2bDir.appendingPathComponent("embed_weight.bin")
        let vl2bEmbedPresent = fileManager.fileExists(atPath: vl2bEmbedURL.path)
        let vl2bAll = (0..<4).allSatisfy { vl2bChunkExists("chunk_\($0)") }
            && vl2bChunkExists("chunk_head")
        if vl2bEmbedPresent && vl2bAll {
            return vl2bDir
        }
        let chunk1 = dir.appendingPathComponent("chunk1.mlmodelc")
        if fileManager.fileExists(atPath: chunk1.appendingPathComponent("weights/weight.bin").path) {
            if isChunkCtxMismatched(modelDir: dir, chunk1Dir: chunk1) {
                try? fileManager.removeItem(at: dir)
                return nil
            }
            return chunk1
        }
        let modelc = dir.appendingPathComponent("model.mlmodelc")
        if fileManager.fileExists(atPath: modelc.appendingPathComponent("weights/weight.bin").path) {
            return modelc
        }
        let pkg = dir.appendingPathComponent("model.mlpackage")
        if fileManager.fileExists(atPath: pkg.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin").path) {
            return pkg
        }
        return nil
    }

    /// Compare chunk1's declared `causal_mask_full` ctx (from model.mil) against
    /// model_config.json's context_length. Returns true on mismatch so callers
    /// can invalidate the cache and force a fresh download.
    private func isChunkCtxMismatched(modelDir: URL, chunk1Dir: URL) -> Bool {
        guard let configData = try? Data(contentsOf: modelDir.appendingPathComponent("model_config.json")),
              let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let expected = json["context_length"] as? Int else {
            return false  // no config to compare against; leave cache alone
        }
        let milURL = chunk1Dir.appendingPathComponent("model.mil")
        guard let mil = try? String(contentsOf: milURL, encoding: .utf8) else {
            return false
        }
        // Look for the causal_mask_full tensor declaration. The shape's last
        // dimension is ctx: e.g. `tensor<fp16, [1, 1, 1, 2048]> causal_mask_full`.
        let pattern = #"tensor<fp16,\s*\[\s*1,\s*1,\s*1,\s*(\d+)\s*\]>\s*causal_mask_full"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: mil, range: NSRange(mil.startIndex..., in: mil)),
              match.numberOfRanges >= 2,
              let range = Range(match.range(at: 1), in: mil),
              let actual = Int(mil[range]) else {
            return false
        }
        return actual != expected
    }

    /// Download a model, skipping files that already exist on disk.
    /// Set `repair: true` to re-check and download any missing files.
    public func download(_ model: ModelInfo, repair: Bool = false) async throws -> URL {
        if !repair, let existing = localModelURL(for: model) { return existing }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.main.async { [weak self] in
                self?.runAfterAdoption {
                    guard let self else { return }
                    // Resume paused download for same model
                    if self.isPaused && self.currentModel?.id == model.id {
                        self.downloadContinuation = continuation
                        self.resumeDownload()
                        return
                    }

                    // Cancel any in-progress download for a different model
                    if self.isDownloading {
                        self.cancelDownload()
                    }

                    self.downloadContinuation = continuation
                    self.currentModel = model
                    self.downloadingModelId = model.id
                    self.isDownloading = true
                    self.isPaused = false
                    self.progress = 0
                    self.status = "Starting..."

                    let dest = self.modelsDirectory.appendingPathComponent(model.folderName)
                    self.destDir = dest
                    try? self.fileManager.createDirectory(at: dest, withIntermediateDirectories: true)

                    if model.downloadURL.contains("huggingface.co") {
                        self.buildHuggingFaceFileList(model)
                        self.fillDownloadSlots()
                    } else {
                        try? self.fileManager.removeItem(at: dest)
                        try? self.fileManager.createDirectory(at: dest, withIntermediateDirectories: true)
                        self.pendingFiles = [DownloadFile(
                            remotePath: model.downloadURL,
                            localPath: "__archive.zip",
                            estimatedSize: 350_000_000
                        )]
                        self.totalBytesForAllFiles = 350_000_000
                        self.completedBytes = 0
                        self.nextFileIndex = 0
                        self.fillDownloadSlots()
                    }
                }
            }
        }
    }

    public func pause() {
        guard isDownloading, !isPaused else { return }
        isPaused = true
        status = "Paused"

        // Cancel all active tasks. On resume, incomplete files are re-downloaded.
        for task in activeDownloadTasks.values {
            task.cancel()
        }
        activeDownloadTasks.removeAll()
        activeTaskFileIndex.removeAll()
        activeTaskBytes.removeAll()
        saveState()
    }

    public func resumeDownload() {
        guard isPaused else { return }
        runAfterAdoption { [weak self] in
            guard let self, self.isPaused else { return }
            self.isPaused = false
            self.status = "Resuming..."

            // Re-scan from beginning — fillDownloadSlots skips completed files
            // on disk and skips files an adopted task is already fetching.
            self.nextFileIndex = 0
            self.completedBytes = 0
            self.fillDownloadSlots()
        }
    }

    public func cancelDownload() {
        for task in activeDownloadTasks.values {
            task.cancel()
        }
        activeDownloadTasks.removeAll()
        activeTaskFileIndex.removeAll()
        activeTaskBytes.removeAll()
        nextFileIndex = 0
        completedBytes = 0
        isDownloading = false
        isPaused = false
        progress = 0
        status = ""
        downloadingModelId = nil
        pendingFiles = []
        cleanupPersistedState()

        downloadContinuation?.resume(throwing: CancellationError())
        downloadContinuation = nil
    }

    public func delete(_ model: ModelInfo) throws {
        if isDownloading && currentModel?.id == model.id {
            cancelDownload()
        }
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        defer { refreshTrigger += 1 }
        guard fileManager.fileExists(atPath: dir.path) else { return }
        try evictToGraveyard(dir)
    }

    /// Remove every model folder under `modelsDirectory`. Used as an escape
    /// hatch when a stale/incompatible artifact from a prior app version
    /// can't be deleted via the per-model trash button.
    public func resetAllModels() throws {
        cancelDownload()
        defer { refreshTrigger += 1 }
        guard fileManager.fileExists(atPath: modelsDirectory.path) else { return }
        let children = (try? fileManager.contentsOfDirectory(
            at: modelsDirectory, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])) ?? []
        var firstError: Error?
        for child in children {
            // Skip the graveyard itself — it gets cleaned asynchronously.
            if child.lastPathComponent.hasPrefix(".graveyard-") { continue }
            do { try evictToGraveyard(child) }
            catch { if firstError == nil { firstError = error } }
        }
        if let e = firstError { throw e }
    }

    /// Move a file / directory out of sight by renaming it into a hidden
    /// graveyard folder, then best-effort delete. Rename succeeds on APFS
    /// even when URLSession background tasks still hold open handles to
    /// files inside — whereas `removeItem` on the same path fails with
    /// "no permission to access" because of those handles.
    ///
    /// The visible model folder disappears immediately. Remaining graveyard
    /// bytes are swept up by `cleanGraveyard` at the next init.
    private func evictToGraveyard(_ url: URL) throws {
        let graveRoot = modelsDirectory.appendingPathComponent(".graveyard", isDirectory: true)
        try? fileManager.createDirectory(at: graveRoot, withIntermediateDirectories: true)
        let grave = graveRoot.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.moveItem(at: url, to: grave)
        try? fileManager.removeItem(at: grave)  // best-effort; leftovers cleaned next launch
    }

    /// Best-effort removal of any graveyard residue from prior sessions.
    /// Called on init — by then the URLSession daemon from the prior app
    /// run has released its file handles, so removeItem usually succeeds.
    private func cleanGraveyard() {
        let graveRoot = modelsDirectory.appendingPathComponent(".graveyard", isDirectory: true)
        guard fileManager.fileExists(atPath: graveRoot.path) else { return }
        if let children = try? fileManager.contentsOfDirectory(
            at: graveRoot, includingPropertiesForKeys: nil, options: []) {
            for c in children { try? fileManager.removeItem(at: c) }
        }
        try? fileManager.removeItem(at: graveRoot)
    }

    // MARK: - Parallel Download Scheduling

    /// Start downloads until we have `maxConcurrentDownloads` active tasks,
    /// or all files are dispatched. Skips files that already exist on disk.
    private func fillDownloadSlots() {
        guard !isPaused else {
            saveState()
            return
        }

        // Files currently being fetched by an adopted or in-flight task.
        // Without this guard, restarting the loop after a pause/relaunch could
        // hand the same file to a second task and double-count its bytes.
        var activeIndices = Set(activeTaskFileIndex.values)

        while activeDownloadTasks.count < maxConcurrentDownloads && nextFileIndex < pendingFiles.count {
            let idx = nextFileIndex
            nextFileIndex += 1

            if activeIndices.contains(idx) { continue }

            let file = pendingFiles[idx]
            guard let dest = destDir else { continue }
            let destFile = dest.appendingPathComponent(file.localPath)

            // Skip already-downloaded files
            if file.localPath != "__archive.zip" && fileManager.fileExists(atPath: destFile.path) {
                let existingSize = (try? fileManager.attributesOfItem(atPath: destFile.path))?[.size] as? Int64 ?? 0
                if existingSize > 0 {
                    completedBytes += existingSize
                    updateProgress()
                    continue
                }
            }

            // Build URL
            let urlString: String
            if file.remotePath.hasPrefix("http") {
                urlString = file.remotePath
            } else if let model = currentModel {
                urlString = "\(model.downloadURL)/\(file.remotePath)"
            } else {
                continue
            }

            guard let url = URL(string: urlString) else { continue }

            try? fileManager.createDirectory(at: destFile.deletingLastPathComponent(),
                                              withIntermediateDirectories: true)

            let task = session.downloadTask(with: url)
            task.taskDescription = file.localPath
            task.resume()

            activeDownloadTasks[task.taskIdentifier] = task
            activeTaskFileIndex[task.taskIdentifier] = idx
            activeIndices.insert(idx)
        }

        // All files dispatched and all tasks completed → finish
        if activeDownloadTasks.isEmpty && nextFileIndex >= pendingFiles.count {
            finishDownload()
            return
        }

        saveState()
    }

    private func updateProgress() {
        let inFlight = activeTaskBytes.values.reduce(0 as Int64, +)
        let bytes = completedBytes + inFlight
        let total = Double(max(totalBytesForAllFiles, 1))
        progress = min(Double(bytes) / total, 0.99)
        let mbDone = Double(bytes) / 1_000_000
        let mbTotal = Double(totalBytesForAllFiles) / 1_000_000
        status = String(format: "%.0f / %.0f MB", mbDone, mbTotal)

        // Safety stop: estimates and actuals agree to within a few percent on
        // this repo. If we cross 1.5x the estimate, the most likely cause is
        // a duplicate-download bug (e.g. a leftover background task fetching
        // the same 2.35 GB file as the new one). Abort so we don't burn the
        // user's data plan or fill the disk.
        if totalBytesForAllFiles > 0,
           bytes > Int64(Double(totalBytesForAllFiles) * 1.5) {
            abortOversizeDownload(bytes: bytes)
        }
    }

    private func abortOversizeDownload(bytes: Int64) {
        for task in activeDownloadTasks.values { task.cancel() }
        activeDownloadTasks.removeAll()
        activeTaskFileIndex.removeAll()
        activeTaskBytes.removeAll()
        pendingFiles = []
        nextFileIndex = 0
        isDownloading = false
        isPaused = false
        downloadingModelId = nil
        cleanupPersistedState()
        let mbDone = bytes / 1_000_000
        let mbTotal = totalBytesForAllFiles / 1_000_000
        status = "Stopped: \(mbDone) MB exceeds expected \(mbTotal) MB"
        let err = NSError(
            domain: "CoreMLLLM.ModelDownloader", code: -2,
            userInfo: [NSLocalizedDescriptionKey:
                "Download aborted: \(mbDone) MB exceeds expected \(mbTotal) MB by 50%+. " +
                "Likely a duplicate-download bug — restart the app and retry."])
        downloadContinuation?.resume(throwing: err)
        downloadContinuation = nil
    }

    private func finishDownload() {
        guard let model = currentModel, let dest = destDir else { return }

        // Share decode weights with prefill chunks ONLY if prefill metadata
        // (coremldata.bin) was downloaded for that chunk. Models that don't
        // ship prefill (e.g. gemma4-e4b) would otherwise get half-populated
        // prefill_chunk{i}.mlmodelc directories — just weights, no
        // coremldata.bin — which CoreML rejects at load time.
        for i in 1...4 {
            let src = dest.appendingPathComponent("chunk\(i).mlmodelc/weights/weight.bin")
            let prefillDir = dest.appendingPathComponent("prefill_chunk\(i).mlmodelc")
            let coreML = prefillDir.appendingPathComponent("coremldata.bin")
            let dst = prefillDir.appendingPathComponent("weights/weight.bin")
            guard fileManager.fileExists(atPath: coreML.path),
                  fileManager.fileExists(atPath: src.path),
                  !fileManager.fileExists(atPath: dst.path) else { continue }
            try? fileManager.createDirectory(at: dst.deletingLastPathComponent(),
                                              withIntermediateDirectories: true)
            try? fileManager.copyItem(at: src, to: dst)
        }

        // Clean up any stray prefill directories that lack the required
        // metadata. These happen when an older build of the app pulled prefill
        // paths that 404'd on a prefill-less repo — the shared-weight copy
        // above then seeded zero-metadata subdirectories, which CoreML can't
        // open. Removing them here makes the device self-heal on next launch.
        for i in 1...4 {
            let prefillDir = dest.appendingPathComponent("prefill_chunk\(i).mlmodelc")
            let coreML = prefillDir.appendingPathComponent("coremldata.bin")
            if fileManager.fileExists(atPath: prefillDir.path)
                && !fileManager.fileExists(atPath: coreML.path) {
                try? fileManager.removeItem(at: prefillDir)
            }
        }

        cleanupPersistedState()
        isDownloading = false
        isPaused = false
        downloadingModelId = nil
        progress = 1.0
        status = "Ready"

        if let url = localModelURL(for: model) {
            downloadContinuation?.resume(returning: url)
        } else {
            downloadContinuation?.resume(throwing: DownloadError.extractionFailed)
        }
        downloadContinuation = nil
    }

    // MARK: - Persistence

    private var stateURL: URL {
        modelsDirectory.appendingPathComponent(".download_state.json")
    }

    private func saveState() {
        guard let model = currentModel else { return }
        try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        let state = PersistedState(
            modelId: model.id,
            totalBytes: totalBytesForAllFiles,
            files: pendingFiles
        )
        try? JSONEncoder().encode(state).write(to: stateURL)
    }

    private func cleanupPersistedState() {
        try? fileManager.removeItem(at: stateURL)
    }

    private func restorePendingDownload() {
        guard let data = try? Data(contentsOf: stateURL),
              let state = try? JSONDecoder().decode(PersistedState.self, from: data),
              let model = availableModels.first(where: { $0.id == state.modelId }) else { return }

        if localModelURL(for: model) != nil {
            cleanupPersistedState()
            return
        }

        currentModel = model
        destDir = modelsDirectory.appendingPathComponent(model.folderName)
        pendingFiles = state.files
        totalBytesForAllFiles = state.totalBytes
        nextFileIndex = 0
        completedBytes = 0
        downloadingModelId = model.id
        isDownloading = true
        isPaused = true
        // Scan completed bytes from disk for accurate progress
        if let dest = destDir {
            for file in pendingFiles {
                let path = dest.appendingPathComponent(file.localPath).path
                if let attrs = try? fileManager.attributesOfItem(atPath: path),
                   let size = attrs[.size] as? Int64, size > 0 {
                    completedBytes += size
                }
            }
        }
        progress = Double(completedBytes) / Double(max(totalBytesForAllFiles, 1))
        status = "Paused"
    }

    // MARK: - HuggingFace File List

    private func buildHuggingFaceFileList(_ model: ModelInfo) {
        // E4B lives in its own repo with a flat layout (chunks at the root,
        // text-only — no prefill, vision, or audio). E2B lives under swa/ +
        // prefill/ + vision/audio at root.
        if model.id == "gemma4-e4b" {
            buildE4BFileList()
            return
        }
        if model.id == "qwen3.5-0.8b" {
            buildQwen35FileList()
            return
        }
        if model.id == "qwen3.5-2b" {
            buildQwen35_2B_FileList()
            return
        }
        if model.id == "qwen3-vl-2b" {
            buildQwen3VL2BFileList()
            return
        }
        // 2K-context shipping model lives at the repo root on HF:
        //   - Decode chunks:  swa/chunk{1-4}.mlmodelc/
        //   - Prefill chunks: prefill/chunk{1-4}.mlmodelc/  (remote name is
        //     chunk1, local is prefill_chunk1 to avoid colliding with decode)
        // NOTE: `sdpa/` on HF is actually 8K — its metadata.json says 2048
        // but model.mil has ctx=8192 (authoritative). Don't use `sdpa/` or
        // `sdpa-8k/` until the 8K decode path lands (see docs/SPEED_8K.md).

        func mlc(_ remoteDir: String, _ remoteName: String, _ localName: String, weightSize: Int64) -> [DownloadFile] {
            [.init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/weights/weight.bin",
                   localPath: "\(localName).mlmodelc/weights/weight.bin", estimatedSize: weightSize),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/coremldata.bin",
                   localPath: "\(localName).mlmodelc/coremldata.bin", estimatedSize: 1_000),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/model.mil",
                   localPath: "\(localName).mlmodelc/model.mil", estimatedSize: 450_000),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/metadata.json",
                   localPath: "\(localName).mlmodelc/metadata.json", estimatedSize: 8_000),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(localName).mlmodelc/analytics/coremldata.bin", estimatedSize: 250)]
        }

        // Prefill metadata-only (weights are shared with decode chunks and
        // copied in finishDownload). Remote name is `chunk1` under prefill/;
        // local name is `prefill_chunk1` so it doesn't collide with decode.
        func prefillMeta(_ remoteName: String, _ localName: String) -> [DownloadFile] {
            [.init(remotePath: "prefill/\(remoteName).mlmodelc/coremldata.bin",
                   localPath: "\(localName).mlmodelc/coremldata.bin", estimatedSize: 1_000),
             .init(remotePath: "prefill/\(remoteName).mlmodelc/model.mil",
                   localPath: "\(localName).mlmodelc/model.mil", estimatedSize: 450_000),
             .init(remotePath: "prefill/\(remoteName).mlmodelc/metadata.json",
                   localPath: "\(localName).mlmodelc/metadata.json", estimatedSize: 8_000),
             .init(remotePath: "prefill/\(remoteName).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(localName).mlmodelc/analytics/coremldata.bin", estimatedSize: 250)]
        }

        // Sort large files first so they start downloading immediately on separate connections,
        // while small files fill in around them.
        var largeFiles: [DownloadFile] = []
        var smallFiles: [DownloadFile] = []

        let chunkFiles = mlc("swa", "chunk1", "chunk1", weightSize: 155_436_864)
             + mlc("swa", "chunk2", "chunk2", weightSize: 133_963_968)
             + mlc("swa", "chunk3", "chunk3", weightSize: 325_282_880)
             + mlc("swa", "chunk4", "chunk4", weightSize: 526_874_880)
             // 3-chunk decode variant (opt-in at runtime via LLM_3CHUNK=1).
             // Adds ~940 MB on disk but stays out of RAM unless selected.
             // chunk2_3way = merged L8-24 (17 layers), chunk3_3way = LM-head
             // chunk.  See docs/THREE_CHUNK_MAC_BENCH.md for the +8%
             // measurement.  Files 404 on older HF snapshots — optional mlc
             // files are skipped by the downloader, so absence is fine.
             + mlc("swa", "chunk2_3way", "chunk2_3way", weightSize: 459_768_064)
             + mlc("swa", "chunk3_3way", "chunk3_3way", weightSize: 526_874_880)
        let prefillFiles = prefillMeta("chunk1", "prefill_chunk1")
             + prefillMeta("chunk2", "prefill_chunk2")
             + prefillMeta("chunk3", "prefill_chunk3")
             + prefillMeta("chunk4", "prefill_chunk4")
        let extraFiles: [DownloadFile] = [
            .init(remotePath: "model_config.json", localPath: "model_config.json", estimatedSize: 500),
            .init(remotePath: "hf_model/tokenizer.json", localPath: "hf_model/tokenizer.json", estimatedSize: 30_000_000),
            .init(remotePath: "hf_model/tokenizer_config.json", localPath: "hf_model/tokenizer_config.json", estimatedSize: 5_000),
            .init(remotePath: "hf_model/config.json", localPath: "hf_model/config.json", estimatedSize: 5_000),
            .init(remotePath: "embed_tokens_q8.bin", localPath: "embed_tokens_q8.bin", estimatedSize: 402_653_184),
            .init(remotePath: "embed_tokens_scales.bin", localPath: "embed_tokens_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "embed_tokens_per_layer_q8.bin", localPath: "embed_tokens_per_layer_q8.bin", estimatedSize: 2_348_810_240),
            .init(remotePath: "embed_tokens_per_layer_scales.bin", localPath: "embed_tokens_per_layer_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "per_layer_projection.bin", localPath: "per_layer_projection.bin", estimatedSize: 27_525_120),
            .init(remotePath: "per_layer_norm_weight.bin", localPath: "per_layer_norm_weight.bin", estimatedSize: 1_024),
            .init(remotePath: "swa/cos_sliding.npy", localPath: "cos_sliding.npy", estimatedSize: 4_194_432),
            .init(remotePath: "swa/sin_sliding.npy", localPath: "sin_sliding.npy", estimatedSize: 4_194_432),
            .init(remotePath: "swa/cos_full.npy", localPath: "cos_full.npy", estimatedSize: 8_388_736),
            .init(remotePath: "swa/sin_full.npy", localPath: "sin_full.npy", estimatedSize: 8_388_736),
            .init(remotePath: "vision.mlmodelc/weights/weight.bin", localPath: "vision.mlmodelc/weights/weight.bin", estimatedSize: 320_000_000),
            .init(remotePath: "vision.mlmodelc/coremldata.bin", localPath: "vision.mlmodelc/coremldata.bin", estimatedSize: 200_000),
            .init(remotePath: "vision.mlmodelc/model.mil", localPath: "vision.mlmodelc/model.mil", estimatedSize: 50_000),
            .init(remotePath: "vision.mlmodelc/metadata.json", localPath: "vision.mlmodelc/metadata.json", estimatedSize: 1_000),
            .init(remotePath: "vision.mlmodelc/analytics/coremldata.bin", localPath: "vision.mlmodelc/analytics/coremldata.bin", estimatedSize: 1_000),
            // Video-grade vision encoder (Gemma 4's `video_processor` path,
            // 64 tokens/frame natively). When absent, the app transparently
            // falls back to Swift-side 2×2 pooling of the still-image encoder.
            .init(remotePath: "vision_video.mlmodelc/weights/weight.bin", localPath: "vision_video.mlmodelc/weights/weight.bin", estimatedSize: 338_081_024),
            .init(remotePath: "vision_video.mlmodelc/coremldata.bin", localPath: "vision_video.mlmodelc/coremldata.bin", estimatedSize: 418),
            .init(remotePath: "vision_video.mlmodelc/model.mil", localPath: "vision_video.mlmodelc/model.mil", estimatedSize: 711_289),
            .init(remotePath: "vision_video.mlmodelc/metadata.json", localPath: "vision_video.mlmodelc/metadata.json", estimatedSize: 2_721),
            .init(remotePath: "vision_video.mlmodelc/analytics/coremldata.bin", localPath: "vision_video.mlmodelc/analytics/coremldata.bin", estimatedSize: 243),
            // Audio encoder (Conformer 12-layer, INT8)
            .init(remotePath: "audio.mlmodelc/weights/weight.bin", localPath: "audio.mlmodelc/weights/weight.bin", estimatedSize: 295_373_248),
            .init(remotePath: "audio.mlmodelc/coremldata.bin", localPath: "audio.mlmodelc/coremldata.bin", estimatedSize: 1_000),
            .init(remotePath: "audio.mlmodelc/model.mil", localPath: "audio.mlmodelc/model.mil", estimatedSize: 759_000),
            .init(remotePath: "audio.mlmodelc/metadata.json", localPath: "audio.mlmodelc/metadata.json", estimatedSize: 3_000),
            .init(remotePath: "audio.mlmodelc/analytics/coremldata.bin", localPath: "audio.mlmodelc/analytics/coremldata.bin", estimatedSize: 250),
            .init(remotePath: "mel_filterbank.bin", localPath: "mel_filterbank.bin", estimatedSize: 131_584),
            .init(remotePath: "audio_config.json", localPath: "audio_config.json", estimatedSize: 500),
            // Audio projection weights (Swift-side float32 computation)
            .init(remotePath: "output_proj_weight.npy", localPath: "output_proj_weight.npy", estimatedSize: 3_145_856),
            .init(remotePath: "output_proj_bias.npy", localPath: "output_proj_bias.npy", estimatedSize: 3_200),
            .init(remotePath: "embed_proj_weight.npy", localPath: "embed_proj_weight.npy", estimatedSize: 4_718_720),
        ]

        let threshold: Int64 = 10_000_000  // 10 MB
        for file in chunkFiles + prefillFiles + extraFiles {
            if file.estimatedSize >= threshold {
                largeFiles.append(file)
            } else {
                smallFiles.append(file)
            }
        }

        // Large files first (sorted biggest-first) so all 4 connections saturate immediately
        largeFiles.sort { $0.estimatedSize > $1.estimatedSize }
        pendingFiles = largeFiles + smallFiles
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        nextFileIndex = 0
    }

    /// Gemma 4 E4B layout on `mlboydaisuke/gemma-4-E4B-coreml`. Text-only
    /// decoder with a flat directory tree (no `swa/` or `prefill/` prefixes,
    /// no vision/audio towers). Produced by
    /// `conversion/build_gemma4_bundle.py --model gemma4-e4b`.
    /// Qwen3.5-0.8B CoreML layout on `mlboydaisuke/qwen3.5-0.8B-CoreML`.
    /// Default ships the INT8 palettized decode (754 MB, same semantic
    /// precision as fp16 — top-3 parity vs fp32 oracle preserved).
    /// Tokenizer is fetched by swift-transformers at runtime from
    /// `Qwen/Qwen3.5-0.8B` on HF.
    ///
    /// mlpackage structure:
    ///   qwen3_5_0_8b_decode_int8_mseq128.mlpackage/
    ///   ├── Manifest.json
    ///   └── Data/com.apple.CoreML/
    ///       ├── model.mlmodel
    ///       └── weights/weight.bin  (753 MB)
    ///
    /// Local layout after download (under `Models/qwen3.5-0.8b/`):
    ///   qwen3_5_0_8b_decode_int8_mseq128.mlpackage/...  (same structure)
    private func buildQwen35FileList() {
        let pkg = "qwen3_5_0_8b_decode_int8_mseq128.mlpackage"
        pendingFiles = [
            .init(remotePath: "\(pkg)/Manifest.json",
                  localPath: "\(pkg)/Manifest.json",
                  estimatedSize: 700),
            .init(remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                  localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                  estimatedSize: 645_000),
            .init(remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                  localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                  estimatedSize: 753_000_000),
        ]
        // Sort biggest-first so large weight download starts immediately.
        pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        nextFileIndex = 0
    }

    /// Qwen3.5-2B CoreML layout on `mlboydaisuke/qwen3.5-2B-CoreML`.
    /// 4 INT8 transformer chunks + 1 raw fp16 embed sidecar under
    /// `qwen3_5_2b_decode_chunks/`:
    ///   chunk_a..c:       6 layers each (pure transformer body)
    ///   chunk_d:          6 layers + final_norm + lm_head
    ///   embed_weight.bin: raw fp16 embed_tokens, Swift mmaps directly
    /// Embed is NOT an mlpackage so CoreML doesn't dequant its 1 GB
    /// into CPU-resident memory — the mmap'd file stays in clean
    /// virtual pages and only the few rows touched per prompt page in.
    /// Every transformer chunk is ≤ 1 GB fp16, fitting iPhone's ANE
    /// single-mlprogram compile envelope.
    private func buildQwen35_2B_FileList() {
        let root = "qwen3_5_2b_decode_chunks"
        // Per-chunk weight.bin sizes measured from the INT8 palettized
        // output. chunk_d carries 1 GB of lm_head; body chunks are just
        // 6 transformer layers.
        let sizes: [(String, Int64)] = [
            ("chunk_a.mlpackage", 340_000_000),  // 6 layers
            ("chunk_b.mlpackage", 340_000_000),  // 6 layers
            ("chunk_c.mlpackage", 340_000_000),  // 6 layers
            ("chunk_d.mlpackage", 850_000_000),  // 6 layers + lm_head
        ]
        var files: [DownloadFile] = []
        for (chunk, weightSize) in sizes {
            let pkg = "\(root)/\(chunk)"
            files.append(.init(
                remotePath: "\(pkg)/Manifest.json",
                localPath: "\(pkg)/Manifest.json",
                estimatedSize: 700))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                estimatedSize: 900_000))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                estimatedSize: weightSize))
        }
        // Raw fp16 embed sidecar: 248320 × 2048 × 2 bytes ≈ 1.017 GB.
        files.append(.init(
            remotePath: "\(root)/embed_weight.bin",
            localPath: "\(root)/embed_weight.bin",
            estimatedSize: 1_017_000_000))
        pendingFiles = files
        pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        nextFileIndex = 0
    }

    /// Qwen3-VL 2B (text-only) CoreML layout on
    /// `mlboydaisuke/qwen3-vl-2b-coreml`. 4 INT8 body chunks
    /// (6 layers each) + chunk_head (final_norm + lm_head) + raw fp16
    /// embed_weight.bin sidecar under `qwen3_vl_2b_decode_chunks/`.
    /// Same shape contract as Qwen3.5 2B v1.1.0 — Swift mmaps the
    /// embed sidecar to keep its 778 MB out of phys_footprint.
    private func buildQwen3VL2BFileList() {
        let root = "qwen3_vl_2b_decode_chunks"
        // 2B is 28 layers split into 4 body chunks × 7 layers each
        // (vs 4B's 36 layers / 6 chunks / 6 each). Per-chunk weight.bin
        // sizes measured from the INT8 palettized output.
        var sizes: [(String, Int64)] = (0..<4).map { i in
            ("chunk_\(i).mlpackage", Int64(353_000_000))  // 7 layers each
        }
        sizes.append(("chunk_head.mlpackage", 311_000_000))  // final_norm + lm_head
        // DeepStack-aware chunk_0 replacement for the vision path —
        // same weight footprint as chunk_0 (353 MB). Shipped alongside
        // the regular chunk_0 so vision can be toggled on per-prompt.
        sizes.append(("chunk_0_vision.mlpackage", 353_000_000))
        // Batched-prefill chunks (T=32) — optional, enables ~10× TTFT
        // improvement for image prompts. Same per-layer weight budget
        // as the decode chunks (they share backbone params just with a
        // T-axis added to the activations), INT8-palettized.
        for i in 0..<4 {
            sizes.append(("prefill_chunk_\(i).mlpackage", 353_000_000))
        }
        sizes.append(("prefill_chunk_0_vision.mlpackage", 353_000_000))
        var files: [DownloadFile] = []
        for (chunk, weightSize) in sizes {
            let pkg = "\(root)/\(chunk)"
            files.append(.init(
                remotePath: "\(pkg)/Manifest.json",
                localPath: "\(pkg)/Manifest.json",
                estimatedSize: 700))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                estimatedSize: 900_000))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                estimatedSize: weightSize))
        }
        // Raw fp16 embed sidecar: 151936 × 2048 × 2 bytes ≈ 622 MB.
        files.append(.init(
            remotePath: "\(root)/embed_weight.bin",
            localPath: "\(root)/embed_weight.bin",
            estimatedSize: 622_000_000))
        // Vision encoder (ships alongside the decode chunks).
        // Input: pixel_values (3, 2, 448, 448) fp16, output: merger
        // hidden + 3 DeepStack slices. ~388 MB INT8 palettized.
        let visionPkg = "qwen3_vl_2b_vision/vision.mlpackage"
        files.append(.init(
            remotePath: "\(visionPkg)/Manifest.json",
            localPath: "\(visionPkg)/Manifest.json",
            estimatedSize: 700))
        files.append(.init(
            remotePath: "\(visionPkg)/Data/com.apple.CoreML/model.mlmodel",
            localPath: "\(visionPkg)/Data/com.apple.CoreML/model.mlmodel",
            estimatedSize: 400_000))
        files.append(.init(
            remotePath: "\(visionPkg)/Data/com.apple.CoreML/weights/weight.bin",
            localPath: "\(visionPkg)/Data/com.apple.CoreML/weights/weight.bin",
            estimatedSize: 406_000_000))
        pendingFiles = files
        pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        nextFileIndex = 0
    }

    private func buildE4BFileList() {
        func mlc(_ name: String, weightSize: Int64) -> [DownloadFile] {
            [.init(remotePath: "\(name).mlmodelc/weights/weight.bin",
                   localPath: "\(name).mlmodelc/weights/weight.bin", estimatedSize: weightSize),
             .init(remotePath: "\(name).mlmodelc/coremldata.bin",
                   localPath: "\(name).mlmodelc/coremldata.bin", estimatedSize: 1_200),
             .init(remotePath: "\(name).mlmodelc/model.mil",
                   localPath: "\(name).mlmodelc/model.mil", estimatedSize: 1_250_000),
             .init(remotePath: "\(name).mlmodelc/metadata.json",
                   localPath: "\(name).mlmodelc/metadata.json", estimatedSize: 25_000),
             .init(remotePath: "\(name).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(name).mlmodelc/analytics/coremldata.bin", estimatedSize: 250)]
        }

        // Chunk weight sizes (observed from the shipping bundle; larger than E2B
        // because hidden=2560 and intermediate=10240 doubles the MLP wide).
        let chunkFiles: [DownloadFile] =
              mlc("chunk1", weightSize: 586_000_000)   // 558.8 MB
            + mlc("chunk2", weightSize: 572_000_000)   // 545.7 MB
            + mlc("chunk3", weightSize: 413_000_000)   // 393.6 MB
            + mlc("chunk4", weightSize: 754_000_000)   // 718.9 MB (includes LM head)

        let extraFiles: [DownloadFile] = [
            .init(remotePath: "model_config.json", localPath: "model_config.json", estimatedSize: 800),
            .init(remotePath: "hf_model/tokenizer.json", localPath: "hf_model/tokenizer.json", estimatedSize: 32_200_000),
            .init(remotePath: "hf_model/tokenizer_config.json", localPath: "hf_model/tokenizer_config.json", estimatedSize: 2_200),
            .init(remotePath: "hf_model/config.json", localPath: "hf_model/config.json", estimatedSize: 5_200),
            .init(remotePath: "hf_model/generation_config.json", localPath: "hf_model/generation_config.json", estimatedSize: 300),
            .init(remotePath: "embed_tokens_q8.bin", localPath: "embed_tokens_q8.bin", estimatedSize: 671_088_640),
            .init(remotePath: "embed_tokens_scales.bin", localPath: "embed_tokens_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "embed_tokens_per_layer_q8.bin", localPath: "embed_tokens_per_layer_q8.bin", estimatedSize: 2_825_912_320),
            .init(remotePath: "embed_tokens_per_layer_scales.bin", localPath: "embed_tokens_per_layer_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "per_layer_projection.bin", localPath: "per_layer_projection.bin", estimatedSize: 55_050_240),
            .init(remotePath: "per_layer_norm_weight.bin", localPath: "per_layer_norm_weight.bin", estimatedSize: 512),
            .init(remotePath: "cos_sliding.npy", localPath: "cos_sliding.npy", estimatedSize: 2_097_280),
            .init(remotePath: "sin_sliding.npy", localPath: "sin_sliding.npy", estimatedSize: 2_097_280),
            .init(remotePath: "cos_full.npy", localPath: "cos_full.npy", estimatedSize: 4_194_432),
            .init(remotePath: "sin_full.npy", localPath: "sin_full.npy", estimatedSize: 4_194_432),
        ]

        var largeFiles: [DownloadFile] = []
        var smallFiles: [DownloadFile] = []
        let threshold: Int64 = 10_000_000
        for file in chunkFiles + extraFiles {
            if file.estimatedSize >= threshold {
                largeFiles.append(file)
            } else {
                smallFiles.append(file)
            }
        }
        largeFiles.sort { $0.estimatedSize > $1.estimatedSize }
        pendingFiles = largeFiles + smallFiles
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        nextFileIndex = 0
    }

    // MARK: - ZIP

    private func unzipFile(_ zipURL: URL, to destDir: URL) throws {
        #if os(macOS)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = ["-xk", zipURL.path, destDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        #else
        // iOS (device + simulator), visionOS, tvOS, watchOS — Foundation's
        // `Process` is macOS-only, so unzip ourselves via the ZIP central
        // directory. Previously this branch used `#if targetEnvironment(simulator)
        // || os(macOS)` which broke iOS Simulator builds with "cannot find
        // 'Process' in scope".
        try extractZipNative(from: zipURL, to: destDir)
        #endif
    }

    #if !os(macOS)
    private func extractZipNative(from zipURL: URL, to destDir: URL) throws {
        let data = try Data(contentsOf: zipURL)
        guard data.count > 22 else { throw DownloadError.extractionFailed }
        var eocdOffset = data.count - 22
        while eocdOffset >= 0 {
            if data[eocdOffset] == 0x50 && data[eocdOffset+1] == 0x4B &&
               data[eocdOffset+2] == 0x05 && data[eocdOffset+3] == 0x06 { break }
            eocdOffset -= 1
        }
        guard eocdOffset >= 0 else { throw DownloadError.extractionFailed }
        let cdOffset = Int(data[eocdOffset+16..<eocdOffset+20].withUnsafeBytes { $0.load(as: UInt32.self) })
        let cdCount = Int(data[eocdOffset+10..<eocdOffset+12].withUnsafeBytes { $0.load(as: UInt16.self) })
        var pos = cdOffset
        for _ in 0..<cdCount {
            guard data[pos] == 0x50, data[pos+1] == 0x4B else { break }
            let uncompSize = Int(data[pos+24..<pos+28].withUnsafeBytes { $0.load(as: UInt32.self) })
            let nameLen = Int(data[pos+28..<pos+30].withUnsafeBytes { $0.load(as: UInt16.self) })
            let extraLen = Int(data[pos+30..<pos+32].withUnsafeBytes { $0.load(as: UInt16.self) })
            let commentLen = Int(data[pos+32..<pos+34].withUnsafeBytes { $0.load(as: UInt16.self) })
            let localOffset = Int(data[pos+42..<pos+46].withUnsafeBytes { $0.load(as: UInt32.self) })
            let name = String(data: data[pos+46..<pos+46+nameLen], encoding: .utf8) ?? ""
            let destPath = destDir.appendingPathComponent(name)
            if name.hasSuffix("/") {
                try fileManager.createDirectory(at: destPath, withIntermediateDirectories: true)
            } else {
                try fileManager.createDirectory(at: destPath.deletingLastPathComponent(), withIntermediateDirectories: true)
                let lnl = Int(data[localOffset+26..<localOffset+28].withUnsafeBytes { $0.load(as: UInt16.self) })
                let lel = Int(data[localOffset+28..<localOffset+30].withUnsafeBytes { $0.load(as: UInt16.self) })
                let ds = localOffset + 30 + lnl + lel
                try Data(data[ds..<ds+uncompSize]).write(to: destPath)
            }
            pos += 46 + nameLen + extraLen + commentLen
        }
    }
    #endif

    /// Files inside an mlmodelc that CoreML doesn't require to load the model.
    /// A 404 on these shouldn't abort the whole download — the upload process
    /// for W8A8 has historically produced HF repos missing `coremldata.bin`
    /// (since fixed) and `metadata.json` (still missing in some uploads); the
    /// latter is purely descriptive. Keep this list conservative — anything
    /// not listed here is treated as required.
    private func isOptionalMlmodelcFile(_ localPath: String) -> Bool {
        // metadata.json and analytics/coremldata.bin are descriptive and
        // missing from some historical uploads — always optional.
        if localPath.hasSuffix(".mlmodelc/metadata.json")
            || localPath.hasSuffix(".mlmodelc/analytics/coremldata.bin") {
            return true
        }
        // 3-chunk variant files are entirely optional — they enable the
        // LLM_3CHUNK=1 opt-in path but are not needed for the default
        // 4-chunk decoder. If HF doesn't yet have them (older snapshot),
        // skip so existing bundles still install cleanly.
        let optionalMlmodelcPrefixes = [
            "chunk2_3way.mlmodelc/",
            "chunk3_3way.mlmodelc/",
        ]
        return optionalMlmodelcPrefixes.contains { localPath.hasPrefix($0) }
    }

    private var modelsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("Models")
    }
}

// MARK: - URLSession Delegate

extension ModelDownloader: URLSessionDownloadDelegate {
    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didFinishDownloadingTo location: URL) {
        guard let localPath = downloadTask.taskDescription,
              let dest = destDir else { return }

        // HTTP status check: URLSessionDownloadTask writes the response body
        // to `location` regardless of status code. A 404 from HuggingFace is
        // a short HTML page ("Entry not found") that would otherwise be
        // saved verbatim and later fail a checksum / signature check with
        // a misleading error. Catch it here with a clear error.
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode >= 400 {
            // Read a small excerpt of the body for the error message.
            let snippet: String = (try? String(contentsOf: location, encoding: .utf8))?
                .prefix(200).trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            try? fileManager.removeItem(at: location)
            let url = downloadTask.originalRequest?.url?.absoluteString ?? "(unknown)"
            let taskId = downloadTask.taskIdentifier

            // Optional files: metadata.json and analytics/coremldata.bin inside
            // an mlmodelc are descriptive, not functional — CoreML loads fine
            // without them. Treat 404 on these as non-fatal so a slightly
            // incomplete HF upload doesn't abort the entire download.
            if http.statusCode == 404 && isOptionalMlmodelcFile(localPath) {
                print("[Download] Skipping optional missing file: \(localPath) (404)")
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.activeDownloadTasks.removeValue(forKey: taskId)
                    self.activeTaskFileIndex.removeValue(forKey: taskId)
                    self.activeTaskBytes.removeValue(forKey: taskId)
                    self.fillDownloadSlots()
                }
                return
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.activeDownloadTasks.removeValue(forKey: taskId)
                self.activeTaskFileIndex.removeValue(forKey: taskId)
                self.activeTaskBytes.removeValue(forKey: taskId)
                self.status = "Error: HTTP \(http.statusCode) for \(localPath)"
                // Surface the actual server message so the user sees *why* it failed
                // (e.g., "Entry not found. Please check the file URL.").
                let err = NSError(domain: "CoreMLLLM.ModelDownloader", code: http.statusCode,
                                  userInfo: [
                                    NSLocalizedDescriptionKey:
                                        "HTTP \(http.statusCode) fetching \(url). " +
                                        (snippet.isEmpty ? "" : "Server: \(snippet)"),
                                  ])
                self.isDownloading = false
                self.downloadingModelId = nil
                self.downloadContinuation?.resume(throwing: err)
                self.downloadContinuation = nil
            }
            return
        }

        let destFile = dest.appendingPathComponent(localPath)

        // Must move synchronously before this method returns
        try? fileManager.createDirectory(at: destFile.deletingLastPathComponent(),
                                          withIntermediateDirectories: true)
        try? fileManager.removeItem(at: destFile)
        try? fileManager.moveItem(at: location, to: destFile)

        let downloadedSize = (try? fileManager.attributesOfItem(atPath: destFile.path))?[.size] as? Int64 ?? 0
        let isZip = localPath == "__archive.zip"
        let taskId = downloadTask.taskIdentifier

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.activeDownloadTasks.removeValue(forKey: taskId)
            self.activeTaskFileIndex.removeValue(forKey: taskId)
            self.activeTaskBytes.removeValue(forKey: taskId)
            self.completedBytes += downloadedSize
            self.updateProgress()

            if isZip {
                self.status = "Extracting..."
                DispatchQueue.global(qos: .userInitiated).async {
                    try? self.unzipFile(destFile, to: dest)
                    try? self.fileManager.removeItem(at: destFile)
                    DispatchQueue.main.async {
                        self.fillDownloadSlots()
                    }
                }
            } else {
                self.fillDownloadSlots()
            }
        }
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                           totalBytesExpectedToWrite: Int64) {
        let taskId = downloadTask.taskIdentifier
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.activeTaskBytes[taskId] = totalBytesWritten
            self.updateProgress()
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        let taskId = task.taskIdentifier
        if (error as NSError).code == NSURLErrorCancelled {
            DispatchQueue.main.async { [weak self] in
                self?.activeDownloadTasks.removeValue(forKey: taskId)
                self?.activeTaskFileIndex.removeValue(forKey: taskId)
                self?.activeTaskBytes.removeValue(forKey: taskId)
            }
            return
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.activeDownloadTasks.removeValue(forKey: taskId)
            self.activeTaskFileIndex.removeValue(forKey: taskId)
            self.activeTaskBytes.removeValue(forKey: taskId)

            // If other tasks are still active, just log and continue
            if !self.activeDownloadTasks.isEmpty || self.nextFileIndex < self.pendingFiles.count {
                self.fillDownloadSlots()
                return
            }

            // All tasks failed
            self.status = "Error: \(error.localizedDescription)"
            self.isDownloading = false
            self.isPaused = false
            self.downloadingModelId = nil
            self.cleanupPersistedState()
            self.downloadContinuation?.resume(throwing: error)
            self.downloadContinuation = nil
        }
    }

    public func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async { [weak self] in
            self?.backgroundCompletionHandler?()
            self?.backgroundCompletionHandler = nil
        }
    }
}

public enum DownloadError: LocalizedError {
    case invalidURL, extractionFailed
    public var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid download URL"
        case .extractionFailed: return "Failed to extract model"
        }
    }
}
