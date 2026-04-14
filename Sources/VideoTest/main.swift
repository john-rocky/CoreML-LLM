import CoreMLLLM
import Foundation

// Mac smoke test for the Gemma 4 E2B video input path.
//
// Usage:
//   swift run video-test <model_dir> <video_path> [--prompt "..."] \
//                        [--fps 1.0] [--max-frames 6] [--audio]
//
// The model_dir must contain chunk1.mlmodelc … chunk4.mlmodelc plus
// vision.mlmodelc (and optionally audio.mlmodelc), exactly as produced by
// conversion/convert_gemma4_multimodal.py.

@main
struct VideoTest {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            FileHandle.standardError.write(Data("""
                usage: video-test <model_dir> <video_path> [--prompt "..."] \
                [--fps 1.0] [--max-frames 6] [--audio]

                """.utf8))
            exit(64)
        }
        let modelDir = URL(fileURLWithPath: args[1])
        let videoURL = URL(fileURLWithPath: args[2])

        var prompt = "Describe what happens in this video."
        var fps: Double = 1.0
        var maxFrames = 6
        var includeAudio = false
        var dryRun = false
        var compareWithSingleFrame = false

        var i = 3
        while i < args.count {
            switch args[i] {
            case "--prompt":
                if i + 1 < args.count { prompt = args[i + 1]; i += 2 } else { i += 1 }
            case "--fps":
                if i + 1 < args.count, let v = Double(args[i + 1]) { fps = v; i += 2 } else { i += 1 }
            case "--max-frames":
                if i + 1 < args.count, let v = Int(args[i + 1]) { maxFrames = v; i += 2 } else { i += 1 }
            case "--audio":
                includeAudio = true; i += 1
            case "--dry-run":
                dryRun = true; i += 1
            case "--compare":
                compareWithSingleFrame = true; i += 1
            default:
                FileHandle.standardError.write(Data("unknown arg: \(args[i])\n".utf8))
                i += 1
            }
        }

        guard FileManager.default.fileExists(atPath: videoURL.path) else {
            FileHandle.standardError.write(Data("video not found: \(videoURL.path)\n".utf8))
            exit(66)
        }
        if !dryRun && !FileManager.default.fileExists(atPath: modelDir.path) {
            FileHandle.standardError.write(Data("model dir not found: \(modelDir.path)\n".utf8))
            exit(66)
        }

        print("[video-test] model : \(modelDir.path)")
        print("[video-test] video : \(videoURL.path)")
        print("[video-test] prompt: \(prompt)")
        print("[video-test] fps=\(fps) maxFrames=\(maxFrames) audio=\(includeAudio)")

        do {
            // Dry-run the extractor so we know how many frames actually
            // made it through before the LLM path swallows any errors.
            let opts = VideoProcessor.Options(
                fps: fps, maxFrames: maxFrames, includeAudio: includeAudio)
            let frames = try await VideoProcessor.extractFrames(
                from: videoURL, options: opts)
            print("[video-test] extracted \(frames.count) frames:")
            for f in frames {
                print("    t=\(String(format: "%.2f", f.timestampSeconds))s " +
                      "\(f.image.width)×\(f.image.height)")
            }
            if includeAudio {
                let pcm = try await VideoProcessor.extractAudioPCM16k(from: videoURL)
                print("[video-test] audio samples: \(pcm?.count ?? 0) " +
                      "(~\(String(format: "%.2f", Double(pcm?.count ?? 0) / 16000))s)")
            }

            if dryRun {
                print("[video-test] --dry-run: skipping model load and generation")
                return
            }

            print("[video-test] loading model (first load may compile chunks)...")
            let t0 = Date()
            let llm = try await CoreMLLLM.load(from: modelDir) { print("  [load] \($0)") }
            print("[video-test] loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
            print("[video-test] ctx=\(llm.contextLength) name=\(llm.modelName)")

            if compareWithSingleFrame, let first = frames.first {
                print("\n[compare] Pass 1/2: SINGLE FRAME (frame 0 only)")
                print("────────────────────────────────────────")
                let t1 = Date()
                for try await tok in try await llm.stream(
                    prompt, image: first.image, maxTokens: 256
                ) { print(tok, terminator: ""); fflush(stdout) }
                print("\n──── single-frame done in " +
                      "\(String(format: "%.1f", Date().timeIntervalSince(t1)))s ────")
                llm.reset()
                llm.clearImageCache()
            }

            print("\n[video-test] Pass \(compareWithSingleFrame ? "2/2: MULTI-FRAME" : "")")
            print("────────────────────────────────────────")
            let tGen = Date()
            var chars = 0
            for try await tok in try await llm.stream(
                prompt, videoURL: videoURL, videoOptions: opts, maxTokens: 256
            ) {
                print(tok, terminator: "")
                chars += tok.count
                fflush(stdout)
            }
            print("\n────────────────────────────────────────")
            let dt = Date().timeIntervalSince(tGen)
            print("[video-test] done in \(String(format: "%.2f", dt))s, " +
                  "\(String(format: "%.1f", llm.tokensPerSecond)) tok/s, " +
                  "\(chars) chars")
        } catch {
            FileHandle.standardError.write(
                Data("[video-test] ERROR: \(error)\n".utf8))
            exit(1)
        }
    }
}
