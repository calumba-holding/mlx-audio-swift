import Foundation
import HuggingFace
import Hub
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioTTS

@main
@MainActor
enum App {
    static func main() async {
        do {
            let args = try CLI.parse()
            try await run(
                model: args.model,
                text: args.text,
                voice: args.voice
            )
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func run(
        model: String,
        text: String,
        voice: String?,
        hfToken: String? = nil
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024
        
        print("Loading model (\(model))…")
        
        // Check for HF token in environment (macOS) or Info.plist (iOS) as a fallback
        let hfToken: String? = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"] ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        
        guard let repoID = Repo.ID(rawValue: model) else {
            throw NSError(domain: "MLXAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(model)"])
        }
        let modelType = try await ModelUtils.resolveModelType(repoID: repoID, hfToken: hfToken)
        let modelURL = try await ModelUtils.resolveOrDownloadModel(repoID: repoID, requiredExtension: "safetensors", hfToken: hfToken)
        let configData = try Data(contentsOf: modelURL.appendingPathComponent("config.json"))
        
        let loadedModel: Module
        
        switch modelType {
        case "qwen_tts":
            loadedModel = try await Qwen3Model.fromPretrained(model)
        default:
            throw NSError(domain: "MLXAudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unsupported model type: \(String(describing: modelType))"])
        }
        
        let player = AudioPlayerManager()

        print("Generating…")
        let started = CFAbsoluteTimeGetCurrent()
        
        loadedModel
        
        // ...

        print(String(format: "Finished generation in %0.2fs", CFAbsoluteTimeGetCurrent() - started))
        print("Memory usage:\n\(Memory.snapshot())")
        player.stopStreaming()

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        print(String(format: "Done. Elapsed: %.2fs", elapsed))
    }
}

// MARK: - Minimal CLI parser

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)

    var description: String {
        switch self {
        case .missingValue(let k): "Missing value for \(k)"
        case .unknownOption(let k): "Unknown option \(k)"
        }
    }
}

struct CLI {
    let model: String
    let text: String
    let voice: String?

    static func parse() throws -> CLI {
        var text: String?
        var voice: String? = nil
        var model = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--text", "-t":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                text = v
            case "--voice", "-v":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                voice = v
            case "--model":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                model = v
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if text == nil, !arg.hasPrefix("-") {
                    text = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        guard let finalText = text, !finalText.isEmpty else {
            throw CLIError.missingValue("--text")
        }

        return CLI(model: model, text: finalText, voice: voice)
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "marvis-tts-cli"
        print("""
        Usage:
          \(exe) --text "Hello world" [--voice conversational_b] [--repo-id <hf-repo>]

        Options:
          -t, --text <string>           Text to synthesize (required if not passed as trailing arg)
          -v, --voice <name>            Voice id
              --repo-id <repo>          HF repo id. Default: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
          -h, --help                    Show this help
        """)
    }
}
