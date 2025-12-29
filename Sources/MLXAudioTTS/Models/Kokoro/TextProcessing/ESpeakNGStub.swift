// Stub for ESpeakNG until a replacement is implemented
// TODO: Replace with alternative phonemization engine

import Foundation

enum ESpeakNGEngineError: Error {
    case notImplemented
    case languageNotSupported
}

class ESpeakNGEngine {
    enum LanguageDialect: String, CaseIterable {
        case none = ""
        case enUS = "en-us"
        case enGB = "en-gb"
        case jaJP = "ja"
        case znCN = "yue"
        case frFR = "fr-fr"
        case hiIN = "hi"
        case itIT = "it"
        case ptBR = "pt-br"
        case esES = "es"
    }

    init() throws {
        // Stub implementation
        throw ESpeakNGEngineError.notImplemented
    }

    func setLanguage(for voice: TTSVoice) throws {
        throw ESpeakNGEngineError.notImplemented
    }

    func languageForVoice(voice: TTSVoice) throws -> LanguageDialect {
        throw ESpeakNGEngineError.languageNotSupported
    }

    func phonemize(text: String) throws -> String {
        throw ESpeakNGEngineError.notImplemented
    }
}
