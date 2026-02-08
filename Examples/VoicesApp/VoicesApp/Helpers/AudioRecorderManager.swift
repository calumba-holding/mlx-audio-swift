import Foundation
import AVFoundation
import MLX

@MainActor
@Observable
class AudioRecorderManager: NSObject {
    var isRecording = false
    var recordingDuration: TimeInterval = 0
    var audioLevel: Float = 0

    private var audioRecorder: AVAudioRecorder?
    private var timer: Timer?
    private var recordingStartTime: Date?
    private var recordingURL: URL?

    private let targetSampleRate: Double = 16000

    func startRecording() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true)
        #endif

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("stt_recording_\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: targetSampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]

        let recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder.delegate = self
        recorder.isMeteringEnabled = true

        guard recorder.record() else {
            throw RecordingError.failedToStart
        }

        audioRecorder = recorder
        recordingURL = url
        isRecording = true
        recordingStartTime = Date()
        recordingDuration = 0
        audioLevel = 0

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let recorder = self.audioRecorder, recorder.isRecording else { return }
                if let start = self.recordingStartTime {
                    self.recordingDuration = Date().timeIntervalSince(start)
                }
                recorder.updateMeters()
                // Convert dB level (-160...0) to 0...1 range
                let db = recorder.averagePower(forChannel: 0)
                let linear = max(0, (db + 50) / 50)  // -50dB floor
                self.audioLevel = min(linear, 1.0)
            }
        }
    }

    func stopRecording() -> MLXArray? {
        guard isRecording, let recorder = audioRecorder else { return nil }

        recorder.stop()
        audioRecorder = nil

        timer?.invalidate()
        timer = nil

        isRecording = false
        audioLevel = 0
        recordingStartTime = nil

        guard let url = recordingURL else { return nil }
        recordingURL = nil

        defer {
            try? FileManager.default.removeItem(at: url)
        }

        // Read the recorded WAV file into an MLXArray
        do {
            let file = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(file.length)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
                return nil
            }
            try file.read(into: buffer)

            let floatData: [Float]
            if let floatChannelData = buffer.floatChannelData {
                floatData = Array(UnsafeBufferPointer(
                    start: floatChannelData[0],
                    count: Int(buffer.frameLength)
                ))
            } else if let int16ChannelData = buffer.int16ChannelData {
                // Convert Int16 PCM to Float
                let int16Pointer = UnsafeBufferPointer(
                    start: int16ChannelData[0],
                    count: Int(buffer.frameLength)
                )
                floatData = int16Pointer.map { Float($0) / Float(Int16.max) }
            } else {
                return nil
            }

            guard !floatData.isEmpty else { return nil }
            return MLXArray(floatData)
        } catch {
            return nil
        }
    }

    func cancelRecording() {
        guard isRecording, let recorder = audioRecorder else { return }

        recorder.stop()
        audioRecorder = nil

        timer?.invalidate()
        timer = nil

        isRecording = false
        audioLevel = 0
        recordingStartTime = nil

        if let url = recordingURL {
            try? FileManager.default.removeItem(at: url)
            recordingURL = nil
        }
    }

    enum RecordingError: LocalizedError {
        case failedToStart

        var errorDescription: String? {
            switch self {
            case .failedToStart:
                return "Failed to start recording. Check microphone permissions in System Settings."
            }
        }
    }
}

extension AudioRecorderManager: AVAudioRecorderDelegate {
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        // Recording finished (e.g., due to interruption)
    }

    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: (any Error)?) {
        Task { @MainActor [weak self] in
            self?.isRecording = false
            self?.audioLevel = 0
        }
    }
}
