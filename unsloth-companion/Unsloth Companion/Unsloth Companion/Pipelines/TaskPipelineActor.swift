import AVFoundation
import CoreImage
import Foundation
import UIKit
@preconcurrency import Vision

struct PreparedTaskInput: Sendable {
    let prompt: String
    let mediaPayloads: [Data]
    let derivedResult: [String: JSONValue]
    let requireJSONObject: Bool
}

enum TaskPipelineError: LocalizedError {
    case missingText, missingMedia, unsupportedMedia, rawMediaNotAuthorized, invalidStructuredOutput

    var errorDescription: String? {
        switch self {
        case .missingText: return String(localized: "The task does not contain text input.")
        case .missingMedia: return String(localized: "The task does not contain a media file.")
        case .unsupportedMedia: return String(localized: "The media format cannot be processed.")
        case .rawMediaNotAuthorized: return String(localized: "Raw media processing was not authorized on this iPhone.")
        case .invalidStructuredOutput: return String(localized: "The model did not return a complete JSON result.")
        }
    }
}

actor TaskPipelineActor {
    static let shared = TaskPipelineActor()
    private let imageContext = CIContext(options: [.cacheIntermediates: false])

    func prepare(task: CompanionTask, taskDirectory: URL, allowRawMedia: Bool) async throws -> PreparedTaskInput {
        let text = task.input["text"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let instruction = task.input["instruction"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        switch task.kind {
        case .subagent:
            guard !text.isEmpty || !instruction.isEmpty else { throw TaskPipelineError.missingText }
            let requireJSONObject = !task.resultSchema.isEmpty
            let formatInstruction: String
            if requireJSONObject {
                let schema = resultSchemaText(task.resultSchema)
                formatInstruction = "Return exactly one complete JSON object that conforms to this result schema: \(schema)"
            } else {
                formatInstruction = "Follow the delegated objective's requested output format exactly. Return only the final deliverable. Do not add a JSON wrapper, metadata, commentary, or extra fields unless the delegated objective explicitly asks for them."
            }
            return PreparedTaskInput(
                prompt: """
                You are an execution subagent for the Mac orchestrator. Complete the delegated objective using the Mac-supplied context below. The delegated objective is an instruction, not text to reproduce.

                Start directly with the requested answer or deliverable. Do not quote, repeat, paraphrase, summarize, or preface the delegated objective. Do not answer by describing the assignment. If the objective asks what is happening "now", "here", or "in the current context", interpret that as the MAC-SUPPLIED CONTEXT below, not as hidden model internals. Do not give a generic explanation about being a language model when the supplied context is sufficient. Preserve every constraint and identify genuine uncertainty without inventing facts. \(formatInstruction)

                DELEGATED OBJECTIVE:
                \(instruction)

                MAC-SUPPLIED CONTEXT:
                \(text)
                """,
                mediaPayloads: [],
                derivedResult: [:],
                requireJSONObject: requireJSONObject
            )
        case .classification:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Classify the input. Return a JSON object with label, confidence from 0 to 1, and concise rationale. \(instruction)")
        case .summary:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Summarize faithfully. Return a JSON object with summary, keyPoints, and omittedDetails. \(instruction)")
        case .contextCompression:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Compress the context without losing constraints, decisions, names, numbers, or unresolved items. Return JSON with compressedContext and preservedFacts. \(instruction)")
        case .extraction:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Extract only supported facts. Return a JSON object conforming to the requested fields and include evidence spans. \(instruction)")
        case .verification:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Verify internal consistency only from the supplied material. Return JSON with verdict, claims, contradictions, and missingEvidence. \(instruction)")
        case .reranking:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Rerank the candidates for the stated query. Return JSON with ordered candidate IDs and scores. \(instruction)")
        case .lightweightPlanning:
            guard !text.isEmpty else { throw TaskPipelineError.missingText }
            return textTask(text, instruction: "Create an executable bounded plan. Return JSON with ordered steps, dependencies, risks, and acceptance checks. \(instruction)")
        case .ocr:
            let media = try mediaURLs(task: task, directory: taskDirectory)
            let recognized = try await media.asyncMap { try await recognizeText(in: $0) }.joined(separator: "\n")
            return PreparedTaskInput(prompt: "", mediaPayloads: [], derivedResult: ["ocrText": .string(recognized)], requireJSONObject: false)
        case .vision:
            try requireRawAuthorization(task, allowRawMedia: allowRawMedia)
            let media = try mediaURLs(task: task, directory: taskDirectory)
            let payloads = try media.map(preprocessImage)
            let recognized = try await media.asyncMap { try await recognizeText(in: $0) }.joined(separator: "\n")
            return PreparedTaskInput(prompt: "Analyze the images for the requested task. OCR found: \(recognized). \(instruction)", mediaPayloads: payloads, derivedResult: ["ocrText": .string(recognized)], requireJSONObject: !task.resultSchema.isEmpty)
        case .videoSummary:
            try requireRawAuthorization(task, allowRawMedia: allowRawMedia)
            guard let video = try mediaURLs(task: task, directory: taskDirectory).first else { throw TaskPipelineError.missingMedia }
            let frames = try await extractAdaptiveFrames(video)
            var ocr: [String] = []
            var payloads: [Data] = []
            for frame in frames {
                payloads.append(try encodeJPEG(frame.image, maximumDimension: 1_280))
                let text = try await recognizeText(in: frame.image)
                if !text.isEmpty { ocr.append("[\(String(format: "%.2f", frame.seconds))s] \(text)") }
            }
            return PreparedTaskInput(prompt: "Summarize the sampled video in temporal order. Frame OCR:\n\(ocr.joined(separator: "\n"))\n\(instruction)", mediaPayloads: payloads, derivedResult: ["sampledFrames": .number(Double(frames.count)), "ocrText": .string(ocr.joined(separator: "\n"))], requireJSONObject: !task.resultSchema.isEmpty)
        case .audioTranscription, .audioAnalysis:
            try requireRawAuthorization(task, allowRawMedia: allowRawMedia)
            guard let audio = try mediaURLs(task: task, directory: taskDirectory).first else { throw TaskPipelineError.missingMedia }
            let payload = try Data(contentsOf: audio, options: .mappedIfSafe)
            return PreparedTaskInput(prompt: task.kind == .audioTranscription ? "Transcribe this audio accurately. \(instruction)" : "Analyze the content and acoustic events in this audio. \(instruction)", mediaPayloads: [payload], derivedResult: [:], requireJSONObject: !task.resultSchema.isEmpty)
        case .dsp:
            try requireRawAuthorization(task, allowRawMedia: allowRawMedia)
            guard let audio = try mediaURLs(task: task, directory: taskDirectory).first else { throw TaskPipelineError.missingMedia }
            let pcm = try decodeAudio(audio)
            let metrics = try AudioDSPAnalyzer.analyze(samples: pcm.samples, sampleRate: pcm.sampleRate)
            let value = try JSONValue.encode(metrics)
            return PreparedTaskInput(prompt: "", mediaPayloads: [], derivedResult: value.object ?? [:], requireJSONObject: false)
        }
    }

    private func textTask(_ text: String, instruction: String, requireJSONObject: Bool = true) -> PreparedTaskInput {
        PreparedTaskInput(prompt: "\(instruction)\n\nINPUT:\n\(text)", mediaPayloads: [], derivedResult: [:], requireJSONObject: requireJSONObject)
    }

    private func resultSchemaText(_ schema: [String: JSONValue]) -> String {
        guard !schema.isEmpty,
              let data = try? ProtocolCoding.encoder.encode(schema),
              let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }

    private func requireRawAuthorization(_ task: CompanionTask, allowRawMedia: Bool) throws {
        if task.mediaPolicy == .rawMedia && !allowRawMedia { throw TaskPipelineError.rawMediaNotAuthorized }
    }

    private func mediaURLs(task: CompanionTask, directory: URL) throws -> [URL] {
        let names = task.input["mediaFiles"]?.array?.compactMap(\.string) ?? []
        let urls = names.compactMap { name -> URL? in
            let candidate = directory.appending(path: name).standardizedFileURL
            guard candidate.path.hasPrefix(directory.standardizedFileURL.path + "/"), FileManager.default.fileExists(atPath: candidate.path) else { return nil }
            return candidate
        }
        guard !urls.isEmpty else { throw TaskPipelineError.missingMedia }
        return urls
    }

    private func preprocessImage(_ url: URL) throws -> Data {
        guard let image = UIImage(contentsOfFile: url.path), let cgImage = image.cgImage else { throw TaskPipelineError.unsupportedMedia }
        return try encodeJPEG(cgImage, maximumDimension: 1_536)
    }

    private func encodeJPEG(_ image: CGImage, maximumDimension: CGFloat) throws -> Data {
        let scale = min(1, maximumDimension / max(CGFloat(image.width), CGFloat(image.height)))
        let width = max(1, Int(CGFloat(image.width) * scale))
        let height = max(1, Int(CGFloat(image.height) * scale))
        let ciImage = CIImage(cgImage: image).transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let resized = imageContext.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: width, height: height)),
              let data = UIImage(cgImage: resized).jpegData(compressionQuality: 0.82) else { throw TaskPipelineError.unsupportedMedia }
        return data
    }

    private func recognizeText(in url: URL) async throws -> String {
        guard let image = UIImage(contentsOfFile: url.path)?.cgImage else { throw TaskPipelineError.unsupportedMedia }
        return try await recognizeText(in: image)
    }

    private func recognizeText(in image: CGImage) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error { continuation.resume(throwing: error); return }
                let text = (request.results as? [VNRecognizedTextObservation])?
                    .compactMap { $0.topCandidates(1).first?.string }
                    .joined(separator: "\n") ?? ""
                continuation.resume(returning: text)
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            DispatchQueue.global(qos: .userInitiated).async {
                do { try VNImageRequestHandler(cgImage: image).perform([request]) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func extractAdaptiveFrames(_ url: URL) async throws -> [(seconds: Double, image: CGImage)] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else { throw TaskPipelineError.unsupportedMedia }
        let count = min(24, max(3, Int(ceil(duration / 5))))
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1_280, height: 1_280)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.25, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.25, preferredTimescale: 600)
        var frames: [(Double, CGImage)] = []
        for index in 0..<count {
            let seconds = duration * (Double(index) + 0.5) / Double(count)
            let response = try await generator.image(at: CMTime(seconds: seconds, preferredTimescale: 600))
            frames.append((response.actualTime.seconds, response.image))
        }
        return frames
    }

    private func decodeAudio(_ url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let capacity = AVAudioFrameCount(min(file.length, 65_536))
        guard capacity > 0 else { throw AudioDSPError.emptyAudio }
        var mono: [Float] = []
        while file.framePosition < file.length {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { throw TaskPipelineError.unsupportedMedia }
            try file.read(into: buffer)
            guard let channels = buffer.floatChannelData else { throw TaskPipelineError.unsupportedMedia }
            for frame in 0..<Int(buffer.frameLength) {
                var value: Float = 0
                for channel in 0..<Int(format.channelCount) { value += channels[channel][frame] }
                mono.append(value / Float(max(1, format.channelCount)))
            }
        }
        return (mono, format.sampleRate)
    }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self { result.append(try await transform(element)) }
        return result
    }
}
