import CryptoKit
import Foundation

enum CompanionProtocol {
    static let version = 1
    static let serviceType = "_unsloth-cp._tcp"
    static let heartbeatInterval: TimeInterval = 5
    static let defaultLease: TimeInterval = 30
}

enum CompanionMessageType: String, Codable, Sendable {
    case hello, challenge, challengeResponse = "challenge_response"
    case pairingOffer = "pairing_offer", pairingConfirm = "pairing_confirm", pairingResult = "pairing_result"
    case capabilities, heartbeat, clientDraining = "client_draining"
    case taskSubmit = "task_submit", taskAccepted = "task_accepted", taskProgress = "task_progress"
    case taskToken = "task_token", leaseRenew = "lease_renew", taskCancel = "task_cancel"
    case taskCancelled = "task_cancelled", taskCompleted = "task_completed", taskFailed = "task_failed"
    case blobBegin = "blob_begin", blobEnd = "blob_end"
}

struct CompanionEnvelope: Codable, Sendable {
    let protocolVersion: Int
    let messageID: UUID
    let sentAt: Date
    let type: CompanionMessageType
    let payload: JSONValue

    init(type: CompanionMessageType, payload: JSONValue) {
        protocolVersion = CompanionProtocol.version
        messageID = UUID()
        sentAt = Date()
        self.type = type
        self.payload = payload
    }
}

enum JSONValue: Codable, Sendable, Equatable {
    case string(String), number(Double), bool(Bool), object([String: JSONValue]), array([JSONValue]), null

    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let object = try? value.decode([String: JSONValue].self) { self = .object(object) }
        else if let array = try? value.decode([JSONValue].self) { self = .array(array) }
        else if let bool = try? value.decode(Bool.self) { self = .bool(bool) }
        else if let number = try? value.decode(Double.self) { self = .number(number) }
        else { self = .string(try value.decode(String.self)) }
    }

    func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .string(let string): try value.encode(string)
        case .number(let number): try value.encode(number)
        case .bool(let bool): try value.encode(bool)
        case .object(let object): try value.encode(object)
        case .array(let array): try value.encode(array)
        case .null: try value.encodeNil()
        }
    }

    var object: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    var string: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    var number: Double? {
        guard case .number(let value) = self else { return nil }
        return value
    }

    var bool: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    var array: [JSONValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    static func encode<T: Encodable>(_ value: T) throws -> JSONValue {
        let data = try ProtocolCoding.encoder.encode(value)
        return try ProtocolCoding.decoder.decode(JSONValue.self, from: data)
    }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        let data = try ProtocolCoding.encoder.encode(self)
        return try ProtocolCoding.decoder.decode(T.self, from: data)
    }
}

enum CompanionTaskKind: String, Codable, CaseIterable, Identifiable, Sendable {
    case subagent, classification, summary, contextCompression = "context_compression", extraction, verification
    case reranking, lightweightPlanning = "lightweight_planning", vision, ocr
    case videoSummary = "video_summary", audioTranscription = "audio_transcription"
    case audioAnalysis = "audio_analysis", dsp
    var id: String { rawValue }
}

enum MediaPolicy: String, Codable, CaseIterable, Sendable {
    case semanticOnly = "semantic_only", derivedMedia = "derived_media", rawMedia = "raw_media"
}

struct CompanionTaskEnvelope: Codable, Identifiable, Sendable {
    let taskID: UUID
    let parentTaskID: UUID?
    let shardIndex: Int?
    let shardCount: Int?
    let idempotencyKey: String
    let kind: CompanionTaskKind
    let priority: Int
    let timeoutSeconds: TimeInterval
    let leaseSeconds: TimeInterval
    let mediaPolicy: MediaPolicy
    let input: [String: JSONValue]
    let resultSchema: [String: JSONValue]
    var id: UUID { taskID }
}

typealias CompanionTask = CompanionTaskEnvelope

struct CompanionTaskResult: Codable, Sendable {
    let taskID: UUID
    let result: [String: JSONValue]
    let sha256: String
    let durationMS: Int
    let tokensGenerated: Int
}

enum CompanionErrorCode: String, Codable, Sendable {
    case protocolIncompatible = "protocol_incompatible", notPaired = "not_paired"
    case authenticationFailed = "authentication_failed", capabilityUnavailable = "capability_unavailable"
    case modelMissing = "model_missing", checksumFailed = "checksum_failed"
    case storageInsufficient = "storage_insufficient", memoryPressure = "memory_pressure"
    case thermalPressure = "thermal_pressure", permissionDenied = "permission_denied"
    case payloadCorrupt = "payload_corrupt", leaseExpired = "lease_expired", timeout, cancelled
    case connectionLost = "connection_lost", runtimeError = "runtime_error"
}

struct CompanionTaskFailure: Codable, Error, Sendable {
    let taskID: UUID
    let code: CompanionErrorCode
    let message: String
    let retryable: Bool
}

struct CompanionCapabilities: Codable, Sendable, Equatable {
    let protocolVersion: Int
    let deviceID: UUID
    let deviceName: String
    let runtimeVersion: String
    let taskKinds: Set<CompanionTaskKind>
    let selectedModelID: String?
    let supportsVision: Bool
    let supportsAudio: Bool
    let contextSize: Int
    let maxRawMediaBytes: Int64
}

struct TaskLease: Codable, Sendable {
    let taskID: UUID
    var expiresAt: Date
    var renewal: Int
    mutating func renew(seconds: TimeInterval) { expiresAt = Date().addingTimeInterval(seconds); renewal += 1 }
    var isExpired: Bool { expiresAt <= Date() }
}

enum TaskCancellationReason: String, Codable, Sendable {
    case user, companionDisabled = "companion_disabled", desktopRequest = "desktop_request"
    case backgrounded, leaseExpired = "lease_expired", thermalPressure = "thermal_pressure"
    case memoryPressure = "memory_pressure", lowBattery = "low_battery", connectionLost = "connection_lost"
}

struct BlobDescriptor: Codable, Sendable {
    let blobID: UUID
    let taskID: UUID
    let name: String
    let mimeType: String
    let size: Int64
    let sha256: String
}

enum ProtocolCoding {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()
    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { value in
            let container = try value.singleValueContainer()
            let string = try container.decode(String.self)
            let fractional = ISO8601DateFormatter()
            fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = fractional.date(from: string) { return date }
            let plain = ISO8601DateFormatter()
            plain.formatOptions = [.withInternetDateTime]
            if let date = plain.date(from: string) { return date }
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid ISO-8601 date")
        }
        return decoder
    }()

    static func digest<T: Encodable>(_ value: T) throws -> String {
        SHA256.hash(data: try encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
    }
}
