import Foundation

struct CompanionSettings: Codable, Sendable, Equatable {
    var serviceEnabled = true
    var keepScreenAwake = true
    var guardScreenEnabled = true
    var allowCellularDownloads = false
    var mediaPolicy: MediaPolicy = .semanticOnly
    var allowRawMedia = false
    var computeMode: RuntimeComputeMode = .automatic

    private enum CodingKeys: String, CodingKey {
        case serviceEnabled, keepScreenAwake, guardScreenEnabled, allowCellularDownloads
        case mediaPolicy, allowRawMedia, computeMode
    }

    init(
        serviceEnabled: Bool = true,
        keepScreenAwake: Bool = true,
        guardScreenEnabled: Bool = true,
        allowCellularDownloads: Bool = false,
        mediaPolicy: MediaPolicy = .semanticOnly,
        allowRawMedia: Bool = false,
        computeMode: RuntimeComputeMode = .automatic
    ) {
        self.serviceEnabled = serviceEnabled
        self.keepScreenAwake = keepScreenAwake
        self.guardScreenEnabled = guardScreenEnabled
        self.allowCellularDownloads = allowCellularDownloads
        self.mediaPolicy = mediaPolicy
        self.allowRawMedia = allowRawMedia
        self.computeMode = computeMode
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        serviceEnabled = try values.decodeIfPresent(Bool.self, forKey: .serviceEnabled) ?? true
        keepScreenAwake = try values.decodeIfPresent(Bool.self, forKey: .keepScreenAwake) ?? true
        guardScreenEnabled = try values.decodeIfPresent(Bool.self, forKey: .guardScreenEnabled) ?? true
        allowCellularDownloads = try values.decodeIfPresent(Bool.self, forKey: .allowCellularDownloads) ?? false
        mediaPolicy = try values.decodeIfPresent(MediaPolicy.self, forKey: .mediaPolicy) ?? .semanticOnly
        allowRawMedia = try values.decodeIfPresent(Bool.self, forKey: .allowRawMedia) ?? false
        computeMode = try values.decodeIfPresent(RuntimeComputeMode.self, forKey: .computeMode) ?? .automatic
    }
}

struct CompanionDevice: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    var name: String
    var enabled: Bool
    var selected: Bool
    var connected: Bool
    var score: Double
    var capabilities: CompanionCapabilities?
}

enum CompanionConnectionMode: String, Codable, CaseIterable, Sendable {
    case automaticBest = "automatic_best"
    case multiple = "multiple"
}
