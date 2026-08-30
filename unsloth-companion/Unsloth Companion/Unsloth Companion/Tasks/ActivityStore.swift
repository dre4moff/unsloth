import Foundation

enum ActivityState: String, Codable, Sendable {
    case accepted, running, completed, failed, cancelled
}

struct CompanionActivity: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    let kind: CompanionTaskKind
    let startedAt: Date
    var finishedAt: Date?
    var state: ActivityState
    var detail: String
    var durationMS: Int?
    var tokens: Int?
}

actor ActivityStore {
    static let shared = ActivityStore(storage: .shared)
    private let storage: StorageBudgetManager
    private let fm = FileManager.default

    init(storage: StorageBudgetManager) { self.storage = storage }

    func list() async throws -> [CompanionActivity] {
        let root = storage.activitiesRoot
        guard fm.fileExists(atPath: root.path) else { return [] }
        let files = try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).filter { $0.pathExtension == "json" }
        return files.compactMap { try? ProtocolCoding.decoder.decode(CompanionActivity.self, from: Data(contentsOf: $0)) }
            .sorted { $0.startedAt > $1.startedAt }
    }

    func save(_ activity: CompanionActivity) async throws {
        let root = storage.activitiesRoot
        let url = root.appending(path: activity.id.uuidString + ".json")
        try ProtocolCoding.encoder.encode(activity).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        try await prune()
    }

    func delete(id: UUID) async throws {
        let root = storage.activitiesRoot
        let url = root.appending(path: id.uuidString + ".json")
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
    }

    func deleteAll() async throws { _ = try await storage.delete(.activities) }

    private func prune() async throws {
        let values = try await list()
        let cutoff = Date().addingTimeInterval(-30 * 24 * 60 * 60)
        for value in values.enumerated() where value.offset >= 2_000 || value.element.startedAt < cutoff {
            try await delete(id: value.element.id)
        }
    }
}
