import CryptoKit
import Foundation

enum StorageCategory: String, Codable, CaseIterable, Identifiable, Sendable {
    case models, downloads, staging, taskCache, temporary, activities, logs, resumeMetadata
    var id: String { rawValue }
}

struct StorageSnapshot: Codable, Sendable {
    let bytes: [StorageCategory: Int64]
    let freeBytes: Int64
    let reservedBytes: Int64
    var totalManagedBytes: Int64 { bytes.values.reduce(0, +) }
}

enum DeletionRequest: String, Codable, Sendable {
    case temporaryCache, temporaryFiles, partialDownloads, activities, logs
    case everythingExceptModelsAndPairings, allModels, resetCompanion
}

struct DeletionResult: Codable, Sendable {
    let request: DeletionRequest
    let reclaimedBytes: Int64
    let skippedProtectedPaths: [String]
}

enum CompanionStorageError: LocalizedError {
    case registryMissing, insufficientSpace(required: Int64, available: Int64)
    case taskLimitExceeded, invalidArtifact, protectedResource, checksumMismatch
    case activeDownloadExists
    case simulatorDownloadUnsupported
    case capacityUnavailable

    var errorDescription: String? {
        switch self {
        case .registryMissing: return String(localized: "The model registry is missing.")
        case .insufficientSpace(let required, let available):
            return String(localized: "Not enough storage: \(ByteCountFormatter.string(fromByteCount: required, countStyle: .file)) required, \(ByteCountFormatter.string(fromByteCount: available, countStyle: .file)) available.")
        case .taskLimitExceeded: return String(localized: "The task storage limit was exceeded.")
        case .invalidArtifact: return String(localized: "The downloaded artifact is invalid.")
        case .protectedResource: return String(localized: "The resource is currently in use.")
        case .checksumMismatch: return String(localized: "The file checksum does not match the trusted registry.")
        case .activeDownloadExists: return String(localized: "Another model download is active or paused.")
        case .simulatorDownloadUnsupported: return String(localized: "Model downloads are available only on a physical iPhone.")
        case .capacityUnavailable: return String(localized: "Available device storage could not be determined.")
        }
    }
}

actor StorageBudgetManager {
    static let shared = StorageBudgetManager()
    static let taskCacheLimit: Int64 = 1_073_741_824
    static let rawMediaPerTaskLimit: Int64 = 2_147_483_648
    static let logLimit: Int64 = 20 * 1_048_576
    static let resumeMetadataLimit: Int64 = 10 * 1_048_576
    static let minimumFreeReserve: Int64 = 2_147_483_648

    let applicationSupport: URL
    let modelsRoot: URL
    let blobsRoot: URL
    let manifestsRoot: URL
    let stagingRoot: URL
    let downloadsRoot: URL
    let tasksRoot: URL
    let temporaryRoot: URL
    let activitiesRoot: URL
    let logsRoot: URL
    let resumeRoot: URL

    private let fm: FileManager
    private let capacityOverride: Int64?
    private var protectedPaths = Set<String>()
    private var reservations: [UUID: Int64] = [:]

    init(fileManager: FileManager = .default, rootOverride: URL? = nil, availableCapacityOverride: Int64? = nil) {
        fm = fileManager
        capacityOverride = availableCapacityOverride
        let managedRoot = rootOverride ?? URL.applicationSupportDirectory.appending(path: "UnslothCompanion", directoryHint: .isDirectory)
        applicationSupport = managedRoot
        modelsRoot = applicationSupport.appending(path: "Models", directoryHint: .isDirectory)
        blobsRoot = modelsRoot.appending(path: "Blobs", directoryHint: .isDirectory)
        manifestsRoot = modelsRoot.appending(path: "Manifests", directoryHint: .isDirectory)
        stagingRoot = applicationSupport.appending(path: "Staging", directoryHint: .isDirectory)
        downloadsRoot = stagingRoot.appending(path: "Downloads", directoryHint: .isDirectory)
        tasksRoot = rootOverride?.appending(path: "Caches/Tasks", directoryHint: .isDirectory)
            ?? URL.cachesDirectory.appending(path: "UnslothCompanion/Tasks", directoryHint: .isDirectory)
        temporaryRoot = rootOverride?.appending(path: "Temporary", directoryHint: .isDirectory)
            ?? fileManager.temporaryDirectory.appending(path: "UnslothCompanion", directoryHint: .isDirectory)
        activitiesRoot = applicationSupport.appending(path: "Activities", directoryHint: .isDirectory)
        logsRoot = applicationSupport.appending(path: "Logs", directoryHint: .isDirectory)
        resumeRoot = applicationSupport.appending(path: "Resume", directoryHint: .isDirectory)
    }

    func bootstrap() throws {
        for directory in [applicationSupport, modelsRoot, blobsRoot, manifestsRoot, stagingRoot, downloadsRoot, tasksRoot, temporaryRoot, activitiesRoot, logsRoot, resumeRoot] {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        try excludeFromBackup(modelsRoot)
        try excludeFromBackup(stagingRoot)
        try recoverInterruptedDeletions()
        try recoverInterruptedInstallations()
        try sweepOrphanedTaskDirectories(activeTaskIDs: [])
        try enforceBudgets()
    }

    func reserve(bytes: Int64) throws -> UUID {
        let available = try availableCapacity()
        let alreadyReserved = reservations.values.reduce(0, +)
        guard available - alreadyReserved - bytes >= Self.minimumFreeReserve else {
            throw CompanionStorageError.insufficientSpace(required: bytes + Self.minimumFreeReserve, available: available - alreadyReserved)
        }
        let id = UUID(); reservations[id] = bytes; return id
    }

    func releaseReservation(_ id: UUID) { reservations[id] = nil }

    func restoreReservation(_ id: UUID, bytes: Int64) {
        reservations[id] = max(0, bytes)
    }

    func taskDirectory(taskID: UUID, expectedBytes: Int64, rawMedia: Bool) throws -> URL {
        let perTaskLimit = rawMedia ? Self.rawMediaPerTaskLimit : Self.taskCacheLimit
        guard expectedBytes <= perTaskLimit else { throw CompanionStorageError.taskLimitExceeded }
        let current = try directorySize(tasksRoot)
        guard current + expectedBytes <= Self.taskCacheLimit || rawMedia else { throw CompanionStorageError.taskLimitExceeded }
        let directory = tasksRoot.appending(path: taskID.uuidString, directoryHint: .isDirectory)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let journal = ["taskID": taskID.uuidString, "createdAt": ISO8601DateFormatter().string(from: Date())]
        try JSONSerialization.data(withJSONObject: journal, options: [.sortedKeys]).write(to: directory.appending(path: ".journal"), options: .atomic)
        protectedPaths.insert(directory.standardizedFileURL.path)
        return directory
    }

    func finishTask(_ taskID: UUID) throws {
        let directory = tasksRoot.appending(path: taskID.uuidString, directoryHint: .isDirectory)
        protectedPaths.remove(directory.standardizedFileURL.path)
        try deleteAtomically(directory)
        try deleteContents(of: temporaryRoot)
        try enforceBudgets()
    }

    func protect(_ url: URL) { protectedPaths.insert(url.standardizedFileURL.path) }
    func unprotect(_ url: URL) { protectedPaths.remove(url.standardizedFileURL.path) }
    func isProtected(_ url: URL) -> Bool {
        let path = url.standardizedFileURL.path
        return protectedPaths.contains(path) || protectedPaths.contains { $0.hasPrefix(path + "/") }
    }

    func deleteManagedItem(_ url: URL) throws {
        let path = url.standardizedFileURL.path
        let managedRoots = [applicationSupport, tasksRoot, temporaryRoot].map { $0.standardizedFileURL.path }
        guard managedRoots.contains(where: { path == $0 || path.hasPrefix($0 + "/") }) else {
            throw CompanionStorageError.invalidArtifact
        }
        guard !isProtected(url) else { throw CompanionStorageError.protectedResource }
        try deleteAtomically(url)
    }

    func snapshot() throws -> StorageSnapshot {
        let mapping: [StorageCategory: URL] = [
            .models: modelsRoot, .downloads: downloadsRoot, .taskCache: tasksRoot,
            .temporary: temporaryRoot, .activities: activitiesRoot, .logs: logsRoot,
            .resumeMetadata: resumeRoot
        ]
        var values: [StorageCategory: Int64] = [:]
        for (category, url) in mapping { values[category] = try directorySize(url) }
        values[.staging] = max(0, try directorySize(stagingRoot) - (values[.downloads] ?? 0))
        return StorageSnapshot(bytes: values, freeBytes: try availableCapacity(), reservedBytes: reservations.values.reduce(0, +))
    }

    func delete(_ request: DeletionRequest) throws -> DeletionResult {
        let before = try snapshot().totalManagedBytes
        var skipped: [String] = []
        let targets: [URL]
        switch request {
        case .temporaryCache: targets = [tasksRoot]
        case .temporaryFiles: targets = [temporaryRoot]
        case .partialDownloads: targets = [downloadsRoot, resumeRoot]
        case .activities: targets = [activitiesRoot]
        case .logs: targets = [logsRoot]
        case .everythingExceptModelsAndPairings: targets = [stagingRoot, tasksRoot, temporaryRoot, activitiesRoot, logsRoot, resumeRoot]
        case .allModels: targets = [modelsRoot]
        case .resetCompanion: targets = [applicationSupport, tasksRoot, temporaryRoot]
        }
        for target in targets {
            let protected = protectedPaths.filter { $0 == target.path || $0.hasPrefix(target.path + "/") }
            if !protected.isEmpty {
                skipped.append(contentsOf: protected)
                if protected.contains(target.standardizedFileURL.path) { continue }
                try deleteContents(of: target)
                continue
            }
            try deleteAtomically(target)
            try fm.createDirectory(at: target, withIntermediateDirectories: true)
        }
        if targets.contains(modelsRoot) { try excludeFromBackup(modelsRoot) }
        if targets.contains(stagingRoot) { try excludeFromBackup(stagingRoot) }
        let after = try snapshot().totalManagedBytes
        return DeletionResult(request: request, reclaimedBytes: max(0, before - after), skippedProtectedPaths: skipped.sorted())
    }

    func sweepOrphanedTaskDirectories(activeTaskIDs: Set<UUID>) throws {
        for child in try contents(of: tasksRoot) where child.hasDirectoryPath {
            guard let id = UUID(uuidString: child.lastPathComponent), activeTaskIDs.contains(id) else {
                if !protectedPaths.contains(child.path) { try deleteAtomically(child) }
                continue
            }
        }
    }

    func hash(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 4 * 1_048_576), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func directorySize(_ url: URL) throws -> Int64 {
        guard fm.fileExists(atPath: url.path) else { return 0 }
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileAllocatedSizeKey, .totalFileAllocatedSizeKey]
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: Array(keys), options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            let values = try file.resourceValues(forKeys: keys)
            if values.isRegularFile == true { total += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0) }
        }
        return total
    }

    private func enforceBudgets() throws {
        if try directorySize(logsRoot) > Self.logLimit { _ = try delete(.logs) }
        if try directorySize(resumeRoot) > Self.resumeMetadataLimit { _ = try delete(.partialDownloads) }
    }

    private func deleteContents(of directory: URL) throws {
        for child in try contents(of: directory) where !isProtected(child) {
            try deleteAtomically(child)
        }
    }

    private func availableCapacity() throws -> Int64 {
        if let capacityOverride { return max(0, capacityOverride) }
        let values = try applicationSupport.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey
        ])
        if let important = values.volumeAvailableCapacityForImportantUsage, important > 0 {
            return Int64(important)
        }
        if let ordinary = values.volumeAvailableCapacity, ordinary > 0 {
            return Int64(ordinary)
        }
        let attributes = try fm.attributesOfFileSystem(forPath: applicationSupport.path)
        if let free = attributes[.systemFreeSize] as? NSNumber, free.int64Value > 0 {
            return free.int64Value
        }
        throw CompanionStorageError.capacityUnavailable
    }

    private func excludeFromBackup(_ url: URL) throws {
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var mutable = url; try mutable.setResourceValues(values)
    }

    private func deleteAtomically(_ url: URL) throws {
        guard fm.fileExists(atPath: url.path) else { return }
        if isProtected(url) { throw CompanionStorageError.protectedResource }
        let tombstone = url.deletingLastPathComponent().appending(path: ".deleting-\(UUID().uuidString)")
        try fm.moveItem(at: url, to: tombstone)
        try fm.removeItem(at: tombstone)
    }

    private func recoverInterruptedDeletions() throws {
        let roots = [applicationSupport, tasksRoot.deletingLastPathComponent(), temporaryRoot.deletingLastPathComponent()]
        var tombstones: [URL] = []
        for root in roots where fm.fileExists(atPath: root.path) {
            guard let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey], options: []) else { continue }
            for case let item as URL in enumerator where item.lastPathComponent.hasPrefix(".deleting-") {
                tombstones.append(item)
                if (try? item.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    enumerator.skipDescendants()
                }
            }
        }
        for tombstone in tombstones.sorted(by: { $0.pathComponents.count > $1.pathComponents.count }) {
            try? fm.removeItem(at: tombstone)
        }
    }

    private func recoverInterruptedInstallations() throws {
        for item in try contents(of: blobsRoot) where item.lastPathComponent.hasPrefix(".installing-") {
            try deleteAtomically(item)
        }
    }

    private func contents(of url: URL) throws -> [URL] {
        guard fm.fileExists(atPath: url.path) else { return [] }
        return try fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey])
    }
}
