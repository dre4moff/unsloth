import Combine
@preconcurrency import Foundation
import Network

enum ModelDownloadPhase: String, Codable, Sendable {
    case idle, reserving, waitingForWiFi, waitingForNetwork, downloadingModel, downloadingMMProj, paused, verifying, installing, completed, failed
}

struct ModelDownloadStatus: Codable, Sendable, Equatable {
    let modelID: String
    var phase: ModelDownloadPhase
    var completedBytes: Int64
    let totalBytes: Int64
    var errorDescription: String?

    var progress: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1, Double(completedBytes) / Double(totalBytes))
    }
}

private struct DownloadJournal: Codable {
    let modelID: String
    var phase: ModelDownloadPhase
    var modelStagingPath: String?
    var resumeData: Data?
    var completedBytes: Int64
    var reservationID: UUID?
}

@MainActor
final class ModelDownloadManager: NSObject, ObservableObject {
    static let shared = ModelDownloadManager()
    static var supportsModelDownloads: Bool {
        #if targetEnvironment(simulator)
        false
        #else
        true
        #endif
    }

    @Published private(set) var status: ModelDownloadStatus?
    @Published private(set) var installedModels: [InstalledModel] = []

    private let storage = StorageBudgetManager.shared
    private let store = ModelStore.shared
    private var registry: ModelRegistryDocument?
    private var journal: DownloadJournal?
    private var task: URLSessionDownloadTask?
    private var backgroundCompletionHandler: (() -> Void)?
    private let pathMonitor = NWPathMonitor()
    private let pathMonitorQueue = DispatchQueue(label: "com.OvenTeam.app.Unsloth-Companion.model-network")
    private var networkAvailable = true
    private var usingCellular = false
    private var allowsCellularDownloads = false
    private var policyRestartToken: UUID?

    private lazy var wifiSession = makeSession(
        identifier: "com.OvenTeam.app.Unsloth-Companion.models",
        allowsCellularAccess: false
    )
    private lazy var cellularSession = makeSession(
        identifier: "com.OvenTeam.app.Unsloth-Companion.models.cellular",
        allowsCellularAccess: true
    )

    override private init() {
        super.init()
        pathMonitor.pathUpdateHandler = { [weak self] path in
            let available = path.status == .satisfied
            let cellular = path.usesInterfaceType(.cellular)
            Task { @MainActor [weak self] in
                self?.networkPathChanged(available: available, cellular: cellular)
            }
        }
        pathMonitor.start(queue: pathMonitorQueue)
    }

    deinit { pathMonitor.cancel() }

    private func makeSession(identifier: String, allowsCellularAccess: Bool) -> URLSession {
        let configuration = URLSessionConfiguration.background(withIdentifier: identifier)
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.allowsCellularAccess = allowsCellularAccess
        configuration.allowsExpensiveNetworkAccess = allowsCellularAccess
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    func setAllowsCellularDownloads(_ allowed: Bool) {
        guard allowsCellularDownloads != allowed else {
            refreshVisibleNetworkPhase()
            return
        }
        allowsCellularDownloads = allowed
        refreshVisibleNetworkPhase()
        restartActiveTaskForNetworkPolicy()
    }

    func bootstrap() async throws {
        try await storage.bootstrap()
        registry = try ModelRegistry.load()
        installedModels = try await store.installedModels()
        #if targetEnvironment(simulator)
        _ = try? await storage.delete(.partialDownloads)
        journal = nil
        status = nil
        return
        #else
        try await restoreJournal()
        _ = wifiSession
        _ = cellularSession
        await restoreBackgroundTask()
        #endif
    }

    func start(modelID: String) async throws {
        guard Self.supportsModelDownloads else { throw CompanionStorageError.simulatorDownloadUnsupported }
        guard task == nil, journal == nil else { throw CompanionStorageError.activeDownloadExists }
        guard let model = registry?.models.first(where: { $0.id == modelID }) else { throw CompanionStorageError.registryMissing }
        let total = model.model.size + model.mmproj.size
        status = ModelDownloadStatus(modelID: modelID, phase: .reserving, completedBytes: 0, totalBytes: total)
        let reservation = try await storage.reserve(bytes: total)
        do {
            journal = DownloadJournal(modelID: modelID, phase: .downloadingModel, modelStagingPath: nil, resumeData: nil, completedBytes: 0, reservationID: reservation)
            try await persistJournal()
            status = ModelDownloadStatus(modelID: modelID, phase: .downloadingModel, completedBytes: 0, totalBytes: total)
            startTask(url: model.url(for: model.model))
        } catch {
            await storage.releaseReservation(reservation)
            journal = nil
            _ = try? await storage.delete(.partialDownloads)
            status = ModelDownloadStatus(modelID: modelID, phase: .failed, completedBytes: 0, totalBytes: total, errorDescription: error.localizedDescription)
            throw error
        }
    }

    func pause() {
        guard var journal else { return }
        policyRestartToken = nil
        let active = task
        task = nil
        journal.phase = .paused
        self.journal = journal
        if var status = self.status { status.phase = .paused; self.status = status }
        Task { try? await self.persistJournal() }
        if let active {
            cancelForPause(active, modelID: journal.modelID)
        } else {
            Task { await self.findAndCancelBackgroundTaskForPause(modelID: journal.modelID) }
        }
    }

    private func cancelForPause(_ active: URLSessionDownloadTask, modelID: String) {
        active.cancel { [weak self] resumeData in
            Task { @MainActor in
                guard let self, self.task == nil, var journal = self.journal,
                      journal.modelID == modelID, journal.phase == .paused else { return }
                journal.resumeData = resumeData
                self.journal = journal
                try? await self.persistJournal()
            }
        }
    }

    private func findAndCancelBackgroundTaskForPause(modelID: String) async {
        let wifiTasks = await wifiSession.allTasks.compactMap { $0 as? URLSessionDownloadTask }
        let cellularTasks = await cellularSession.allTasks.compactMap { $0 as? URLSessionDownloadTask }
        guard journal?.modelID == modelID, journal?.phase == .paused else { return }
        let downloads = wifiTasks + cellularTasks
        for extra in downloads.dropFirst() { extra.cancel() }
        if let active = downloads.first { cancelForPause(active, modelID: modelID) }
    }

    func resume() async throws {
        guard task == nil, var journal, journal.phase == .paused else { return }
        let resumeData = journal.resumeData
        journal.resumeData = nil
        journal.phase = journal.modelStagingPath == nil ? .downloadingModel : .downloadingMMProj
        self.journal = journal
        if var status { status.phase = journal.phase; self.status = status }
        try await persistJournal()
        if let resumeData {
            let next = downloadSession.downloadTask(withResumeData: resumeData)
            task = next
            next.resume()
        } else if let model = registry?.models.first(where: { $0.id == journal.modelID }) {
            startTask(url: model.url(for: journal.modelStagingPath == nil ? model.model : model.mmproj))
        }
    }

    func cancelAndDelete() async {
        policyRestartToken = nil
        task?.cancel(); task = nil
        if let reservationID = journal?.reservationID { await storage.releaseReservation(reservationID) }
        journal = nil; status = nil
        _ = try? await storage.delete(.partialDownloads)
    }

    func setBackgroundCompletionHandler(_ handler: @escaping () -> Void) {
        backgroundCompletionHandler = handler
    }

    private func startTask(url: URL) {
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.allowsCellularAccess = allowsCellularDownloads
        request.timeoutInterval = 7 * 24 * 60 * 60
        let next = downloadSession.downloadTask(with: request)
        task = next
        next.resume()
        refreshVisibleNetworkPhase()
    }

    private var downloadSession: URLSession {
        allowsCellularDownloads ? cellularSession : wifiSession
    }

    private func networkPathChanged(available: Bool, cellular: Bool) {
        networkAvailable = available
        usingCellular = cellular
        refreshVisibleNetworkPhase()
    }

    private func refreshVisibleNetworkPhase() {
        guard var status, let journal else { return }
        guard [.downloadingModel, .downloadingMMProj].contains(journal.phase) else { return }
        if !networkAvailable {
            status.phase = .waitingForNetwork
        } else if usingCellular && !allowsCellularDownloads {
            status.phase = .waitingForWiFi
        } else {
            status.phase = journal.phase
        }
        self.status = status
    }

    private func restartActiveTaskForNetworkPolicy() {
        guard let active = task, let journal else { return }
        let token = UUID()
        policyRestartToken = token
        task = nil
        active.cancel { [weak self] resumeData in
            Task { @MainActor in
                guard let self, self.policyRestartToken == token,
                      self.journal?.modelID == journal.modelID else { return }
                self.policyRestartToken = nil
                var updated = journal
                updated.resumeData = nil
                self.journal = updated
                try? await self.persistJournal()
                if let resumeData, self.allowsCellularDownloads {
                    let next = self.downloadSession.downloadTask(withResumeData: resumeData)
                    self.task = next
                    next.resume()
                } else if let model = self.registry?.models.first(where: { $0.id == journal.modelID }) {
                    self.startTask(url: model.url(for: journal.modelStagingPath == nil ? model.model : model.mmproj))
                }
                self.refreshVisibleNetworkPhase()
            }
        }
    }

    private func persistJournal() async throws {
        let root = storage.resumeRoot
        let url = root.appending(path: "active-model-download.json")
        if let journal {
            try ProtocolCoding.encoder.encode(journal).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        } else if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    private func restoreJournal() async throws {
        let root = storage.resumeRoot
        let url = root.appending(path: "active-model-download.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        var restored = try ProtocolCoding.decoder.decode(DownloadJournal.self, from: Data(contentsOf: url))
        restored.phase = .paused
        journal = restored
        if let model = registry?.models.first(where: { $0.id == restored.modelID }) {
            if let reservationID = restored.reservationID {
                await storage.restoreReservation(reservationID, bytes: model.model.size + model.mmproj.size)
            }
            status = ModelDownloadStatus(
                modelID: restored.modelID,
                phase: .paused,
                completedBytes: restored.completedBytes,
                totalBytes: model.model.size + model.mmproj.size
            )
        } else {
            await cancelAndDelete()
        }
    }

    private func restoreBackgroundTask() async {
        guard journal != nil else { return }
        let wifiDownloads = await wifiSession.allTasks.compactMap { $0 as? URLSessionDownloadTask }
        let cellularDownloads = await cellularSession.allTasks.compactMap { $0 as? URLSessionDownloadTask }
        let downloads = wifiDownloads + cellularDownloads
        guard let active = downloads.first else { return }
        for extra in downloads.dropFirst() { extra.cancel() }
        task = active
        if var journal {
            journal.phase = journal.modelStagingPath == nil ? .downloadingModel : .downloadingMMProj
            self.journal = journal
            if var status { status.phase = journal.phase; self.status = status }
            try? await persistJournal()
        }
        let restoredAllowsCellular = active.currentRequest?.allowsCellularAccess
            ?? active.originalRequest?.allowsCellularAccess
            ?? false
        active.resume()
        if restoredAllowsCellular != allowsCellularDownloads {
            restartActiveTaskForNetworkPolicy()
        } else {
            refreshVisibleNetworkPhase()
        }
    }

    private func pauseAfterNetworkError(_ error: Error, resumeData: Data?) async {
        task = nil
        guard var journal else { return }
        journal.phase = .paused
        journal.resumeData = resumeData
        self.journal = journal
        if var status {
            status.phase = .paused
            status.errorDescription = error.localizedDescription
            self.status = status
        }
        try? await persistJournal()
    }

    private func consumeDownloadedFile(_ location: URL, from completedTask: URLSessionDownloadTask) async {
        defer {
            if FileManager.default.fileExists(atPath: location.path) {
                try? FileManager.default.removeItem(at: location)
            }
        }
        guard task === completedTask else {
            return
        }
        guard var journal, let model = registry?.models.first(where: { $0.id == journal.modelID }) else { return }
        task = nil
        do {
            let root = storage.downloadsRoot
            let staging = root.appending(path: journal.modelID, directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            if journal.modelStagingPath == nil {
                let target = staging.appending(path: "model.part")
                try replaceDownloadedFile(location, target: target)
                journal.modelStagingPath = target.path
                journal.completedBytes = model.model.size
                journal.phase = .downloadingMMProj
                self.journal = journal
                status = ModelDownloadStatus(modelID: model.id, phase: .downloadingMMProj, completedBytes: model.model.size, totalBytes: model.model.size + model.mmproj.size)
                try await persistJournal()
                startTask(url: model.url(for: model.mmproj))
                return
            }
            let mmproj = staging.appending(path: "mmproj.part")
            try replaceDownloadedFile(location, target: mmproj)
            journal.phase = .verifying
            journal.completedBytes = model.model.size + model.mmproj.size
            self.journal = journal
            status = ModelDownloadStatus(modelID: model.id, phase: .verifying, completedBytes: journal.completedBytes, totalBytes: journal.completedBytes)
            try await persistJournal()
            guard let modelPath = journal.modelStagingPath else { throw CompanionStorageError.invalidArtifact }
            if var status { status.phase = .installing; self.status = status }
            _ = try await store.install(registryModel: model, modelStagingURL: URL(fileURLWithPath: modelPath), mmprojStagingURL: mmproj)
            installedModels = try await store.installedModels()
            if let reservationID = journal.reservationID { await storage.releaseReservation(reservationID) }
            self.journal = nil
            try await persistJournal()
            try? FileManager.default.removeItem(at: staging)
            status = ModelDownloadStatus(modelID: model.id, phase: .completed, completedBytes: model.model.size + model.mmproj.size, totalBytes: model.model.size + model.mmproj.size)
        } catch {
            await fail(error)
        }
    }

    private func fail(_ error: Error) async {
        task = nil
        if shouldDiscardArtifacts(after: error) {
            let failedStatus = status.map {
                var value = $0
                value.phase = .failed
                value.errorDescription = error.localizedDescription
                return value
            }
            if let reservationID = journal?.reservationID { await storage.releaseReservation(reservationID) }
            journal = nil
            _ = try? await storage.delete(.partialDownloads)
            status = failedStatus
            return
        }
        if var journal { journal.phase = .failed; self.journal = journal }
        if var status { status.phase = .failed; status.errorDescription = error.localizedDescription; self.status = status }
        try? await persistJournal()
    }

    private func shouldDiscardArtifacts(after error: Error) -> Bool {
        guard let storageError = error as? CompanionStorageError else { return false }
        switch storageError {
        case .checksumMismatch, .invalidArtifact: return true
        default: return false
        }
    }

    private func replaceDownloadedFile(_ source: URL, target: URL) throws {
        if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
        try FileManager.default.moveItem(at: source, to: target)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var mutable = target; try mutable.setResourceValues(values)
    }
}

extension ModelDownloadManager: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            // URLSession owns `location` only for the duration of this delegate
            // callback. Claim it synchronously before returning; otherwise the
            // asynchronous MainActor hop can observe a deleted CFNetwork temp file.
            let temporaryRoot = FileManager.default.temporaryDirectory
                .appending(path: "UnslothCompanion", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
            let claimed = temporaryRoot.appending(path: "model-download-\(UUID().uuidString).tmp")
            try FileManager.default.moveItem(at: location, to: claimed)
            Task { @MainActor [weak self] in
                await self?.consumeDownloadedFile(claimed, from: downloadTask)
            }
        } catch {
            Task { @MainActor [weak self] in
                guard let self, self.task === downloadTask else { return }
                await self.fail(error)
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        Task { @MainActor [weak self] in
            guard let self, self.task === downloadTask,
                  var journal = self.journal, var status = self.status else { return }
            let prior = journal.modelStagingPath == nil ? 0 : (self.registry?.models.first(where: { $0.id == journal.modelID })?.model.size ?? 0)
            let observedBytes = prior + totalBytesWritten
            status.completedBytes = max(status.completedBytes, observedBytes)
            journal.completedBytes = max(journal.completedBytes, observedBytes)
            self.journal = journal
            status.phase = journal.phase
            self.status = status
            self.refreshVisibleNetworkPhase()
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        let nsError = error as NSError
        if nsError.code == NSURLErrorCancelled { return }
        let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        Task { @MainActor [weak self] in
            guard let self, self.task === task else { return }
            await self.pauseAfterNetworkError(error, resumeData: resumeData)
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor [weak self] in
            self?.backgroundCompletionHandler?()
            self?.backgroundCompletionHandler = nil
        }
    }
}
