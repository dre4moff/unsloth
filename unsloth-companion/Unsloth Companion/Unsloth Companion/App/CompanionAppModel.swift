import Combine
import Foundation

@MainActor
final class CompanionAppModel: ObservableObject {
    @Published private(set) var registryModels: [RegistryModel] = []
    @Published private(set) var installedModels: [InstalledModel] = []
    @Published private(set) var recoverableModelBytes: [String: Int64] = [:]
    @Published private(set) var activities: [CompanionActivity] = []
    @Published private(set) var storageSnapshot: StorageSnapshot?
    @Published private(set) var pairedDesktops: [PairedDesktop] = []
    @Published private(set) var loadedModelID: String?
    @Published private(set) var lastDeletionResult: DeletionResult?
    @Published private(set) var hasCompletedOnboarding: Bool
    @Published var presentedError: String?
    @Published var isBusy = false

    let service = CompanionServiceModel()
    let downloads = ModelDownloadManager.shared
    private let storage = StorageBudgetManager.shared
    private let models = ModelStore.shared
    private let activityStore = ActivityStore.shared
    private let identity = CompanionIdentityStore.shared

    init() {
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("-ShowCompanionOnboarding") {
            hasCompletedOnboarding = false
        } else {
            hasCompletedOnboarding = arguments.contains("-SkipCompanionOnboarding") || UserDefaults.standard.bool(forKey: "companion-onboarding-completed")
        }
    }

    func bootstrap() async {
        isBusy = true
        defer { isBusy = false }
        do {
            downloads.setAllowsCellularDownloads(service.settings.allowCellularDownloads)
            try await downloads.bootstrap()
            registryModels = try ModelRegistry.load().models
            try await refresh()
            if hasCompletedOnboarding { service.start() }
        } catch { presentedError = error.localizedDescription }
    }

    func refresh() async throws {
        installedModels = try await models.installedModels()
        var recoverable: [String: Int64] = [:]
        for model in installedModels { recoverable[model.id] = try await models.recoverableBytes(deleting: model) }
        recoverableModelBytes = recoverable
        activities = try await activityStore.list()
        storageSnapshot = try await storage.snapshot()
        pairedDesktops = try await identity.pairedDesktops()
    }

    func startDownload(_ model: RegistryModel) async {
        do { try await downloads.start(modelID: model.id); try await refresh() }
        catch { presentedError = error.localizedDescription }
    }

    func load(_ model: InstalledModel) async {
        isBusy = true; defer { isBusy = false }
        do {
            try await service.setLoadedModel(model)
            loadedModelID = model.id
        } catch { presentedError = error.localizedDescription }
    }

    func unloadModel() async {
        do { try await service.setLoadedModel(nil); loadedModelID = nil }
        catch { presentedError = error.localizedDescription }
    }

    func importGGUF(urls: [URL]) async {
        guard let modelURL = urls.first(where: { !$0.lastPathComponent.lowercased().contains("mmproj") }) else { return }
        let mmproj = urls.first(where: { $0.lastPathComponent.lowercased().contains("mmproj") })
        let scoped = urls.filter { $0.startAccessingSecurityScopedResource() }
        defer { scoped.forEach { $0.stopAccessingSecurityScopedResource() } }
        isBusy = true; defer { isBusy = false }
        do {
            _ = try await models.importGGUF(from: modelURL, displayName: modelURL.deletingPathExtension().lastPathComponent, mmproj: mmproj)
            try await refresh()
        } catch { presentedError = error.localizedDescription }
    }

    func deleteModel(_ model: InstalledModel) async {
        isBusy = true; defer { isBusy = false }
        do {
            if model.id == loadedModelID { await unloadModel() }
            try await models.deleteModel(id: model.id)
            try await refresh()
        } catch { presentedError = error.localizedDescription }
    }

    func deleteAllModels() async {
        isBusy = true; defer { isBusy = false }
        do { await unloadModel(); try await models.deleteAllModels(); try await refresh() }
        catch { presentedError = error.localizedDescription }
    }

    func deleteStorage(_ request: DeletionRequest) async {
        isBusy = true; defer { isBusy = false }
        do { lastDeletionResult = try await storage.delete(request); try await refresh() }
        catch { presentedError = error.localizedDescription }
    }

    func deleteActivity(_ activity: CompanionActivity) async {
        do { try await activityStore.delete(id: activity.id); try await refresh() }
        catch { presentedError = error.localizedDescription }
    }

    func revoke(_ desktop: PairedDesktop) async {
        do { try await identity.revoke(desktopID: desktop.id); try await refresh() }
        catch { presentedError = error.localizedDescription }
    }

    func completeOnboarding() {
        UserDefaults.standard.set(true, forKey: "companion-onboarding-completed")
        hasCompletedOnboarding = true
        service.start()
    }

    func setAllowsCellularDownloads(_ allowed: Bool) {
        service.settings.allowCellularDownloads = allowed
        downloads.setAllowsCellularDownloads(allowed)
    }

    func resetCompanion() async {
        isBusy = true; defer { isBusy = false }
        await service.stop()
        await unloadModel()
        do {
            await downloads.cancelAndDelete()
            try await identity.reset()
            _ = try await storage.delete(.resetCompanion)
            UserDefaults.standard.removeObject(forKey: "companion-settings")
            UserDefaults.standard.removeObject(forKey: "companion-onboarding-completed")
            service.settings = CompanionSettings()
            downloads.setAllowsCellularDownloads(false)
            hasCompletedOnboarding = false
            try await downloads.bootstrap()
            try await refresh()
        } catch { presentedError = error.localizedDescription }
    }
}
