import SwiftUI

struct StorageView: View {
    @ObservedObject var appModel: CompanionAppModel
    @State private var pendingDeletion: DeletionRequest?
    @State private var resetStage = 0

    var body: some View {
        NavigationStack {
            List {
                Section("Managed storage") {
                    if let snapshot = appModel.storageSnapshot {
                        ForEach(StorageCategory.allCases) { category in
                            LabeledContent(label(category), value: format(snapshot.bytes[category] ?? 0))
                        }
                        LabeledContent("Reserved", value: format(snapshot.reservedBytes))
                        LabeledContent("Free", value: format(snapshot.freeBytes))
                        LabeledContent("Managed total", value: format(snapshot.totalManagedBytes))
                    } else { ProgressView() }
                }
                Section("Cache and temporary data") {
                    deletionButton("Clear temporary cache", .temporaryCache)
                    deletionButton("Delete temporary files", .temporaryFiles)
                    deletionButton("Delete partial downloads", .partialDownloads)
                    deletionButton("Clear logs", .logs)
                    deletionButton("Clear all except models and pairings", .everythingExceptModelsAndPairings)
                    if appModel.service.currentTaskID != nil {
                        Button("Cancel task and clean", role: .destructive) {
                            Task {
                                await appModel.service.cancelCurrentTask()
                                await appModel.deleteStorage(.temporaryCache)
                            }
                        }
                    }
                }
                Section("History") { deletionButton("Clear all activity", .activities) }
                if let result = appModel.lastDeletionResult {
                    Section("Last cleanup") {
                        LabeledContent("Recovered", value: format(result.reclaimedBytes))
                        if !result.skippedProtectedPaths.isEmpty { Text("Protected files were skipped because a task is using them.").foregroundStyle(.orange) }
                    }
                }
                Section("Reset") {
                    Button("Reset Companion", role: .destructive) { resetStage = 1 }
                    Text("Deletes models, cache, activity, preferences, device identity, and every pairing.").font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Storage")
            .refreshable { try? await appModel.refresh() }
            .alert("Confirm cleanup", isPresented: Binding(get: { pendingDeletion != nil }, set: { if !$0 { pendingDeletion = nil } })) {
                Button("Cancel", role: .cancel) { pendingDeletion = nil }
                Button("Delete", role: .destructive) { if let request = pendingDeletion { pendingDeletion = nil; Task { await appModel.deleteStorage(request) } } }
            } message: { Text(cleanupDescription) }
            .alert("Reset Companion?", isPresented: Binding(get: { resetStage > 0 }, set: { if !$0 { resetStage = 0 } })) {
                Button("Cancel", role: .cancel) { resetStage = 0 }
                Button(resetStage == 1 ? String(localized: "Continue") : String(localized: "Reset now"), role: .destructive) {
                    if resetStage == 1 { resetStage = 2 } else { resetStage = 0; Task { await appModel.resetCompanion() } }
                }
            } message: { Text(resetStage == 1 ? String(localized: "This removes all local Companion data, including models and pairings.") : String(localized: "Final confirmation: this action cannot be reversed.")) }
        }
    }

    private func deletionButton(_ title: LocalizedStringKey, _ request: DeletionRequest) -> some View {
        Button(role: .destructive) { pendingDeletion = request } label: {
            HStack { Text(title); Spacer(); Text(format(recoverableBytes(request))).foregroundStyle(.secondary) }
        }
    }

    private var cleanupDescription: String {
        guard let request = pendingDeletion else { return "" }
        return String(localized: "This action can recover approximately \(format(recoverableBytes(request))). Files protected by a running task will be skipped.")
    }

    private func recoverableBytes(_ request: DeletionRequest) -> Int64 {
        guard let snapshot = appModel.storageSnapshot else { return 0 }
        switch request {
        case .temporaryCache: return snapshot.bytes[.taskCache] ?? 0
        case .temporaryFiles: return snapshot.bytes[.temporary] ?? 0
        case .partialDownloads: return (snapshot.bytes[.downloads] ?? 0) + (snapshot.bytes[.resumeMetadata] ?? 0)
        case .activities: return snapshot.bytes[.activities] ?? 0
        case .logs: return snapshot.bytes[.logs] ?? 0
        case .everythingExceptModelsAndPairings: return snapshot.totalManagedBytes - (snapshot.bytes[.models] ?? 0)
        case .allModels: return snapshot.bytes[.models] ?? 0
        case .resetCompanion: return snapshot.totalManagedBytes
        }
    }

    private func label(_ category: StorageCategory) -> LocalizedStringKey {
        switch category { case .models: "Models"; case .downloads: "Active and paused downloads"; case .staging: "Staging"; case .taskCache: "Task cache"; case .temporary: "Temporary files"; case .activities: "Activity"; case .logs: "Logs"; case .resumeMetadata: "Resume metadata" }
    }

    private func format(_ bytes: Int64) -> String { ByteCountFormatter.string(fromByteCount: max(0, bytes), countStyle: .file) }
}
