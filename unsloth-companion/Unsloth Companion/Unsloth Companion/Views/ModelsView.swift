import SwiftUI
import UniformTypeIdentifiers

struct ModelsView: View {
    @ObservedObject var appModel: CompanionAppModel
    @ObservedObject private var downloads: ModelDownloadManager
    @State private var importing = false
    @State private var deletingModel: InstalledModel?
    @State private var deleteModelStage = 0
    @State private var deleteAllStage = 0

    init(appModel: CompanionAppModel) {
        self.appModel = appModel
        _downloads = ObservedObject(wrappedValue: appModel.downloads)
    }

    var body: some View {
        NavigationStack {
            List {
                if let status = downloads.status {
                    Section("Download") {
                        ProgressView(value: status.progress)
                        LabeledContent(downloadPhase(status.phase), value: ByteCountFormatter.string(fromByteCount: status.completedBytes, countStyle: .file) + " / " + ByteCountFormatter.string(fromByteCount: status.totalBytes, countStyle: .file))
                        if let error = status.errorDescription { Text(error).foregroundStyle(.red) }
                        HStack {
                            if status.phase == .paused { Button("Resume") { Task { try? await downloads.resume() } } }
                            else if [.waitingForWiFi, .waitingForNetwork, .downloadingModel, .downloadingMMProj].contains(status.phase) { Button("Pause") { downloads.pause() } }
                            Button("Cancel and delete", role: .destructive) { Task { await downloads.cancelAndDelete(); try? await appModel.refresh() } }
                        }
                        .buttonStyle(.borderless)
                    }
                }
                Section("Installed") {
                    if appModel.installedModels.isEmpty { Text("No model installed").foregroundStyle(.secondary) }
                    ForEach(appModel.installedModels) { model in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(model.displayName).font(.headline)
                                    Text(model.modelFileName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                }
                                Spacer()
                                if appModel.loadedModelID == model.id { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).accessibilityLabel("Loaded") }
                            }
                            HStack {
                                Button(appModel.loadedModelID == model.id ? String(localized: "Unload") : String(localized: "Load")) {
                                    Task { appModel.loadedModelID == model.id ? await appModel.unloadModel() : await appModel.load(model) }
                                }
                                Spacer()
                                Button("Delete", role: .destructive) { deletingModel = model; deleteModelStage = 1 }
                            }.buttonStyle(.bordered)
                            if appModel.loadedModelID == model.id {
                                Label("Protected while loaded", systemImage: "lock.fill").font(.caption).foregroundStyle(.secondary)
                            }
                        }.padding(.vertical, 4)
                    }
                }
                Section("Catalog") {
                    if !ModelDownloadManager.supportsModelDownloads {
                        Text("Model downloads are available only on a physical iPhone.").font(.footnote).foregroundStyle(.secondary)
                    }
                    ForEach(appModel.registryModels) { model in
                        let installed = appModel.installedModels.contains { $0.id == model.id }
                        VStack(alignment: .leading, spacing: 5) {
                            Text(model.displayName).font(.headline)
                            Text("\(model.profile.capitalized) · \(model.source.capitalized) · \(ByteCountFormatter.string(fromByteCount: model.model.size + model.mmproj.size, countStyle: .file))")
                                .font(.caption).foregroundStyle(.secondary)
                            if !installed {
                                Button("Download") { Task { await appModel.startDownload(model) } }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(!ModelDownloadManager.supportsModelDownloads)
                            }
                        }.padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Models")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button { importing = true } label: { Image(systemName: "square.and.arrow.down") }.accessibilityLabel("Import GGUF")
                    Button(role: .destructive) { deleteAllStage = 1 } label: { Image(systemName: "trash") }.disabled(appModel.installedModels.isEmpty).accessibilityLabel("Delete all models")
                }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: [UTType(filenameExtension: "gguf") ?? .data], allowsMultipleSelection: true) { result in
                if case .success(let urls) = result { Task { await appModel.importGGUF(urls: urls) } }
                if case .failure(let error) = result { appModel.presentedError = error.localizedDescription }
            }
            .alert("Delete model?", isPresented: Binding(get: { deletingModel != nil }, set: { if !$0 { deletingModel = nil; deleteModelStage = 0 } })) {
                Button("Cancel", role: .cancel) { deletingModel = nil; deleteModelStage = 0 }
                Button(deleteModelStage == 1 ? String(localized: "Continue") : String(localized: "Delete model now"), role: .destructive) {
                    if deleteModelStage == 1 {
                        deleteModelStage = 2
                    } else if let value = deletingModel {
                        deletingModel = nil; deleteModelStage = 0
                        Task { await appModel.deleteModel(value) }
                    }
                }
            } message: {
                if let model = deletingModel {
                    Text(deleteModelStage == 1
                         ? String(localized: "This removes the model and unshared mmproj data, recovering approximately \(format(appModel.recoverableModelBytes[model.id] ?? 0)).")
                         : String(localized: "Final confirmation: downloading or importing will be required to restore this model."))
                }
            }
            .alert("Delete every model?", isPresented: Binding(get: { deleteAllStage > 0 }, set: { if !$0 { deleteAllStage = 0 } })) {
                Button("Cancel", role: .cancel) { deleteAllStage = 0 }
                Button(deleteAllStage == 1 ? String(localized: "Continue") : String(localized: "Delete all models"), role: .destructive) {
                    if deleteAllStage == 1 { deleteAllStage = 2 } else { deleteAllStage = 0; Task { await appModel.deleteAllModels() } }
                }
            } message: { Text(deleteAllStage == 1 ? String(localized: "All installed models will be unloaded. Approximately \(format(appModel.storageSnapshot?.bytes[.models] ?? 0)) can be recovered.") : String(localized: "Final confirmation: all model files will be deleted now.")) }
        }
    }

    private func downloadPhase(_ phase: ModelDownloadPhase) -> String {
        switch phase {
        case .idle: String(localized: "Idle")
        case .reserving: String(localized: "Reserving space")
        case .waitingForWiFi: String(localized: "Waiting for Wi-Fi")
        case .waitingForNetwork: String(localized: "Waiting for a network connection")
        case .downloadingModel: String(localized: "Downloading model")
        case .downloadingMMProj: String(localized: "Downloading mmproj")
        case .paused: String(localized: "Paused")
        case .verifying: String(localized: "Verifying")
        case .installing: String(localized: "Installing")
        case .completed: String(localized: "Completed")
        case .failed: String(localized: "Failed")
        }
    }

    private func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: max(0, bytes), countStyle: .file)
    }
}
