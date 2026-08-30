import Foundation

enum ModelStoreError: LocalizedError {
    case modelNotInstalled, modelInUse, invalidImport

    var errorDescription: String? {
        switch self {
        case .modelNotInstalled: return String(localized: "The selected model is not installed.")
        case .modelInUse: return String(localized: "The model is loaded by an active task.")
        case .invalidImport: return String(localized: "The selected file is not a valid GGUF model.")
        }
    }
}

actor ModelStore {
    static let shared = ModelStore(storage: .shared)

    private let storage: StorageBudgetManager
    private let fm: FileManager

    init(storage: StorageBudgetManager, fileManager: FileManager = .default) {
        self.storage = storage
        fm = fileManager
    }

    func installedModels() async throws -> [InstalledModel] {
        let root = storage.manifestsRoot
        guard fm.fileExists(atPath: root.path) else { return [] }
        let urls = try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        return try urls.map { try ProtocolCoding.decoder.decode(InstalledModel.self, from: Data(contentsOf: $0)) }
            .sorted { $0.installedAt > $1.installedAt }
    }

    func install(registryModel: RegistryModel, modelStagingURL: URL, mmprojStagingURL: URL) async throws -> InstalledModel {
        try await verify(modelStagingURL, artifact: registryModel.model)
        try await verify(mmprojStagingURL, artifact: registryModel.mmproj)
        let modelBlob = try await commitBlob(from: modelStagingURL, artifact: registryModel.model)
        let mmprojBlob = try await commitBlob(from: mmprojStagingURL, artifact: registryModel.mmproj)
        let installed = InstalledModel(
            id: registryModel.id,
            displayName: registryModel.displayName,
            modelSHA256: registryModel.model.sha256,
            mmprojSHA256: registryModel.mmproj.sha256,
            modelFileName: registryModel.model.file,
            mmprojFileName: registryModel.mmproj.file,
            installedAt: Date()
        )
        do {
            try await writeManifest(installed)
        } catch {
            try? await removeBlobIfUnreferenced(modelBlob, excludingModelID: installed.id)
            try? await removeBlobIfUnreferenced(mmprojBlob, excludingModelID: installed.id)
            throw error
        }
        return installed
    }

    func importGGUF(from source: URL, displayName: String, mmproj: URL? = nil) async throws -> InstalledModel {
        guard source.pathExtension.lowercased() == "gguf", try hasGGUFMagic(source) else { throw ModelStoreError.invalidImport }
        let modelHash = try await storage.hash(of: source)
        let modelArtifact = RegistryArtifact(file: source.lastPathComponent, size: fileSize(source), sha256: modelHash)
        let modelBlob = try await commitBlob(from: source, artifact: modelArtifact, copySource: true)
        var mmprojBlob: URL?
        var mmprojHash: String?
        var mmprojName: String?
        if let mmproj {
            guard mmproj.pathExtension.lowercased() == "gguf", try hasGGUFMagic(mmproj) else { throw ModelStoreError.invalidImport }
            let hash = try await storage.hash(of: mmproj)
            let artifact = RegistryArtifact(file: mmproj.lastPathComponent, size: fileSize(mmproj), sha256: hash)
            mmprojBlob = try await commitBlob(from: mmproj, artifact: artifact, copySource: true)
            mmprojHash = hash
            mmprojName = mmproj.lastPathComponent
        }
        let installed = InstalledModel(
            id: "imported-\(modelHash.prefix(16))",
            displayName: displayName,
            modelSHA256: modelHash,
            mmprojSHA256: mmprojHash,
            modelFileName: source.lastPathComponent,
            mmprojFileName: mmprojName,
            installedAt: Date()
        )
        do { try await writeManifest(installed) }
        catch {
            try? await removeBlobIfUnreferenced(modelBlob, excludingModelID: installed.id)
            if let mmprojBlob { try? await removeBlobIfUnreferenced(mmprojBlob, excludingModelID: installed.id) }
            throw error
        }
        return installed
    }

    func URLs(for model: InstalledModel) async -> (model: URL, mmproj: URL?) {
        let root = storage.blobsRoot
        let modelURL = root.appending(path: model.modelSHA256 + ".gguf")
        let mmprojURL = model.mmprojSHA256.map { root.appending(path: $0 + ".gguf") }
        return (modelURL, mmprojURL)
    }

    func recoverableBytes(deleting model: InstalledModel) async throws -> Int64 {
        let others = try await installedModels().filter { $0.id != model.id }
        let urls = await URLs(for: model)
        var total: Int64 = 0
        if !others.contains(where: { $0.modelSHA256 == model.modelSHA256 || $0.mmprojSHA256 == model.modelSHA256 }) {
            total += fileSize(urls.model)
        }
        if let mmprojHash = model.mmprojSHA256, let mmprojURL = urls.mmproj,
           !others.contains(where: { $0.modelSHA256 == mmprojHash || $0.mmprojSHA256 == mmprojHash }) {
            total += fileSize(mmprojURL)
        }
        return total
    }

    func setRuntimeProtection(for model: InstalledModel, enabled: Bool) async {
        let urls = await URLs(for: model)
        if enabled {
            await storage.protect(urls.model)
            if let mmproj = urls.mmproj { await storage.protect(mmproj) }
        } else {
            await storage.unprotect(urls.model)
            if let mmproj = urls.mmproj { await storage.unprotect(mmproj) }
        }
    }

    func deleteModel(id: String) async throws {
        guard let installed = try await installedModels().first(where: { $0.id == id }) else { throw ModelStoreError.modelNotInstalled }
        let urls = await URLs(for: installed)
        if await storage.isProtected(urls.model) { throw ModelStoreError.modelInUse }
        if let mmproj = urls.mmproj, await storage.isProtected(mmproj) { throw ModelStoreError.modelInUse }
        let manifests = storage.manifestsRoot
        let manifest = manifests.appending(path: safeManifestName(id))
        try await storage.deleteManagedItem(manifest)
        try await removeBlobIfUnreferenced(urls.model, excludingModelID: nil)
        if let mmproj = urls.mmproj { try await removeBlobIfUnreferenced(mmproj, excludingModelID: nil) }
    }

    func deleteAllModels() async throws {
        for model in try await installedModels() { try await deleteModel(id: model.id) }
    }

    private func verify(_ url: URL, artifact: RegistryArtifact) async throws {
        guard fileSize(url) == artifact.size else { throw CompanionStorageError.invalidArtifact }
        guard try await storage.hash(of: url) == artifact.sha256 else { throw CompanionStorageError.checksumMismatch }
        guard try hasGGUFMagic(url) else { throw CompanionStorageError.invalidArtifact }
    }

    private func commitBlob(from source: URL, artifact: RegistryArtifact, copySource: Bool = false) async throws -> URL {
        let root = storage.blobsRoot
        let destination = root.appending(path: artifact.sha256 + ".gguf")
        if fm.fileExists(atPath: destination.path) {
            guard try await storage.hash(of: destination) == artifact.sha256 else {
                try await storage.deleteManagedItem(destination)
                throw CompanionStorageError.checksumMismatch
            }
            if !copySource { try? fm.removeItem(at: source) }
            return destination
        }
        let installing = root.appending(path: ".installing-\(UUID().uuidString)")
        if copySource { try fm.copyItem(at: source, to: installing) } else { try fm.moveItem(at: source, to: installing) }
        do { try fm.moveItem(at: installing, to: destination) }
        catch { try? fm.removeItem(at: installing); throw error }
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var mutable = destination; try mutable.setResourceValues(values)
        return destination
    }

    private func writeManifest(_ model: InstalledModel) async throws {
        let root = storage.manifestsRoot
        let destination = root.appending(path: safeManifestName(model.id))
        try ProtocolCoding.encoder.encode(model).write(to: destination, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }

    private func removeBlobIfUnreferenced(_ blob: URL, excludingModelID: String?) async throws {
        let hash = blob.deletingPathExtension().lastPathComponent
        let manifestsRoot = blob.deletingLastPathComponent().deletingLastPathComponent().appending(path: "Manifests")
        guard fm.fileExists(atPath: manifestsRoot.path) else {
            try await storage.deleteManagedItem(blob)
            return
        }
        let manifests = try fm.contentsOfDirectory(at: manifestsRoot, includingPropertiesForKeys: nil)
        for manifest in manifests where manifest.pathExtension == "json" {
            let model = try ProtocolCoding.decoder.decode(InstalledModel.self, from: Data(contentsOf: manifest))
            if model.id != excludingModelID && (model.modelSHA256 == hash || model.mmprojSHA256 == hash) { return }
        }
        try await storage.deleteManagedItem(blob)
    }

    private func hasGGUFMagic(_ url: URL) throws -> Bool {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        return try handle.read(upToCount: 4) == Data([0x47, 0x47, 0x55, 0x46])
    }

    private func fileSize(_ url: URL) -> Int64 {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
    }

    private func safeManifestName(_ id: String) -> String {
        id.replacingOccurrences(of: "/", with: "_") + ".json"
    }
}
