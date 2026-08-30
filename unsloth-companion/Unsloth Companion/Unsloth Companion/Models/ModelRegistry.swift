import Foundation

struct RegistryArtifact: Codable, Hashable, Sendable {
    let file: String
    let size: Int64
    let sha256: String
}

struct RegistryModel: Codable, Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let profile: String
    let source: String
    let license: String
    let repository: String
    let revision: String
    let model: RegistryArtifact
    let mmproj: RegistryArtifact
    let capabilities: [String]

    func url(for artifact: RegistryArtifact) -> URL {
        let escaped = artifact.file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? artifact.file
        return URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(escaped)")!
    }
}

struct ModelRegistryDocument: Codable, Sendable {
    let schemaVersion: Int
    let updatedAt: Date
    let models: [RegistryModel]
}

enum ModelRegistry {
    static func load() throws -> ModelRegistryDocument {
        guard let url = Bundle.main.url(forResource: "model-registry", withExtension: "json") else {
            throw CompanionStorageError.registryMissing
        }
        return try ProtocolCoding.decoder.decode(ModelRegistryDocument.self, from: Data(contentsOf: url))
    }
}

struct InstalledModel: Codable, Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let modelSHA256: String
    let mmprojSHA256: String?
    let modelFileName: String
    let mmprojFileName: String?
    let installedAt: Date
}
