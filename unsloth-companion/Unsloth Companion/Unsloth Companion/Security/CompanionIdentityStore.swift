import CryptoKit
import Foundation
import Security

struct PairedDesktop: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    var name: String
    let signingPublicKey: Data
    let certificateSHA256: String
    let pairedAt: Date
}

struct CompanionIdentity: Sendable {
    let deviceID: UUID
    let publicKey: Data
}

enum IdentityStoreError: LocalizedError {
    case keychain(OSStatus), invalidKey, desktopNotPaired, signatureInvalid

    var errorDescription: String? {
        switch self {
        case .keychain(let status): return String(localized: "Keychain failed with status \(status).")
        case .invalidKey: return String(localized: "The device identity is invalid.")
        case .desktopNotPaired: return String(localized: "This Mac is not paired.")
        case .signatureInvalid: return String(localized: "The signed challenge is invalid.")
        }
    }
}

actor CompanionIdentityStore {
    static let shared = CompanionIdentityStore()

    private let service = "com.OvenTeam.app.Unsloth-Companion.identity"
    private let keyAccount = "p256-signing-key"
    private let idAccount = "device-id"
    private let pairingAccount = "paired-desktops"

    func identity() throws -> CompanionIdentity {
        let id: UUID
        if let data = try read(account: idAccount), let value = String(data: data, encoding: .utf8).flatMap(UUID.init(uuidString:)) {
            id = value
        } else {
            id = UUID()
            try write(Data(id.uuidString.utf8), account: idAccount)
        }
        return CompanionIdentity(deviceID: id, publicKey: try publicKey())
    }

    func publicKey() throws -> Data {
        if SecureEnclave.isAvailable { return try secureEnclaveKey().publicKey.x963Representation }
        return try softwareKey().publicKey.x963Representation
    }

    func sign(_ data: Data) throws -> Data {
        if SecureEnclave.isAvailable { return try secureEnclaveKey().signature(for: data).derRepresentation }
        return try softwareKey().signature(for: data).derRepresentation
    }

    func verify(signature: Data, data: Data, publicKey: Data) throws {
        let key = try P256.Signing.PublicKey(x963Representation: publicKey)
        let value = try P256.Signing.ECDSASignature(derRepresentation: signature)
        guard key.isValidSignature(value, for: data) else { throw IdentityStoreError.signatureInvalid }
    }

    func pairedDesktops() throws -> [PairedDesktop] {
        guard let data = try read(account: pairingAccount) else { return [] }
        return try ProtocolCoding.decoder.decode([PairedDesktop].self, from: data)
    }

    func pairedDesktop(id: UUID) throws -> PairedDesktop {
        guard let desktop = try pairedDesktops().first(where: { $0.id == id }) else { throw IdentityStoreError.desktopNotPaired }
        return desktop
    }

    func savePairing(_ desktop: PairedDesktop) throws {
        var values = try pairedDesktops().filter { $0.id != desktop.id }
        values.append(desktop)
        try write(ProtocolCoding.encoder.encode(values), account: pairingAccount)
    }

    func revoke(desktopID: UUID) throws {
        let values = try pairedDesktops().filter { $0.id != desktopID }
        try write(ProtocolCoding.encoder.encode(values), account: pairingAccount)
    }

    func reset() throws {
        for account in [keyAccount, idAccount, pairingAccount] { try delete(account: account) }
    }

    static func pairingCode(serverPublicKey: Data, devicePublicKey: Data, nonce: Data, certificateSHA256: String) -> String {
        var input = Data()
        input.append(serverPublicKey)
        input.append(devicePublicKey)
        input.append(nonce)
        input.append(Data(certificateSHA256.utf8))
        let prefix = SHA256.hash(data: input).prefix(4).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        return String(format: "%06u", prefix % 1_000_000)
    }

    private func secureEnclaveKey() throws -> SecureEnclave.P256.Signing.PrivateKey {
        if let data = try read(account: keyAccount) { return try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: data) }
        var error: Unmanaged<CFError>?
        guard let access = SecAccessControlCreateWithFlags(nil, kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly, .privateKeyUsage, &error) else {
            throw error?.takeRetainedValue() ?? IdentityStoreError.invalidKey
        }
        let key = try SecureEnclave.P256.Signing.PrivateKey(accessControl: access)
        try write(key.dataRepresentation, account: keyAccount)
        return key
    }

    private func softwareKey() throws -> P256.Signing.PrivateKey {
        if let data = try read(account: keyAccount) { return try P256.Signing.PrivateKey(rawRepresentation: data) }
        let key = P256.Signing.PrivateKey()
        try write(key.rawRepresentation, account: keyAccount)
        return key
    }

    private func read(account: String) throws -> Data? {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data else { throw IdentityStoreError.keychain(status) }
        return data
    }

    private func write(_ data: Data, account: String) throws {
        let base: [CFString: Any] = [kSecClass: kSecClassGenericPassword, kSecAttrService: service, kSecAttrAccount: account]
        let update = SecItemUpdate(base as CFDictionary, [kSecValueData: data] as CFDictionary)
        if update == errSecSuccess { return }
        if update != errSecItemNotFound { throw IdentityStoreError.keychain(update) }
        var create = base
        create[kSecValueData] = data
        create[kSecAttrAccessible] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(create as CFDictionary, nil)
        guard status == errSecSuccess else { throw IdentityStoreError.keychain(status) }
    }

    private func delete(account: String) throws {
        let query: [CFString: Any] = [kSecClass: kSecClassGenericPassword, kSecAttrService: service, kSecAttrAccount: account]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw IdentityStoreError.keychain(status) }
    }
}
