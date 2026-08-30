import CryptoKit
import Foundation
@preconcurrency import Network
import Security

enum CompanionTransportEvent: Sendable {
    case envelope(CompanionEnvelope)
    case binary(Data)
}

enum CompanionTransportError: LocalizedError {
    case connectionFailed(String), certificateMismatch, invalidFrame, closed

    var errorDescription: String? {
        switch self {
        case .connectionFailed(let reason): return String(localized: "Connection failed: \(reason)")
        case .certificateMismatch: return String(localized: "The Mac certificate does not match the paired identity.")
        case .invalidFrame: return String(localized: "The Mac sent an invalid protocol frame.")
        case .closed: return String(localized: "The Mac connection is closed.")
        }
    }
}

final class CompanionTransport: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.unsloth.companion.transport", qos: .userInitiated)
    private let lock = NSLock()
    private var connection: NWConnection?
    private var continuation: AsyncThrowingStream<CompanionTransportEvent, Error>.Continuation?
    private var certificateHash: String?

    var observedCertificateSHA256: String? { lock.withLock { certificateHash } }

    func connect(
        endpoint: NWEndpoint,
        expectedCertificateSHA256: String?,
        allowUntrustedPairing: Bool
    ) async throws -> AsyncThrowingStream<CompanionTransportEvent, Error> {
        let tls = NWProtocolTLS.Options()
        let verifyQueue = queue
        sec_protocol_options_set_verify_block(tls.securityProtocolOptions, { [weak self] _, trust, complete in
            guard let self,
                  let certificate = SecTrustCopyCertificateChain(sec_trust_copy_ref(trust).takeRetainedValue()) as? [SecCertificate],
                  let leaf = certificate.first else {
                complete(false); return
            }
            let hash = SHA256.hash(data: SecCertificateCopyData(leaf) as Data).map { String(format: "%02x", $0) }.joined()
            self.lock.withLock { self.certificateHash = hash }
            if let expectedCertificateSHA256 {
                complete(hash.caseInsensitiveCompare(expectedCertificateSHA256) == .orderedSame)
            } else {
                complete(allowUntrustedPairing)
            }
        }, verifyQueue)

        let parameters = NWParameters(tls: tls, tcp: NWProtocolTCP.Options())
        let websocket = NWProtocolWebSocket.Options()
        websocket.autoReplyPing = true
        websocket.maximumMessageSize = 4 * 1_048_576
        parameters.defaultProtocolStack.applicationProtocols.insert(websocket, at: 0)
        parameters.includePeerToPeer = true
        let connection = NWConnection(to: endpoint, using: parameters)
        self.connection = connection

        try await withCheckedThrowingContinuation { (ready: CheckedContinuation<Void, Error>) in
            let gate = ConnectionReadyGate(continuation: ready)
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    gate.resume()
                case .failed(let error), .waiting(let error):
                    gate.resume(throwing: CompanionTransportError.connectionFailed(error.localizedDescription))
                case .cancelled:
                    gate.resume(throwing: CompanionTransportError.closed)
                default: break
                }
            }
            connection.start(queue: queue)
            queue.asyncAfter(deadline: .now() + 8) {
                if gate.resume(throwing: CompanionTransportError.connectionFailed(String(localized: "connection timeout"))) {
                    connection.cancel()
                }
            }
        }

        let stream = AsyncThrowingStream<CompanionTransportEvent, Error> { continuation in
            self.continuation = continuation
            continuation.onTermination = { [weak self] _ in self?.close() }
        }
        receiveNext()
        return stream
    }

    func send(_ envelope: CompanionEnvelope) async throws {
        try await send(data: ProtocolCoding.encoder.encode(envelope), opcode: .text)
    }

    func sendBinary(_ data: Data) async throws {
        guard data.count <= 1_048_576 else { throw CompanionTransportError.invalidFrame }
        try await send(data: data, opcode: .binary)
    }

    func close() {
        lock.withLock {
            connection?.cancel()
            connection = nil
            continuation?.finish()
            continuation = nil
        }
    }

    private func send(data: Data, opcode: NWProtocolWebSocket.Opcode) async throws {
        guard let connection = lock.withLock({ connection }) else { throw CompanionTransportError.closed }
        let metadata = NWProtocolWebSocket.Metadata(opcode: opcode)
        let context = NWConnection.ContentContext(identifier: UUID().uuidString, metadata: [metadata])
        try await withCheckedThrowingContinuation { (sent: CheckedContinuation<Void, Error>) in
            connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { error in
                if let error { sent.resume(throwing: CompanionTransportError.connectionFailed(error.localizedDescription)) }
                else { sent.resume() }
            })
        }
    }

    private func receiveNext() {
        guard let connection else { return }
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self else { return }
            if let error {
                self.continuation?.finish(throwing: CompanionTransportError.connectionFailed(error.localizedDescription))
                return
            }
            guard let data,
                  let metadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition) as? NWProtocolWebSocket.Metadata else {
                self.continuation?.finish(throwing: CompanionTransportError.invalidFrame)
                return
            }
            switch metadata.opcode {
            case .text:
                do { self.continuation?.yield(.envelope(try ProtocolCoding.decoder.decode(CompanionEnvelope.self, from: data))) }
                catch { self.continuation?.finish(throwing: CompanionTransportError.invalidFrame); return }
            case .binary: self.continuation?.yield(.binary(data))
            case .close: self.continuation?.finish(); return
            default: break
            }
            self.receiveNext()
        }
    }
}

private final class ConnectionReadyGate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?

    init(continuation: CheckedContinuation<Void, Error>) { self.continuation = continuation }

    @discardableResult
    func resume(throwing error: Error? = nil) -> Bool {
        let value = lock.withLock { () -> CheckedContinuation<Void, Error>? in
            defer { continuation = nil }
            return continuation
        }
        guard let value else { return false }
        if let error { value.resume(throwing: error) } else { value.resume() }
        return true
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock(); defer { unlock() }
        return try body()
    }
}
