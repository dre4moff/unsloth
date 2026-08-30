import Combine
import Foundation
import UIKit

enum CompanionServiceState: String, Sendable {
    case offline, discovering, pairing, ready, leased, running, draining, suspended
}

private struct PairingOffer: Codable {
    let desktopID: UUID
    let desktopName: String
    let signingPublicKey: Data
    let nonce: Data
    let certificateSHA256: String
}

private struct PendingPairing {
    let offer: PairingOffer
    let code: String
}

@MainActor
final class CompanionServiceModel: ObservableObject {
    @Published private(set) var state: CompanionServiceState = .offline { didSet { applyIdleTimer() } }
    @Published private(set) var statusDetail = String(localized: "Offline")
    @Published private(set) var pairingCode: String?
    @Published private(set) var connectedDesktopName: String?
    @Published private(set) var lastError: String?
    @Published private(set) var currentTaskID: UUID?
    @Published private(set) var runtimeProbe = RuntimeProbe(supportsMetal: false, supportsVision: false, supportsAudio: false, contextSize: 0)
    @Published var settings: CompanionSettings {
        didSet {
            persistSettings()
            applyIdleTimer()
            Task { [weak self] in await self?.settingsDidChange() }
        }
    }

    let discovery = CompanionDiscoveryService()
    private let identity = CompanionIdentityStore.shared
    private let coordinator = CompanionTaskCoordinator()
    private let runtime = ModelRuntimeActor.shared
    private var transport: CompanionTransport?
    private var receiveTask: Task<Void, Never>?
    private var monitorTask: Task<Void, Never>?
    private var pendingPairing: PendingPairing?
    private var authenticatedDesktop: PairedDesktop?
    private var currentBlobID: UUID?
    private var seenMessageIDs = Set<UUID>()
    private var seenOrder: [UUID] = []
    private var lastHeartbeat = Date()
    private var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid
    private var reconnectEndpoint: DiscoveredDesktop?
    private var loadedModelID: String?
    private var loadedModel: InstalledModel?

    init() {
        if let data = UserDefaults.standard.data(forKey: "companion-settings"),
           let value = try? ProtocolCoding.decoder.decode(CompanionSettings.self, from: data) {
            settings = value
        } else {
            settings = CompanionSettings()
        }
        UIDevice.current.isBatteryMonitoringEnabled = true
        monitorTask = Task { [weak self] in await self?.monitorDevice() }
    }

    deinit { monitorTask?.cancel(); receiveTask?.cancel() }

    func start() {
        guard settings.serviceEnabled else { state = .offline; return }
        state = .discovering
        statusDetail = String(localized: "Searching for Unsloth Desktop")
        discovery.start()
        applyIdleTimer()
    }

    func stop(reason: TaskCancellationReason = .companionDisabled) async {
        state = .draining
        await coordinator.drain(reason: reason)
        await closeConnection()
        discovery.stop()
        state = .offline
        statusDetail = String(localized: "Offline")
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func connect(to desktop: DiscoveredDesktop) async {
        reconnectEndpoint = desktop
        lastError = nil
        statusDetail = String(localized: "Connecting to Mac")
        do {
            let paired = try await identity.pairedDesktops().first { $0.name == desktop.name }
            let transport = CompanionTransport()
            let stream = try await transport.connect(
                endpoint: desktop.endpoint,
                expectedCertificateSHA256: paired?.certificateSHA256,
                allowUntrustedPairing: paired == nil
            )
            self.transport = transport
            let deviceIdentity = try await identity.identity()
            var hello: [String: JSONValue] = [
                "deviceID": .string(deviceIdentity.deviceID.uuidString),
                "deviceName": .string(UIDevice.current.name),
                "devicePublicKey": .string(deviceIdentity.publicKey.base64EncodedString())
            ]
            if let paired { hello["pairedDesktopID"] = .string(paired.id.uuidString) }
            try await transport.send(CompanionEnvelope(type: .hello, payload: .object(hello)))
            authenticatedDesktop = nil
            connectedDesktopName = desktop.name
            lastHeartbeat = Date()
            statusDetail = paired == nil ? String(localized: "Waiting for secure pairing") : String(localized: "Authenticating Mac")
            state = paired == nil ? .pairing : .discovering
            receiveTask?.cancel()
            receiveTask = Task { [weak self, transport] in
                do {
                    for try await event in stream { try await self?.handle(event) }
                    await self?.connectionEnded(error: nil, source: transport)
                } catch { await self?.connectionEnded(error: error, source: transport) }
            }
        } catch {
            lastError = error.localizedDescription
            statusDetail = String(localized: "Connection failed")
        }
    }

    func confirmPairing() async {
        guard let pendingPairing, let transport else { return }
        do {
            let device = try await identity.identity()
            let signature = try await identity.sign(pendingPairing.offer.nonce)
            let payload: [String: JSONValue] = [
                "desktopID": .string(pendingPairing.offer.desktopID.uuidString),
                "deviceID": .string(device.deviceID.uuidString),
                "deviceName": .string(UIDevice.current.name),
                "devicePublicKey": .string(device.publicKey.base64EncodedString()),
                "signature": .string(signature.base64EncodedString()),
                "confirmed": .bool(true)
            ]
            try await transport.send(CompanionEnvelope(type: .pairingConfirm, payload: .object(payload)))
            statusDetail = String(localized: "Confirming pairing on Mac")
        } catch { lastError = error.localizedDescription }
    }

    func rejectPairing() async {
        if let transport {
            try? await transport.send(CompanionEnvelope(type: .pairingConfirm, payload: .object(["confirmed": .bool(false)])))
        }
        pendingPairing = nil; pairingCode = nil
        await closeConnection()
        state = .discovering
    }

    func setLoadedModel(_ model: InstalledModel?) async throws {
        guard currentTaskID == nil else { throw ModelRuntimeError.busy }
        if model?.id == loadedModel?.id { return }
        let previous = loadedModel
        if let model {
            let urls = await ModelStore.shared.URLs(for: model)
            await ModelStore.shared.setRuntimeProtection(for: model, enabled: true)
            do {
                runtimeProbe = try await runtime.load(modelURL: urls.model, mmprojURL: urls.mmproj, computeMode: settings.computeMode)
            } catch {
                await ModelStore.shared.setRuntimeProtection(for: model, enabled: false)
                if let previous { await ModelStore.shared.setRuntimeProtection(for: previous, enabled: false) }
                loadedModel = nil
                loadedModelID = nil
                throw error
            }
            if let previous { await ModelStore.shared.setRuntimeProtection(for: previous, enabled: false) }
            loadedModel = model
        } else {
            try await runtime.unload()
            if let previous { await ModelStore.shared.setRuntimeProtection(for: previous, enabled: false) }
            loadedModel = nil
            runtimeProbe = RuntimeProbe(supportsMetal: false, supportsVision: false, supportsAudio: false, contextSize: 0)
        }
        loadedModelID = model?.id
        try? await sendCapabilities()
    }

    func cancelCurrentTask() async {
        guard let currentTaskID else { return }
        await coordinator.cancel(taskID: currentTaskID, reason: .user)
        self.currentTaskID = nil
        state = .ready
    }

    func sceneDidBecomeActive() {
        if settings.serviceEnabled {
            if authenticatedDesktop != nil {
                state = currentTaskID == nil ? .ready : .running
                statusDetail = currentTaskID == nil ? String(localized: "Ready for tasks") : String(localized: "Task running")
            } else {
                start()
                if let endpoint = reconnectEndpoint { Task { [weak self] in await self?.connect(to: endpoint) } }
            }
            Task { try? await StorageBudgetManager.shared.sweepOrphanedTaskDirectories(activeTaskIDs: currentTaskID.map { [$0] } ?? []) }
        }
        endBackgroundTask()
        applyIdleTimer()
    }

    func sceneWillResignActive() {
        state = .draining
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func sceneDidEnterBackground() {
        beginCleanupBackgroundTask()
        Task { [weak self] in
            await self?.sendDrainingAndStop(reason: .backgrounded)
        }
    }

    func memoryWarning() {
        Task { [weak self] in await self?.sendDrainingAndStop(reason: .memoryPressure) }
    }

    private func handle(_ event: CompanionTransportEvent) async throws {
        switch event {
        case .binary(let data):
            guard authenticatedDesktop != nil, let currentBlobID else { throw CompanionTransportError.invalidFrame }
            try await coordinator.appendBlobChunk(blobID: currentBlobID, data: data)
        case .envelope(let envelope):
            guard envelope.protocolVersion == CompanionProtocol.version,
                  abs(envelope.sentAt.timeIntervalSinceNow) <= 60,
                  !seenMessageIDs.contains(envelope.messageID) else { throw CompanionTransportError.invalidFrame }
            remember(envelope.messageID)
            try await handleEnvelope(envelope)
        }
    }

    private func handleEnvelope(_ envelope: CompanionEnvelope) async throws {
        switch envelope.type {
        case .pairingOffer:
            guard authenticatedDesktop == nil else { throw CompanionTransportError.invalidFrame }
            let offer = try envelope.payload.decode(PairingOffer.self)
            guard offer.certificateSHA256 == transport?.observedCertificateSHA256 else { throw CompanionTransportError.certificateMismatch }
            let device = try await identity.identity()
            let code = CompanionIdentityStore.pairingCode(serverPublicKey: offer.signingPublicKey, devicePublicKey: device.publicKey, nonce: offer.nonce, certificateSHA256: offer.certificateSHA256)
            pendingPairing = PendingPairing(offer: offer, code: code)
            pairingCode = code
            connectedDesktopName = offer.desktopName
            state = .pairing
            let signature = try await identity.sign(offer.nonce)
            try await transport?.send(CompanionEnvelope(type: .pairingConfirm, payload: .object([
                "preview": .bool(true),
                "confirmed": .bool(false),
                "deviceID": .string(device.deviceID.uuidString),
                "deviceName": .string(UIDevice.current.name),
                "devicePublicKey": .string(device.publicKey.base64EncodedString()),
                "nonce": .string(offer.nonce.base64EncodedString()),
                "signature": .string(signature.base64EncodedString())
            ])))
        case .pairingResult:
            guard let pendingPairing,
                  envelope.payload.object?["accepted"]?.bool == true else { throw IdentityStoreError.signatureInvalid }
            let paired = PairedDesktop(
                id: pendingPairing.offer.desktopID,
                name: pendingPairing.offer.desktopName,
                signingPublicKey: pendingPairing.offer.signingPublicKey,
                certificateSHA256: pendingPairing.offer.certificateSHA256,
                pairedAt: Date()
            )
            try await identity.savePairing(paired)
            authenticatedDesktop = paired
            self.pendingPairing = nil; pairingCode = nil
            try await authenticatedReady()
        case .challenge:
            guard let object = envelope.payload.object,
                  let id = object["desktopID"]?.string.flatMap(UUID.init(uuidString:)),
                  let nonce = object["nonce"]?.string.flatMap({ Data(base64Encoded: $0) }),
                  let signature = object["signature"]?.string.flatMap({ Data(base64Encoded: $0) }) else { throw CompanionTransportError.invalidFrame }
            let paired = try await identity.pairedDesktop(id: id)
            guard paired.certificateSHA256 == transport?.observedCertificateSHA256 else { throw CompanionTransportError.certificateMismatch }
            try await identity.verify(signature: signature, data: nonce, publicKey: paired.signingPublicKey)
            let device = try await identity.identity()
            let response = try await identity.sign(nonce)
            try await transport?.send(CompanionEnvelope(type: .challengeResponse, payload: .object([
                "deviceID": .string(device.deviceID.uuidString),
                "signature": .string(response.base64EncodedString())
            ])))
            authenticatedDesktop = paired
            try await authenticatedReady()
        case .heartbeat:
            guard authenticatedDesktop != nil else { throw IdentityStoreError.desktopNotPaired }
            lastHeartbeat = Date()
            guard let echo = envelope.payload.object?["echoMonotonic"]?.number else { throw CompanionTransportError.invalidFrame }
            let storage = try? await StorageBudgetManager.shared.snapshot()
            try await transport?.send(CompanionEnvelope(type: .heartbeat, payload: .object(deviceStatusPayload(echoMonotonic: echo, storageFreeBytes: storage?.freeBytes ?? 0))))
        case .taskSubmit:
            try requireAuthenticated()
            let task = try envelope.payload.decode(CompanionTask.self)
            guard currentTaskID == nil, state == .ready else {
                try await sendTaskFailure(taskID: task.taskID, code: .runtimeError, message: String(localized: "The iPhone runtime is busy."), retryable: true)
                return
            }
            guard supportedTaskKinds().contains(task.kind) else {
                try await sendTaskFailure(taskID: task.taskID, code: .capabilityUnavailable, message: String(localized: "The requested capability is not currently available."), retryable: true)
                return
            }
            currentTaskID = task.taskID
            state = .leased
            do {
                try await coordinator.submit(task)
                state = .running
            } catch {
                currentTaskID = nil
                state = .ready
                let code: CompanionErrorCode = error is CompanionStorageError ? .storageInsufficient : (error as? CompanionTaskFailure)?.code ?? .runtimeError
                try await sendTaskFailure(taskID: task.taskID, code: code, message: error.localizedDescription, retryable: false)
            }
        case .leaseRenew:
            try requireAuthenticated()
            guard let id = envelope.payload.object?["taskID"]?.string.flatMap(UUID.init(uuidString:)) else { throw CompanionTransportError.invalidFrame }
            await coordinator.renewLease(taskID: id)
        case .taskCancel:
            try requireAuthenticated()
            guard let id = envelope.payload.object?["taskID"]?.string.flatMap(UUID.init(uuidString:)) else { throw CompanionTransportError.invalidFrame }
            let explicitUser = envelope.payload.object?["explicitUser"]?.bool == true
            await coordinator.cancel(taskID: id, reason: explicitUser ? .user : .desktopRequest)
            currentTaskID = nil; state = .ready
        case .blobBegin:
            try requireAuthenticated()
            let descriptor = try envelope.payload.decode(BlobDescriptor.self)
            do {
                try await coordinator.beginBlob(descriptor)
                currentBlobID = descriptor.blobID
            } catch {
                await coordinator.fail(taskID: descriptor.taskID, error: error)
                currentTaskID = nil; currentBlobID = nil; state = .ready
            }
        case .blobEnd:
            try requireAuthenticated()
            guard let currentBlobID else { throw CompanionTransportError.invalidFrame }
            do {
                try await coordinator.finishBlob(blobID: currentBlobID)
                self.currentBlobID = nil
            } catch {
                let taskID = self.currentTaskID
                self.currentBlobID = nil
                if let taskID { await coordinator.fail(taskID: taskID, error: error) }
                currentTaskID = nil; state = .ready
            }
        default: break
        }
    }

    private func authenticatedReady() async throws {
        guard let transport else { return }
        await coordinator.configure(
            sender: { envelope in try await transport.send(envelope) },
            mediaPolicy: settings.mediaPolicy,
            allowRawMedia: settings.allowRawMedia,
            completionHandler: { [weak self] taskID in
                guard let self, self.currentTaskID == taskID else { return }
                self.currentTaskID = nil
                self.state = .ready
            }
        )
        try await sendCapabilities()
        state = .ready
        statusDetail = String(localized: "Ready for tasks")
    }

    private func sendCapabilities() async throws {
        guard authenticatedDesktop != nil, let transport else { return }
        let device = try await identity.identity()
        let kinds = supportedTaskKinds()
        let capabilities = CompanionCapabilities(
            protocolVersion: CompanionProtocol.version,
            deviceID: device.deviceID,
            deviceName: UIDevice.current.name,
            runtimeVersion: "llama.cpp-3173a56471c1",
            taskKinds: kinds,
            selectedModelID: loadedModelID,
            supportsVision: runtimeProbe.supportsVision,
            supportsAudio: runtimeProbe.supportsAudio,
            contextSize: runtimeProbe.contextSize,
            maxRawMediaBytes: StorageBudgetManager.rawMediaPerTaskLimit
        )
        try await transport.send(CompanionEnvelope(type: .capabilities, payload: try JSONValue.encode(capabilities)))
    }

    private func sendDrainingAndStop(reason: TaskCancellationReason) async {
        if let transport { try? await transport.send(CompanionEnvelope(type: .clientDraining, payload: .object(["reason": .string(reason.rawValue)]))) }
        await coordinator.drain(reason: reason)
        currentTaskID = nil
        await closeConnection()
        try? await StorageBudgetManager.shared.sweepOrphanedTaskDirectories(activeTaskIDs: [])
        state = .suspended
        UIApplication.shared.isIdleTimerDisabled = false
        endBackgroundTask()
    }

    private func connectionEnded(error: Error?, source: CompanionTransport) async {
        guard transport === source else { return }
        receiveTask = nil
        if let error { lastError = error.localizedDescription }
        await coordinator.drain(reason: .connectionLost)
        currentTaskID = nil
        transport = nil; authenticatedDesktop = nil
        source.close()
        if settings.serviceEnabled, UIApplication.shared.applicationState == .active, let endpoint = reconnectEndpoint {
            for delay in [0.5, 1.0, 2.0, 5.0] {
                try? await Task.sleep(for: .seconds(delay))
                guard transport == nil else { return }
                await connect(to: endpoint)
                if transport != nil { return }
            }
        }
        state = settings.serviceEnabled ? .discovering : .offline
    }

    private func closeConnection() async {
        let activeTask = receiveTask
        let activeTransport = transport
        receiveTask = nil; transport = nil
        activeTask?.cancel()
        activeTransport?.close()
        authenticatedDesktop = nil
        currentBlobID = nil
    }

    private func monitorDevice() async {
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(5))
            if authenticatedDesktop != nil, Date().timeIntervalSince(lastHeartbeat) > CompanionProtocol.heartbeatInterval * 3 {
                if let transport { await connectionEnded(error: CompanionTransportError.connectionFailed(String(localized: "heartbeat timeout")), source: transport) }
            }
            let battery = UIDevice.current.batteryLevel
            if ProcessInfo.processInfo.isLowPowerModeEnabled || (battery >= 0 && battery < 0.10) {
                if state != .suspended { await sendDrainingAndStop(reason: .lowBattery) }
            } else if ProcessInfo.processInfo.thermalState == .serious || ProcessInfo.processInfo.thermalState == .critical {
                if state != .suspended { await sendDrainingAndStop(reason: .thermalPressure) }
            } else if state == .suspended, settings.serviceEnabled, UIApplication.shared.applicationState == .active {
                start()
                if let endpoint = reconnectEndpoint { await connect(to: endpoint) }
            }
        }
    }

    private func deviceStatusPayload(echoMonotonic: Double, storageFreeBytes: Int64) -> [String: JSONValue] {
        [
            "echoMonotonic": .number(echoMonotonic),
            "battery": .number(Double(max(0, UIDevice.current.batteryLevel))),
            "lowPowerMode": .bool(ProcessInfo.processInfo.isLowPowerModeEnabled),
            "thermalState": .number(Double(ProcessInfo.processInfo.thermalState.rawValue)),
            "state": .string(state.rawValue),
            "storageFreeBytes": .number(Double(storageFreeBytes))
        ]
    }

    private func requireAuthenticated() throws {
        guard authenticatedDesktop != nil else { throw IdentityStoreError.desktopNotPaired }
    }

    private func remember(_ id: UUID) {
        seenMessageIDs.insert(id); seenOrder.append(id)
        if seenOrder.count > 2_048 {
            let excess = seenOrder.count - 2_048
            let removed = seenOrder.prefix(excess)
            seenOrder.removeFirst(excess)
            seenMessageIDs.subtract(removed)
        }
    }

    private func persistSettings() {
        if let data = try? ProtocolCoding.encoder.encode(settings) { UserDefaults.standard.set(data, forKey: "companion-settings") }
    }

    private func applyIdleTimer() {
        let available = state == .ready || state == .leased || state == .running
        UIApplication.shared.isIdleTimerDisabled = settings.serviceEnabled && settings.keepScreenAwake && available && UIApplication.shared.applicationState == .active
    }

    private func settingsDidChange() async {
        guard let transport, authenticatedDesktop != nil else { return }
        await coordinator.configure(
            sender: { envelope in try await transport.send(envelope) },
            mediaPolicy: settings.mediaPolicy,
            allowRawMedia: settings.allowRawMedia,
            completionHandler: { [weak self] taskID in
                guard let self, self.currentTaskID == taskID else { return }
                self.currentTaskID = nil
                self.state = .ready
            }
        )
        try? await sendCapabilities()
    }

    private func supportedTaskKinds() -> Set<CompanionTaskKind> {
        var kinds: Set<CompanionTaskKind> = [.ocr, .dsp]
        guard loadedModelID != nil, runtimeProbe.contextSize > 0 else { return kinds }
        kinds.formUnion([.subagent, .classification, .summary, .contextCompression, .extraction, .verification, .reranking, .lightweightPlanning])
        if runtimeProbe.supportsVision { kinds.formUnion([.vision, .videoSummary]) }
        if runtimeProbe.supportsAudio { kinds.formUnion([.audioTranscription, .audioAnalysis]) }
        return kinds
    }

    private func sendTaskFailure(taskID: UUID, code: CompanionErrorCode, message: String, retryable: Bool) async throws {
        guard let transport else { throw CompanionTransportError.closed }
        let failure = CompanionTaskFailure(taskID: taskID, code: code, message: message, retryable: retryable)
        try await transport.send(CompanionEnvelope(type: .taskFailed, payload: try JSONValue.encode(failure)))
    }

    private func beginCleanupBackgroundTask() {
        endBackgroundTask()
        backgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: "CompanionDrain") { [weak self] in
            Task { @MainActor in self?.endBackgroundTask() }
        }
    }

    private func endBackgroundTask() {
        guard backgroundTaskID != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTaskID)
        backgroundTaskID = .invalid
    }
}
