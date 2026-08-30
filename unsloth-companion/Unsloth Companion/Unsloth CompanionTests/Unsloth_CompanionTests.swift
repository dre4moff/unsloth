import CryptoKit
import Foundation
import Testing
@testable import Unsloth_Companion

@Suite(.serialized)
struct ProtocolAndStorageTests {
    @Test func protocolRoundTripAndFractionalDate() throws {
        let task = CompanionTask(
            taskID: UUID(uuidString: "42f8a43c-b251-4f90-98e3-ce6ae8c6a4fd")!,
            parentTaskID: nil,
            shardIndex: nil,
            shardCount: nil,
            idempotencyKey: "summary-fixture-00000001",
            kind: .summary,
            priority: 50,
            timeoutSeconds: 120,
            leaseSeconds: 30,
            mediaPolicy: .semanticOnly,
            input: ["text": .string("Unsloth Companion processes this text locally.")],
            resultSchema: ["type": .string("object")]
        )
        let envelope = CompanionEnvelope(type: .taskSubmit, payload: try JSONValue.encode(task))
        let roundTrip = try ProtocolCoding.decoder.decode(CompanionEnvelope.self, from: ProtocolCoding.encoder.encode(envelope))
        #expect(roundTrip.protocolVersion == 1)
        #expect(try roundTrip.payload.decode(CompanionTask.self).idempotencyKey == task.idempotencyKey)

        let fractional = Data(#"{"protocolVersion":1,"messageID":"6d1748ee-0831-4dd4-92af-349ef5c46118","sentAt":"2026-08-29T16:00:00.123456Z","type":"heartbeat","payload":{"echoMonotonic":1}}"#.utf8)
        #expect(try ProtocolCoding.decoder.decode(CompanionEnvelope.self, from: fractional).sentAt.timeIntervalSince1970 > 0)
    }

    @Test func taskCleanupProtectsActiveDirectoryAndRemovesOrphans() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        let activeID = UUID()
        let active = try await storage.taskDirectory(taskID: activeID, expectedBytes: 64, rawMedia: false)
        try Data(repeating: 1, count: 64).write(to: active.appending(path: "active.bin"))
        let tasksRoot = await storage.tasksRoot
        let orphan = tasksRoot.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: true)
        try Data(repeating: 2, count: 64).write(to: orphan.appending(path: "orphan.bin"))

        let result = try await storage.delete(.temporaryCache)
        #expect(FileManager.default.fileExists(atPath: active.path))
        #expect(!FileManager.default.fileExists(atPath: orphan.path))
        #expect(result.skippedProtectedPaths.contains(active.path))

        try await storage.finishTask(activeID)
        #expect(!FileManager.default.fileExists(atPath: active.path))
    }

    @Test func oneHundredTasksLeaveNoTaskOrTemporaryBytes() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        for _ in 0..<100 {
            let id = UUID()
            let directory = try await storage.taskDirectory(taskID: id, expectedBytes: 1_024, rawMedia: false)
            try Data(repeating: 7, count: 1_024).write(to: directory.appending(path: "payload.bin"))
            try await storage.finishTask(id)
        }
        let snapshot = try await storage.snapshot()
        #expect(snapshot.bytes[.taskCache] == 0)
        #expect(snapshot.bytes[.temporary] == 0)
        #expect((snapshot.bytes[.logs] ?? 0) <= StorageBudgetManager.logLimit)
    }

    @Test func interruptedDeletionIsRecoveredAtBootstrap() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        let applicationSupport = await storage.applicationSupport
        let tombstone = applicationSupport.appending(path: ".deleting-crash")
        try FileManager.default.createDirectory(at: tombstone, withIntermediateDirectories: true)
        try Data("unfinished".utf8).write(to: tombstone.appending(path: "file"))
        try await storage.bootstrap()
        #expect(!FileManager.default.fileExists(atPath: tombstone.path))

        let blobsRoot = await storage.blobsRoot
        let nestedTombstone = blobsRoot.appending(path: ".deleting-model-crash")
        let interruptedInstall = blobsRoot.appending(path: ".installing-model-crash")
        try FileManager.default.createDirectory(at: nestedTombstone, withIntermediateDirectories: true)
        try Data("unfinished model deletion".utf8).write(to: nestedTombstone.appending(path: "file"))
        try Data("unfinished install".utf8).write(to: interruptedInstall)
        try await storage.bootstrap()
        #expect(!FileManager.default.fileExists(atPath: nestedTombstone.path))
        #expect(!FileManager.default.fileExists(atPath: interruptedInstall.path))
    }

    @Test func sharedGGUFBlobsAreReferenceCounted() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        let store = ModelStore(storage: storage)
        let modelData = Data([0x47, 0x47, 0x55, 0x46]) + Data(repeating: 3, count: 32)
        let mmprojData = Data([0x47, 0x47, 0x55, 0x46]) + Data(repeating: 4, count: 32)
        let modelHash = try await writeAndHash(modelData, name: "model-a.gguf", root: root, storage: storage)
        let mmprojHash = try await writeAndHash(mmprojData, name: "mmproj-a.gguf", root: root, storage: storage)
        let modelArtifact = RegistryArtifact(file: "model.gguf", size: Int64(modelData.count), sha256: modelHash)
        let mmprojArtifact = RegistryArtifact(file: "mmproj.gguf", size: Int64(mmprojData.count), sha256: mmprojHash)

        for id in ["shared-a", "shared-b"] {
            let modelURL = root.appending(path: "\(id)-model.gguf")
            let mmprojURL = root.appending(path: "\(id)-mmproj.gguf")
            try modelData.write(to: modelURL)
            try mmprojData.write(to: mmprojURL)
            let registry = RegistryModel(id: id, displayName: id, profile: "test", source: "test", license: "test", repository: "test/test", revision: String(repeating: "a", count: 40), model: modelArtifact, mmproj: mmprojArtifact, capabilities: ["text"])
            _ = try await store.install(registryModel: registry, modelStagingURL: modelURL, mmprojStagingURL: mmprojURL)
        }

        try await store.deleteModel(id: "shared-a")
        let blobsRoot = await storage.blobsRoot
        #expect(FileManager.default.fileExists(atPath: blobsRoot.appending(path: modelHash + ".gguf").path))
        #expect(FileManager.default.fileExists(atPath: blobsRoot.appending(path: mmprojHash + ".gguf").path))
        try await store.deleteModel(id: "shared-b")
        #expect(!FileManager.default.fileExists(atPath: blobsRoot.appending(path: modelHash + ".gguf").path))
        #expect(!FileManager.default.fileExists(atPath: blobsRoot.appending(path: mmprojHash + ".gguf").path))
    }

    @Test func embeddedRegistryHasFivePinnedArtifacts() throws {
        let registry = try ModelRegistry.load()
        #expect(registry.schemaVersion == 1)
        #expect(registry.models.count == 5)
        for model in registry.models {
            #expect(model.revision.count == 40)
            #expect(model.model.size > 0 && model.mmproj.size > 0)
            #expect(model.model.sha256.wholeMatch(of: /^[a-f0-9]{64}$/) != nil)
            #expect(model.mmproj.sha256.wholeMatch(of: /^[a-f0-9]{64}$/) != nil)
            #expect(!model.repository.contains("example"))
        }
    }

    @MainActor @Test func simulatorCannotStartModelDownloads() {
        #if targetEnvironment(simulator)
        #expect(!ModelDownloadManager.supportsModelDownloads)
        #else
        #expect(ModelDownloadManager.supportsModelDownloads)
        #endif
    }

    @Test func settingsDecodePreservesSafeDefaultsFromOlderInstallations() throws {
        let legacy = Data(#"{"serviceEnabled":true,"keepScreenAwake":true,"guardScreenEnabled":true,"mediaPolicy":"semantic_only","allowRawMedia":false,"computeMode":"automatic"}"#.utf8)
        let settings = try ProtocolCoding.decoder.decode(CompanionSettings.self, from: legacy)
        #expect(!settings.allowCellularDownloads)
    }

    @Test func storageReservationsKeepTheFreeSpaceReserve() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let available = StorageBudgetManager.minimumFreeReserve + 1_024
        let storage = StorageBudgetManager(rootOverride: root, availableCapacityOverride: available)
        try await storage.bootstrap()
        let reservation = try await storage.reserve(bytes: 1_024)
        await storage.releaseReservation(reservation)
        do {
            _ = try await storage.reserve(bytes: 1_025)
            Issue.record("A reservation consumed the mandatory free-space reserve")
        } catch let error as CompanionStorageError {
            guard case .insufficientSpace = error else {
                Issue.record("Unexpected storage error: \(error)")
                return
            }
        }
    }

    @Test func dspMeasuresSineAndClipping() throws {
        let sampleRate = 48_000.0
        let samples = (0..<48_000).map { Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / sampleRate)) }
        let result = try AudioDSPAnalyzer.analyze(samples: samples, sampleRate: sampleRate)
        #expect(abs((result.estimatedPitchHz ?? 0) - 440) < 12)
        #expect(abs(result.rmsDBFS - (-9.03)) < 0.5)
        #expect(result.clippedSampleCount == 0)
        let clipped = try AudioDSPAnalyzer.analyze(samples: [1, -1, 0, 0], sampleRate: sampleRate)
        #expect(clipped.clippedSampleCount == 2)
    }

    @Test func coordinatorWaitsForEveryDeclaredBlobAndCleansFailure() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        let recorder = EnvelopeRecorder()
        let coordinator = CompanionTaskCoordinator(
            storage: storage,
            runtime: ModelRuntimeActor(),
            pipeline: TaskPipelineActor(),
            activities: ActivityStore(storage: storage)
        )
        await coordinator.configure(
            sender: { envelope in await recorder.append(envelope) },
            mediaPolicy: .derivedMedia,
            allowRawMedia: false
        )
        let taskID = UUID()
        let firstData = Data([1, 2, 3, 4])
        let secondData = Data([5, 6, 7, 8])
        let task = CompanionTask(
            taskID: taskID,
            parentTaskID: nil,
            shardIndex: nil,
            shardCount: nil,
            idempotencyKey: "two-blob-coordinator-test",
            kind: .dsp,
            priority: 50,
            timeoutSeconds: 30,
            leaseSeconds: 30,
            mediaPolicy: .derivedMedia,
            input: [
                "mediaFiles": .array([.string("0000-first.wav"), .string("0001-second.wav")]),
                "expectedMediaBytes": .number(8)
            ],
            resultSchema: [:]
        )
        try await coordinator.submit(task)

        let first = BlobDescriptor(blobID: UUID(), taskID: taskID, name: "0000-first.wav", mimeType: "audio/wav", size: 4, sha256: digest(firstData))
        try await coordinator.beginBlob(first)
        try await coordinator.appendBlobChunk(blobID: first.blobID, data: firstData)
        try await coordinator.finishBlob(blobID: first.blobID)
        try await Task.sleep(for: .milliseconds(100))
        #expect(!(await recorder.messageTypes()).contains(.taskProgress))

        let second = BlobDescriptor(blobID: UUID(), taskID: taskID, name: "0001-second.wav", mimeType: "audio/wav", size: 4, sha256: digest(secondData))
        try await coordinator.beginBlob(second)
        try await coordinator.appendBlobChunk(blobID: second.blobID, data: secondData)
        try await coordinator.finishBlob(blobID: second.blobID)
        for _ in 0..<100 {
            if (await recorder.messageTypes()).contains(.taskFailed) { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        let messageTypes = await recorder.messageTypes()
        #expect(messageTypes.contains(.taskProgress))
        #expect(messageTypes.contains(.taskFailed))
        let tasksRoot = await storage.tasksRoot
        #expect(!FileManager.default.fileExists(atPath: tasksRoot.appending(path: taskID.uuidString).path))
    }

    @Test func coordinatorEnforcesLocalMediaPolicyBeforeAdmission() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        let coordinator = CompanionTaskCoordinator(
            storage: storage,
            runtime: ModelRuntimeActor(),
            pipeline: TaskPipelineActor(),
            activities: ActivityStore(storage: storage)
        )
        await coordinator.configure(sender: { _ in }, mediaPolicy: .semanticOnly, allowRawMedia: false)
        let task = CompanionTask(
            taskID: UUID(), parentTaskID: nil, shardIndex: nil, shardCount: nil,
            idempotencyKey: "rejected-derived-media-test", kind: .vision, priority: 50,
            timeoutSeconds: 30, leaseSeconds: 30, mediaPolicy: .derivedMedia,
            input: [:], resultSchema: [:]
        )
        do {
            try await coordinator.submit(task)
            Issue.record("Derived media was admitted while the local policy was semantic-only")
        } catch let failure as CompanionTaskFailure {
            #expect(failure.code == .permissionDenied)
        }
        let tasksRoot = await storage.tasksRoot
        #expect(!FileManager.default.fileExists(atPath: tasksRoot.appending(path: task.taskID.uuidString).path))
    }

    @Test func explicitCancellationEmitsOneTerminalMessageAndCleansTask() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = StorageBudgetManager(rootOverride: root)
        try await storage.bootstrap()
        let recorder = EnvelopeRecorder()
        let coordinator = CompanionTaskCoordinator(
            storage: storage,
            runtime: ModelRuntimeActor(),
            pipeline: TaskPipelineActor(),
            activities: ActivityStore(storage: storage)
        )
        await coordinator.configure(sender: { envelope in await recorder.append(envelope) }, mediaPolicy: .derivedMedia, allowRawMedia: false)
        let taskID = UUID()
        let task = CompanionTask(
            taskID: taskID, parentTaskID: nil, shardIndex: nil, shardCount: nil,
            idempotencyKey: "explicit-cancellation-test", kind: .ocr, priority: 50,
            timeoutSeconds: 30, leaseSeconds: 30, mediaPolicy: .derivedMedia,
            input: ["mediaFiles": .array([.string("0000-image.png")]), "expectedMediaBytes": .number(4)],
            resultSchema: [:]
        )
        try await coordinator.submit(task)
        await coordinator.cancel(taskID: taskID, reason: .user)
        let terminal = await recorder.envelopes(ofType: .taskCancelled)
        #expect(terminal.count == 1)
        #expect(terminal.first?.payload.object?["reason"]?.string == TaskCancellationReason.user.rawValue)
        let tasksRoot = await storage.tasksRoot
        #expect(!FileManager.default.fileExists(atPath: tasksRoot.appending(path: taskID.uuidString).path))
    }

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "UnslothCompanionTests-\(UUID().uuidString)", directoryHint: .isDirectory)
    }

    private func writeAndHash(_ data: Data, name: String, root: URL, storage: StorageBudgetManager) async throws -> String {
        let url = root.appending(path: name)
        try data.write(to: url)
        return try await storage.hash(of: url)
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private actor EnvelopeRecorder {
    private var envelopes: [CompanionEnvelope] = []
    func append(_ envelope: CompanionEnvelope) { envelopes.append(envelope) }
    func messageTypes() -> [CompanionMessageType] { envelopes.map(\.type) }
    func envelopes(ofType type: CompanionMessageType) -> [CompanionEnvelope] { envelopes.filter { $0.type == type } }
}
