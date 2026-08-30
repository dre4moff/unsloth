import Foundation

private struct IncomingBlob {
    let descriptor: BlobDescriptor
    let partialURL: URL
    let handle: FileHandle
    var bytesWritten: Int64
}

actor CompanionTaskCoordinator {
    typealias Sender = @Sendable (CompanionEnvelope) async throws -> Void
    typealias CompletionHandler = @MainActor @Sendable (UUID) -> Void

    private let storage: StorageBudgetManager
    private let runtime: ModelRuntimeActor
    private let pipeline: TaskPipelineActor
    private let activities: ActivityStore
    private var sender: Sender?
    private var completionHandler: CompletionHandler?
    private var activeTask: CompanionTask?
    private var activeTaskDirectory: URL?
    private var execution: Task<Void, Never>?
    private var cancellationReason: TaskCancellationReason?
    private var lease: TaskLease?
    private var leaseWatchdog: Task<Void, Never>?
    private var incomingBlobs: [UUID: IncomingBlob] = [:]
    private var allowRawMedia = false
    private var permittedMediaPolicy: MediaPolicy = .semanticOnly
    private var expectedBlobCount = 0
    private var completedBlobCount = 0
    private var expectedMediaBytes: Int64 = 0
    private var declaredBlobBytes: Int64 = 0
    private var receivedBlobNames = Set<String>()

    init(
        storage: StorageBudgetManager = .shared,
        runtime: ModelRuntimeActor = .shared,
        pipeline: TaskPipelineActor = .shared,
        activities: ActivityStore = .shared
    ) {
        self.storage = storage
        self.runtime = runtime
        self.pipeline = pipeline
        self.activities = activities
    }

    func configure(sender: Sender?, mediaPolicy: MediaPolicy, allowRawMedia: Bool, completionHandler: CompletionHandler? = nil) {
        self.sender = sender
        permittedMediaPolicy = mediaPolicy
        self.allowRawMedia = allowRawMedia
        self.completionHandler = completionHandler
    }

    func submit(_ task: CompanionTask) async throws {
        guard activeTask == nil else {
            throw CompanionTaskFailure(taskID: task.taskID, code: .runtimeError, message: String(localized: "The iPhone runtime is busy."), retryable: true)
        }
        guard permits(task.mediaPolicy) else {
            throw CompanionTaskFailure(taskID: task.taskID, code: .permissionDenied, message: String(localized: "The task media policy is not enabled on this iPhone."), retryable: false)
        }
        let mediaFiles = task.input["mediaFiles"]?.array?.compactMap(\.string) ?? []
        guard task.mediaPolicy != .semanticOnly || mediaFiles.isEmpty else {
            throw CompanionTaskFailure(taskID: task.taskID, code: .permissionDenied, message: String(localized: "Semantic-only tasks cannot include media files."), retryable: false)
        }
        let expectedValue = task.input["expectedMediaBytes"]?.number ?? 0
        guard expectedValue.isFinite, expectedValue >= 0, expectedValue <= Double(Int64.max) else {
            throw CompanionTaskFailure(taskID: task.taskID, code: .payloadCorrupt, message: String(localized: "The declared media size is invalid."), retryable: false)
        }
        let expected = Int64(expectedValue)
        let directory = try await storage.taskDirectory(taskID: task.taskID, expectedBytes: expected, rawMedia: task.mediaPolicy == .rawMedia)
        activeTask = task
        activeTaskDirectory = directory
        expectedBlobCount = mediaFiles.count
        completedBlobCount = 0
        expectedMediaBytes = expected
        declaredBlobBytes = 0
        receivedBlobNames.removeAll()
        lease = TaskLease(taskID: task.taskID, expiresAt: Date().addingTimeInterval(task.leaseSeconds), renewal: 0)
        do {
            try await saveActivity(task: task, state: .accepted, detail: "Accepted")
            try await send(.taskAccepted, ["taskID": .string(task.taskID.uuidString), "leaseSeconds": .number(task.leaseSeconds)])
            startLeaseWatchdog(taskID: task.taskID)
            if expectedBlobCount == 0 { startExecution() }
        } catch {
            try? await saveActivity(task: task, state: .failed, detail: error.localizedDescription)
            try? await cleanup(taskID: task.taskID)
            throw error
        }
    }

    func beginBlob(_ descriptor: BlobDescriptor) async throws {
        guard let task = activeTask, task.taskID == descriptor.taskID, let directory = activeTaskDirectory else {
            throw CompanionTaskFailure(taskID: descriptor.taskID, code: .payloadCorrupt, message: String(localized: "Blob does not belong to the active task."), retryable: false)
        }
        guard incomingBlobs[descriptor.blobID] == nil, completedBlobCount + incomingBlobs.count < expectedBlobCount else {
            throw CompanionTaskFailure(taskID: descriptor.taskID, code: .payloadCorrupt, message: String(localized: "The task received an unexpected media blob."), retryable: false)
        }
        let limit = task.mediaPolicy == .rawMedia ? StorageBudgetManager.rawMediaPerTaskLimit : StorageBudgetManager.taskCacheLimit
        guard descriptor.size >= 0, declaredBlobBytes + descriptor.size <= limit,
              declaredBlobBytes + descriptor.size <= expectedMediaBytes else {
            throw CompanionTaskFailure(taskID: descriptor.taskID, code: .storageInsufficient, message: String(localized: "Blob exceeds the per-task media limit."), retryable: false)
        }
        let safeName = URL(fileURLWithPath: descriptor.name).lastPathComponent
        let expectedNames = Set(task.input["mediaFiles"]?.array?.compactMap(\.string) ?? [])
        guard !safeName.isEmpty, safeName == descriptor.name, expectedNames.contains(safeName), !receivedBlobNames.contains(safeName) else {
            throw CompanionTaskFailure(taskID: descriptor.taskID, code: .payloadCorrupt, message: String(localized: "The blob name is not part of the task."), retryable: false)
        }
        let partial = directory.appending(path: ".blob-\(descriptor.blobID.uuidString).part")
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        incomingBlobs[descriptor.blobID] = IncomingBlob(descriptor: descriptor, partialURL: partial, handle: try FileHandle(forWritingTo: partial), bytesWritten: 0)
        declaredBlobBytes += descriptor.size
        receivedBlobNames.insert(safeName)
    }

    func appendBlobChunk(blobID: UUID, data: Data) throws {
        guard data.count <= 1_048_576, var blob = incomingBlobs[blobID] else { throw CompanionStorageError.invalidArtifact }
        guard blob.bytesWritten + Int64(data.count) <= blob.descriptor.size else { throw CompanionStorageError.invalidArtifact }
        try blob.handle.write(contentsOf: data)
        blob.bytesWritten += Int64(data.count)
        incomingBlobs[blobID] = blob
    }

    func finishBlob(blobID: UUID) async throws {
        guard let blob = incomingBlobs.removeValue(forKey: blobID), let directory = activeTaskDirectory else { throw CompanionStorageError.invalidArtifact }
        try blob.handle.close()
        guard blob.bytesWritten == blob.descriptor.size,
              try await storage.hash(of: blob.partialURL) == blob.descriptor.sha256 else {
            try? FileManager.default.removeItem(at: blob.partialURL)
            throw CompanionStorageError.checksumMismatch
        }
        let destination = directory.appending(path: blob.descriptor.name)
        if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
        try FileManager.default.moveItem(at: blob.partialURL, to: destination)
        completedBlobCount += 1
        if incomingBlobs.isEmpty, completedBlobCount == expectedBlobCount { startExecution() }
    }

    func renewLease(taskID: UUID) {
        guard var lease, lease.taskID == taskID else { return }
        lease.renew(seconds: CompanionProtocol.defaultLease)
        self.lease = lease
    }

    func cancel(taskID: UUID, reason: TaskCancellationReason) async {
        guard activeTask?.taskID == taskID else { return }
        cancellationReason = reason
        let activeExecution = execution
        activeExecution?.cancel()
        await runtime.cancel(taskID: taskID)
        if let activeExecution {
            await activeExecution.value
        } else {
            await completeCancellation(taskID: taskID, reason: reason)
        }
    }

    func drain(reason: TaskCancellationReason) async {
        if let id = activeTask?.taskID { await cancel(taskID: id, reason: reason) }
    }

    func fail(taskID: UUID, error: Error) async {
        guard activeTask?.taskID == taskID else { return }
        execution?.cancel()
        await runtime.cancel(taskID: taskID)
        let retryable = (error as? CompanionTaskFailure)?.retryable ?? !(error is TaskPipelineError)
        let failure = CompanionTaskFailure(taskID: taskID, code: errorCode(error), message: error.localizedDescription, retryable: retryable)
        if let task = activeTask { try? await saveActivity(task: task, state: .failed, detail: error.localizedDescription) }
        try? await cleanup(taskID: taskID)
        try? await send(.taskFailed, (try? JSONValue.encode(failure).object) ?? [:])
    }

    private func startExecution() {
        guard execution == nil, let task = activeTask, let directory = activeTaskDirectory else { return }
        execution = Task { [weak self] in await self?.execute(task: task, directory: directory) }
    }

    private func execute(task: CompanionTask, directory: URL) async {
        do {
            try await saveActivity(task: task, state: .running, detail: "Running")
            try await send(.taskProgress, ["taskID": .string(task.taskID.uuidString), "progress": .number(0.05)])
            let prepared = try await pipeline.prepare(task: task, taskDirectory: directory, allowRawMedia: allowRawMedia)
            var result = prepared.derivedResult
            var generatedTokens = 0
            var durationMS = 0
            if !prepared.prompt.isEmpty {
                let requestedValue = task.input["maximumTokens"]?.number ?? 4_096
                let requestedTokens = requestedValue.isFinite ? Int(requestedValue) : 4_096
                let generation = try await runtime.generate(
                    taskID: task.taskID,
                    prompt: prepared.prompt,
                    mediaPayloads: prepared.mediaPayloads,
                    maximumTokens: max(256, min(16_384, requestedTokens)),
                    requireJSONObject: prepared.requireJSONObject
                ) { [weak self] token in
                    try? await self?.send(.taskToken, ["taskID": .string(task.taskID.uuidString), "token": .string(token)])
                }
                generatedTokens = generation.tokensGenerated
                durationMS = generation.durationMS
                if prepared.requireJSONObject {
                    guard let data = generation.text.data(using: .utf8),
                          let object = try? ProtocolCoding.decoder.decode([String: JSONValue].self, from: data) else {
                        throw TaskPipelineError.invalidStructuredOutput
                    }
                    result.merge(object) { _, new in new }
                } else {
                    result["text"] = .string(generation.text)
                }
            }
            let hash = try ProtocolCoding.digest(result)
            let response = CompanionTaskResult(taskID: task.taskID, result: result, sha256: hash, durationMS: durationMS, tokensGenerated: generatedTokens)
            try await send(.taskCompleted, try JSONValue.encode(response).object ?? [:])
            try await saveActivity(task: task, state: .completed, detail: "Completed", durationMS: durationMS, tokens: generatedTokens)
            try await cleanup(taskID: task.taskID)
        } catch is CancellationError {
            await completeCancellation(taskID: task.taskID, reason: cancellationReason ?? .desktopRequest)
        } catch ModelRuntimeError.cancelled {
            await completeCancellation(taskID: task.taskID, reason: cancellationReason ?? .desktopRequest)
        } catch {
            await fail(taskID: task.taskID, error: error)
        }
    }

    private func completeCancellation(taskID: UUID, reason: TaskCancellationReason) async {
        if let task = activeTask { try? await saveActivity(task: task, state: .cancelled, detail: reason.rawValue) }
        try? await cleanup(taskID: taskID)
        try? await send(.taskCancelled, ["taskID": .string(taskID.uuidString), "reason": .string(reason.rawValue)])
    }

    private func cleanup(taskID: UUID) async throws {
        leaseWatchdog?.cancel(); leaseWatchdog = nil
        execution = nil; lease = nil
        cancellationReason = nil
        for blob in incomingBlobs.values { try? blob.handle.close() }
        incomingBlobs.removeAll()
        activeTask = nil; activeTaskDirectory = nil
        expectedBlobCount = 0; completedBlobCount = 0
        expectedMediaBytes = 0; declaredBlobBytes = 0
        receivedBlobNames.removeAll()
        try await storage.finishTask(taskID)
        await completionHandler?(taskID)
    }

    private func startLeaseWatchdog(taskID: UUID) {
        leaseWatchdog?.cancel()
        leaseWatchdog = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard let self else { return }
                if await self.lease?.isExpired == true { await self.cancel(taskID: taskID, reason: .leaseExpired); return }
            }
        }
    }

    private func send(_ type: CompanionMessageType, _ payload: [String: JSONValue]) async throws {
        guard let sender else { throw CompanionTaskFailure(taskID: activeTask?.taskID ?? UUID(), code: .connectionLost, message: String(localized: "The Mac connection is closed."), retryable: true) }
        try await sender(CompanionEnvelope(type: type, payload: .object(payload)))
    }

    private func saveActivity(task: CompanionTask, state: ActivityState, detail: String, durationMS: Int? = nil, tokens: Int? = nil) async throws {
        let previous = try await activities.list().first(where: { $0.id == task.taskID })
        let value = CompanionActivity(
            id: task.taskID,
            kind: task.kind,
            startedAt: previous?.startedAt ?? Date(),
            finishedAt: [.completed, .failed, .cancelled].contains(state) ? Date() : nil,
            state: state,
            detail: detail,
            durationMS: durationMS,
            tokens: tokens
        )
        try await activities.save(value)
    }

    private func errorCode(_ error: Error) -> CompanionErrorCode {
        if let failure = error as? CompanionTaskFailure { return failure.code }
        if error is CompanionStorageError { return .storageInsufficient }
        if error is TaskPipelineError { return .payloadCorrupt }
        if error is ModelRuntimeError { return .runtimeError }
        return .runtimeError
    }

    private func permits(_ requested: MediaPolicy) -> Bool {
        switch permittedMediaPolicy {
        case .semanticOnly: return requested == .semanticOnly
        case .derivedMedia: return requested != .rawMedia
        case .rawMedia: return requested != .rawMedia || allowRawMedia
        }
    }
}
