import Foundation
import llama

enum RuntimeComputeMode: String, Codable, CaseIterable, Sendable {
    case automatic, cpu, metal
}

enum ModelRuntimeState: Sendable, Equatable {
    case unloaded, loading, ready, generating(UUID), draining, failed(String)
}

enum ModelRuntimeError: LocalizedError {
    case modelNotLoaded, modelLoadFailed, contextCreationFailed, samplerCreationFailed
    case tokenizationFailed, promptTooLong, decodeFailed(Int32), cancelled, emptyOutput, outputTruncated(Int), busy

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return String(localized: "No model is loaded.")
        case .modelLoadFailed: return String(localized: "llama.cpp could not load this GGUF model.")
        case .contextCreationFailed: return String(localized: "llama.cpp could not create the model context.")
        case .samplerCreationFailed: return String(localized: "llama.cpp could not create the sampler.")
        case .tokenizationFailed: return String(localized: "The prompt could not be tokenized.")
        case .promptTooLong: return String(localized: "The delegated prompt does not fit in the iPhone model context.")
        case .decodeFailed(let code): return String(localized: "llama.cpp decode failed with code \(code).")
        case .cancelled: return String(localized: "The task was cancelled.")
        case .emptyOutput: return String(localized: "The model returned an empty result.")
        case .outputTruncated(let limit): return String(localized: "The model reached its \(limit)-token output budget before completing the answer.")
        case .busy: return String(localized: "The runtime is already processing a task.")
        }
    }
}

struct RuntimeProbe: Sendable, Equatable {
    let supportsMetal: Bool
    let supportsVision: Bool
    let supportsAudio: Bool
    let contextSize: Int
}

struct RuntimeGeneration: Sendable {
    let text: String
    let tokensGenerated: Int
    let durationMS: Int
}

actor ModelRuntimeActor {
    static let shared = ModelRuntimeActor()

    private static var backendInitialized = false
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var loadedModelURL: URL?
    private var loadedMMProjURL: URL?
    private var cancelledTaskIDs = Set<UUID>()
    private(set) var state: ModelRuntimeState = .unloaded
    private(set) var probe = RuntimeProbe(supportsMetal: false, supportsVision: false, supportsAudio: false, contextSize: 0)

    func load(modelURL: URL, mmprojURL: URL?, computeMode: RuntimeComputeMode = .automatic) throws -> RuntimeProbe {
        switch state {
        case .loading, .generating, .draining: throw ModelRuntimeError.busy
        default: break
        }
        unloadResources()
        state = .loading
        if !Self.backendInitialized {
            llama_backend_init()
            Self.backendInitialized = true
        }

        var modelParameters = llama_model_default_params()
        let metalAvailable = llama_supports_gpu_offload()
        #if targetEnvironment(simulator)
        modelParameters.n_gpu_layers = 0
        #else
        switch computeMode {
        case .cpu: modelParameters.n_gpu_layers = 0
        case .automatic, .metal: modelParameters.n_gpu_layers = metalAvailable ? -1 : 0
        }
        #endif
        guard let loadedModel = modelURL.path.withCString({ llama_model_load_from_file($0, modelParameters) }) else {
            state = .failed(ModelRuntimeError.modelLoadFailed.localizedDescription)
            throw ModelRuntimeError.modelLoadFailed
        }

        var contextParameters = llama_context_default_params()
        contextParameters.n_ctx = UInt32(preferredContextSize(model: loadedModel))
        contextParameters.n_batch = 512
        contextParameters.n_ubatch = 512
        let threads = Int32(max(2, ProcessInfo.processInfo.processorCount - 1))
        contextParameters.n_threads = threads
        contextParameters.n_threads_batch = threads
        guard let loadedContext = llama_init_from_model(loadedModel, contextParameters) else {
            llama_model_free(loadedModel)
            state = .failed(ModelRuntimeError.contextCreationFailed.localizedDescription)
            throw ModelRuntimeError.contextCreationFailed
        }

        model = loadedModel
        context = loadedContext
        loadedModelURL = modelURL
        loadedMMProjURL = mmprojURL
        let media = probeMedia(model: loadedModel, mmprojURL: mmprojURL, useGPU: modelParameters.n_gpu_layers != 0)
        probe = RuntimeProbe(
            supportsMetal: metalAvailable,
            supportsVision: media.vision,
            supportsAudio: media.audio,
            contextSize: Int(llama_n_ctx(loadedContext))
        )
        state = .ready
        return probe
    }

    private func preferredContextSize(model: OpaquePointer) -> Int {
        let gibibyte = UInt64(1_073_741_824)
        let memory = ProcessInfo.processInfo.physicalMemory
        let deviceLimit: Int
        if memory >= 10 * gibibyte {
            deviceLimit = 16_384
        } else if memory >= 6 * gibibyte {
            deviceLimit = 8_192
        } else {
            deviceLimit = 4_096
        }
        let trainedContext = Int(llama_model_n_ctx_train(model))
        guard trainedContext > 0 else { return deviceLimit }
        return max(2_048, min(deviceLimit, trainedContext))
    }

    func unload() throws {
        if case .generating = state { throw ModelRuntimeError.busy }
        if state == .loading || state == .draining { throw ModelRuntimeError.busy }
        unloadResources()
    }

    private func unloadResources() {
        state = .draining
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        context = nil
        model = nil
        loadedModelURL = nil
        loadedMMProjURL = nil
        cancelledTaskIDs.removeAll()
        probe = RuntimeProbe(supportsMetal: llama_supports_gpu_offload(), supportsVision: false, supportsAudio: false, contextSize: 0)
        state = .unloaded
    }

    func cancel(taskID: UUID) { cancelledTaskIDs.insert(taskID) }

    func generate(
        taskID: UUID,
        prompt: String,
        mediaPayloads: [Data] = [],
        maximumTokens: Int,
        requireJSONObject: Bool,
        onToken: @escaping @Sendable (String) async -> Void
    ) async throws -> RuntimeGeneration {
        guard state == .ready else {
            if case .generating = state { throw ModelRuntimeError.busy }
            throw ModelRuntimeError.modelNotLoaded
        }
        guard let model, let context, let vocabulary = llama_model_get_vocab(model) else { throw ModelRuntimeError.modelNotLoaded }
        state = .generating(taskID)
        cancelledTaskIDs.remove(taskID)
        let started = ContinuousClock.now
        defer {
            cancelledTaskIDs.remove(taskID)
            if self.model != nil { state = .ready }
        }

        llama_memory_clear(llama_get_memory(context), true)
        let contextCapacity = Int(llama_n_ctx(context))
        let consumedPromptTokens: Int
        if mediaPayloads.isEmpty {
            let renderedPrompt = applyChatTemplate(model: model, userPrompt: prompt) ?? "User: \(prompt)\nAssistant:"
            consumedPromptTokens = try decodeTextPrompt(
                renderedPrompt,
                vocabulary: vocabulary,
                context: context,
                contextCapacity: contextCapacity,
                taskID: taskID
            )
        } else {
            consumedPromptTokens = try decodeMediaPrompt(
                prompt: prompt,
                payloads: mediaPayloads,
                model: model,
                context: context,
                taskID: taskID
            )
        }

        guard let sampler = makeSampler(vocabulary: vocabulary, requireJSONObject: requireJSONObject) else {
            throw ModelRuntimeError.samplerCreationFailed
        }
        defer { llama_sampler_free(sampler) }

        var output = ""
        var tokenCount = 0
        var finishedNaturally = false
        let available = max(1, contextCapacity - consumedPromptTokens - 8)
        let outputLimit = max(1, min(maximumTokens, available))
        for _ in 0..<outputLimit {
            try checkCancellation(taskID)
            let token = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocabulary, token) {
                finishedNaturally = true
                break
            }
            let piece = tokenPiece(token, vocabulary: vocabulary)
            if !piece.isEmpty {
                output += piece
                tokenCount += 1
                await onToken(piece)
            }
            llama_sampler_accept(sampler, token)
            var mutableToken = token
            let code = withUnsafeMutablePointer(to: &mutableToken) { llama_decode(context, llama_batch_get_one($0, 1)) }
            guard code >= 0 else { throw ModelRuntimeError.decodeFailed(code) }
        }
        guard finishedNaturally else { throw ModelRuntimeError.outputTruncated(outputLimit) }
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw ModelRuntimeError.emptyOutput }
        let elapsed = started.duration(to: .now)
        return RuntimeGeneration(text: text, tokensGenerated: tokenCount, durationMS: Int(elapsed.components.seconds * 1_000) + Int(elapsed.components.attoseconds / 1_000_000_000_000_000))
    }

    private func checkCancellation(_ taskID: UUID) throws {
        if Task.isCancelled || cancelledTaskIDs.contains(taskID) { throw ModelRuntimeError.cancelled }
    }

    private func probeMedia(model: OpaquePointer, mmprojURL: URL?, useGPU: Bool) -> (vision: Bool, audio: Bool) {
        guard let mmprojURL else { return (false, false) }
        var parameters = mtmd_context_params_default()
        parameters.use_gpu = useGPU
        parameters.print_timings = false
        parameters.n_threads = Int32(max(2, ProcessInfo.processInfo.processorCount - 1))
        parameters.warmup = false
        guard let mediaContext = mmprojURL.path.withCString({ mtmd_init_from_file($0, model, parameters) }) else { return (false, false) }
        defer { mtmd_free(mediaContext) }
        return (mtmd_support_vision(mediaContext), mtmd_support_audio(mediaContext))
    }

    private func decodeTextPrompt(
        _ prompt: String,
        vocabulary: OpaquePointer,
        context: OpaquePointer,
        contextCapacity: Int,
        taskID: UUID
    ) throws -> Int {
        let promptTokens = try tokenize(prompt, vocabulary: vocabulary)
        guard promptTokens.count < contextCapacity - 256 else { throw ModelRuntimeError.promptTooLong }
        let batchSize = max(1, Int(llama_n_batch(context)))
        var cursor = 0
        while cursor < promptTokens.count {
            try checkCancellation(taskID)
            let end = min(cursor + batchSize, promptTokens.count)
            var chunk = Array(promptTokens[cursor..<end])
            let code = chunk.withUnsafeMutableBufferPointer { llama_decode(context, llama_batch_get_one($0.baseAddress, Int32($0.count))) }
            guard code >= 0 else { throw ModelRuntimeError.decodeFailed(code) }
            cursor = end
        }
        return promptTokens.count
    }

    private func decodeMediaPrompt(
        prompt: String,
        payloads: [Data],
        model: OpaquePointer,
        context: OpaquePointer,
        taskID: UUID
    ) throws -> Int {
        guard let mmprojURL = loadedMMProjURL else { throw CompanionTaskFailure(taskID: taskID, code: .capabilityUnavailable, message: String(localized: "The selected model has no mmproj."), retryable: false) }
        var parameters = mtmd_context_params_default()
        parameters.use_gpu = probe.supportsMetal
        parameters.print_timings = false
        parameters.n_threads = Int32(max(2, ProcessInfo.processInfo.processorCount - 1))
        parameters.warmup = false
        guard let mediaContext = mmprojURL.path.withCString({ mtmd_init_from_file($0, model, parameters) }) else {
            throw CompanionTaskFailure(taskID: taskID, code: .runtimeError, message: String(localized: "mtmd could not load mmproj."), retryable: false)
        }
        defer { mtmd_free(mediaContext) }
        let marker = mtmd_default_marker().map(String.init(cString:)) ?? "<__media__>"
        let mediaPrompt = Array(repeating: marker, count: payloads.count).joined(separator: "\n") + "\n" + prompt
        var wrappers: [mtmd_helper_bitmap_wrapper] = []
        wrappers.reserveCapacity(payloads.count)
        for payload in payloads {
            try checkCancellation(taskID)
            let wrapper = payload.withUnsafeBytes { bytes in
                mtmd_helper_bitmap_init_from_buf(
                    mediaContext,
                    bytes.bindMemory(to: UInt8.self).baseAddress,
                    bytes.count,
                    false,
                    mtmd_helper_init_opt_default()
                )
            }
            guard wrapper.bitmap != nil else {
                throw CompanionTaskFailure(taskID: taskID, code: .payloadCorrupt, message: String(localized: "mtmd rejected a media payload."), retryable: false)
            }
            wrappers.append(wrapper)
        }
        defer {
            for wrapper in wrappers {
                if let bitmap = wrapper.bitmap { mtmd_bitmap_free(bitmap) }
                if let video = wrapper.video_ctx { mtmd_helper_video_free(video) }
            }
        }
        guard let chunks = mtmd_input_chunks_init() else { throw ModelRuntimeError.contextCreationFailed }
        defer { mtmd_input_chunks_free(chunks) }
        var bitmaps: [OpaquePointer?] = wrappers.map(\.bitmap)
        let tokenizationCode: Int32 = mediaPrompt.withCString { text in
            var input = mtmd_input_text(text: text, text_len: strlen(text), add_special: true, parse_special: true)
            return bitmaps.withUnsafeMutableBufferPointer { buffer in
                mtmd_tokenize(mediaContext, chunks, &input, buffer.baseAddress, buffer.count)
            }
        }
        guard tokenizationCode == 0 else {
            throw CompanionTaskFailure(taskID: taskID, code: .payloadCorrupt, message: String(localized: "mtmd tokenization failed with code \(tokenizationCode)."), retryable: false)
        }
        var past: llama_pos = 0
        let code = mtmd_helper_eval_chunks(
            mediaContext,
            context,
            chunks,
            0,
            0,
            Int32(max(1, Int(llama_n_batch(context)))),
            true,
            &past
        )
        guard code == 0 else { throw ModelRuntimeError.decodeFailed(code) }
        return max(Int(past), Int(mtmd_helper_get_n_tokens(chunks)))
    }

    private func makeSampler(vocabulary: OpaquePointer, requireJSONObject: Bool) -> UnsafeMutablePointer<llama_sampler>? {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return nil }
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(llama_vocab_n_tokens(vocabulary), 128, 1.08, 0, 0))
        if requireJSONObject {
            let grammar = #"""
            root ::= object
            value ::= object | array | string | number | ("true" | "false" | "null") ws
            object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
            array ::= "[" ws (value ("," ws value)*)? "]" ws
            string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
            number ::= ("-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?) ws
            ws ::= ([ \t\n] ws)?
            """#
            let grammarSampler = grammar.withCString { llama_sampler_init_grammar(vocabulary, $0, "root") }
            guard let grammarSampler else {
                llama_sampler_free(chain)
                return nil
            }
            llama_sampler_chain_add(chain, grammarSampler)
        }
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(64))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.65))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 1...UInt32.max)))
        return chain
    }

    private func tokenize(_ text: String, vocabulary: OpaquePointer) throws -> [llama_token] {
        var tokens = [llama_token](repeating: 0, count: max(64, text.utf8.count + 16))
        var count = text.withCString { llama_tokenize(vocabulary, $0, Int32(strlen($0)), &tokens, Int32(tokens.count), true, true) }
        if count < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-count) + 8)
            count = text.withCString { llama_tokenize(vocabulary, $0, Int32(strlen($0)), &tokens, Int32(tokens.count), true, true) }
        }
        guard count > 0 else { throw ModelRuntimeError.tokenizationFailed }
        return Array(tokens.prefix(Int(count)))
    }

    private func tokenPiece(_ token: llama_token, vocabulary: OpaquePointer) -> String {
        var buffer = [CChar](repeating: 0, count: 64)
        var written = llama_token_to_piece(vocabulary, token, &buffer, Int32(buffer.count), 0, true)
        if written < 0 {
            buffer = [CChar](repeating: 0, count: Int(-written) + 8)
            written = llama_token_to_piece(vocabulary, token, &buffer, Int32(buffer.count), 0, true)
        }
        guard written > 0 else { return "" }
        return String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    private func applyChatTemplate(model: OpaquePointer, userPrompt: String) -> String? {
        guard let template = llama_model_chat_template(model, nil) else { return nil }
        let role = strdup("user")
        let content = strdup(userPrompt)
        guard let role, let content else { return nil }
        defer { free(role); free(content) }
        var message = llama_chat_message(role: UnsafePointer(role), content: UnsafePointer(content))
        var probe = [CChar](repeating: 0, count: 1)
        let required = llama_chat_apply_template(template, &message, 1, true, &probe, Int32(probe.count))
        guard required > 0 else { return nil }
        var output = [CChar](repeating: 0, count: Int(required) + 1)
        guard llama_chat_apply_template(template, &message, 1, true, &output, Int32(output.count)) > 0 else { return nil }
        return String(cString: output)
    }
}
