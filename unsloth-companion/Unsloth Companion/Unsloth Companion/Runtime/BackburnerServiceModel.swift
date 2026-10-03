import Backburner
import Combine
import Foundation
import UIKit

@MainActor
final class BackburnerServiceModel: ObservableObject {
    @Published private(set) var selected = false
    @Published private(set) var running = false
    @Published private(set) var transitioning = false
    @Published private(set) var cableAddress = ""
    @Published private(set) var tailState = "offline"
    @Published private(set) var detail = ""
    @Published private(set) var tokens: UInt64 = 0
    @Published private(set) var tokensPerSecond: Double = 0
    @Published private(set) var heldKeys: UInt64 = 0
    @Published private(set) var macPhase = ""
    @Published private(set) var availableBytes: UInt64 = 0
    private var foreground = true
    private var monitor: Task<Void, Never>?
    private var agentModel: InstalledModel?

    init() {
        monitor = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                await self?.refresh()
            }
        }
    }

    deinit { monitor?.cancel() }

    func setSelected(_ value: Bool, companion: CompanionServiceModel) async {
        guard !transitioning, selected != value else { return }
        transitioning = true
        defer { transitioning = false }
        if value {
            companion.accelerationSelected = true
            await companion.stop()
            agentModel = companion.currentLoadedModel
            do {
                try await companion.setLoadedModel(nil)
            } catch {
                detail = error.localizedDescription
                companion.accelerationSelected = false
                agentModel = nil
                if foreground { companion.start() }
                return
            }
            selected = true
            if foreground { await start() }
        } else {
            await stop()
            selected = false
            companion.accelerationSelected = false
            if let previous = agentModel {
                do { try await companion.setLoadedModel(previous) }
                catch { detail = error.localizedDescription }
            }
            agentModel = nil
            if foreground { companion.start() }
        }
    }

    func sceneChanged(active: Bool) async {
        foreground = active
        if active, selected { await start() }
        else if !active { await stop() }
    }

    private func start() async {
        guard selected, !running else { return }
        cableAddress = SidecarRPC.cableAddress()
        guard !cableAddress.isEmpty else {
            detail = String(localized: "Connect iPhone to Mac with a 10 Gb/s USB-C cable.")
            return
        }
        let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Backburner")
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        await Task.detached(priority: .userInitiated) { SidecarRPC.beginServices() }.value
        running = SidecarRPC.servicesRunning()
        UIApplication.shared.isIdleTimerDisabled = running
    }

    private func stop() async {
        await Task.detached(priority: .userInitiated) { SidecarRPC.endServices() }.value
        running = false
        tailState = "offline"
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func refresh() async {
        guard !transitioning else { return }
        let address = SidecarRPC.cableAddress()
        if selected, running, address != cableAddress { await stop() }
        cableAddress = address
        if selected, foreground, !running, !address.isEmpty { await start() }
        guard running else { return }
        let tail = SidecarRPC.tailStatus()
        let attention = SidecarRPC.phoneAttnStatus()
        let mac = SidecarRPC.macStatus()
        tailState = tail["state"] as? String ?? ""
        detail = tail["detail"] as? String ?? ""
        tokens = (tail["tokens"] as? NSNumber)?.uint64Value ?? 0
        tokensPerSecond = (tail["lastTokS"] as? NSNumber)?.doubleValue ?? 0
        heldKeys = (attention["heldKeys"] as? NSNumber)?.uint64Value ?? 0
        macPhase = mac["phase"] as? String ?? ""
        availableBytes = (SidecarRPC.memoryStats()["availableBytes"] as? NSNumber)?.uint64Value ?? 0
    }
}
