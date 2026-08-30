import SwiftUI

struct DashboardView: View {
    @ObservedObject var appModel: CompanionAppModel
    @ObservedObject private var service: CompanionServiceModel
    @ObservedObject private var discovery: CompanionDiscoveryService

    init(appModel: CompanionAppModel) {
        self.appModel = appModel
        _service = ObservedObject(wrappedValue: appModel.service)
        _discovery = ObservedObject(wrappedValue: appModel.service.discovery)
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Companion") {
                    Toggle("Use this iPhone", isOn: Binding(
                        get: { service.settings.serviceEnabled },
                        set: { enabled in
                            service.settings.serviceEnabled = enabled
                            Task { enabled ? service.start() : await service.stop() }
                        }
                    ))
                    LabeledContent("State", value: stateLabel(service.state))
                    LabeledContent("Status", value: service.statusDetail)
                    if let name = service.connectedDesktopName { LabeledContent("Mac", value: name) }
                    if let task = service.currentTaskID {
                        LabeledContent("Task", value: String(task.uuidString.prefix(8)))
                        Button("Cancel task", role: .destructive) { Task { await service.cancelCurrentTask() } }
                    }
                }
                if let code = service.pairingCode {
                    Section("Secure pairing") {
                        Text(code).font(.system(size: 42, weight: .semibold, design: .rounded)).monospacedDigit().frame(maxWidth: .infinity)
                            .accessibilityLabel("Pairing code \(code)")
                        Text("Confirm that the same six digits are visible in Unsloth Desktop.").font(.footnote).foregroundStyle(.secondary)
                        Button("Confirm pairing") { Task { await service.confirmPairing() } }
                        Button("Reject", role: .destructive) { Task { await service.rejectPairing() } }
                    }
                }
                Section("Available Macs") {
                    if discovery.desktops.isEmpty {
                        ContentUnavailableView("No Mac found", systemImage: "network.slash", description: Text("Open Unsloth Desktop on the same local network."))
                    } else {
                        ForEach(discovery.desktops) { desktop in
                            Button { Task { await service.connect(to: desktop) } } label: { Label(desktop.name, systemImage: "macbook") }
                        }
                    }
                }
                Section("Runtime") {
                    LabeledContent("Model", value: appModel.loadedModelID.flatMap { id in appModel.installedModels.first(where: { $0.id == id })?.displayName } ?? String(localized: "None"))
                    LabeledContent("Metal", value: service.runtimeProbe.supportsMetal ? String(localized: "Available") : String(localized: "Unavailable"))
                    LabeledContent("Vision", value: service.runtimeProbe.supportsVision ? String(localized: "Ready") : String(localized: "Unavailable"))
                    LabeledContent("Audio", value: service.runtimeProbe.supportsAudio ? String(localized: "Ready") : String(localized: "Unavailable"))
                    LabeledContent("Context", value: service.runtimeProbe.contextSize.formatted())
                }
            }
            .navigationTitle("Unsloth Companion")
            .refreshable { try? await appModel.refresh() }
        }
    }

    private func stateLabel(_ state: CompanionServiceState) -> String {
        switch state {
        case .offline: String(localized: "Offline")
        case .discovering: String(localized: "Discovering")
        case .pairing: String(localized: "Pairing")
        case .ready: String(localized: "Ready")
        case .leased: String(localized: "Leased")
        case .running: String(localized: "Running")
        case .draining: String(localized: "Draining")
        case .suspended: String(localized: "Suspended")
        }
    }
}
