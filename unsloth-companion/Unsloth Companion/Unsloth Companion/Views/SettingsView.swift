import SwiftUI

struct SettingsView: View {
    @ObservedObject var appModel: CompanionAppModel
    @ObservedObject private var service: CompanionServiceModel
    @State private var revoking: PairedDesktop?

    init(appModel: CompanionAppModel) {
        self.appModel = appModel
        _service = ObservedObject(wrappedValue: appModel.service)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Continuity") {
                    Toggle("Keep screen awake", isOn: setting(\.keepScreenAwake))
                    Toggle("Dark Companion guard", isOn: setting(\.guardScreenEnabled))
                    Text("The idle timer is disabled only while Companion is enabled and the app is in the foreground.").font(.footnote).foregroundStyle(.secondary)
                }
                Section("Model downloads") {
                    Toggle("Allow downloads over cellular", isOn: Binding(
                        get: { service.settings.allowCellularDownloads },
                        set: { appModel.setAllowsCellularDownloads($0) }
                    ))
                    Text("Model files are several gigabytes. When disabled, downloads wait for Wi-Fi and can be paused without deleting partial data.").font(.footnote).foregroundStyle(.secondary)
                }
                Section("Media privacy") {
                    Picker("Transfer policy", selection: setting(\.mediaPolicy)) {
                        Text("Semantic only").tag(MediaPolicy.semanticOnly)
                        Text("Derived media").tag(MediaPolicy.derivedMedia)
                        Text("Raw media").tag(MediaPolicy.rawMedia)
                    }
                    Toggle("Allow raw media", isOn: setting(\.allowRawMedia))
                    Text("Camera and microphone are never activated remotely. Raw files are accepted only when this switch is enabled.").font(.footnote).foregroundStyle(.secondary)
                }
                Section("Compute") {
                    Picker("Runtime", selection: setting(\.computeMode)) {
                        Text("Automatic").tag(RuntimeComputeMode.automatic)
                        Text("CPU").tag(RuntimeComputeMode.cpu)
                        Text("Metal").tag(RuntimeComputeMode.metal)
                    }
                    Text("Changing compute mode takes effect the next time a model is loaded.").font(.footnote).foregroundStyle(.secondary)
                }
                Section("Paired Macs") {
                    if appModel.pairedDesktops.isEmpty { Text("No paired Mac").foregroundStyle(.secondary) }
                    ForEach(appModel.pairedDesktops) { desktop in
                        HStack {
                            VStack(alignment: .leading) { Text(desktop.name); Text(desktop.pairedAt.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(.secondary) }
                            Spacer(); Button("Revoke", role: .destructive) { revoking = desktop }
                        }
                    }
                }
                Section("About") {
                    LabeledContent("Version", value: "0.0.1")
                    LabeledContent("Protocol", value: "v\(CompanionProtocol.version)")
                    LabeledContent("Runtime commit", value: "3173a56471c1")
                }
            }
            .navigationTitle("Settings")
            .alert("Revoke this Mac?", isPresented: Binding(get: { revoking != nil }, set: { if !$0 { revoking = nil } })) {
                Button("Cancel", role: .cancel) { revoking = nil }
                Button("Revoke", role: .destructive) { if let value = revoking { revoking = nil; Task { await appModel.revoke(value) } } }
            } message: { Text("The Mac must be paired again before it can send tasks.") }
        }
    }

    private func setting<Value>(_ keyPath: WritableKeyPath<CompanionSettings, Value>) -> Binding<Value> {
        Binding(get: { service.settings[keyPath: keyPath] }, set: { service.settings[keyPath: keyPath] = $0 })
    }
}
