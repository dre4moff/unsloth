import SwiftUI

struct BackburnerView: View {
    @ObservedObject var appModel: CompanionAppModel
    @ObservedObject private var acceleration: BackburnerServiceModel

    init(appModel: CompanionAppModel) {
        self.appModel = appModel
        _acceleration = ObservedObject(wrappedValue: appModel.acceleration)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Increase speed", isOn: Binding(
                        get: { acceleration.selected },
                        set: { selected in Task { await appModel.selectAcceleration(selected) } }
                    ))
                    .disabled(acceleration.transitioning)
                    .accessibilityIdentifier("backburner-mode-toggle")
                    Text("This mode stops the iPhone subagent and uses its GPU and Neural Engine with your Mac. Keep Companion open and unlocked.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("USB connection") {
                    LabeledContent("Cable", value: acceleration.cableAddress.isEmpty ? String(localized: "Not connected") : acceleration.cableAddress)
                    Text("Use an iPhone 15 Pro or newer with USB at 10 Gb/s. Thunderbolt is not required. The USB 2 cable included with iPhone is too slow.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("Acceleration") {
                    LabeledContent("Prefill", value: acceleration.tailState)
                    if !acceleration.detail.isEmpty { Text(acceleration.detail).font(.footnote) }
                    LabeledContent("Tokens read", value: acceleration.tokens.formatted())
                    LabeledContent("Last chunk") { Text("\(acceleration.tokensPerSecond, specifier: "%.1f") tok/s") }
                    LabeledContent("Phone-held context", value: acceleration.heldKeys.formatted())
                    LabeledContent("Mac", value: acceleration.macPhase.isEmpty ? String(localized: "Waiting") : acceleration.macPhase)
                    Text("On Mac, open Tools → iPhone → Increase speed. Prepare your Qwen3.8-27B GGUF (supported quantization, including abliterated/uncensored variants) and DFlash2 draft, then switch this page off and on after the USB transfer.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Text("Long prompts are read in parallel while context fits on the Mac. iPhone holds older context beyond the Mac's local cache: 64k in the original profile, up to 8k in the 16 GB profile. A 50k total context is supported when phone memory allows.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("Credits") {
                    Link("Backburner by StayLameBro", destination: URL(string: "https://github.com/StayLameBro/backburner")!)
                    Text("Original MIT-licensed engine, kernels and protocols by StayLameBro. Integrated with a separate pinned runtime in Unsloth Companion.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Increase speed")
        }
    }
}
