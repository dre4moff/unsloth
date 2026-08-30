import SwiftUI

struct OnboardingView: View {
    let complete: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 10) {
                    Image(systemName: "iphone.and.arrow.forward")
                        .font(.system(size: 52, weight: .medium))
                        .foregroundStyle(.green)
                    Text("Unsloth Companion").font(.largeTitle.bold())
                    Text("Turn this iPhone into a private local co-processor for Unsloth Desktop.")
                        .font(.title3).foregroundStyle(.secondary)
                }

                feature("lock.shield", "Private by design", "Pairing uses a six-digit code and signed device identities. Tasks stay on your local network.")
                feature("sun.max", "Foreground continuity", "Keep Companion open while it works. The screen stays awake when the service is ready; backgrounding safely returns unfinished work to the Mac.")
                feature("internaldrive", "Controlled storage", "Models download only on a physical iPhone. Cache is bounded, visible, and removable at any time.")

                Button(action: complete) {
                    Text("Continue").frame(maxWidth: .infinity).padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .accessibilityHint("Opens the Companion dashboard")
            }
            .padding(28)
            .frame(maxWidth: 620, alignment: .leading)
        }
        .background(Color(.systemBackground))
    }

    private func feature(_ symbol: String, _ title: LocalizedStringKey, _ detail: LocalizedStringKey) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: symbol).font(.title2).foregroundStyle(.green).frame(width: 34)
            VStack(alignment: .leading, spacing: 5) {
                Text(title).font(.headline)
                Text(detail).foregroundStyle(.secondary)
            }
        }
    }
}
