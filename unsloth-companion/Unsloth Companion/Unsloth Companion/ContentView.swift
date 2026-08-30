import SwiftUI

struct ContentView: View {
    @ObservedObject var appModel: CompanionAppModel
    @ObservedObject private var service: CompanionServiceModel
    @State private var lastInteraction = Date()
    @State private var guardVisible = false

    init(appModel: CompanionAppModel) {
        self.appModel = appModel
        _service = ObservedObject(wrappedValue: appModel.service)
    }

    var body: some View {
        ZStack {
            if appModel.hasCompletedOnboarding {
                TabView {
                    DashboardView(appModel: appModel).tabItem { Label("Dashboard", systemImage: "iphone.and.arrow.forward") }
                    ModelsView(appModel: appModel).tabItem { Label("Models", systemImage: "cpu") }
                    ActivitiesView(appModel: appModel).tabItem { Label("Activity", systemImage: "clock.arrow.circlepath") }
                    StorageView(appModel: appModel).tabItem { Label("Storage", systemImage: "internaldrive") }
                    SettingsView(appModel: appModel).tabItem { Label("Settings", systemImage: "gearshape") }
                }
                .disabled(appModel.isBusy)
            } else {
                OnboardingView { appModel.completeOnboarding() }
            }
            if appModel.isBusy {
                Color.black.opacity(0.18).ignoresSafeArea()
                ProgressView().controlSize(.large).padding(28).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
            }
            if guardVisible {
                CompanionGuardView(service: service) { guardVisible = false; lastInteraction = Date() }.transition(.opacity)
            }
        }
        .simultaneousGesture(DragGesture(minimumDistance: 0).onEnded { _ in lastInteraction = Date(); guardVisible = false })
        .task {
            await appModel.bootstrap()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(10))
                if service.settings.guardScreenEnabled, service.settings.serviceEnabled,
                   service.state == .ready, Date().timeIntervalSince(lastInteraction) >= 60 {
                    withAnimation(.easeInOut(duration: 0.35)) { guardVisible = true }
                }
                try? await appModel.refresh()
            }
        }
        .alert("Error", isPresented: Binding(get: { appModel.presentedError != nil }, set: { if !$0 { appModel.presentedError = nil } })) {
            Button("OK", role: .cancel) { appModel.presentedError = nil }
        } message: { Text(appModel.presentedError ?? "") }
    }
}

private struct CompanionGuardView: View {
    @ObservedObject var service: CompanionServiceModel
    let dismiss: () -> Void

    var body: some View {
        ZStack {
            Color(red: 0.004, green: 0.006, blue: 0.008).ignoresSafeArea()
            VStack(spacing: 10) {
                Image(systemName: service.currentTaskID == nil ? "bolt.horizontal.circle" : "waveform.path.ecg")
                    .font(.system(size: 24, weight: .light)).foregroundStyle(.green.opacity(0.55))
                Text(service.currentTaskID == nil ? String(localized: "Companion ready") : String(localized: "Task running")).font(.footnote).foregroundStyle(.white.opacity(0.38))
                Text("Tap to return").font(.caption2).foregroundStyle(.white.opacity(0.2))
            }
        }
        .contentShape(Rectangle()).onTapGesture(perform: dismiss)
        .accessibilityLabel("Companion guard screen").accessibilityHint("Double tap to return to the app")
    }
}
