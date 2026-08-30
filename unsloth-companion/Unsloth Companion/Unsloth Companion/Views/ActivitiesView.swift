import SwiftUI

struct ActivitiesView: View {
    @ObservedObject var appModel: CompanionAppModel
    @ObservedObject private var service: CompanionServiceModel
    @State private var clearHistory = false

    init(appModel: CompanionAppModel) {
        self.appModel = appModel
        _service = ObservedObject(wrappedValue: appModel.service)
    }

    var body: some View {
        NavigationStack {
            List {
                if let task = service.currentTaskID {
                    Section("Current task") {
                        LabeledContent("Identifier", value: String(task.uuidString.prefix(12)))
                        LabeledContent("State", value: serviceStateLabel(service.state))
                        Button("Cancel task", role: .destructive) { Task { await service.cancelCurrentTask() } }
                    }
                }
                Section("History") {
                    if appModel.activities.isEmpty { ContentUnavailableView("No activity", systemImage: "clock") }
                    ForEach(appModel.activities) { activity in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack { Text(kindLabel(activity.kind)).font(.headline); Spacer(); Text(activityStateLabel(activity.state)).foregroundStyle(color(activity.state)) }
                            Text(activity.startedAt.formatted(date: .abbreviated, time: .standard)).font(.caption).foregroundStyle(.secondary)
                            Text(detailLabel(activity.detail)).font(.footnote).lineLimit(3)
                            if let duration = activity.durationMS { Text("\(duration) ms · \(activity.tokens ?? 0) tokens").font(.caption2).foregroundStyle(.secondary) }
                        }
                        .swipeActions { Button("Delete", role: .destructive) { Task { await appModel.deleteActivity(activity) } } }
                    }
                }
            }
            .navigationTitle("Activity")
            .toolbar { Button(role: .destructive) { clearHistory = true } label: { Image(systemName: "trash") }.disabled(appModel.activities.isEmpty).accessibilityLabel("Clear history") }
            .refreshable { try? await appModel.refresh() }
            .alert("Clear all activity?", isPresented: $clearHistory) {
                Button("Cancel", role: .cancel) {}
                Button("Clear", role: .destructive) { Task { await appModel.deleteStorage(.activities) } }
            } message: { Text("This removes every saved activity entry but does not delete models.") }
        }
    }

    private func color(_ state: ActivityState) -> Color {
        switch state { case .completed: .green; case .failed: .red; case .cancelled: .orange; default: .secondary }
    }

    private func kindLabel(_ kind: CompanionTaskKind) -> String {
        switch kind {
        case .subagent: String(localized: "Subagent")
        case .classification: String(localized: "Classification")
        case .summary: String(localized: "Summary")
        case .contextCompression: String(localized: "Context compression")
        case .extraction: String(localized: "Extraction")
        case .verification: String(localized: "Verification")
        case .reranking: String(localized: "Reranking")
        case .lightweightPlanning: String(localized: "Lightweight planning")
        case .vision: String(localized: "Vision")
        case .ocr: String(localized: "OCR")
        case .videoSummary: String(localized: "Video summary")
        case .audioTranscription: String(localized: "Audio transcription")
        case .audioAnalysis: String(localized: "Audio analysis")
        case .dsp: String(localized: "DSP")
        }
    }

    private func activityStateLabel(_ state: ActivityState) -> String {
        switch state {
        case .accepted: String(localized: "Accepted")
        case .running: String(localized: "Running")
        case .completed: String(localized: "Completed")
        case .failed: String(localized: "Failed")
        case .cancelled: String(localized: "Cancelled")
        }
    }

    private func detailLabel(_ detail: String) -> String {
        switch detail {
        case "Accepted": String(localized: "Accepted")
        case "Running": String(localized: "Running")
        case "Completed": String(localized: "Completed")
        default: detail
        }
    }

    private func serviceStateLabel(_ state: CompanionServiceState) -> String {
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
