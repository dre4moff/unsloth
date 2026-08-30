import SwiftUI
import UIKit

extension Notification.Name { static let companionMemoryWarning = Notification.Name("CompanionMemoryWarning") }

final class CompanionAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        Task { @MainActor in ModelDownloadManager.shared.setBackgroundCompletionHandler(completionHandler) }
    }

    func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
        NotificationCenter.default.post(name: .companionMemoryWarning, object: nil)
    }
}

@main
struct Unsloth_CompanionApp: App {
    @UIApplicationDelegateAdaptor(CompanionAppDelegate.self) private var appDelegate
    @StateObject private var appModel = CompanionAppModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(appModel: appModel)
                .onReceive(NotificationCenter.default.publisher(for: .companionMemoryWarning)) { _ in appModel.service.memoryWarning() }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: appModel.service.sceneDidBecomeActive()
            case .inactive: appModel.service.sceneWillResignActive()
            case .background: appModel.service.sceneDidEnterBackground()
            @unknown default: break
            }
        }
    }
}
