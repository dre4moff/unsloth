import Combine
import Foundation
@preconcurrency import Network

struct DiscoveredDesktop: Identifiable, Equatable {
    let id: String
    let name: String
    let endpoint: NWEndpoint

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id && lhs.name == rhs.name }
}

@MainActor
final class CompanionDiscoveryService: ObservableObject {
    @Published private(set) var desktops: [DiscoveredDesktop] = []
    @Published private(set) var errorDescription: String?
    private var browser: NWBrowser?
    private let queue = DispatchQueue(label: "com.unsloth.companion.discovery")

    func start() {
        guard browser == nil else { return }
        let parameters = NWParameters.tcp
        parameters.includePeerToPeer = true
        let browser = NWBrowser(for: .bonjourWithTXTRecord(type: CompanionProtocol.serviceType, domain: "local."), using: parameters)
        browser.stateUpdateHandler = { [weak self] state in
            if case .failed(let error) = state {
                guard let owner = self else { return }
                Task { @MainActor in owner.errorDescription = error.localizedDescription }
            }
        }
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            let values = results.compactMap { result -> DiscoveredDesktop? in
                guard case .service(let name, _, _, _) = result.endpoint else { return nil }
                let endpoint: NWEndpoint
                if case .bonjour(let record) = result.metadata,
                   let host = record["host"], !host.isEmpty,
                   let portText = record["port"],
                   let portValue = UInt16(portText),
                   let port = NWEndpoint.Port(rawValue: portValue) {
                    endpoint = .hostPort(host: NWEndpoint.Host(host), port: port)
                } else {
                    endpoint = result.endpoint
                }
                return DiscoveredDesktop(id: result.endpoint.debugDescription, name: name, endpoint: endpoint)
            }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
            guard let owner = self else { return }
            Task { @MainActor in owner.desktops = values }
        }
        browser.start(queue: queue)
        self.browser = browser
    }

    func stop() {
        browser?.cancel()
        browser = nil
        desktops = []
    }
}
