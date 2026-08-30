import XCTest

final class Unsloth_CompanionUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testEnglishNavigationAndSimulatorDownloadGuard() throws {
        let app = launch(language: "en", locale: "en_US")

        XCTAssertTrue(app.tabBars.buttons["Dashboard"].waitForExistence(timeout: 15))
        XCTAssertFalse(app.alerts.firstMatch.exists)
        XCTAssertTrue(app.switches["Use this iPhone"].exists)
        XCTAssertTrue(app.tabBars.buttons["Models"].exists)
        XCTAssertTrue(app.tabBars.buttons["Activity"].exists)
        XCTAssertTrue(app.tabBars.buttons["Storage"].exists)
        XCTAssertTrue(app.tabBars.buttons["Settings"].exists)

        app.tabBars.buttons["Models"].tap()
        XCTAssertTrue(app.staticTexts["Model downloads are available only on a physical iPhone."].waitForExistence(timeout: 5))
        let downloadButton = app.buttons["Download"].firstMatch
        XCTAssertTrue(downloadButton.exists)
        XCTAssertFalse(downloadButton.isEnabled)

        app.tabBars.buttons["Storage"].tap()
        XCTAssertTrue(app.staticTexts["Managed storage"].waitForExistence(timeout: 5))
        XCTAssertTrue(button(in: app, startingWith: "Clear temporary cache").waitForExistence(timeout: 5))
        XCTAssertTrue(scrollToButton(in: app, startingWith: "Reset Companion"))
    }

    @MainActor
    func testItalianNavigationAndSimulatorDownloadGuard() throws {
        let app = launch(language: "it", locale: "it_IT")

        XCTAssertTrue(app.tabBars.buttons["Dashboard"].waitForExistence(timeout: 15))
        XCTAssertFalse(app.alerts.firstMatch.exists)
        XCTAssertTrue(app.switches["Usa questo iPhone"].exists)
        XCTAssertTrue(app.tabBars.buttons["Modelli"].exists)
        XCTAssertTrue(app.tabBars.buttons["Attività"].exists)
        XCTAssertTrue(app.tabBars.buttons["Spazio"].exists)
        XCTAssertTrue(app.tabBars.buttons["Impostazioni"].exists)

        app.tabBars.buttons["Modelli"].tap()
        XCTAssertTrue(app.staticTexts["I download dei modelli sono disponibili solo su un iPhone fisico."].waitForExistence(timeout: 5))
        let downloadButton = app.buttons["Scarica"].firstMatch
        XCTAssertTrue(downloadButton.exists)
        XCTAssertFalse(downloadButton.isEnabled)

        app.tabBars.buttons["Spazio"].tap()
        XCTAssertTrue(app.staticTexts["Spazio gestito"].waitForExistence(timeout: 5))
        XCTAssertTrue(button(in: app, startingWith: "Pulisci cache temporanea").waitForExistence(timeout: 5))
        XCTAssertTrue(scrollToButton(in: app, startingWith: "Ripristina Companion"))
    }

    @MainActor
    func testFreshInstallShowsOnboardingBeforeStartingCompanion() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "-ShowCompanionOnboarding"]
        app.launch()
        XCTAssertTrue(app.staticTexts["Private by design"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Foreground continuity"].exists)
        XCTAssertTrue(app.staticTexts["Controlled storage"].exists)
        XCTAssertTrue(app.buttons["Continue"].exists)
        XCTAssertFalse(app.tabBars.firstMatch.exists)
    }

    @MainActor
    private func launch(language: String, locale: String) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale, "-SkipCompanionOnboarding"]
        app.launch()
        return app
    }

    @MainActor
    private func button(in app: XCUIApplication, startingWith label: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", label)).firstMatch
    }

    @MainActor
    private func scrollToButton(in app: XCUIApplication, startingWith label: String) -> Bool {
        let target = button(in: app, startingWith: label)
        for _ in 0..<4 where !target.exists { app.swipeUp() }
        return target.exists
    }
}
