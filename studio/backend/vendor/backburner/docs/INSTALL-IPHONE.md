# Installing the iPhone (or iPad) app

Two ways. Both give the same app; the Mac scripts find it on the phone either way (`scripts/bundle-id.sh`).

| | AltStore + the release IPA | Xcode |
|---|---|---|
| Apple account | your free Apple ID | an Apple developer team id |
| Build anything for the phone | no | yes (`scripts/build-iphone.sh`, ~10 min) |
| Re-sign | every 7 days (AltStore does it) | every 7 days on a free team, yearly on a paid one |

You need an iPhone 15 Pro or newer, or an iPad with an M-series chip (M1 or newer). The A19 Pro (iPhone 17 Pro / Pro Max)
has the GPU matrix units that make the phone's half of prefill 2.4x faster; the A18 Pro works but helps less.

**iPads** install the same app (v0.0.2 and newer) and every step below is the same. iPads are **not tested yet**: we
don't know how much memory iPadOS gives the app (step 5 prints it), or how fast an M-series iPad runs its layers. If you
try one, please [post your results](https://github.com/StayLameBro/backburner/issues/new?template=results.yml), including
the `app budget` line. The app runs full screen on iPad (no Split View), because it has to stay in front while the Mac uses it.

## With AltStore (no developer account)

> **Not yet tested by us.** We've verified the IPA carries the memory entitlement, and that a free Apple account gets the
> full ~6 GB through Xcode; AltStore says version 2.2+ keeps that entitlement when it signs. If you install this way, please
> [post your results](https://github.com/StayLameBro/backburner/issues/new?template=results.yml) with the `app budget` line
> from step 5, so we can confirm it.

1. **AltServer on the Mac.** Download it from [altstore.io](https://altstore.io) and open it. It has no window, only a
   diamond-shaped icon in the menu bar. No icon? On a MacBook with a notch, a full menu bar hides icons behind it: quit a
   menu-bar app or two (or hold ⌘ and drag icons off the bar) until it appears. Also check System Settings → Menu Bar →
   Allow in the Menu Bar → AltServer.
2. **AltStore on the iPhone.** Plug the phone in, unlock it, then in the AltServer menu: *Install AltStore* → your iPhone.
   AltServer asks for your Apple ID: that is how free sideloading signs apps. Use **AltStore 2.2 or newer**.
3. **Developer Mode.** On the iPhone: Settings → Privacy & Security → Developer Mode → on, then restart when asked.
   If iOS says the developer isn't trusted: Settings → General → VPN & Device Management → your Apple ID → Trust.
4. **Backburner.** In AltStore: **Sources → +**, paste this source URL, then install Backburner from it. AltStore then tells you
   when there's a new version:

   ```
   https://raw.githubusercontent.com/StayLameBro/backburner/main/altstore/source.json
   ```

   Or download `Backburner.ipa` from the [releases page](https://github.com/StayLameBro/backburner/releases) onto the iPhone
   (or AirDrop it over), then in AltStore: My Apps → **+** → `Backburner.ipa`.
5. **Check the memory budget.** Open Backburner, keep it in front, plug the phone into a 10 Gb/s USB-C port, and on the Mac:

   ```bash
   scripts/phone-up.sh
   # phone-up: wired iPhone at 169.254.x.x: phone-attn v…, prefill tail down, … MiB system wired, 6xxx MiB app budget
   ```

   The app budget should be around 6,000 MiB on a 17 Pro Max. Around 3,000 means the app lost its increased-memory-limit
   entitlement on the way: the phone then has no room for its half of the model. That happens with **SideStore** (free
   accounts, [SideStore#1616](https://github.com/SideStore/SideStore/issues/1616)) and AltStore older than 2.2. Reinstall with
   AltStore 2.2+.

Then continue with step 4 of the README's Setup (the phone's half of the model).

**Every 7 days** a free Apple ID signature expires and the app stops opening. AltStore refreshes it in the background when the
phone and a Mac running AltServer are on the same Wi-Fi, or tap *Refresh All* in AltStore. Free accounts can have 3 sideloaded
apps at a time (AltStore counts as one).

## With Xcode (developer team id)

```bash
export DEVELOPMENT_TEAM=<your team id>       # Xcode > Settings > Accounts
UDID=<your iPhone's UDID> scripts/build-iphone.sh
```

It builds llama.cpp for iOS, the SME2 attention kernel and the app, signs it as `app.backburner.<team id>` and installs it.

## Making the IPA (maintainers)

```bash
IPA=1 scripts/build-iphone.sh    # -> ios/build/Backburner.ipa
```

The IPA is unsigned except for an ad-hoc signature that carries the entitlements (`ios/Backburner/Sidecar/Sidecar.entitlements`),
so AltStore can request increased-memory-limit when it signs the app with the user's Apple ID. The script stops if the
entitlement is missing.

For a new release: bump `MARKETING_VERSION` (and `CURRENT_PROJECT_VERSION`) in the Xcode project, build, attach the IPA to the
GitHub release, then add a version to `altstore/source.json` with the same version and build numbers, the IPA's size in bytes
and its download URL. AltStore refuses an install whose version or permissions don't match the source.
