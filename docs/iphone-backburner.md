# iPhone Companion: Backburner

The **Speed / Aumenta velocità** page integrates [Backburner by StayLameBro](https://github.com/StayLameBro/backburner). It is mutually exclusive with Companion's existing subagent runtime. The Mac selector is in **Tools → iPhone**. A compatible wired iPhone must be present before the speed option appears; after a disconnect the selected mode remains visible so you can return to Agent.

## Requirements and first use

1. Use Apple Silicon and macOS 14 or later for acceleration (the rest of the Mac app retains its macOS 12 deployment target). The original reference configuration is a 24 GB M4 Pro MacBook Pro and an A19 Pro/A18 Pro iPhone. The Companion app requires iOS 18.6 or later. An iPhone must negotiate USB at **10 Gb/s or faster**; ordinary USB 2 iPhone cables and non-Pro models with USB 2 cannot enable this mode. Thunderbolt certification alone is not the check.
2. Install Companion 0.0.2 by signing/sideloading its IPA. Preserve the `com.apple.developer.kernel.increased-memory-limit` entitlement, as in the original Backburner app. Keep the phone unlocked and Companion in the foreground.
3. Obtain the **original Qwen3.8-27B IQ4_XS GGUF** and **dflash2-v2-q4km-self16.gguf**, following the pinned upstream README and model preparation instructions in `studio/backend/vendor/backburner`. Weights are not included. Other models and quantizations are rejected.
4. Follow upstream's Mac GPU memory setup: `sudo sysctl iogpu.wired_limit_mb=20480` after each reboot, on a Mac with enough memory for the original profile. Unsloth checks this prerequisite and does not apply it automatically.
5. On iPhone open **Speed → Increase speed**. This drains and unloads the subagent before starting Backburner. On Mac open **Tools → iPhone → Prepare iPhone**, enter the original model/draft paths, and choose **Prepare and copy over USB**. The exact upstream split script produces L40 for iPhone18,* (A19 generation), L52 otherwise; it copies the tail, original 16k ANE template, and source identity through upstream's USB transfer protocol.
6. When copying finishes, turn Increase speed off and on on the phone, then wait for the tail to load. Load the same original Qwen model on Mac and select **Increase speed** in Tools. Changing modes requires idle Mac inference and no running/queued Companion tasks. Select **Agent** to reload the standard Mac engine and restore your Companion preferences.

## Original behavior and integration boundaries

The integration invokes upstream `serve.sh` without changing its inference flags: 64k local q8_0 KV, one slot, DFlash2 plus n-gram speculation, original Metal/SME kernels, split prefill, phone-held KV, and the original SSD prompt-cache proxy. Remote context is sized using upstream's 9,400 MiB wired ceiling and 512 MiB app reserve, capped at 262,144 tokens.

Below 64k context the phone accelerates sufficiently long prompt reads; writing speed is the Mac's. Beyond 64k the phone holds old KV and computes attention using its GPU/Neural Engine. The original automatic 60-second phone fallback and algorithmic limitations are preserved. Performance numbers in the upstream README are upstream measurements, not measurements of this integration.

The selector controls one wired iPhone per Mac session. Upstream's separate experimental generic RPC/draft/FFN modes and multi-phone command-line orchestration are not exposed. The integrated function is the upstream default profile, not a new general-purpose accelerator for every Unsloth model.

Unmodified upstream sources and their SHA-256 checksums are in `studio/backend/vendor/backburner/UPSTREAM.json`. Backburner is pinned at `c78e38b6bf74c5672aaaf5c66668b6540b97bd37`; its llama.cpp engine is pinned at `f8c76832b9bb7710db52143d9c89adcda5402718`. MIT licenses and attribution are included in the Mac backend and iPhone notices/page.

The native adapter changes service ownership, cancellation, USB-only listener binding, environment restoration and sandbox paths to `Documents/Backburner`. Its compute kernels and binary wire protocols remain upstream code. The private Backburner framework exports only its Objective-C bridge, preventing C-symbol collisions with Companion's existing llama.cpp. The simulator adapter explicitly reports unavailable acceleration.

The Mac runtime is bundled in the release, verified against its manifest, and installed under `<studio root>/backburner/<pinned engine commit>/runtime`. The standard llama.cpp updater uses a different directory and cannot overwrite this engine. Its models, draft setting and cache also live separately. Switching to Agent restores the caller's original Mac load settings.

## Build and validation

Run `unsloth-companion/scripts/build_backburner_runtime.sh` before building the Mac app or IPA. It fetches the pinned engine and generates the private device/simulator framework and Mac binaries. For the original ANE template set `UNSLOTH_BACKBURNER_COREML_PYTHON` to a build-only Python environment with `coremltools==9.0` and `numpy==2.2.6`. No such build tooling is needed by end users.

Backend tests: `studio/backend/tests/test_backburner_companion.py`. Native lifecycle regression: `python3 unsloth-companion/scripts/test_backburner_lifecycle.py` after the runtime build. The latter uses actual TCP clients and the generated native adapter on macOS; only the endpoint, sandbox home and iOS-only available-memory query are substituted. It verifies three start/stop cycles, open clients, closed ports, protocol version and environment restoration. It does not test phone inference.

The release includes build and automated-test evidence. A physical compatible iPhone was unavailable during validation. USB discovery on hardware, model transfer, Metal/ANE inference, background/disconnect recovery and performance on real Mac/iPhone pairs remain acceptance checks; the release is marked prerelease for that reason.
