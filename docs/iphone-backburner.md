# iPhone Companion: Backburner

The **Speed / Aumenta velocità** page integrates [Backburner by StayLameBro](https://github.com/StayLameBro/backburner). It is mutually exclusive with Companion's existing subagent runtime. The Mac selector is in **Tools → iPhone**. A compatible wired iPhone must be present before the speed option appears; after a disconnect the selected mode remains visible so you can return to Agent.

## Requirements and first use

1. Use Apple Silicon and macOS 14 or later for acceleration (the rest of the Mac app retains its macOS 12 deployment target). The original reference configuration is a 24 GB M4 Pro MacBook Pro and an A19 Pro/A18 Pro iPhone. The Companion app requires iOS 18.6 or later. An iPhone must negotiate USB at **10 Gb/s or faster**; ordinary USB 2 iPhone cables and non-Pro models with USB 2 cannot enable this mode. Thunderbolt certification alone is not the check.
2. Install Companion 0.0.3 by signing/sideloading its IPA. Preserve the `com.apple.developer.kernel.increased-memory-limit` entitlement, as in the original Backburner app. Keep the phone unlocked and Companion in the foreground.
3. Obtain the **Qwen3.8-27B GGUF or a compatible derivative (including abliterated/uncensored)** and **dflash2-v2-q4km-self16.gguf**, following the pinned upstream README and model preparation instructions in `studio/backend/vendor/backburner`. Weights are not included. Quantization is checked per tensor: mixed/dynamic quantizations such as UD and GSQ keep their exact weights; filename and `general.file_type` do not gate compatibility.
4. The original 64k + DFlash2 profile keeps the upstream minimum 20,000 MiB GPU wired-limit check (reference: a 24 GB Mac); accepting a smaller target quantization does not remove that requirement. A 16 GB Mac does not satisfy this original profile, and larger quantizations may require more than 24 GB. Follow upstream's Mac GPU memory setup: `sudo sysctl iogpu.wired_limit_mb=20480` after each reboot, on a Mac with enough memory for the original profile. Unsloth checks this prerequisite and does not apply it automatically.
5. On iPhone open **Speed → Increase speed**. This drains and unloads the subagent before starting Backburner. On Mac open **Tools → iPhone → Prepare iPhone**, enter your model and the original draft paths, and choose **Prepare and copy over USB**. The exact upstream split script uses L40 for iPhone18,* (A19 generation), L52 otherwise, and drops optional MTP blocks. Larger weights move the split later in multiples of four to fit conservative 6/3 GiB tail-weight budgets; it copies the tail, original 16k ANE template, and source identity through upstream's USB transfer protocol.
6. When copying finishes, turn Increase speed off and on on the phone, then wait for the tail to load. Load the same Qwen GGUF on Mac and select **Increase speed** in Tools. Changing modes requires idle Mac inference and no running/queued Companion tasks. Select **Agent** to reload the standard Mac engine and restore your Companion preferences.

## Interchangeable target weights

Supported weight types in the pinned Metal engine: F32, F16, BF16; Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0; Q2_K, Q3_K, Q4_K, Q5_K, Q6_K; IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS; MXFP4 and TQ2_0. File-level labels such as Q4_K_M or IQ3_M are mixtures of these tensor types. NVFP4 and TQ1_0 lack the required Metal kernels in this engine and are rejected before transfer; intermediate Q8_1/Q8_K and integer/F64 weights are also rejected. Formats accepted by the validator still require enough Mac/iPhone memory. Merge sharded GGUFs into one file before preparation.

Compatibility requires Qwen3.8 ancestry in model metadata, 64 trunk layers (additional MTP layers are excluded from the tail), and the original 27B dimensions, attention/GDN geometry and four-layer full-attention interval. The embedding/output vocabulary stays at 248,320 tokens for the original draft; derivatives that resize it are rejected. Other Qwen sizes/architectures are not covered. The DFlash2 draft remains the original Qwen3.8-27B draft; acceptance rate may differ for fine-tunes. No conversion to IQ4_XS or replacement/download of the selected target occurs. The upstream SME matrix optimization applies only to IQ4_XS tensors; other weight types use the upstream Metal path. The ANE attention template operates on KV and is independent of target weight quantization.

To change weights, return to Agent, load the desired compatible GGUF on Mac, prepare/copy that GGUF's tail, restart Speed on iPhone, then enable Increase speed on Mac. A source SHA mismatch blocks acceleration until both devices use the same weights.

## Original behavior and integration boundaries

The integration invokes upstream `serve.sh` without changing its inference flags: 64k local q8_0 KV, one slot, DFlash2 plus n-gram speculation, original Metal/SME kernels, split prefill, phone-held KV, and the original SSD prompt-cache proxy. Remote context is sized using upstream's 9,400 MiB wired ceiling and 512 MiB app reserve, capped at 262,144 tokens.

Below 64k context the phone accelerates sufficiently long prompt reads; writing speed is the Mac's. Beyond 64k the phone holds old KV and computes attention using its GPU/Neural Engine. The original automatic 60-second phone fallback and algorithmic limitations are preserved. Performance numbers in the upstream README are upstream measurements, not measurements of this integration.

The selector controls one wired iPhone per Mac session. Upstream's separate experimental generic RPC/draft/FFN modes and multi-phone command-line orchestration are not exposed. The integrated function is the upstream default profile, not a new general-purpose accelerator for every Unsloth model.

Unmodified upstream sources and their SHA-256 checksums are in `studio/backend/vendor/backburner/UPSTREAM.json`. Backburner is pinned at `c78e38b6bf74c5672aaaf5c66668b6540b97bd37`; its llama.cpp engine is pinned at `f8c76832b9bb7710db52143d9c89adcda5402718`. MIT licenses and attribution are included in the Mac backend and iPhone notices/page.

The native adapter changes service ownership, cancellation, USB-only listener binding, environment restoration and sandbox paths to `Documents/Backburner`. Its compute kernels and binary wire protocols remain upstream code. The private Backburner framework exports only its Objective-C bridge, preventing C-symbol collisions with Companion's existing llama.cpp. The simulator adapter explicitly reports unavailable acceleration.

The Mac runtime is bundled in the release, verified against its manifest, and installed under `<studio root>/backburner/<pinned engine commit>/runtime`. The standard llama.cpp updater uses a different directory and cannot overwrite this engine. Its models, draft setting and cache also live separately. SSD prompt/KV caches are isolated by target content SHA-256 and draft file identity; switching quantization or fine-tune cannot reuse another model's states. Switching to Agent restores the caller's original Mac load settings.

## Build and validation

Run `unsloth-companion/scripts/build_backburner_runtime.sh` before building the Mac app or IPA. It fetches the pinned engine and generates the private device/simulator framework and Mac binaries. For the original ANE template set `UNSLOTH_BACKBURNER_COREML_PYTHON` to a build-only Python environment with `coremltools==9.0` and `numpy==2.2.6`. No such build tooling is needed by end users.

Backend tests: `studio/backend/tests/test_backburner_companion.py`. Native lifecycle regression: `python3 unsloth-companion/scripts/test_backburner_lifecycle.py` after the runtime build. The latter uses actual TCP clients and the generated native adapter on macOS; only the endpoint, sandbox home and iOS-only available-memory query are substituted. It verifies three start/stop cycles, open clients, closed ports, protocol version and environment restoration. It does not test phone inference.

The release includes build and automated-test evidence. A physical compatible iPhone was unavailable during validation. USB discovery on hardware, model transfer, Metal/ANE inference, background/disconnect recovery and performance on real Mac/iPhone pairs remain acceptance checks; the release is marked prerelease for that reason.
