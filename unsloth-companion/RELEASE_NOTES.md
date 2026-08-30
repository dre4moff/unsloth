# Unsloth Companion prototype release notes

Release tag: `v0.1.800-mlx.9-companion.1`

## The release

This public prototype combines model-aware compaction, persistent objectives, and
an expandable network of local iPhone workers. The emphasis is the Companion
architecture: one or more authenticated phones can act as offline subagents while
the Mac remains the orchestrator and fallback.

The complete source is included: Swift app, desktop manager and UI, protocol and
schema, llama.cpp runtime bridge, model registry, scheduler, pairing/security,
storage recovery, tests, build scripts, and implementation documentation.

## Why it matters

Additional PCs, GPUs, and servers are expensive. Compatible phones that a user
already owns can run bounded parallel work—especially while connected to power
and otherwise idle. The current iOS lifecycle requires the app to stay foregrounded;
the longer-term design covers explicit overnight queues, mixed iPhone generations,
future Android workers, and future capability-routed image generation.

Parallelism is task-level: separate subagent tasks, chunks, frames, batches, or
shards may run on different devices. A single generation and KV cache are not split
between phones.

## Included artifacts

- `Unsloth_0.1.800-mlx.9_aarch64.dmg`: community Apple Silicon/macOS 12+ build,
  ad-hoc signed and not Apple-notarized.
- `Unsloth-Companion_0.0.1_unsigned.ipa`: arm64/iOS 18.6 prototype, unsigned and
  requiring a user-supplied signing identity before installation.
- `SHA256SUMS.txt`: release artifact digests.
- GitHub-generated source archives containing the open implementation and docs.

## Validation boundary

Public-release verification passed 13 desktop protocol/manager tests, 13 Swift
unit tests, and 3 signed iOS UI tests. The unsigned IPA was rebuilt from the tagged
source. No new physical iPhone subagent run was performed. A signed physical build
and resumed 61.8 MB transfer had already been verified; the full five-model
physical smoke/stress matrix remains open. See `IMPLEMENTATION_STATUS.md` for the
exact evidence.

This is a community prototype and not an official Unsloth mobile release.
