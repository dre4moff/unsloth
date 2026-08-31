# Unsloth Companion prototype release notes

Release tag: `v0.1.800-mlx.11-companion.3`

## The release

This public prototype combines model-aware compaction, persistent objectives, and
an expandable network of local iPhone workers. The emphasis is the Companion
architecture: one or more authenticated phones can act as offline subagents while
the Mac remains the orchestrator and fallback.

This corrective release makes the asynchronous contract authoritative to the main model
and turns the connected phones into a visible worker pool. Every model pass receives live
connected/eligible/idle/busy counts and per-kind parallel capacity. The orchestrator fans
independent `items` out across compatible idle phones, immediately continues its own Mac
branch, receives mailbox state changes between passes, and uses bounded `wait` only as the
final fork/join barrier. The runtime prompt explicitly forbids describing submission as
blocking or the parallelism as conceptual.

`submit` still returns only acceptance and job IDs before phone inference completes.
`status` and `collect` remain non-blocking; stopping the Mac turn does not cancel accepted
iPhone work, and completed, partial, or failed outcomes remain available for evaluation
and fallback.

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

- `Unsloth_0.1.800-mlx.11_aarch64.dmg`: community Apple Silicon/macOS 12+ build,
  ad-hoc signed and not Apple-notarized.
- `Unsloth-Companion_0.0.1_unsigned.ipa`: arm64/iOS 18.6 prototype, unsigned and
  requiring a user-supplied signing identity before installation.
- `SHA256SUMS.txt`: release artifact digests.
- GitHub-generated source archives containing the open implementation and docs.

## Validation boundary

No automated or physical-device tests were run for `companion11`, as requested by the
user. The release build and final-artifact integrity checks are not presented as behavioral
tests. The 15 Desktop protocol/manager tests from `companion10`, plus 13 Swift unit and
3 signed iOS UI results from the earlier unchanged iPhone implementation, are carried
forward only as historical evidence. The iPhone wire protocol and Swift app remain
unchanged and the existing unsigned IPA is reused byte for byte. See
`IMPLEMENTATION_STATUS.md` for the exact boundary.

This is a community prototype and not an official Unsloth mobile release.
