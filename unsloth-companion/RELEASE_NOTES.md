# Unsloth Companion prototype release notes

Release tag: `v0.1.800-mlx.10-companion.2`

## The release

This public prototype combines model-aware compaction, persistent objectives, and
an expandable network of local iPhone workers. The emphasis is the Companion
architecture: one or more authenticated phones can act as offline subagents while
the Mac remains the orchestrator and fallback.

This corrective release makes the Desktop subagent contract genuinely asynchronous.
`submit` returns a job ID immediately, the Mac continues working, and later non-blocking
`status`/`collect` calls read results from a mailbox scoped to the originating chat.
Stopping the Mac turn no longer implicitly cancels accepted iPhone work. Completed,
partial, and failed outcomes remain available so the Mac can evaluate or fall back when
the result is actually needed.

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

- `Unsloth_0.1.800-mlx.10_aarch64.dmg`: community Apple Silicon/macOS 12+ build,
  ad-hoc signed and not Apple-notarized.
- `Unsloth-Companion_0.0.1_unsigned.ipa`: arm64/iOS 18.6 prototype, unsigned and
  requiring a user-supplied signing identity before installation.
- `SHA256SUMS.txt`: release artifact digests.
- GitHub-generated source archives containing the open implementation and docs.

## Validation boundary

Corrective-release verification passed 15 desktop protocol/manager tests, including
the synchronous-to-asynchronous boundary, delayed collection, chat isolation, failure
retention, and temporary-media cleanup. The iPhone wire protocol and Swift app are
unchanged; the 13 Swift unit and 3 signed iOS UI results remain evidence from the prior
prototype release. No new physical iPhone subagent run was performed because final
device testing is reserved to the user. See `IMPLEMENTATION_STATUS.md` for the exact
boundary.

This is a community prototype and not an official Unsloth mobile release.
