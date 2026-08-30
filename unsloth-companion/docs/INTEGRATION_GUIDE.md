# Integration guide for offline LLM applications

This guide extracts the reusable design from the Unsloth-specific implementation.
It is intended for developers who want to add local phone workers, robust context
compaction, or objective continuity to another offline LLM app.

## 1. Reuse the boundaries, not the product names

Treat the system as four interfaces:

| Interface | Responsibility | Reference implementation |
| --- | --- | --- |
| Context fitter | Keep each model pass inside the real active budget | `studio/backend/core/inference/context_window.py`, `checkpoint.py` |
| Objective store | Preserve goal/checklist state across compaction and reload | `studio/backend/core/inference/turn_checkpoint.py` |
| Worker protocol | Describe capabilities, tasks, leases, results, and cancellation | `unsloth-companion/protocol/` |
| Scheduler/transport | Pair devices, route tasks, validate results, and fall back | `studio/backend/core/companion/` |

An app can adopt these independently. The protocol does not require the Unsloth
frontend, and the context fitter does not require a phone.

## 2. Context compaction contract

Do not compact after a fixed number of messages. Compute a budget for the model
and request that will actually run:

```text
available prompt tokens = model context window
                        - requested reply reserve
                        - safety margin
```

Count the rendered prompt, including the active chat template, system message,
tool catalog, reasoning controls, and any runtime-specific wrapper. Re-run fitting
before every model pass, including passes after tool results. Cumulative processed
token usage is not the same as resident context occupancy.

When a checkpoint is safe:

1. preserve the system contract and newest user turn;
2. carry forward bounded, explicit instructions and the active objective/checklist;
3. archive older turns only if the running model can actually retrieve them;
4. fall back to rolling truncation when archive/search tools are not reachable;
5. store compaction notices as UI metadata rather than conversation content;
6. never store or reinject private chain-of-thought.

The detailed Unsloth behavior is documented in
`studio/AUTOMATIC_CONTEXT_COMPACTION.md`.

## 3. Objective and progress contract

Persist a small public execution record:

```json
{
  "objective": "Produce the requested release",
  "checklist": [
    {"item": "Verify source", "status": "completed"},
    {"item": "Publish artifacts", "status": "in_progress"}
  ],
  "strategy": "Build from the verified branch and publish a prerelease"
}
```

The record should survive context fitting and application reload. It should be
separate from hidden model reasoning and small enough to reinsert on every pass.

A bounded action counter can trigger a review, but it should not impose an
arbitrary stop. At review time, require one of:

- an actual checklist state transition;
- a concrete strategy change that addresses the stalled step;
- completion;
- a genuine blocker after practical alternatives are exhausted.

Repeating the same plan in different words is not progress.

## 4. Worker capability contract

Never assign work based only on a device name. Require an authenticated capability
report containing at least:

- protocol and runtime versions;
- task kinds;
- selected model and model digest;
- context and output limits;
- supported modalities;
- available storage;
- battery, charging, low-power, and thermal state;
- queue depth and observed latency.

Keep optional capabilities additive so a text-only Android worker can participate
without implementing iPhone-specific media APIs.

## 5. Task model

Every task should have:

- a globally unique task ID;
- an idempotency key;
- a kind and versioned input;
- an optional result schema;
- priority, timeout, and lease duration;
- optional parent/shard identity;
- input and output size limits;
- one terminal result: success, error, or cancellation.

Use content digests for transferred blobs and canonical result payloads. Reject
late results after cancellation or lease expiry. Retry only idempotent units.

Do not split a single autoregressive generation or KV cache across phones. Split at
semantic boundaries: documents, frames, audio chunks, candidates, batches, or
other independently verifiable shards.

## 6. Pairing and transport

Bonjour or another discovery layer may advertise presence, but discovery is not
authentication. The reference design uses:

1. a long-lived P-256 identity per endpoint;
2. a local TLS certificate and certificate pinning;
3. a fresh nonce signed by the phone;
4. a six-digit short authentication string displayed on both devices;
5. explicit confirmation on both endpoints;
6. signed challenges for later sessions;
7. replay windows and timestamp validation.

Use encrypted local transport for task metadata and blobs. Keep private keys in
the platform key store when available. Support revocation and key rotation.

## 7. Scheduler and fallback

Filter workers first, then score them. A worker is eligible only when it is paired,
authenticated, ready, compatible with the task, within storage limits, and not in
a disallowed power or thermal state.

A simple score can combine normalized values:

```text
score = model/capability fit
      + available context and storage
      + charging/battery preference
      - thermal penalty
      - queue penalty
      - latency penalty
```

Use deterministic tie-breaking. Record the assignment reason for diagnostics
without logging task payloads.

The desktop remains the authority. If no worker qualifies, run locally. If a
non-divisible task disconnects, discard partial output and either retry from the
start or fall back. If the user explicitly cancels, do not automatically reassign.

## 8. Output integrity

The orchestrator chooses an output budget that the worker advertises it can
support. The worker must report why generation stopped. Treat context overflow,
token-cap exhaustion without EOS, malformed schema output, checksum mismatch, and
runtime cancellation as unusable results.

Structured output is optional. Free-text subagents should return free text by
default and use JSON only when the caller supplied a result schema.

## 9. Mobile lifecycle and storage

Mobile operating systems are not general-purpose servers. The iOS reference app
accepts work only while foregrounded and uses background time solely to drain and
clean up. Do not fake persistence through silent audio, location, or unrelated
background modes.

Centralize storage ownership. Separate immutable model blobs, manifests, staging,
task caches, activity history, logs, and resume metadata. Use atomic rename and a
journal for install/delete recovery. Protect in-use files and delete terminal task
caches promptly.

## 10. Porting to Android

Keep the JSON protocol and task semantics stable. Replace only platform layers:

- Network.framework/Bonjour with Android-compatible discovery and TLS transport;
- Keychain/Secure Enclave with Android Keystore hardware-backed keys;
- iOS lifecycle state with a transparent Android foreground-service policy;
- Metal runtime with an Android-capable local runtime and GPU/NPU delegates;
- Vision/AVFoundation/Accelerate with platform equivalents.

Advertise Android-specific capabilities instead of branching core task schemas by
platform. Mixed iPhone/Android networks should be ordinary heterogeneous worker
pools from the scheduler's perspective.

## 11. Adding image generation

Image generation should be a new capability and task kind, not an assumption that
all workers support it. Advertise model digest, supported dimensions, precision,
maximum steps, expected memory, and thermal/power constraints. Transfer prompts
and optional conditioning assets through the same authenticated blob path. Return
an image digest plus typed metadata, and preserve desktop fallback.

## 12. Minimum verification matrix

Before calling a port ready, verify:

- cross-language schema round trips;
- pairing, replay rejection, revocation, and certificate pinning;
- lease expiry, disconnects, explicit cancellation, and late results;
- scheduler behavior with mixed capabilities and multiple phones;
- context fitting before every ordinary and tool-loop generation;
- objective persistence and no-progress strategy changes;
- output-cap and malformed-result rejection;
- crash recovery and zero terminal task-cache residue;
- real-device lifecycle, network loss, thermal, and storage behavior;
- source and artifact scans for credentials and local machine paths.

The reference test and implementation status files deliberately distinguish
automated coverage from physical-device validation. Preserve that distinction in
derivative projects.
