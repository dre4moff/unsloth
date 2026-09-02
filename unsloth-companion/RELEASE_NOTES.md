# Unsloth Companion prototype release notes

GitHub prerelease: `v0.1.800-mlx.20-companion.4`

## The release

This public prototype combines model-aware compaction, persistent objectives, and
an expandable network of local iPhone workers. The emphasis is the Companion
architecture: one or more authenticated phones can act as offline subagents while
the Mac remains the orchestrator and fallback.

The Mac slowdown was traced from the real llama-server timings, not inferred from the
short visible reasoning. The stopped request had already compacted 86,678 prompt tokens and
dropped 50 messages, but replayed an old epoch that still contained 26,846 tokens. llama.cpp
then spent 348.9 seconds evaluating that prompt to generate only 118 tokens, and the next pass
began another full prefill. The previous deterministic-slot experiment did not help because
the prefix itself changed between passes, so that experiment has been removed.

The complete context-window and checkpoint policy now matches releases `.7` through `.11`
exactly, before and after every searchable checkpoint. There is no separate performance target,
20% limit, 10K threshold, or prefill environment override. The configured context and original
reply-token reservation alone determine when a checkpoint is replayed or a new epoch begins.
The active-turn ledger is appended to the
newest user suffix instead of rewriting the system prefix after each tool action, and the GGUF
loop keeps one stable tool catalogue throughout ordinary plan/tool/final passes. Explicit
iPhone delegation also skips the redundant `update_plan` generation and enters the named
Companion call directly. The Companion schema remains stable while a phone moves between idle
and busy; volatile worker and mailbox state stays out of the cached tool prefix.

The latest real-device report also identified why a simple fairy-tale assignment could be
returned as a JSON paraphrase instead of being performed. A text-only subagent call put
`Scrivi...` in `text`; the previous iPhone prompt always rendered `text` as context and
`instruction` as the task, so it presented an empty task to the small worker. The correction
introduced in build 7 and included in build 8 treats `text` as the objective when no separate
instruction exists, while still separating
source context from an explicit objective when both are present. The live Desktop schema
now directs the Mac model to put assignments in `instruction` and reserve `text` for source
material; instruction-only tasks no longer acquire unrelated conversation context.

Output validation is also semantic enough for the observed near-copy: it inspects string
leaves inside plain, fenced, and structured JSON, accepts minor article or imperative changes
as an echo, and compares against both context and objective. The iPhone retries once with the
complete output-format contract; repeated meta-output such as `Self-Correction`, `Final attempt`,
and promises to generate the answer is stopped during generation rather than allowed to consume
the whole budget. The Desktop independently rejects any surviving echo instead of presenting it
as a completed deliverable. This is a production-path correction, not merely an error detector:
the first prompt now asks the worker to execute a non-empty objective.

This candidate also retains the Desktop correction for checkpoint identity across tool-using turns.
Studio now sends the ordered IDs of the active message branch as private request metadata;
the backend restores the exact persisted checkpoint from those IDs and never forwards them
to the model. Reasoning and tool calls may still expand one saved assistant message into
several wire messages, but that lossy text projection can no longer make a 9K active window
look like a fresh 50K transcript and trigger a second immediate compaction. Older API clients
remain supported through the existing conservative text fallback.

The Companion schema was already present in every enabled model request, so searching the
terminal or provoking a validation error was incorrect model behaviour. The runtime prompt
now says explicitly that the complete live schema is attached and must be used directly.
Explicit dispatch also recognises the common `compaion` typo and retry/test wording, while
automatic proactive delegation remains available whenever eligible phones are connected.

This corrective candidate closes the two gaps observed in a real Studio transcript. The
Mac model generated more than 20,000 reasoning tokens while repeatedly discussing an
imaginary Companion schema, but never emitted an `iphone_companion` call. Explicit requests
now enter a named function-call path with only the live Companion schema available, a
4096-token ceiling per dispatch attempt, and a bounded retry. The stable schema explains that
`submit` remains valid while every compatible phone is busy and that separate submissions are
queued; runtime validation still requires `kind` plus `maximum_tokens` for a submission.
This hard guarantee does not restrict normal orchestration: for useful independent work,
the Mac model can still select available iPhones proactively even when the user did not
name Companion.

The second gap was presentation. A batch previously appeared as one generic tool card and
rendered `parallelResults` as undivided JSON. Background jobs now retain per-item state and
worker assignment. The Desktop renders a parent job card plus one child card per subagent,
including task ID, iPhone name, live state, duration, token count, error, and isolated output.

The real-device report exposed a terminal-order race: the iPhone sent `taskCompleted` before
clearing its active runtime, so the Desktop immediately dequeued the next item into a phone
that still reported busy. Build 4 now clears the coordinator and publishes `ready` before the
completion envelope. The Desktop also retries this specific transient response for compatibility
with earlier IPA builds. The queue now also spans separate `submit` calls: one busy iPhone no
longer makes the next job fail, but processes queued work serially as it becomes free. Multiple
eligible iPhones consume independent queued items concurrently.

Every delegated task retains the previous Mac-selected output ceiling from 8192 to 16384 tokens;
this was deliberately not reduced because the measured Mac delay was prompt prefill. The
Desktop may only lower the transmitted ceiling when the selected phone's physical context cannot
fit prompt plus output. The
subagent prompt no longer ends with a named "final deliverable" slot: a small Gemma worker had
copied that semantic slot as the literal marker `[Final deliverable output]`. Exact-literal tasks
use a minimal prompt and byte-for-byte acceptance check. General tasks reject objective echoes and
placeholder-only output, retry once with a marker-free correction, then fail instead of being
labelled Completed. Desktop applies the same validation defensively, so an older IPA cannot turn
a template marker into a successful result.
`update_plan` now emits a terminal tool event, including a reload fallback for old messages, so
its card cannot spin after completion. Context compaction uses the exact `.7`-`.11` policy for
fresh chats and existing checkpoint epochs; reply and iPhone task budgets are unchanged.

Planning is now an independent `Plan` tool in the Studio `+` menu. Its persisted switch controls
the existing internal planner without changing compaction or other tools, and queued turns retain
the setting they had when submitted. The objective/checklist remains visible while a turn runs and
disappears as soon as generation finishes, including completed plans.

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

- `Unsloth_0.1.800-mlx.20_aarch64.dmg`: Apple Silicon/macOS 12+ build,
  ad-hoc signed and not Apple-notarized.
- `Unsloth-Companion_0.0.1_unsigned.ipa`: arm64/iOS 18.6 prototype, unsigned and
  requiring a user-supplied signing identity before installation.
- `SHA256SUMS.txt`: release artifact digests.
- The repository source containing the implementation and docs, published under tag
  `v0.1.800-mlx.20-companion.4`.

## Validation boundary

No fresh automated, simulator, or physical-device behavioral tests were run for `companion20`.
Validation includes the production frontend/DMG build, unsigned iPhoneOS build 8, and final
artifact-integrity checks. The prior
`companion18` candidate had passed 18 Desktop Companion tests, 388 safetensors/request-tool tests,
and 297 GGUF/context tests; those results are historical and are not claimed for this build.
The protocol remains v1 and the unsigned iPhone IPA was rebuilt from current source as build 8. See
`IMPLEMENTATION_STATUS.md` for the exact boundary and final artifact checks.

This is a community prototype and not an official Unsloth mobile release.
