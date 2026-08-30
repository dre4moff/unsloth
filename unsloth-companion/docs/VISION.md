# A household edge-agent network

This document records the idea behind Unsloth Companion, proposed by GitHub user
[@dre4moff](https://github.com/dre4moff): the devices people already own should be
reusable parts of a private offline AI system, not isolated screens that remain
idle whenever they are not in somebody's hand.

## The thesis

Buying another workstation, GPU, or home server is expensive. Many homes already
contain several generations of phones with capable CPUs, GPUs, neural engines,
storage, cameras, microphones, and mature power management. Those devices spend a
large part of each day unused, often connected to power overnight.

An offline LLM application can treat those phones as a heterogeneous network of
subagents:

- the main computer owns the user request, private conversation, objective, and
  final answer;
- each phone advertises what it can safely do now;
- the orchestrator assigns bounded, independent work to suitable phones;
- results return over the local network and are validated before use;
- work falls back to the main computer when a phone disconnects, overheats, locks,
  runs out of context, or returns an incomplete result.

This does not turn several phones into one giant GPU. It creates something more
modular: a network of small local workers whose capacities can be combined at the
task level.

## Why compaction and objectives belong in the same system

Long autonomous work fails for reasons beyond raw inference speed. The main model
can lose the original goal when context fills, repeat actions without advancing,
or accept a truncated subagent response as if it were complete.

The prototype therefore joins three mechanisms:

```text
model-aware compaction
        keeps the active request within the real context budget
                         |
persistent objective/checklist
        survives compaction and makes progress observable
                         |
capability-routed phone subagents
        execute independent work while the main model orchestrates
```

Compaction is recalculated for the model and request actually in use, including
tool definitions and reply reserve. The visible objective/checklist is carried
across compaction without storing private chain-of-thought. Tool activity alone
does not count as progress: after a bounded number of actions, the orchestrator
must advance the checklist or adopt a concretely different strategy.

The same discipline applies at the phone boundary. A delegated result must fit a
declared schema or finish cleanly; a token-cap termination is not silently treated
as a usable answer.

## Parallelism people can afford

The useful unit of parallel work is an independent task or shard, for example:

- summarize separate document sections;
- classify or verify batches of records;
- inspect different images or sampled video ranges;
- transcribe and analyze separate audio segments;
- generate candidate plans and critiques;
- rerank independent result groups;
- run OCR or deterministic DSP beside text inference.

The Mac can keep reasoning while phones work, and several phones can process
different items simultaneously. Newer devices can receive deeper or multimodal
jobs; less powerful compatible devices can still perform short summaries,
classification, OCR, verification, or deterministic signal analysis.

## The overnight scenario

A future queue can turn charging hours into useful local compute:

1. The user chooses an explicit overnight task and allowed data scope.
2. The orchestrator breaks it into resumable, idempotent units.
3. Only opted-in phones that are charging, cool enough, and on an approved local
   network accept work.
4. Each unit is journaled, checksummed, and safe to retry after interruption.
5. The main computer compacts its working context and preserves the objective as
   the queue progresses.
6. In the morning, the user receives the consolidated result and a record of what
   ran where.

The current iOS prototype intentionally does not claim unattended background
execution: iOS lifecycle rules require the app to remain foregrounded and
unlocked. The overnight design is a direction for user-controlled deployments,
dedicated-device configurations, future platform capabilities, and Android—not a
hidden-background promise in this release.

## Modularity is the point

Every layer is replaceable:

- the main model can be GGUF, MLX, or another local runtime;
- the phone runtime can evolve independently of the desktop runtime;
- discovery and transport are separate from task semantics;
- the protocol advertises capabilities instead of assuming one device class;
- model registries pin immutable artifacts but are not hardwired to one model;
- schedulers consume capability, battery, thermal, latency, queue, and storage
  signals rather than a specific iPhone name;
- deterministic pipelines can coexist with LLM and multimodal inference.

This lets developers reuse what already works. An app can adopt only the protocol
and scheduler, only the compaction and objective design, or the complete desktop
plus iPhone stack.

## Future network

The architecture is intended to grow in several directions:

- **Android workers:** the same task protocol and scheduling contract over a
  platform-native secure transport and lifecycle implementation.
- **Image generation:** capability-advertised diffusion or other image pipelines
  when a mobile device has the runtime, memory, thermals, and model assets to run
  them safely.
- **Specialist devices:** dedicate one phone to OCR/vision, another to text, and
  another to audio or verification.
- **Energy-aware scheduling:** prefer charging devices, honor user-set battery and
  thermal ceilings, and estimate whether a job is worth assigning.
- **Resumable long tasks:** checkpoint at task boundaries so a network can shrink
  or grow without losing completed work.
- **Portable orchestration:** reuse the protocol in other offline LLM apps rather
  than binding the idea to one desktop UI.

## Non-negotiable boundaries

More devices must not mean less control. A serious implementation should preserve:

- explicit pairing and revocation;
- local-only operation by default;
- no remote activation of camera or microphone;
- no sensitive payloads in logs;
- capability and model integrity checks;
- deterministic cancellation and cleanup;
- honest lifecycle behavior;
- a main-computer fallback;
- no claim that independent devices share one KV cache;
- no claim of successful hardware validation without real evidence.

## What this release leaves behind

The goal of this open prototype is not to declare the network finished. It is to
leave a complete, inspectable starting point: code, protocol, security model,
scheduler, UI, runtime bridge, build scripts, tests, and the limitations observed
on physical hardware.

If offline AI is going to be broadly accessible, it should be able to grow from
the hardware people already have. A phone that would otherwise sit on a charger
can become a private subagent. A drawer of compatible devices can become a small
local network. The main model stays in control, and every reused module makes the
next implementation cheaper to build.
