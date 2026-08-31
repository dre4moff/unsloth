# Unsloth Companion Protocol v1

The canonical machine-readable contract is [`schema-v1.json`](schema-v1.json). Swift and Python use the same field names and reject any envelope that does not have `protocolVersion: 1`.

## Transport and envelope

Bonjour advertises `_unsloth-cp._tcp.`. All control messages use JSON text frames over the pinned WSS connection; media chunks use binary frames on that same authenticated session.

```json
{
  "protocolVersion": 1,
  "messageID": "6d1748ee-0831-4dd4-92af-349ef5c46118",
  "sentAt": "2026-08-29T16:00:00Z",
  "type": "heartbeat",
  "payload": { "echoMonotonic": 42.5 }
}
```

`messageID` must be unique. Receivers reject a duplicate ID, a timestamp more than 60 seconds from local time, binary control envelopes, and unknown or malformed payload fields.

## Pairing and authentication

1. iPhone sends `hello` with its UUID, display name, and uncompressed P-256 signing public key.
2. Desktop responds with `pairing_offer`: desktop identity, P-256 public key, random nonce, and WSS certificate SHA-256.
3. iPhone verifies the certificate pin, signs the nonce, and sends a preview `pairing_confirm`.
4. Both screens derive the same six-digit SAS from both public keys, the nonce, and certificate hash.
5. Pairing completes only after local confirmation on both devices and a second signed `pairing_confirm` bound to both identities.
6. Later sessions use `challenge` and `challenge_response`. Revocation removes the trusted key and closes the session.

Pairing offers expire after 60 seconds and pending SAS confirmation after five minutes.

## Capability and health messages

- `capabilities` reports task kinds, selected model, probed vision/audio support, context size, and raw-media limit.
- `heartbeat` runs every five seconds and carries echo time, battery, Low Power Mode, thermal state, service state, and free storage.
- Three missed heartbeats close the session and trigger shard reassignment or Mac fallback.
- `client_draining` stops admission before backgrounding, low battery, serious thermal pressure, or memory pressure.

## Tasks and leases

`task_submit` carries a UUID, idempotency key, task kind, priority, timeout, 30-second lease by default, media policy, typed input, and a JSON result schema. `parentTaskID`, `shardIndex`, and `shardCount` identify divisible work.

The iPhone may send `task_accepted`, `task_progress`, and provisional `task_token` messages. The Desktop renews active leases with `lease_renew` every heartbeat. A task ends with exactly one of:

- `task_completed`, containing a structured result and SHA-256 of canonical JSON;
- `task_failed`, containing a stable error code and retryability;
- `task_cancelled`, containing a stable cancellation reason.

`task_cancel` distinguishes explicit user cancellation from service disablement. Explicit user cancellation does not cause automatic fallback; disabling Companion transfers eligible work to the Mac.

The wire protocol remains completion-based, but the Desktop caller is asynchronous:
after `task_submit` is admitted, the Mac-side tool returns a job handle instead of
waiting for `task_completed`. Terminal envelopes are retained in a per-chat Desktop
mailbox and collected later. This mailbox is an orchestrator concern and does not change
protocol version 1 or require a new iPhone message type. Worker-count advertisement,
per-kind scheduling capacity, and the Desktop `wait` join barrier are likewise entirely
Desktop-side concerns, so this release does not require an IPA rebuild.

## Binary media

`blob_begin` declares blob UUID, task UUID, collision-safe remote name, MIME type, byte count, and SHA-256. Binary frames then carry chunks of at most 1 MiB under the transport send lock, providing backpressure. `blob_end` closes the blob. The iPhone verifies declared size and digest before exposing a completed file to a task.

The default media policy is `semantic_only`; `derived_media` is optional and `raw_media` requires explicit authorization. Camera and microphone capture always require confirmation on the iPhone.
