# Kaggle TPU / OpenAI-compatible remote backend

Unsloth Studio can use a model running outside the Mac while keeping the agent runtime on the Mac. Only inference crosses the network:

```text
Unsloth Studio (chat, context, tools, MCP, web, code, files)
    -> OpenAI-compatible HTTPS request
    -> Cloudflare tunnel
    -> Kaggle TPU v5e-8 / vLLM
    -> Qwen3.8-27B
```

Tool requests are returned to Studio and executed by Studio's existing local tool loop. The TPU does not receive filesystem access, MCP credentials, browser control, shell access, or a Python execution environment.

## Prerequisites for Kaggle managed mode

Create a Kaggle API token in **Kaggle Settings -> API**. Studio bundles the
MIT-licensed `kaggle-tpu-lab` launcher and the matching Kaggle CLI dependency,
so an installed app does not require a separate clone, `pip install`, or
`~/.kaggle/kaggle.json`. The token is encrypted in Studio's installation
credential store and passed to the launcher only as `KAGGLE_API_TOKEN`.

The Kaggle account must be eligible for TPU use. Quota, session duration, model availability, startup time, and tunnel behavior are controlled by Kaggle and `kaggle-tpu-lab`, not by Studio.

## Managed Kaggle connection

1. Open **Settings -> Connections**.
2. Add a connection and select **Kaggle TPU**.
3. Paste the **Kaggle API token**. There is no launcher path to configure.
4. Keep `qwen3.8-27b` and `262144` as the defaults, or lower the context length if preferred.
5. Choose the lifecycle options:
   - **Auto-start** starts a configured connection when the Studio backend starts.
   - **Auto-stop** asks `launch.py stop` to end the Kaggle session during a clean Studio shutdown.
   - **Keepalive** is passed directly to `launch.py --keepalive-min`; Studio does not create a second heartbeat.
   - **Text only** starts faster but disables image input.
   - **Fast start** asks the launcher to skip startup precompilation.
6. Save the connection. Studio starts it immediately and keeps the progress panel visible. Later, use **Start** to start a stopped session, **Reconnect** to reattach, or **Stop** to end it. Adding the connection requires only the Kaggle token; all other values have defaults.

Studio reports `STARTING`, `PROVISIONING`, `LOADING`, `READY`, `DISCONNECTED`, `ERROR`, `STOPPING`, or `STOPPED`. After the launcher reports READY and Studio verifies the authenticated `/v1/models` endpoint, Studio stores the generated API key through the existing credential store and saves the endpoint, model ID, and context length. The key is not returned by the lifecycle status API.

The bundled launcher has a small, documented local `--json` adapter. It reads both ntfy progress and the official Kaggle logs, including a bounded snapshot of the live log stream. The legacy human-readable READY parser remains available for an external launcher override. It calls upstream `serve`, `status`, and `stop`; TPU setup, vLLM and MTP stay in the unchanged upstream serving kernel. Each connection owns a distinct private notebook. Stop follows the upstream behavior of deleting that managed notebook and terminating its session; it is idempotent and reports failures.

On a Studio restart, a saved active session is reattached without provisioning another TPU. New sessions are started from chat only when Auto-start is enabled. A failed or stopped session otherwise requires Start. State is written atomically with mode 0600; JSON-mode launcher state does not retain the plaintext inference key. Connection credentials remain encrypted in Studio.

Developers can still override the bundled launcher for compatibility testing:

```bash
export UNSLOTH_KAGGLE_TPU_LAB="/absolute/path/to/kaggle-tpu-lab"
```

## Manual Kaggle or generic OpenAI-compatible connection

Managed startup is optional. Any OpenAI-compatible endpoint can be connected without a launcher:

1. Start the remote server yourself and obtain its base URL, API key, and model ID.
2. In **Settings -> Connections**, select **OpenAI Compatible**.
3. Enter a base URL ending in `/v1`, the Bearer API key, model ID, context length, and only the capabilities that the server actually supports.
4. Use **Test Connection**. Studio first requests `/v1/models`; manual model entry remains available when discovery is unsupported.
5. Select the remote model in chat. Remote entries show their configured context window in the model selector.

This generic mode can also be used for vLLM, llama.cpp, LM Studio, Ollama's OpenAI-compatible API, and private compatible servers. Studio does not assume that an arbitrary endpoint has Kaggle's context length or capabilities.

## Runtime behavior

- Streaming uses the existing Studio event path, including text, reasoning, tool-call fragments, usage, completion, and errors.
- Qwen reasoning is exposed as `off`, `low`, `medium`, and `xhigh`. Kaggle requests map this to `chat_template_kwargs.reasoning_effort`; `off` sends `chat_template_kwargs.enable_thinking: false` (and the compatible top-level switch).
- Studio tools remain enabled only when the saved connection declares tool-calling support. Tool calls are accumulated and executed locally by the existing agent loop.
- Images are sent only when vision is enabled. A managed text-only launch updates both image and vision capabilities to false.
- The configured context length feeds Studio's normal context accounting. For remote requests, `truncate_oldest` invokes the existing instruction pins, checkpoint/rolling policy, conversation archive and recall before each inference request, including tool follow-ups. Branch boundaries and the active-turn execution ledger are preserved. Token counts are estimates because the remote tokenizer is not loaded on the Mac; an irreducible oversized prompt returns a safe context error.
- URLs and upstream exception text are redacted from routine provider logs for private compatible endpoints. Common authentication, timeout, rate-limit, tunnel, and server failures are normalized into user-facing errors.

There is no silent provider switch. If a remote endpoint becomes unavailable, Studio preserves the conversation and reports the failure so the user can reconnect or explicitly choose another model. An automatic local-model fallback should only be added with an explicit per-connection opt-in and a visible model-change event.

## Validation boundary — mlx.23

The real Kaggle launch on 5 September reproduced `Temporary failure in name resolution` inside the Kaggle VM, affecting cloudflared, ntfy and package installation. Kaggle metadata already declared Internet enabled and `TpuV5E8`; this is an external VM/network prerequisite, not a missing local API key. The test notebook was stopped and stopped status verified. Real TPU inference, image inference and throughput remain unverified until Kaggle networking works. Check notebook Internet access and account eligibility on Kaggle; retry Start when service access is available.

The local implementation is covered by provider persistence, capability, launcher-command, lifecycle-output, credential-redaction, reasoning, local-tool-loop, context-compaction, frontend capability, type-check, and production-build checks. A real Kaggle acceptance run still requires the user's Kaggle credentials, available TPU quota, and a live tunnel; it cannot be represented as completed by mocked tests.


## mlx.24: actual TPU allocation and startup diagnostics

The integration continues to use [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) at the commit recorded in `backend/vendor/kaggle_tpu_lab/UPSTREAM`. Studio explicitly requests `TpuV5E8`, checks for TPU device nodes before installing the official runtime, then verifies eight actual JAX TPU devices before upstream starts vLLM. The upstream kernel file and embedded MTP patch remain unchanged; `studio_preflight.py` is embedded by the launcher with validated source anchors.

A retryable preflight failure triggers at most one automatic retry, only after the previous session has ended. If Kaggle continues to assign a CPU, the connection reports the missing TPU promptly. A phone-verified account and valid token do not guarantee available TPU capacity or entitlement: this must be available on Kaggle. No other model or CPU fallback is selected.

Run markers exclude old persisted logs during a new submission. Server/compile failures are classified by their actual stage and error context. Historical successful installation messages and the upstream-documented GCE metadata warnings are not treated as installation or Internet failures. Newly announced authenticated tunnels receive bounded readiness retries.


## mlx.25: queue monitoring

Kaggle `QUEUED` means the model has not started installing. Studio now displays
elapsed queue time on each poll. Read-only status/log calls have 15-second
deadlines; a timeout no longer terminates the launcher with a generic error.
Transient status failures retry automatically, preserving the last model state;
five consecutive unreadable statuses detach as DISCONNECTED. Reconnect checks
the existing notebook rather than submitting another TPU request. A terminal
Kaggle error before any boot logs is reported as a pre-start allocation error.

On 5 September at 18:13 UTC the user's session was confirmed QUEUED with no
execution logs. The user subsequently stopped it at 18:14 UTC, so the exact
preceding generic error cannot be recovered. The user now confirms identity
verification completed and TPU v5e-8 selectable; the earlier unavailable-account
condition must not be assumed to persist. Availability in the editor does not
guarantee an immediate batch allocation.

## mlx.29: automatic execution through the upstream batch flow

Managed Start again submits the complete configured upstream serving script with
`kernels push --accelerator TpuV5E8`. Kaggle runs it when capacity becomes
available, independently of whether Studio remains open. The private kernel,
attached datasets, hardware/network preflight, structured events, authenticated
readiness check, local tools, and duplicate-session protection are retained.
Reconnect follows the same saved version; it does not submit another run.

The mlx.26-28 interactive bootstrap was a regression for unattended startup:
closing the local launcher during allocation lost the subsequent Jupyter execute
request. Its `Tpu1VmV38` identifier also differed from the requested v5e-8 shape.
The mlx.28 worker status interpretation was incorrect: `kernels/status` refers
to the committed version, not the pending interactive operation. Old interactive
sessions remain recoverable/stoppable; a closed operation is now terminal rather
than an endless PROVISIONING status. New Starts use batch execution.

Live diagnosis on 6 September 2026 confirmed available account quota and the
correct TPU v5e-8/Internet settings. A minimal batch submitted with upstream's
metadata returned QUEUED; the Kaggle web editor independently displayed position
20 in the TPU queue. A selectable accelerator and free quota do not guarantee
immediate capacity. Studio cannot bypass that queue or infer a queue position
from the public batch status API. Real model readiness is a separate acceptance
step from successful submission, tests, and packaging.
