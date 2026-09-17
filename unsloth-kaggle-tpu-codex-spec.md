# Unsloth — Remote Kaggle TPU / OpenAI-Compatible Backend
## Implementation specification for Codex

### Objective

Implement a first-class **Remote OpenAI-Compatible inference backend** in the `dre4moff/unsloth` fork.

Primary target:

```text
Unsloth Studio
        ↓
Remote OpenAI-compatible provider
        ↓
Cloudflare Tunnel
        ↓
Kaggle TPU v5e-8
        ↓
vLLM TPU
        ↓
Qwen3.8-27B
```

The implementation MUST preserve all existing Unsloth agent-side functionality.

The remote model is ONLY responsible for model inference. Unsloth remains responsible for:

- conversation state
- system prompts
- tool definitions
- tool execution
- MCP
- web search
- code execution
- agent loops
- context management
- context compaction
- reasoning display
- attachments where supported
- UI
- authentication
- provider selection
- error handling
- streaming

Do NOT fork or replace the existing agent/tool architecture.

---

# 1. Design principle

Do NOT create a provider called only `Kaggle TPU`.

Instead create a generic provider:

```text
OpenAI Compatible
```

with an optional convenience integration:

```text
Kaggle TPU
```

The generic provider must work with:

- kaggle-tpu-lab
- vLLM
- llama.cpp
- Ollama OpenAI-compatible endpoints
- LM Studio
- OpenRouter-compatible endpoints
- arbitrary private OpenAI-compatible servers

Example:

```text
Provider:
    OpenAI Compatible

Name:
    Kaggle Qwen3.8-27B

Base URL:
    https://xxxx-yyyy.trycloudflare.com/v1

API Key:
    sk-xxxxxxxx

Model:
    qwen3.8-27b
```

---

# 2. Do not treat the remote backend as a local model

Do not attempt to:

- download Qwen3.8-27B locally
- load it through llama.cpp
- load it through Transformers
- quantize it
- initialize a local tokenizer for inference
- allocate local GPU/CPU memory for the remote model

The model is already running remotely.

The local Unsloth application should only instantiate an OpenAI-compatible client.

Conceptually:

```python
client = OpenAI(
    base_url=provider.base_url,
    api_key=provider.api_key,
)
```

Reuse the existing Unsloth provider infrastructure rather than introducing a second HTTP client if an appropriate abstraction already exists.

---

# 3. Provider configuration

Add a provider configuration equivalent to:

```yaml
providers:
  kaggle-qwen38:
    type: openai_compatible
    name: Kaggle TPU — Qwen3.8-27B
    base_url: https://xxxx-yyyy.trycloudflare.com/v1
    api_key: sk-xxxxxxxx
    default_model: qwen3.8-27b

    capabilities:
      chat_completions: true
      streaming: true
      tool_calling: true
      reasoning: true
      vision: true
      context_length: 262144
```

Do not hard-code the tunnel URL or API key.

Do not commit credentials.

---

# 4. Provider capabilities

Implement explicit capability metadata.

Minimum fields:

```text
supports_streaming
supports_tool_calling
supports_reasoning
supports_vision
supports_images
context_length
api_mode
```

For the Kaggle TPU backend:

```text
streaming = true
tool_calling = true
reasoning = true
vision = true
context_length = 262144
api_mode = chat_completions
```

The model is Qwen3.8-27B served by vLLM TPU.

Do not globally assume every OpenAI-compatible endpoint has 262144 context.

---

# 5. Tool calling — CRITICAL

Do NOT disable tools for remote providers.

Forward the following through the OpenAI-compatible request whenever supported:

```json
{
  "tools": [...],
  "tool_choice": "auto"
}
```

Also support:

```text
tool_choice = none
tool_choice = required
specific function selection
```

Conceptually:

```python
client.chat.completions.create(
    model="qwen3.8-27b",
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    stream=True,
)
```

Do NOT execute tools remotely.

The model only requests the tool.

Unsloth receives the tool call and executes it locally.

Expected flow:

```text
User
 ↓
Unsloth
 ↓
Qwen TPU
 ↓
tool_call: execute_python(...)
 ↓
Unsloth tool executor
 ↓
Python sandbox
 ↓
tool result
 ↓
Qwen TPU
 ↓
final response
```

This separation is mandatory.

---

# 6. Preserve MCP

MCP must remain entirely local to Unsloth.

Do not send MCP implementation details to the TPU backend.

Convert available MCP tools into the same OpenAI function-tool representation already used by the existing Unsloth agent loop.

The remote model sees:

```json
{
  "type": "function",
  "function": {
    "name": "...",
    "description": "...",
    "parameters": {...}
  }
}
```

Unsloth remains responsible for actually executing the MCP tool.

---

# 7. Preserve web search

Web search must remain an Unsloth-side tool.

Do not assume that Qwen has direct Internet access.

The model should request:

```text
web_search(...)
```

and Unsloth executes it.

The result is then appended to the conversation and sent back to the remote model.

---

# 8. Preserve code execution

Same architecture:

```text
Qwen
 ↓
tool call
 ↓
Unsloth
 ↓
sandbox
 ↓
stdout/stderr/files
 ↓
tool result
 ↓
Qwen
```

Never execute code on the Kaggle TPU server merely because the model is remote.

The remote endpoint must be treated as an untrusted inference backend.

---

# 9. Reasoning support

Qwen3.8 supports reasoning levels.

The Kaggle server supports:

```text
xhigh
medium
low
off
```

and accepts:

```json
{
  "chat_template_kwargs": {
    "reasoning_effort": "medium"
  }
}
```

The provider should expose:

```text
Reasoning effort:
    xhigh
    medium
    low
    off
```

Map the UI setting to the existing request metadata mechanism.

Do not implement a second reasoning system.

If the existing Unsloth code already has a normalized reasoning setting, convert it to:

```python
extra_body = {
    "chat_template_kwargs": {
        "reasoning_effort": reasoning_effort
    }
}
```

When thinking should be disabled, support the backend's corresponding option:

```json
{
  "enable_thinking": false
}
```

Use the existing Unsloth reasoning abstraction rather than exposing raw Qwen-specific fields throughout the application.

---

# 10. Streaming

Streaming MUST remain enabled.

Use the existing Unsloth streaming abstraction.

Do not implement a separate UI streaming renderer for Kaggle.

The provider should transform OpenAI streaming chunks into the internal Unsloth stream/event representation.

The UI must continue receiving:

```text
text deltas
reasoning deltas
tool calls
tool arguments
finish events
usage
errors
```

exactly like local providers.

---

# 11. Tool-call streaming

Pay special attention to fragmented tool arguments.

For example:

```text
chunk 1:
{"name":"python","arguments":"{\"x"

chunk 2:
"\": 10"

chunk 3:
"}"}
```

The provider must accumulate fragments until the tool call is complete.

Do not attempt to execute partial JSON.

Use the existing Unsloth tool-call accumulator if one exists.

If one does not exist, introduce a provider-independent accumulator.

It must support multiple simultaneous tool calls and stable tool-call IDs.

---

# 12. Model discovery

When the user enters:

```text
https://xxxx-yyyy.trycloudflare.com/v1
```

provide a:

```text
Test Connection
```

button.

First try:

```http
GET /v1/models
```

using the supplied Bearer token.

If successful, populate the model selector from the returned model IDs.

Expected model:

```text
qwen3.8-27b
```

If `/v1/models` is unavailable, allow manual model entry.

Never require model discovery.

---

# 13. Model metadata

If `/v1/models` returns context metadata, use it.

Otherwise allow:

```text
Context length:
262144
```

to be configured manually.

For the built-in Kaggle TPU profile use 262144 as the default.

---

# 14. Vision

The generic OpenAI-compatible provider must support multimodal OpenAI messages where the backend advertises vision.

Example:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "Describe this image"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,..."
      }
    }
  ]
}
```

Do not send images when:

```text
supports_vision = false
```

For text-only Kaggle launches, correctly report vision as unavailable.

---

# 15. Context compaction

Remote inference must NOT bypass Unsloth's existing context-compaction mechanism.

Architecture:

```text
Conversation
      ↓
Unsloth context manager
      ↓
context budget
      ↓
compaction if necessary
      ↓
OpenAI-compatible request
      ↓
Kaggle TPU
```

Do not rely exclusively on Qwen's 262k context.

The configured remote context length should feed into the existing context manager.

Example:

```text
provider.context_length = 262144
```

---

# 16. Prompt caching

Do not implement provider-specific prompt caching unless the existing architecture already supports it.

Do not accidentally disable existing caching logic.

If OpenAI-compatible caching metadata is already supported, pass it through.

Otherwise keep current Unsloth behavior.

---

# 17. Error handling

Normalize errors from the remote server.

Handle:

```text
401 Unauthorized
403 Forbidden
404 Not Found
408 Timeout
429 Too Many Requests
500 Server Error
502 Bad Gateway
503 Service Unavailable
504 Gateway Timeout
connection reset
DNS failure
Cloudflare tunnel unavailable
```

Show user-friendly errors.

Example:

```text
Kaggle TPU backend unavailable.

The Cloudflare tunnel could not be reached.

Check:
• Kaggle session is running
• Cloudflare tunnel is still active
• API key is correct
• Base URL includes /v1
```

Do not expose raw stack traces in the UI unless developer mode is enabled.

---

# 18. Cloudflare tunnel integration

Implement TWO modes.

## Mode A — manual remote endpoint

The user enters:

```text
Base URL:
https://xxxx-yyyy.trycloudflare.com/v1

API key:
sk-xxxx

Model:
qwen3.8-27b
```

This is the primary generic implementation.

## Mode B — Kaggle TPU managed connection

Add an optional provider type:

```text
kaggle_tpu
```

This provider should be a thin orchestration layer around the existing `kaggle-tpu-lab` project.

Do NOT copy the Kaggle serving implementation into Unsloth.

Do NOT duplicate:

```text
vLLM
Pallas
XLA
MTP
TPU setup
```

Those remain in `kaggle-tpu-lab`.

Unsloth only orchestrates the launcher.

---

# 19. Kaggle TPU managed mode

Configuration:

```yaml
providers:
  kaggle-tpu:
    type: kaggle_tpu
    project: qwen38
    model: qwen3.8-27b
    auto_start: false
```

UI:

```text
Kaggle TPU

Model:
Qwen3.8-27B

Status:
● Disconnected

[Start TPU]

[Connect]

[Stop TPU]
```

When the user presses Start TPU, invoke the existing launcher:

```bash
python launch.py serve
```

or the equivalent configured command.

Do not assume a fixed filesystem path.

Allow:

```text
Kaggle TPU Lab path:
[ /path/to/kaggle-tpu-lab ]
```

or:

```text
UNSLOTH_KAGGLE_TPU_LAB=/path/to/kaggle-tpu-lab
```

---

# 20. Never reimplement launch.py

The existing Kaggle project already handles:

- Kaggle authentication
- TPU provisioning
- dataset attachment
- environment setup
- XLA cache
- model weights
- vLLM
- Cloudflare tunnel
- health checks
- API key
- endpoint discovery

Reuse that.

Unsloth should consume the launcher output rather than duplicate its logic.

---

# 21. Machine-readable launcher output

If possible, extend `kaggle-tpu-lab` with:

```bash
python launch.py serve --json
```

returning:

```json
{
  "status": "ready",
  "base_url": "https://xxxx-yyyy.trycloudflare.com/v1",
  "api_key": "sk-xxxx",
  "model": "qwen3.8-27b",
  "context_length": 262144
}
```

Preferred architecture:

```text
Unsloth
   ↓
subprocess
   ↓
launch.py --json
   ↓
JSON
   ↓
provider configuration
```

If modifying `kaggle-tpu-lab` is undesirable, implement a robust fallback parser for the existing READY output.

---

# 22. Lifecycle

Managed Kaggle provider states:

```text
STOPPED
STARTING
PROVISIONING
LOADING
READY
ERROR
STOPPING
```

UI should display the current state.

When READY:

```text
● Connected

Qwen3.8-27B
Kaggle TPU v5e-8
262k context
Cloudflare HTTPS
```

When the tunnel dies:

```text
○ Disconnected

Kaggle TPU session may still be running.
[Reconnect]
```

Do not automatically start a new TPU unless the user explicitly enabled Auto-start.

---

# 23. Keepalive

Expose:

```text
Keep TPU alive:
[ On / Off ]
```

Do not implement a custom heartbeat if the Kaggle launcher already handles this.

If the server disappears, mark the provider unavailable.

---

# 24. Security

Treat the Cloudflare URL and API key as credentials.

Never:

- log the API key
- store it in plaintext logs
- send it to analytics
- include it in crash reports
- commit it to config files
- expose it to browser JavaScript unnecessarily

Store credentials using the same secure credential storage already used by Unsloth.

---

# 25. Do not proxy tools through Cloudflare

The tunnel carries:

```text
LLM inference requests
```

It must NOT carry:

```text
MCP calls
Python execution
Bash execution
filesystem operations
browser control
```

Those remain local.

Correct architecture:

```text
                 INTERNET
                    │
                    ▼
             Cloudflare Tunnel
                    │
                    ▼
              Kaggle vLLM
                    │
                    ▼
                 Qwen

Unsloth
 ├── MCP
 ├── Python
 ├── Bash
 ├── web search
 ├── filesystem
 └── agent loop
```

---

# 26. Provider abstraction

Extend the existing provider interface rather than creating a parallel system.

Conceptually:

```python
class InferenceProvider:
    async def stream_chat(...):
        ...

    async def list_models(...):
        ...

    async def test_connection(...):
        ...

    def capabilities(...):
        ...
```

Then:

```python
class OpenAICompatibleProvider(InferenceProvider):
    ...
```

Kaggle becomes:

```python
class KaggleTPUProvider(OpenAICompatibleProvider):
    ...
```

The Kaggle provider adds lifecycle management only.

It must NOT duplicate inference logic.

---

# 27. Avoid Qwen-specific code in the generic provider

Do not write:

```python
if model == "qwen3.8-27b":
    ...
```

inside the generic OpenAI client.

Qwen-specific reasoning fields should live inside the capability/profile configuration.

For example:

```python
provider.extra_request_fields = {
    "chat_template_kwargs": {
        "reasoning_effort": ...
    }
}
```

Keep the backend reusable.

---

# 28. Qwen3.8 MTP

Do not implement MTP inside Unsloth.

MTP is handled server-side by the Kaggle TPU/vLLM stack.

Unsloth should see normal streamed model responses.

Do not duplicate the TPU-side MTP/DeltaNet implementation.

---

# 29. API mode

Default:

```text
chat_completions
```

because the Kaggle endpoint is OpenAI-compatible.

Keep the abstraction extensible so future providers can use:

```text
responses
chat_completions
anthropic_messages
```

Do not force every provider through `/v1/responses`.

---

# 30. UI

Add:

```text
Settings
 → Models
   → Add Provider
      → OpenAI Compatible
```

Fields:

```text
Provider name
Base URL
API key
Model
Context length
Reasoning
Vision
Tool calling
```

Add:

```text
[Test Connection]
```

and:

```text
[Fetch Models]
```

Optional:

```text
☑ Kaggle TPU managed backend
```

When enabled:

```text
Kaggle TPU Lab path
Auto-start
Auto-stop
Keepalive
```

---

# 31. Model selector

Display remote models distinctly:

```text
Models

LOCAL
  Qwen3.6-27B
  Gemma ...

REMOTE
  Qwen3.8-27B
  └── Kaggle TPU
      262k context
      Cloudflare
```

Example label:

```text
Qwen3.8-27B
Remote · Kaggle TPU
262k context
```

---

# 32. Agent compatibility

Existing agent commands/functionality must continue working.

The selected remote provider should become the inference backend without changing agent/tool behavior.

Example:

```text
unsloth start codex
        ↓
Unsloth provider = Kaggle TPU
        ↓
Qwen3.8-27B
```

Do not require manual environment variables when the provider is configured inside Unsloth.

---

# 33. Fallback

Allow:

```text
Primary:
Kaggle Qwen3.8-27B

Fallback:
Local Qwen3.6
```

Only fallback when explicitly enabled.

Do not silently switch providers.

---

# 34. Tests

Add unit tests for:

### Connection

```text
GET /v1/models
```

### Authentication

```text
Authorization: Bearer ...
```

### Chat

```text
messages
```

### Streaming

```text
delta accumulation
```

### Tool calling

```text
tools
tool_choice
tool_call fragments
```

### Reasoning

```text
reasoning content
```

### Vision

```text
image_url
```

### Errors

```text
401
404
429
500
503
timeout
Cloudflare failure
```

### Context

```text
262144
```

### Managed provider

Mock:

```bash
python launch.py --json
```

and verify correct parsing of:

```json
{
  "status": "ready",
  "base_url": "...",
  "api_key": "...",
  "model": "qwen3.8-27b",
  "context_length": 262144
}
```

---

# 35. Integration test

Create an end-to-end test using a mock OpenAI-compatible server.

Test:

```text
User:
"Create a Python file containing hello world."

Expected:

Qwen requests:
tool_call = write_file(...)

Unsloth:
executes tool

Qwen receives:
tool result

Qwen:
final response
```

The test must prove that tools execute on the Unsloth side, not the remote server.

---

# 36. Acceptance test against real Kaggle TPU

Once implementation is complete:

1. Start `kaggle-tpu-lab`.
2. Wait for READY.
3. Connect Unsloth using the generated base URL and API key.
4. Test connection.
5. Fetch models.
6. Select `qwen3.8-27b`.
7. Send a normal prompt.
8. Verify streaming.
9. Enable reasoning.
10. Verify reasoning display.
11. Give the model a tool.
12. Verify tool call.
13. Verify Unsloth executes the tool locally.
14. Send tool result back to Qwen.
15. Verify final response.
16. Test a long conversation.
17. Verify context manager.
18. Verify context compaction still works.
19. Send an image if vision is enabled.
20. Kill the Cloudflare tunnel.
21. Verify graceful connection error.
22. Reconnect.
23. Verify conversation can continue.

---

# 37. Files to inspect before modifying anything

First inspect the repository and identify the actual current architecture.

Do NOT blindly create files.

Find:

```text
provider definitions
provider registry
OpenAI client
Anthropic client
model configuration
model selector
agent loop
tool execution
MCP
context manager
context compaction
stream event types
reasoning parser
settings persistence
credential storage
Studio API
CLI provider selection
```

Use the actual repository structure rather than assuming paths.

---

# 38. Critical implementation rule

Before editing, run:

```bash
git status
git branch --show-current
git log -5 --oneline
```

Then inspect the current implementation of:

```text
OpenAI provider
external provider support
tool calling
context compaction
remote access
Cloudflare
```

Do not overwrite existing work from the `dre4moff` fork.

Integrate into the existing architecture.

---

# 39. Git strategy

Create:

```text
feature/kaggle-tpu-provider
```

Do not modify `main` directly.

Make logically separated commits:

```text
1. Add generic OpenAI-compatible remote provider
2. Preserve tool/reasoning/vision streaming
3. Add provider UI/configuration
4. Add Kaggle TPU lifecycle integration
5. Add tests
6. Add documentation
```

---

# 40. Documentation

Add a section:

```markdown
## Kaggle TPU — Qwen3.8-27B

Unsloth can connect to Qwen3.8-27B running remotely on a Kaggle TPU through the OpenAI-compatible API exposed by kaggle-tpu-lab.

Architecture:

Unsloth
→ Cloudflare
→ Kaggle TPU v5e-8
→ vLLM TPU
→ Qwen3.8-27B

Tools, MCP, web search, code execution and agent orchestration remain local to Unsloth.

The TPU is used only for LLM inference.
```

Include manual configuration and managed-start configuration.

---

# 41. Final acceptance criteria

The feature is complete ONLY when all of these are true:

- [ ] Existing local models still work.
- [ ] Existing OpenAI providers still work.
- [ ] Existing Anthropic providers still work.
- [ ] Existing Cloudflare remote functionality still works.
- [ ] OpenAI-compatible arbitrary endpoints work.
- [ ] Kaggle TPU endpoint works.
- [ ] Qwen3.8-27B is selectable.
- [ ] Streaming works.
- [ ] Reasoning works.
- [ ] Tool calling works.
- [ ] Tool execution remains local.
- [ ] MCP works.
- [ ] Web search works.
- [ ] Code execution works.
- [ ] Context management works.
- [ ] Context compaction works.
- [ ] Vision works when enabled.
- [ ] API keys are not logged.
- [ ] Cloudflare failures are handled gracefully.
- [ ] Kaggle lifecycle can be started/stopped when managed mode is enabled.
- [ ] No Qwen-specific logic leaks into the generic OpenAI-compatible provider.
- [ ] No TPU/vLLM implementation is duplicated inside Unsloth.
- [ ] Tests pass.
- [ ] Existing tests pass.

---

# 42. Desired final UX

```text
┌───────────────────────────────────────────────┐
│ Models                                        │
│                                               │
│ REMOTE                                        │
│                                               │
│  🧠 Qwen3.8-27B                               │
│     Kaggle TPU v5e-8                          │
│     262k context · Cloudflare                 │
│                                               │
│     ● Connected                               │
│                                               │
│     [Stop TPU]  [Reconnect]                   │
│                                               │
└───────────────────────────────────────────────┘
```

The chat must behave exactly like a normal Unsloth agent:

```text
User
 ↓
Unsloth
 ↓
Qwen3.8-27B on Kaggle TPU
 ↓
reasoning
 ↓
tool call
 ↓
Unsloth executes tool locally
 ↓
tool result
 ↓
Qwen3.8-27B
 ↓
final answer
```

The user should NOT have to know that the model is remote while using tools.

The only visible difference should be:

```text
Model: Qwen3.8-27B
Backend: Kaggle TPU
```

---

# Recommended implementation order

Do NOT start with the Kaggle lifecycle integration.

First implement and validate:

```text
Unsloth
    ↓
OpenAI Compatible Provider
    ↓
https://...trycloudflare.com/v1
    ↓
Qwen3.8-27B
```

Verify:

- streaming
- reasoning
- tool calling
- local tool execution
- MCP
- web search
- code execution
- context management
- context compaction
- vision

Only after this works, implement:

```text
Unsloth
    ↓
Kaggle TPU launcher
    ↓
Kaggle
    ↓
Cloudflare
    ↓
OpenAI-compatible endpoint
```

This separates the inference integration from the TPU provisioning/lifecycle problem.

---

# Architectural goal

The final architecture should be:

```text
                    ┌──────────────────────────┐
                    │        UNSLOTH           │
                    │                          │
                    │ Agent loop               │
                    │ Context                  │
                    │ Compaction               │
                    │ MCP                      │
                    │ Web search               │
                    │ Python                   │
                    │ Bash                     │
                    │ Files                    │
                    │ Tool execution            │
                    │ UI                       │
                    └────────────┬─────────────┘
                                 │
                          OpenAI API
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    CLOUDFLARE TUNNEL     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      KAGGLE TPU v5e-8    │
                    │                          │
                    │          vLLM             │
                    │            ↓             │
                    │      Qwen3.8-27B          │
                    │       BF16 / MTP          │
                    └──────────────────────────┘
```

Unsloth is the agent runtime; Kaggle TPU is the remote inference accelerator.

The implementation should remain generic enough that the same provider can later connect to vLLM, llama.cpp, LM Studio, Ollama or any other OpenAI-compatible backend without modifying the agent/tool architecture.
