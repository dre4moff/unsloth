# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract tests for the generic remote backend and Kaggle launcher bridge."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

import httpx

from core.inference import external_provider as external_provider_module
from core.inference.external_provider import (
    ExternalProviderClient,
    _friendly_provider_error_text,
    _provider_url_for_log,
)
from core.inference.kaggle_tpu import (
    KaggleTPULifecycleManager,
    _Runtime,
    _bundled_launcher_root,
    _launcher_command,
    _launcher_environment,
    _parse_launcher_line,
)
from core.inference.providers import (
    effective_provider_capabilities,
    list_available_providers,
    provider_runs_local_tools,
)
from routes.providers import _test_openai_compatible_connection


def test_remote_profiles_declare_the_connection_contract():
    registry = {
        entry["provider_type"]: entry for entry in list_available_providers(include_hidden=True)
    }
    generic = registry["openai_compatible"]
    kaggle = registry["kaggle_tpu"]

    assert generic["hidden"] is True
    assert generic["supports_studio_tools"] is True
    assert generic["context_length"] is None
    assert kaggle["default_models"] == ["qwen3.8-27b"]
    assert kaggle["supports_reasoning"] is True
    assert kaggle["supports_vision"] is True
    assert kaggle["context_length"] == 262144


def test_saved_capability_override_closes_the_local_tool_loop():
    capabilities = effective_provider_capabilities(
        "openai_compatible",
        {
            "supports_tool_calling": False,
            "supports_vision": True,
            "context_length": 65536,
        },
    )
    assert capabilities["supports_vision"] is True
    assert capabilities["context_length"] == 65536
    assert provider_runs_local_tools("openai_compatible", capabilities) is False


def test_launcher_command_delegates_all_tpu_work_to_launch_py(tmp_path):
    launcher = tmp_path / "launch.py"
    launcher.write_text("# launcher stub\n", encoding="utf-8")
    command = _launcher_command(
        {
            "lab_path": str(tmp_path),
            "context_length": 131072,
            "max_num_seqs": 2,
            "mtp_tokens": 1,
            "reasoning_effort_default": "medium",
            "keepalive_minutes": 120,
            "text_only": True,
            "fast_start": True,
        },
        "serve",
    )
    assert command[2:4] == [str(launcher), "serve"]
    assert command[command.index("--max-model-len") + 1] == "131072"
    assert command[command.index("--keepalive-min") + 1] == "120"
    assert "--text-only" in command
    assert "--fast-start" in command


def test_managed_launcher_defaults_to_the_bundled_mit_copy():
    root = _bundled_launcher_root()
    command = _launcher_command({}, "status")
    assert command[2] == str(root / "launch.py")
    assert (root / "kernel" / "serve_qwen38.py").is_file()
    assert "MIT License" in (root / "LICENSE").read_text(encoding="utf-8")


def test_launcher_environment_uses_saved_token_and_scoped_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.inference.kaggle_tpu.credential_secrets.get_kaggle_api_token",
        lambda provider_id: "kaggle-secret" if provider_id == "provider-1" else None,
    )
    monkeypatch.setattr("core.inference.kaggle_tpu.studio_root", lambda: tmp_path)
    env = _launcher_environment("provider-1")
    assert env["KAGGLE_API_TOKEN"] == "kaggle-secret"
    assert env["UNSLOTH_KAGGLE_TPU_STATE_FILE"] == str(
        tmp_path / "kaggle-tpu" / "provider-1.json"
    )


def test_launcher_key_is_parsed_for_storage_but_never_public():
    update = _parse_launcher_line("API key: sk-super-secret-value")
    assert update.pop("api_key") == "sk-super-secret-value"

    runtime = _Runtime(provider_id="provider-1")
    public = runtime.public()
    assert "api_key" not in public
    assert "sk-super-secret-value" not in json.dumps(public)


def test_cloudflare_endpoint_and_upstream_error_are_redacted():
    endpoint = "https://secret-host.trycloudflare.com/v1/chat/completions"
    assert _provider_url_for_log("kaggle_tpu", endpoint) == "<redacted-user-endpoint>"
    message = _friendly_provider_error_text(
        "kaggle_tpu",
        503,
        f"upstream unavailable at {endpoint}",
        model="qwen3.8-27b",
    )
    assert "trycloudflare.com" not in message
    assert "Reconnect" in message
    generic = _friendly_provider_error_text(
        "openai_compatible",
        500,
        f"internal proxy trace at {endpoint}",
        model="private-model",
    )
    assert "trycloudflare.com" not in generic
    assert "internal proxy trace" not in generic


def test_connection_probe_falls_back_to_manual_chat_when_models_is_unavailable():
    calls: list[str] = []

    class FakeClient:
        async def list_models(self):
            calls.append("models")
            raise RuntimeError("This compatible server has no model catalog")

        async def chat_completion(self, **kwargs):
            calls.append(f"chat:{kwargs['model']}")
            return {"choices": [{"message": {"content": "ok"}}]}

    result = asyncio.run(
        _test_openai_compatible_connection(FakeClient(), "manual-model")
    )
    assert result.success is True
    assert result.models_count == 0
    assert calls == ["models", "chat:manual-model"]


def test_managed_launcher_output_persists_the_ready_connection(monkeypatch, tmp_path):
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        """import sys
print('YOUR ENDPOINT IS LIVE', flush=True)
print('base URL : https://unit-test.trycloudflare.com/v1', flush=True)
print('API key  : sk-generated-secret', flush=True)
print('model    : qwen3.8-27b   (context: 262144)', flush=True)
""",
        encoding="utf-8",
    )
    row = {
        "id": "managed-1",
        "provider_type": "kaggle_tpu",
        "base_url": "",
        "models": ["qwen3.8-27b"],
        "capabilities": {"context_length": 262144},
        "managed_config": {"lab_path": str(tmp_path)},
    }
    manager = KaggleTPULifecycleManager()
    persisted: dict = {}

    async def fake_persist(provider_id: str, **ready):
        persisted.update(provider_id=provider_id, **ready)

    monkeypatch.setattr("core.inference.kaggle_tpu.providers_db.get_provider", lambda _id: row)
    monkeypatch.setattr(
        "core.inference.kaggle_tpu.credential_secrets.get_kaggle_api_token",
        lambda _id: "unit-test-kaggle-token",
    )
    monkeypatch.setattr(manager, "_persist_ready_guarded", fake_persist)
    monkeypatch.setattr(manager, "_healthy", AsyncMock(return_value=True))
    monkeypatch.setattr(manager, "_protect_launcher_state_file", lambda _provider_id: None)

    async def run():
        await manager.start("managed-1")
        runtime = manager._runtime["managed-1"]
        assert runtime.task is not None
        await asyncio.wait_for(runtime.task, timeout=5)

    asyncio.run(run())
    assert persisted == {
        "provider_id": "managed-1",
        "base_url": "https://unit-test.trycloudflare.com/v1",
        "api_key": "sk-generated-secret",
        "model": "qwen3.8-27b",
        "context_length": 262144,
        "text_only": False,
    }


def test_missing_managed_endpoint_does_not_start_without_opt_in(monkeypatch):
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes
    from core.inference import kaggle_tpu as kaggle_module

    row = {
        "id": "managed-auto-1",
        "provider_type": "kaggle_tpu",
        "display_name": "Kaggle TPU",
        "base_url": "",
        "is_enabled": True,
        "models": ["qwen3.8-27b"],
        "capabilities": {
            "supports_streaming": True,
            "supports_tool_calling": True,
            "supports_reasoning": True,
            "supports_vision": True,
            "supports_images": True,
            "context_length": 262144,
            "api_mode": "chat_completions",
        },
        "managed_config": {"auto_start": False, "text_only": False},
    }
    started: list[str] = []

    class FakeManager:
        async def status(self, provider_id: str):
            return {"state": "STOPPED", "message": "Use Start to start the TPU."}

        async def start(self, provider_id: str):
            started.append(provider_id)
            return {
                "provider_id": provider_id,
                "state": "STARTING",
                "message": "Starting bundled Kaggle TPU launcher.",
                "base_url": None,
            }

    async def is_disconnected():
        return False

    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(skip_api_monitor=True),
        is_disconnected=is_disconnected,
    )
    payload = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        provider_id="managed-auto-1",
        external_model="qwen3.8-27b",
        stream=True,
        enable_tools=False,
        enabled_tools=[],
        max_tokens=16,
    )
    monkeypatch.setattr(inference_routes.providers_db, "get_provider", lambda _id: dict(row))
    monkeypatch.setattr(kaggle_module, "kaggle_tpu_manager", FakeManager())

    async def run():
        return await inference_routes._proxy_to_external_provider(
            payload, request, current_subject="test"
        )

    try:
        asyncio.run(run())
    except HTTPException as exc:
        assert exc.status_code == 409
        assert "Start or Reconnect" in str(exc.detail)
        assert "Base URL is required" not in str(exc.detail)
    else:
        raise AssertionError("Expected a startup-in-progress response")
    assert started == []


def test_kaggle_reasoning_effort_uses_qwen_chat_template_kwargs(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        external_provider_module,
        "_http_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def run() -> None:
        client = ExternalProviderClient(
            provider_type="kaggle_tpu",
            base_url="http://127.0.0.1:1/v1",
            api_key="secret",
        )
        try:
            async for _line in client.stream_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                model="qwen3.8-27b",
                reasoning_effort="xhigh",
            ):
                pass
        finally:
            await client.close()

    asyncio.run(run())
    assert captured["body"]["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}


def test_kaggle_reasoning_can_be_disabled(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        external_provider_module,
        "_http_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def run() -> None:
        client = ExternalProviderClient(
            provider_type="kaggle_tpu",
            base_url="http://127.0.0.1:1/v1",
            api_key="secret",
        )
        try:
            async for _line in client.stream_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                model="qwen3.8-27b",
                reasoning_effort="none",
            ):
                pass
        finally:
            await client.close()

    asyncio.run(run())
    assert captured["body"]["enable_thinking"] is False
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_remote_connection_uses_its_context_length_for_rolling_compaction(
    monkeypatch,
):
    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    captured: dict = {}
    row = {
        "id": "remote-1",
        "provider_type": "openai_compatible",
        "display_name": "Remote",
        "base_url": "http://127.0.0.1:1/v1",
        "is_enabled": True,
        "capabilities": {
            "supports_streaming": True,
            "supports_tool_calling": True,
            "supports_reasoning": False,
            "supports_vision": False,
            "supports_images": False,
            "context_length": 1024,
            "api_mode": "chat_completions",
        },
        "managed_config": None,
    }

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def stream_chat_completion(self, *, messages, **_kwargs):
            captured["messages"] = messages
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

        async def close(self):
            pass

    async def is_disconnected():
        return False

    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(skip_api_monitor=True),
        is_disconnected=is_disconnected,
    )
    messages = [
        {"role": "system", "content": "Keep this instruction."},
        *[
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"turn-{index} " + ("x" * 500),
            }
            for index in range(10)
        ],
        {"role": "user", "content": "latest question"},
    ]
    payload = ChatCompletionRequest(
        messages=messages,
        provider_id="remote-1",
        external_model="remote-model",
        stream=True,
        enable_tools=False,
        enabled_tools=[],
        context_overflow="truncate_oldest",
        max_tokens=64,
    )
    monkeypatch.setattr(inference_routes.providers_db, "get_provider", lambda _id: dict(row))
    monkeypatch.setattr(
        inference_routes,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "secret",
    )
    monkeypatch.setattr(inference_routes, "ExternalProviderClient", FakeClient)

    async def run():
        response = await inference_routes._proxy_to_external_provider(
            payload, request, current_subject="test"
        )
        chunks = [chunk async for chunk in response.body_iterator]
        return "".join(
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in chunks
        )

    stream = asyncio.run(run())
    assert '"context_truncated"' in stream
    assert len(captured["messages"]) < len(messages)
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][-1]["content"] == "latest question"


def test_remote_stream_error_redacts_credentials_and_tunnel(monkeypatch, caplog):
    secret = 'sk-never-log-this-value'
    endpoint = 'https://private-host.trycloudflare.com'
    def handler(request):
        body = 'data: ' + json.dumps({'error': {'message': secret + ' ' + endpoint}}) + '\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(external_provider_module, '_http_client', http)
            client = ExternalProviderClient('kaggle_tpu', 'http://127.0.0.1:1/v1', 'test')
            return ''.join([x async for x in client.stream_chat_completion(messages=[{'role': 'user', 'content': 'hi'}], model='qwen3.8-27b')])
    result = asyncio.run(run())
    assert 'Reconnect' in result
    assert secret not in result + caplog.text
    assert endpoint not in result + caplog.text


def test_remote_vision_is_forwarded_only_when_declared():
    from routes.inference import _build_external_messages
    from models.inference import ChatMessage
    messages = [ChatMessage(role='user', content=[
        {'type': 'text', 'text': 'Describe this'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGVsbG8='}},
    ])]
    vision = _build_external_messages(messages, True, provider_type='kaggle_tpu')
    text = _build_external_messages(messages, False, provider_type='kaggle_tpu')
    assert 'data:image/png' in json.dumps(vision)
    assert 'data:image/png' not in json.dumps(text)
