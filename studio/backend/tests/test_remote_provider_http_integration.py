# SPDX-License-Identifier: AGPL-3.0-only
"""Actual HTTP/SSE and Studio's real local Python executor (no TPU required)."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from core.inference import external_provider as ep
from core.inference import kaggle_tpu
from models.inference import ChatCompletionRequest
from routes import inference


@pytest.mark.parametrize('provider_type', ['openai_compatible', 'kaggle_tpu'])
def test_http_model_requests_local_file_and_receives_result(monkeypatch, tmp_path, provider_type):
    output = tmp_path / 'hello.py'
    requests = []
    code = f"from pathlib import Path\nPath({str(output)!r}).write_text('print(\"hello world\")\\n')\nprint('LOCAL_FILE_CREATED')"
    args = json.dumps({'code': code})
    def sse(delta=None, finish=None, **extra):
        return ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}], **extra}) + '\n\n').encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            assert self.headers['Authorization'] == 'Bearer test-endpoint-key'
            assert self.path == '/v1/models'
            body = json.dumps({'data': [{'id': 'qwen3.8-27b', 'max_model_len': 262144}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            assert self.headers['Authorization'] == 'Bearer test-endpoint-key'
            assert self.path == '/v1/chat/completions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            if len(requests) == 1:
                assert not output.exists()
                assert any(t['function']['name'] == 'python' for t in body['tools'])
                self.wfile.write(sse({'reasoning_content': 'I will create the file using Studio.'}))
                self.wfile.write(sse({'tool_calls': [{'index': 0, 'id': 'call_local', 'type': 'function', 'function': {'name': 'python', 'arguments': args[:19]}}]}))
                self.wfile.write(sse({'tool_calls': [{'index': 0, 'function': {'arguments': args[19:]}}]}))
                self.wfile.write(sse(finish='tool_calls'))
            else:
                assert output.read_text() == 'print("hello world")\n'
                result = next(m for m in reversed(body['messages']) if m['role'] == 'tool')
                assert result['tool_call_id'] == 'call_local'
                assert 'LOCAL_FILE_CREATED' in result['content']
                self.wfile.write(sse({'content': 'The local file is ready.'}, finish='stop'))
            self.wfile.write(sse(usage={'prompt_tokens': 50, 'completion_tokens': 10, 'total_tokens': 60}))
            self.wfile.write(b'data: [DONE]\n\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}/v1'
    row = {'id': 'remote', 'provider_type': provider_type, 'display_name': 'Remote',
           'base_url': base, 'is_enabled': True, 'managed_config': {},
           'capabilities': {'supports_streaming': True, 'supports_tool_calling': True,
                            'supports_reasoning': True, 'supports_vision': True,
                            'supports_images': True, 'context_length': 262144}}
    monkeypatch.setattr(inference.providers_db, 'get_provider', lambda _: dict(row))
    monkeypatch.setattr(inference, 'resolve_provider_api_key_or_400', lambda *a, **kw: 'test-endpoint-key')
    async def ready(_): return {'state': 'READY', 'base_url': base}
    monkeypatch.setattr(kaggle_tpu.kaggle_tpu_manager, 'status', ready)
    async def connected(): return False
    request = SimpleNamespace(headers={}, state=SimpleNamespace(skip_api_monitor=True), is_disconnected=connected)
    payload = ChatCompletionRequest(
        provider_id='remote', external_model='qwen3.8-27b', stream=True,
        messages=[{'role': 'user', 'content': 'Create hello.py containing hello world.'}],
        enable_tools=True, enabled_tools=['python'], tool_choice='required',
        permission_mode='auto', bypass_permissions=True,
        context_overflow='truncate_oldest', max_tokens=1024, reasoning_effort='low',
    )
    async def run():
        async with httpx.AsyncClient(trust_env=False) as http:
            monkeypatch.setattr(ep, '_http_client', http)
            client = ep.ExternalProviderClient(provider_type, base, 'test-endpoint-key')
            catalog = await client.list_models()
            assert catalog[0]['max_model_len'] == 262144
            response = await inference._proxy_to_external_provider(payload, request, current_subject='test')
            return ''.join([chunk async for chunk in response.body_iterator])
    try:
        stream = asyncio.run(run())
        assert output.read_text() == 'print("hello world")\n', stream
        assert 'The local file is ready.' in stream
        assert 'tool_start' in stream and 'tool_end' in stream
        assert 'I will create the file using Studio.' in stream
        assert len(requests) == 2
        assert requests[0]['tool_choice'] == 'required'
        assert requests[1]['tool_choice'] == 'auto'
        if provider_type == 'kaggle_tpu':
            assert all(r['chat_template_kwargs']['reasoning_effort'] == 'low' for r in requests)
        assert stream.count('data: [DONE]') == 1
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_context_is_refitted_after_local_tool_results(monkeypatch, tmp_path):
    from core.inference.external_tool_transport import OAICompatTransport
    from core.inference.remote_context import RemoteContextFitter
    from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools
    from core.inference import studio_tool_loop
    from routes.inference import _estimate_external_prompt_tokens
    messages = [{'role': 'system', 'content': 'Preserve this system instruction.'}]
    messages += [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'old-' + str(i) + 'x' * 950} for i in range(12)]
    messages += [{'role': 'user', 'content': 'Read the tool result, then answer.'}]
    fitter = RemoteContextFitter(messages, thread_id=None, branch_message_ids=None,
                                context_length=5000, max_tokens=256,
                                count_tokens=_estimate_external_prompt_tokens, tools_enabled=True)
    seen = []
    fit_events = []
    class Client:
        async def stream_chat_completion(self, messages, **kwargs):
            seen.append(json.loads(json.dumps(messages)))
            if len(seen) == 1:
                yield 'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_read', 'function': {'name': 'web_search', 'arguments': '{"query":"test"}'}}]}, 'finish_reason': 'tool_calls'}]})
            else:
                yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}]})
            yield 'data: [DONE]'
    def prepare(msgs, tools):
        fitted, truncation, _ = fitter.fit(msgs, tools)
        fit_events.append(truncation)
        return fitted, [], not truncation or truncation.get('fits') is not False
    monkeypatch.setattr(studio_tool_loop, 'execute_tool', lambda *a, **kw: 'RESULT ' + 'y' * 9000)
    tools = [{'type': 'function', 'function': {'name': 'web_search', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]
    async def run():
        return [x async for x in stream_with_studio_tools(
            OAICompatTransport(Client(), model='remote', prepare_messages=prepare),
            run=ToolLoopRun(messages),
            policy=ToolLoopPolicy(tools, 3, 10, 'off', False, False, None),
            cancel_event=threading.Event(),
        )]
    asyncio.run(run())
    assert len(seen) == 2
    assert len(fit_events) == 2
    assert fit_events[1] and fit_events[1]['fits']
    assert fit_events[1]['dropped_messages'] > 0
    assert seen[1][0]['content'] == 'Preserve this system instruction.'
    assert any(m['role'] == 'tool' and m['content'].startswith('RESULT') for m in seen[1])
    assert len(seen[1]) < len(seen[0])
