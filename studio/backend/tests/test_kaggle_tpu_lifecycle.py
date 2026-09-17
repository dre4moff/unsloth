# SPDX-License-Identifier: AGPL-3.0-only
"""Regressions from the actual failed Kaggle launch, without network credentials."""
import asyncio
import importlib.util
import json
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.inference import kaggle_tpu as mod


def bridge_module():
    path = mod._bundled_launcher_root() / 'studio_bridge.py'
    spec = importlib.util.spec_from_file_location('test_studio_bridge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_network_failure_is_explained_without_printing_secrets(capsys):
    bridge = bridge_module()
    message = bridge.diagnostic('Temporary failure in name resolution sk-secret KGAT_secret')
    assert 'DNS/network' in message
    assert 'sk-secret' not in message
    bridge.JSON_MODE = True
    assert bridge.terminal_status('ERROR', 'Temporary failure in name resolution')
    event = json.loads(capsys.readouterr().out)
    assert event['status'] == 'ERROR'
    assert 'Internet' in event['message']


def test_fallback_reads_phases_from_official_kaggle_json_logs():
    bridge = bridge_module()
    logs = json.dumps([
        {'data': '[12:00] PHASE install {"vllm_tpu":"0.28.0"}\n'},
        {'data': 'Temporary failure in name resolution\n'},
        {'data': '[12:04] PHASE failed {"step":"install"}\n'},
    ])
    events, text = bridge.kernel_logs(lambda *args: SimpleNamespace(returncode=0, stdout=logs), 'user/kernel')
    assert [e['phase'] for e in events] == ['install', 'failed']
    assert 'DNS/network' in bridge.diagnostic(text)


def test_launcher_state_is_private_from_first_write(tmp_path):
    bridge = bridge_module()
    path = tmp_path / 'connection.json'
    bridge.save_state(path, {'api_key': 'secret'})
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())['api_key'] == 'secret'


def test_structured_ready_requires_complete_validated_tunnel():
    good = mod._parse_launcher_line(json.dumps({
        'status': 'READY', 'base_url': 'https://test.trycloudflare.com/v1',
        'api_key': 'sk-test', 'model': 'qwen3.8-27b', 'context_length': 262144,
    }))
    assert good['context_length'] == 262144
    bad = mod._parse_launcher_line(json.dumps({
        'status': 'READY', 'base_url': 'https://test.trycloudflare.com.evil/v1',
        'message': 'failed KGAT_secret sk-secret https://private.invalid',
    }))
    assert 'base_url' not in bad
    assert 'secret' not in bad['message']


def test_dead_tunnel_never_persists_ready(monkeypatch):
    manager = mod.KaggleTPULifecycleManager()
    monkeypatch.setattr(manager, '_healthy', AsyncMock(return_value=False))
    persist = AsyncMock()
    monkeypatch.setattr(manager, '_persist_ready_guarded', persist)
    runtime = mod._Runtime('id', base_url='https://test.trycloudflare.com/v1', model='qwen3.8-27b', context_length=262144)
    assert not asyncio.run(manager._accept_ready(runtime, 'sk-secret', {}))
    assert runtime.state == 'DISCONNECTED'
    persist.assert_not_called()


def test_saved_ready_becomes_disconnected_when_tunnel_dies(monkeypatch):
    manager = mod.KaggleTPULifecycleManager()
    manager._runtime['id'] = mod._Runtime('id', state='READY', base_url='https://test.trycloudflare.com/v1')
    monkeypatch.setattr(manager, '_row', lambda _: {'provider_type': 'kaggle_tpu'})
    monkeypatch.setattr(manager, '_healthy', AsyncMock(return_value=False))
    assert asyncio.run(manager.status('id'))['state'] == 'DISCONNECTED'


def test_start_reattaches_to_saved_session_instead_of_reprovisioning(monkeypatch, tmp_path):
    manager = mod.KaggleTPULifecycleManager()
    state = tmp_path / 'state.json'; state.write_text('{}')
    monkeypatch.setattr(mod, '_launcher_state_file', lambda _: state)
    monkeypatch.setattr(mod, '_kaggle_api_token', lambda _: 'token')
    monkeypatch.setattr(manager, '_row', lambda _: {'id': 'id', 'provider_type': 'kaggle_tpu'})
    async def refresh(runtime, managed):
        runtime.state = 'READY'
    monkeypatch.setattr(manager, '_refresh_locked', refresh)
    calls = []
    monkeypatch.setattr(manager, '_watch', lambda runtime, managed, **kw: calls.append(kw['action']))
    assert asyncio.run(manager.start('id'))['state'] == 'READY'
    assert calls == ['status']


def test_stop_propagates_a_real_launcher_failure(monkeypatch, tmp_path):
    script = tmp_path / 'launch.py'
    script.write_text('import sys\nprint(\'{"status":"ERROR","message":"Kaggle denied access."}\')\nsys.exit(1)\n')
    manager = mod.KaggleTPULifecycleManager()
    monkeypatch.setattr(manager, '_row', lambda _: {'provider_type': 'kaggle_tpu', 'managed_config': {'lab_path': str(tmp_path)}})
    monkeypatch.setattr(mod, '_kaggle_api_token', lambda _: 'fake')
    result = asyncio.run(manager.stop('id'))
    assert result['state'] == 'ERROR'
    assert result['message'] == 'Kaggle denied access.'


def test_status_recovers_ready_older_than_eight_heartbeats(monkeypatch, capsys, tmp_path):
    root = mod._bundled_launcher_root()
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('test_lab_launcher', root / 'launch.py')
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
    monkeypatch.setattr(lab.bridge, 'JSON_MODE', True)
    state = tmp_path / 'state.json'; state.write_text(json.dumps({'kernel': 'u/k', 'topic': 'test'}))
    monkeypatch.setattr(lab, 'STATE_FILE', state)
    monkeypatch.setattr(lab.bridge, 'status', lambda *a: 'RUNNING')
    monkeypatch.setattr(lab.bridge, 'kernel_logs', lambda *a, **kw: ([], ''))
    ready = {'phase': 'ready', 'endpoint': 'https://test.trycloudflare.com/v1', 'api_key': 'sk-test', 'model': 'm', 'max_model_len': 262144}
    monkeypatch.setattr(lab, 'read_events', lambda *a: [(1, ready)] + [(i, {'phase': 'heartbeat'}) for i in range(2, 25)])
    monkeypatch.setattr(lab.bridge, 'health', lambda ev: ev == ready)
    lab.cmd_status(SimpleNamespace(follow=False))
    assert json.loads(capsys.readouterr().out)['status'] == 'READY'


def test_json_state_does_not_store_plaintext_inference_key(tmp_path):
    bridge = bridge_module(); bridge.JSON_MODE = True
    path = tmp_path / 'state.json'
    bridge.save_state(path, {'kernel': 'user/kernel', 'topic': 'test-topic', 'api_key': 'sk-private'})
    assert 'api_key' not in json.loads(path.read_text())
    assert 'sk-private' not in path.read_text()


def test_live_log_snapshot_reports_network_failure_before_kernel_exits():
    bridge = bridge_module()
    calls = []
    def kaggle(*args, **kwargs):
        calls.append((args, kwargs))
        if '--follow' in args:
            return SimpleNamespace(returncode=0, stdout='[12:00] PHASE install {}\nTemporary failure in name resolution\n')
        return SimpleNamespace(returncode=0, stdout='\n')
    events, _ = bridge.kernel_logs(kaggle, 'user/kernel')
    assert events[-1]['phase'] == 'network-unavailable'
    assert calls[-1][1]['timeout'] == 5


def test_compilation_failure_is_not_misclassified_as_installation(capsys):
    bridge = bridge_module(); bridge.JSON_MODE = True
    logs = ('PHASE install {}\nPHASE installed {}\nPHASE compiling {"elapsed_s":1200}\n'
            'PHASE failed {"step":"server","tail":"Engine process failed"}\n')
    bridge.terminal_status('ERROR', logs)
    event = json.loads(capsys.readouterr().out)
    assert event['session_ended'] is True
    assert 'server stopped' in event['message']
    assert 'failed to install' not in event['message']


def test_harmless_upstream_metadata_warning_does_not_become_internet_failure():
    bridge = bridge_module()
    warning = "Unable to poll the TPU GCE Metadata: metadata.google.internal: Name or service not known"
    assert 'DNS/network' not in bridge.diagnostic(warning, step='server')
    assert 'DNS/network' in bridge.diagnostic(warning + '\npypi.org: Temporary failure in name resolution', step='install')


def test_package_versions_and_elapsed_numbers_are_not_http_error_codes():
    bridge = bridge_module()
    for text in ['version 0.401.0 installed', 'elapsed_s: 1429', 'shape 4032', 'Triton not installed']:
        message = bridge.diagnostic(text, step='server')
        assert 'server stopped' in message
        assert 'token' not in message


def test_compile_timeout_and_out_of_memory_have_distinct_diagnostics():
    bridge = bridge_module()
    assert 'deadline' in bridge.diagnostic('install succeeded', step='health-timeout')
    assert 'out of memory' in bridge.diagnostic('RESOURCE_EXHAUSTED: allocation failed', step='server')


def test_watch_preserves_structured_server_failure(monkeypatch, capsys):
    root = mod._bundled_launcher_root()
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('test_watch_failure', root / 'launch.py')
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
    monkeypatch.setattr(lab.bridge, 'JSON_MODE', True)
    failure = {'phase':'failed', 'step':'server', 'tail':'Engine exited'}
    monkeypatch.setattr(lab, 'read_events', lambda *a: [(1, failure)])
    monkeypatch.setattr(lab.bridge, 'kernel_logs', lambda *a, **kw: ([], 'PHASE install {}\nPHASE installed {}'))
    lab.watch('u/k', 'topic')
    events = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    assert len(events) == 1
    assert 'server stopped' in events[0]['message']


def test_compilation_progress_reports_elapsed_minutes(capsys):
    bridge = bridge_module(); bridge.JSON_MODE = True
    bridge.render({'phase':'compiling', 'elapsed_s':1200})
    event = json.loads(capsys.readouterr().out)
    assert '20 minutes elapsed' in event['message']


def preflight_module():
    spec = importlib.util.spec_from_file_location('test_preflight', mod._bundled_launcher_root() / 'studio_preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def launcher_module(monkeypatch, name):
    root = mod._bundled_launcher_root(); monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(name, root / 'launch.py')
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
    monkeypatch.setattr(lab.bridge, 'JSON_MODE', True)
    return lab


def interactive_module():
    path = mod._bundled_launcher_root() / 'studio_interactive.py'
    spec = importlib.util.spec_from_file_location('test_studio_interactive', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interactive_bootstrap_is_cpu_only_and_keeps_fast_start_datasets(tmp_path):
    interactive = interactive_module(); calls = []; metadata = {}
    statuses = iter([
        SimpleNamespace(returncode=1, stdout='', stderr='missing'),
        SimpleNamespace(returncode=0, stdout='"KernelWorkerStatus.COMPLETE"', stderr=''),
    ])
    def cli(*args):
        calls.append(args)
        if args[:2] == ('kernels', 'status'):
            return next(statuses)
        if args[:2] == ('kernels', 'push'):
            root = args[args.index('-p') + 1]
            metadata.update(json.loads(open(root + '/kernel-metadata.json').read()))
            return SimpleNamespace(returncode=0, stdout='ok', stderr='')
        raise AssertionError(args)
    interactive.ensure_bootstrap(
        cli, 'user/private-notebook', 'owner/weights', 'owner/env',
        emit=lambda *_: None,
    )
    assert metadata['enable_tpu'] == 'false'
    assert metadata['enable_gpu'] == 'false'
    assert metadata['dataset_sources'] == ['owner/weights', 'owner/env']
    assert ('kernels', 'push') == calls[1][:2]


def test_interactive_create_session_uses_kaggle_interactive_shape(monkeypatch):
    interactive = interactive_module(); seen = {}; allocated = []
    class JsonObject:
        def __init__(self, value): self.value = value
        def __str__(self): return json.dumps(self.value)
    class KernelApi:
        def create_kernel_session(self, request):
            seen['shape'] = request.machine_shape
            seen['slug'] = request.slug
            return JsonObject({'name':'operations/1','metadata':{'kernelSessionId':17},'done':False})
    class Operations:
        def get_operation(self, name):
            return JsonObject({'name':name,'done':True,'response':{
                'kernelSessionId':17,
                'jupyterUrl':'https://jupyter.invalid',
                'tokenizedJupyterUrl':'https://jupyter.invalid/?token=private',
            }})
    class Client:
        def __init__(self):
            self.kernels = SimpleNamespace(kernels_api_client=KernelApi())
            self.common = SimpleNamespace(operations_client=Operations())
    class Context:
        def __enter__(self): return Client()
        def __exit__(self, *args): pass
    fake_api = SimpleNamespace(build_kaggle_client=lambda: Context())
    monkeypatch.setattr(interactive, '_api', lambda: fake_api)
    session = interactive.create_session(
        'user/private-notebook',
        on_allocated=lambda op, sid: allocated.append((op, sid)),
        emit=lambda *_: None,
    )
    assert seen == {'shape':'TpuV5E8', 'slug':'user/private-notebook'}
    assert session.kernel_session_id == 17
    assert allocated[-1] == ('operations/1', 17)


def test_interactive_create_session_has_no_default_allocation_deadline(monkeypatch):
    interactive = interactive_module(); allocated = []; polls = {'count': 0}
    class JsonObject:
        def __init__(self, value): self.value = value
        def __str__(self): return json.dumps(self.value)
    class KernelApi:
        def create_kernel_session(self, request):
            return JsonObject({'name':'operations/slow','metadata':{'kernelSessionId':23},'done':False})
    class Operations:
        def get_operation(self, name):
            polls['count'] += 1
            if polls['count'] == 1:
                return JsonObject({'name':name,'metadata':{'kernelSessionId':23},'done':False})
            return JsonObject({'name':name,'done':True,'response':{
                'kernelSessionId':23,
                'jupyterUrl':'https://jupyter.invalid',
                'tokenizedJupyterUrl':'https://jupyter.invalid/?token=private',
            }})
    class Client:
        def __init__(self):
            self.kernels = SimpleNamespace(kernels_api_client=KernelApi())
            self.common = SimpleNamespace(operations_client=Operations())
    class Context:
        def __enter__(self): return Client()
        def __exit__(self, *args): pass
    monkeypatch.setattr(interactive, '_api', lambda: SimpleNamespace(build_kaggle_client=lambda: Context()))
    monkeypatch.setattr(interactive.time, 'monotonic', iter((0.0, 1000.0, 2000.0, 3000.0)).__next__)
    monkeypatch.setattr(interactive.time, 'sleep', lambda _: None)
    session = interactive.create_session(
        'user/private-notebook',
        on_allocated=lambda op, sid: allocated.append((op, sid)),
        emit=lambda *_: None,
    )
    assert session.kernel_session_id == 23
    assert polls['count'] == 2
    assert allocated[-1] == ('operations/slow', 23)


def test_interactive_allocation_does_not_use_unrelated_batch_status(monkeypatch):
    interactive = interactive_module(); events = []; polls = {'count': 0}
    class JsonObject:
        def __init__(self, value): self.value = value
        def __str__(self): return json.dumps(self.value)
    class Status:
        status = SimpleNamespace(name='QUEUED')
        failure_message = ''
    class Quota:
        time_used = timedelta(hours=1)
        time_reserved = timedelta(0)
        total_time_allowed = timedelta(hours=20)
    class QuotaResponse:
        tpu_quota = Quota()
    class KernelApi:
        def create_kernel_session(self, request):
            return JsonObject({'name':'operations/queued','metadata':{'kernelSessionId':29},'done':False})
        def get_kernel_session_status(self, request): raise AssertionError('batch status is unrelated')
        def get_accelerator_quota_statistics(self, request): return QuotaResponse()
    class Operations:
        def get_operation(self, name):
            polls['count'] += 1
            if polls['count'] == 1:
                return JsonObject({'name':name,'metadata':{'kernelSessionId':29},'done':False})
            return JsonObject({'name':name,'done':True,'response':{
                'kernelSessionId':29,
                'jupyterUrl':'https://jupyter.invalid',
                'tokenizedJupyterUrl':'https://jupyter.invalid/?token=private',
            }})
    class Client:
        def __init__(self):
            self.kernels = SimpleNamespace(kernels_api_client=KernelApi())
            self.common = SimpleNamespace(operations_client=Operations())
    class Context:
        def __enter__(self): return Client()
        def __exit__(self, *args): pass
    monkeypatch.setattr(interactive, '_api', lambda: SimpleNamespace(build_kaggle_client=lambda: Context()))
    monkeypatch.setattr(interactive.time, 'monotonic', iter((0.0, 61.0, 62.0, 63.0)).__next__)
    monkeypatch.setattr(interactive.time, 'sleep', lambda _: None)
    session = interactive.create_session(
        'user/private-notebook',
        on_allocated=lambda *_: None,
        emit=lambda state, message: events.append((state, message)),
    )
    assert session.kernel_session_id == 29
    assert any('has not completed' in message.lower() for _, message in events)
    assert not any('queued' in message.lower() or 'assigned' in message.lower() for _, message in events)


def test_interactive_allocation_timeout_has_specific_diagnostic():
    bridge = bridge_module()
    message = bridge.diagnostic('Kaggle timed out while allocating the interactive TPU session.')
    assert 'did not finish allocating' in message
    assert 'Reconnect' in message


def test_interactive_stop_cancels_session_without_deleting_bootstrap(monkeypatch, capsys, tmp_path):
    lab = launcher_module(monkeypatch, 'test_interactive_stop')
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({
        'interactive': True, 'kernel': 'u/bootstrap', 'topic': 't',
        'kernel_session_id': 42, 'stopped': False,
    }))
    monkeypatch.setattr(lab, 'STATE_FILE', state)
    cancelled = []
    monkeypatch.setattr(lab, 'studio_interactive', lambda: SimpleNamespace(
        cancel_session=lambda sid: cancelled.append(sid),
    ))
    monkeypatch.setattr(lab.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not delete bootstrap')))
    lab.cmd_stop(SimpleNamespace())
    assert cancelled == [42]
    assert json.loads(state.read_text())['stopped'] is True
    assert json.loads(capsys.readouterr().out)['status'] == 'STOPPED'


def test_preflight_fails_before_install_or_network_on_cpu(monkeypatch):
    import pytest
    preflight = preflight_module(); events = []
    monkeypatch.setattr(preflight._studio_glob, 'glob', lambda _: [])
    monkeypatch.setattr(preflight._studio_time, 'sleep', lambda _: None)
    monkeypatch.setattr(preflight._studio_request, 'urlopen', lambda *a, **k: pytest.fail('CPU must fail before downloads'))
    with pytest.raises(SystemExit):
        preflight.studio_preflight(lambda phase, **fields: events.append({'phase':phase, **fields}))
    assert events[-1]['step'] == 'tpu-devices'
    assert events[-1]['retryable'] is True


def test_jax_validation_rejects_cpu_and_accepts_exactly_eight_tpus(monkeypatch):
    import pytest
    preflight = preflight_module(); events = []
    for platforms in [['cpu'], ['tpu'] * 4, ['tpu'] * 8]:
        monkeypatch.setattr(preflight._studio_subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps({'count':len(platforms), 'platforms':platforms})))
        publish = lambda phase, **fields: events.append({'phase':phase, **fields})
        if len(platforms) == 8:
            preflight.studio_validate_tpu('/tmp/python', publish)
            assert events[-1]['phase'] == 'tpu-verified'
        else:
            with pytest.raises(SystemExit):preflight.studio_validate_tpu('/tmp/python', publish)
            assert events[-1]['step'] == 'tpu-devices'


def test_generated_kernel_guards_upstream_install_and_launch(monkeypatch):
    import ast
    lab = launcher_module(monkeypatch, 'test_prepare')
    source = lab.prepare_kernel({'ntfy_topic':'test-run', 'api_key':'sk-test'})
    ast.parse(source)
    assert source.index('studio_preflight(publish)\n\n#') < source.index('runtime = install_runtime(')
    assert source.index('studio_validate_tpu(PY, publish)') < source.index('server = launch_server(CFG)')
    assert "STUDIO_RUN " in source
    assert 'MTP_PATCH_B64 =' in source


def test_automatic_retry_is_bounded_and_waits_for_terminal_session(monkeypatch, capsys):
    lab = launcher_module(monkeypatch, 'test_retry')
    failure = {'phase':'failed', 'step':'tpu-devices', 'retryable':True}
    calls = []
    monkeypatch.setattr(lab, 'submit_serve', lambda args, attempt: calls.append(attempt) or failure)
    monkeypatch.setattr(lab, 'load_state', lambda: {'kernel':'u/k'})
    monkeypatch.setattr(lab.bridge, 'status', lambda *a: 'ERROR')
    lab.cmd_serve(SimpleNamespace())
    assert calls == [0, 1]
    assert 'attempt 2 of 2' in capsys.readouterr().out
    calls.clear()
    monkeypatch.setattr(lab.bridge, 'status', lambda *a: 'UNKNOWN')
    lab.cmd_serve(SimpleNamespace())
    assert calls == [0]


def test_previous_kernel_version_logs_cannot_fail_new_run():
    bridge = bridge_module()
    old = 'STUDIO_RUN previous\nPHASE failed {"step":"server"}\n'
    new = 'STUDIO_RUN current\nPHASE preflight {}\n'
    def cli(*args, **kw):
        return SimpleNamespace(returncode=0, stdout=new if '--follow' in args else old)
    events, raw = bridge.kernel_logs(cli, 'u/k', expected_run='current')
    assert [e['phase'] for e in events] == ['preflight']
    assert 'previous' not in raw
    events, raw = bridge.kernel_logs(lambda *a, **kw: SimpleNamespace(returncode=0, stdout=old), 'u/k', expected_run='current')
    assert events == [] and raw == ''


def test_tpu_allocation_error_survives_truncated_failure_tail(capsys):
    bridge = bridge_module(); bridge.JSON_MODE = True
    logs = ('PHASE installed {}\nPHASE server-launch {}\n'
            'ValueError: Insufficient devices for 2D mesh: found 1, expected 8\n'
            'PHASE failed {"step":"server","tail":"Engine core initialization failed"}')
    bridge.terminal_status('ERROR', logs)
    assert 'eight TPU devices' in json.loads(capsys.readouterr().out)['message']


def test_ready_retries_transient_tunnel_before_persisting(monkeypatch):
    manager = mod.KaggleTPULifecycleManager()
    monkeypatch.setattr(manager, '_healthy', AsyncMock(side_effect=[False, True]))
    persist = AsyncMock(); monkeypatch.setattr(manager, '_persist_ready_guarded', persist)
    monkeypatch.setattr(mod.asyncio, 'sleep', AsyncMock())
    runtime = mod._Runtime('id', base_url='https://test.trycloudflare.com/v1', model='m', context_length=262144)
    assert asyncio.run(manager._accept_ready(runtime, 'sk-test', {}, attempts=2))
    assert runtime.state == 'READY'
    persist.assert_awaited_once()


def test_read_timeout_does_not_crash_launcher(monkeypatch):
    import subprocess
    lab = launcher_module(monkeypatch, 'test_read_timeout')
    def timeout(cmd, **kw):
        assert kw['timeout'] <= 15
        raise subprocess.TimeoutExpired(cmd, kw['timeout'])
    monkeypatch.setattr(lab.subprocess, 'run', timeout)
    for action in ['status', 'logs']:
        assert lab.kaggle('kernels', action, 'u/k').returncode == 124


def test_queued_monitor_survives_transient_status_failure(monkeypatch, capsys, tmp_path):
    lab = launcher_module(monkeypatch, 'test_queued_retry')
    monkeypatch.setattr(lab, 'STATE_FILE', tmp_path / 'absent.json')
    monkeypatch.setattr(lab, 'read_events', lambda *a: [])
    monkeypatch.setattr(lab.bridge, 'kernel_logs', lambda *a, **kw: ([], ''))
    monkeypatch.setattr(lab.time, 'sleep', lambda _: None)
    responses = iter([
        SimpleNamespace(returncode=0, stdout='"KernelWorkerStatus.QUEUED"', stderr=''),
        SimpleNamespace(returncode=124, stdout='', stderr='request timed out'),
        SimpleNamespace(returncode=0, stdout='"KernelWorkerStatus.QUEUED"', stderr=''),
        SimpleNamespace(returncode=0, stdout='"KernelWorkerStatus.COMPLETE"', stderr=''),
    ])
    monkeypatch.setattr(lab, 'kaggle', lambda *a: next(responses))
    lab.watch('u/k','topic')
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [e['status'] for e in events] == ['PROVISIONING', 'MONITOR_RETRY', 'PROVISIONING', 'STOPPED']
    assert 'minutes in queue' in events[0]['message']
    assert 'installation has not started' in events[0]['message']


def test_persistent_status_failure_detaches_without_restarting(monkeypatch, capsys, tmp_path):
    lab = launcher_module(monkeypatch, 'test_persistent_status')
    monkeypatch.setattr(lab, 'STATE_FILE', tmp_path / 'absent.json')
    monkeypatch.setattr(lab, 'read_events', lambda *a: [])
    monkeypatch.setattr(lab.bridge, 'kernel_logs', lambda *a, **kw: ([], ''))
    monkeypatch.setattr(lab.time, 'sleep', lambda _: None)
    calls = []
    def cli(*args):
        calls.append(args)
        return SimpleNamespace(returncode=124, stdout='', stderr='request timed out')
    monkeypatch.setattr(lab, 'kaggle', cli)
    lab.watch('u/k','topic')
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert len(calls) == 5
    assert events[-1]['status'] == 'DISCONNECTED'
    assert all(c[:2] == ('kernels','status') for c in calls)


def test_monitor_retry_preserves_runtime_state_and_redacts_message():
    update = mod._parse_launcher_line(json.dumps({'status':'MONITOR_RETRY','message':'retry sk-secret https://private.invalid'}))
    assert 'state' not in update
    assert 'secret' not in update['message']


def test_status_timeout_is_disconnected_not_model_failure(capsys):
    bridge = bridge_module(); bridge.JSON_MODE = True
    assert bridge.status(lambda *a: SimpleNamespace(returncode=124,stdout='',stderr='timeout'), 'u/k') == 'UNKNOWN'
    assert json.loads(capsys.readouterr().out)['status'] == 'DISCONNECTED'


def test_kernel_error_before_boot_is_reported_as_allocation_failure(monkeypatch, capsys, tmp_path):
    lab = launcher_module(monkeypatch, 'test_preboot_error')
    monkeypatch.setattr(lab, 'STATE_FILE', tmp_path / 'absent.json')
    monkeypatch.setattr(lab, 'read_events', lambda *a: [])
    monkeypatch.setattr(lab.bridge, 'kernel_logs', lambda *a, **kw: ([], ''))
    monkeypatch.setattr(lab, 'kaggle', lambda *a: SimpleNamespace(returncode=0, stdout='"KernelWorkerStatus.ERROR"', stderr=''))
    lab.watch('u/k','topic')
    event=json.loads(capsys.readouterr().out)
    assert event['session_ended'] is True
    assert 'before the model started' in event['message']


def test_managed_submission_queues_complete_server_without_interactive_bootstrap(monkeypatch, tmp_path):
    lab = launcher_module(monkeypatch, 'test_managed_batch_submission')
    state_file = tmp_path / 'connection.json'
    monkeypatch.setattr(lab, 'STATE_FILE', state_file)
    monkeypatch.setattr(lab, 'check_auth', lambda: None)
    monkeypatch.setattr(lab, 'kaggle_username', lambda _: 'owner')
    monkeypatch.setattr(lab, 'prepare_kernel', lambda cfg: 'SERVER_SOURCE_WITH_CONFIG')
    monkeypatch.setattr(lab, 'studio_interactive', lambda: (_ for _ in ()).throw(AssertionError('must submit server, not bootstrap')))
    seen = {}
    def cli(*args):
        from pathlib import Path
        assert args[:2] == ('kernels', 'push')
        root = Path(args[args.index('-p') + 1])
        seen['metadata'] = json.loads((root / 'kernel-metadata.json').read_text())
        seen['code'] = (root / 'serve_qwen38.py').read_text()
        # The request must already be tracked if the app closes during push.
        seen['state_at_push'] = json.loads(state_file.read_text())
        return SimpleNamespace(returncode=0, stdout='Kernel version 1 successfully pushed', stderr='')
    monkeypatch.setattr(lab, 'kaggle', cli)
    monkeypatch.setattr(lab, 'watch', lambda *args, **kw: seen.update(watch=args))
    args = SimpleNamespace(user=None, slug='managed', max_model_len=262144, max_num_seqs=4,
                           mtp=3, reasoning_effort='xhigh', keepalive_min=480,
                           weights_dataset='owner/weights', no_tools=False,
                           text_only=False, verbose=False, fast_start=False)
    lab.submit_serve(args)
    assert seen['metadata']['machine_shape'] == 'TpuV5E8'
    assert seen['metadata']['is_private'] == 'true'
    assert seen['metadata']['enable_internet'] == 'true'
    assert seen['metadata']['dataset_sources'] == ['owner/weights', lab.ENV_DATASET]
    assert seen['code'] == 'SERVER_SOURCE_WITH_CONFIG'
    assert seen['state_at_push']['kernel'] == 'owner/managed'
    assert 'api_key' not in seen['state_at_push']
    assert not seen['state_at_push'].get('interactive')
    assert seen['watch'][0] == 'owner/managed'


def test_reattach_closed_interactive_operation_is_terminal(monkeypatch, tmp_path, capsys):
    lab = launcher_module(monkeypatch, 'test_closed_interactive_operation')
    interactive = interactive_module()
    def restore(_):
        raise interactive.InteractiveSessionClosed('closed')
    monkeypatch.setattr(interactive, 'restore_session', restore)
    monkeypatch.setattr(lab, 'studio_interactive', lambda: interactive)
    lab.interactive_status({'operation_name': 'operations/closed', 'topic': 'topic'})
    event = json.loads(capsys.readouterr().out)
    assert event['status'] == 'ERROR'
    assert event['session_ended'] is True


def test_batch_reconnect_does_not_call_a_queued_session_running(monkeypatch, tmp_path, capsys):
    lab = launcher_module(monkeypatch, 'test_batch_queue_reconnect')
    state_file = tmp_path / 'state.json'
    state_file.write_text(json.dumps({'kernel': 'u/k', 'topic': 't', 'started_at': 100}))
    monkeypatch.setattr(lab, 'STATE_FILE', state_file)
    monkeypatch.setattr(lab.bridge, 'status', lambda *_: 'QUEUED')
    monkeypatch.setattr(lab.bridge, 'kernel_logs', lambda *a, **kw: ([], ''))
    monkeypatch.setattr(lab, 'read_events', lambda *_: [])
    lab.cmd_status(SimpleNamespace(follow=False))
    event = json.loads(capsys.readouterr().out)
    assert event['status'] == 'PROVISIONING'
    assert 'queued' in event['message']
    assert 'running' not in event['message']
