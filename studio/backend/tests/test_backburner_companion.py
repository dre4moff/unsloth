# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import asyncio
import hashlib
import json
import plistlib
import socket
import struct
import threading
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.companion import backburner as bb
from core.companion.backburner import BackburnerManager, PATN, phone_hello, usb_phones
from core.companion.models import CompanionSettings


def run_async(test):
    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return run


def registry(speed=5, name="iPhone", vendor=0x05AC, serial="00008150001E29A12669401C", interface="en8"):
    return [{"idVendor": vendor, "USB Product Name": name, "USB Serial Number": serial,
             "USBSpeed": speed, "bcdUSB": 0x0320,
             "IORegistryEntryChildren": [{"IORegistryEntryChildren": [{"BSD Name": interface}]}]}]


@pytest.mark.parametrize("speed", [None, 0, 1, 2, 3, 4, 7, "5"])
def test_usb_unknown_or_slow_is_never_accepted(speed):
    assert usb_phones(registry(speed)) == []


@pytest.mark.parametrize("speed,gbps", [(5, 10), (6, 20)])
def test_usb_uses_negotiated_speed_of_exact_iphone(speed, gbps):
    phone = usb_phones(registry(speed))[0]
    assert phone["speedGbps"] == gbps
    assert phone["interface"] == "en8"


@pytest.mark.parametrize("kwargs", [dict(name="iPad"), dict(name="USB SSD"), dict(vendor=123),
                                    dict(serial="wireless-uuid"), dict(interface="awdl0")])
def test_usb_other_devices_and_wireless_interfaces_are_rejected(kwargs):
    assert usb_phones(registry(**kwargs)) == []


def test_fast_hub_cannot_hide_a_slow_iphone():
    tree = {"USBSpeed": 5, "IORegistryEntryChildren": registry(3)}
    assert usb_phones(tree) == []


def test_usb_hyphenated_apple_serial_is_accepted():
    assert usb_phones(registry(serial="00008150-001E29A12669401C"))[0]["speedGbps"] == 10


@pytest.mark.parametrize("speed,available", [(5, True), (4, False)])
def test_discovery_includes_usb_network_descendant_properties(monkeypatch, speed, available):
    monkeypatch.setattr(bb.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(bb.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(bb.platform, "mac_ver", lambda: ("27.0.1", (), ""))

    def run(args, timeout=8):
        if args[0].endswith("ioreg"):
            tree = registry(speed)
            if "-l" not in args:
                # ioreg otherwise omits properties of non-matching descendants.
                tree[0]["IORegistryEntryChildren"] = [{"IORegistryEntryName": "en8"}]
            return plistlib.dumps(tree)
        if args[0].endswith("ifconfig"):
            assert args[1] == "en8"
            return b"inet 169.254.1.1 netmask 0xffff0000\n"
        if args[0].endswith("ping"):
            assert args[1:3] == ["-b", "en8"]
            return b"64 bytes from 169.254.1.2: icmp_seq=0\n"
        if args[0].endswith("route"):
            return b"interface: en8\n"
        raise AssertionError(args)

    monkeypatch.setattr(bb, "_run", run)
    monkeypatch.setattr(bb, "phone_hello", lambda address, source: True)
    monkeypatch.setattr(bb, "phone_command", lambda phone, command: {"tail_state": "down"})
    phone = bb.detect_wired_phone()
    assert (phone is not None) is available
    if available:
        assert phone["ready"] and phone["interface"] == "en8"
        assert phone["speedGbps"] == 10


@pytest.mark.parametrize("version,magic,length,expected", [(3, PATN, 72, True),
                                                          (2, PATN, 72, False),
                                                          (3, 0, 72, False),
                                                          (3, PATN, 1000000000, False)])
def test_phone_hello_framing_and_version_on_real_socket(version, magic, length, expected, monkeypatch):
    server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen()
    port = server.getsockname()[1]
    errors = []
    def serve():
        try:
            with server.accept()[0] as client:
                assert client.recv(16) == struct.pack("<IIQ", PATN, 1, 0)
                data = struct.pack("<IIQ", magic, 2, length)
                if magic == PATN and length == 72:
                    data += struct.pack("<II64s", version, 0, b"iPhone")
                for index in range(0, len(data), 5):
                    client.sendall(data[index:index+5])
        except Exception as exc:
            errors.append(exc)
        finally:
            server.close()
    worker = threading.Thread(target=serve); worker.start()
    original_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", lambda self, address: original_connect(self, (address[0], port)))
    assert phone_hello("127.0.0.1", "127.0.0.1") is expected
    worker.join(5)
    assert not worker.is_alive() and not errors


def test_eof_is_bounded_and_does_not_spin():
    first, second = socket.socketpair()
    second.close()
    with first, pytest.raises(ConnectionError):
        bb._recv_exact(first, 16)


class Companion:
    def __init__(self, enabled=True):
        self.settings = CompanionSettings(enabled=enabled)
        self.pending = False
    def has_pending_work(self):
        return self.pending
    async def update_settings(self, value, *, persist=True):
        self.settings = value
        self.last_persist = persist


class Backend:
    def __init__(self, manager, fail=False):
        self.last_load_intent = object()
        self.is_loaded = True
        self.loads = []
        self.manager = manager
        self.fail = fail
        self.unloads = 0
    def load_model(self, intent):
        self.loads.append(self.manager.mode)
        return not (self.fail and self.manager.mode == "speed")
    def unload_model(self):
        self.unloads += 1
        return True


@run_async
@pytest.mark.parametrize("enabled", [True, False])
async def test_modes_restore_original_agent_preference(enabled, monkeypatch):
    manager = BackburnerManager(); companion = Companion(enabled); backend = Backend(manager)
    monkeypatch.setattr(manager, "preflight", lambda intent: {})
    monkeypatch.setattr(manager, "status", lambda refresh=True: {"mode": manager.mode})
    await manager.select_mode("speed", backend, companion)
    assert manager.mode == "speed" and not companion.settings.enabled
    assert companion.last_persist is False
    await manager.select_mode("agent", backend, companion)
    assert manager.mode == "agent" and companion.settings.enabled is enabled
    assert companion.last_persist is False
    assert backend.loads == ["speed", "agent"]


@run_async
async def test_failed_speed_start_restores_mac_and_companion(monkeypatch):
    manager = BackburnerManager(); companion = Companion(); backend = Backend(manager, fail=True)
    monkeypatch.setattr(manager, "preflight", lambda intent: {})
    with pytest.raises(ValueError, match="failed to start"):
        await manager.select_mode("speed", backend, companion)
    assert backend.loads == ["speed", "agent"]
    assert manager.mode == "agent" and companion.settings.enabled


@run_async
async def test_queued_iphone_work_prevents_model_teardown(monkeypatch):
    manager = BackburnerManager(); companion = Companion(); companion.pending = True
    backend = Backend(manager)
    with pytest.raises(ValueError, match="Finish running"):
        await manager.select_mode("speed", backend, companion)
    assert not backend.loads and not backend.unloads


@run_async
async def test_task_starting_during_preflight_preserves_runtime(monkeypatch):
    manager = BackburnerManager(); companion = Companion(); backend = Backend(manager)
    def race(intent):
        companion.pending = True
    monkeypatch.setattr(manager, "preflight", race)
    with pytest.raises(ValueError, match="task started"):
        await manager.select_mode("speed", backend, companion)
    assert not backend.loads and companion.settings.enabled


def test_disconnect_never_reenables_agent_mid_generation(monkeypatch):
    manager = BackburnerManager(); manager.mode = "speed"
    monkeypatch.setattr(bb, "detect_wired_phone", lambda existing=None: None)
    state = manager.status()
    assert state["mode"] == "speed" and not state["available"] and not state["ready"]
    assert "USB connection lost" in state["error"]


def test_upstream_sources_are_byte_identical():
    for rel, expected in bb.UPSTREAM["files"].items():
        assert hashlib.sha256((bb.VENDOR / rel).read_bytes()).hexdigest() == expected, rel


def test_runtime_is_pinned_and_separate_from_standard_updater(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    binary = vendor / "runtime/bin/llama-server"; binary.parent.mkdir(parents=True)
    binary.write_bytes(b"pinned original engine")
    (vendor / "runtime/MANIFEST.json").write_text(json.dumps({"files": {
        "bin/llama-server": hashlib.sha256(binary.read_bytes()).hexdigest()}}))
    monkeypatch.setattr(bb, "VENDOR", vendor)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    manager = BackburnerManager()
    installed = manager.runtime_directory() / "bin/llama-server"
    assert installed.read_bytes() == binary.read_bytes()
    assert installed.stat().st_mode & 0o111
    standard = tmp_path / "studio/llama.cpp/bin/llama-server"
    standard.parent.mkdir(parents=True); standard.write_bytes(b"automatic update")
    assert installed.read_bytes() == b"pinned original engine"
    binary.write_bytes(b"unattested replacement")
    with pytest.raises(ValueError, match="integrity"):
        manager.runtime_directory()
    assert installed.read_bytes() == b"pinned original engine"


def test_incompatible_model_is_rejected(tmp_path):
    import gguf
    path = tmp_path / "other.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("Other model"); writer.add_block_count(32); writer.add_file_type(30)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with pytest.raises(ValueError, match="requires Qwen3.8"):
        BackburnerManager().validate_model(str(path))


def write_qwen(path, *, name="Qwen3.8-27B", file_type=30, nextn=0, changes=None, tensor_type="F32"):
    from core.companion.backburner_gguf import pinned_gguf
    gguf = pinned_gguf()
    import numpy as np
    from core.companion.backburner_models import PROFILE
    writer = gguf.GGUFWriter(str(path), "qwen35")
    values = {f"qwen35.{key}": value for key, value in PROFILE.items()}
    values.update({"qwen35.block_count": 64 + nextn, "qwen35.nextn_predict_layers": nextn})
    values.update(changes or {})
    writer.add_name(name)
    if file_type is not None:
        writer.add_file_type(file_type)
    for key, value in values.items():
        writer.add_uint32(key, value)
    kind = gguf.GGMLQuantizationType[tensor_type]
    block, size = gguf.GGML_QUANT_SIZES[kind]
    data = np.zeros(5120 // block * size, dtype=np.uint8)
    writer.add_tensor("blk.0.ffn_gate.weight", data, raw_dtype=kind)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    return path


def test_original_model_metadata_is_accepted(tmp_path):
    path = write_qwen(tmp_path / "original.gguf")
    BackburnerManager().validate_model(str(path))


@pytest.mark.parametrize("name", ["Qwen3.8-27B", "Huihui Qwen3.8 27B Abliterated", "Qwen3.8-27B uncensored", "Qwen-3.8-27B fine-tune"])
@pytest.mark.parametrize("nextn", [0, 1])
def test_derivatives_and_optional_mtp_are_accepted(tmp_path, name, nextn):
    path = write_qwen(tmp_path / "renamed.gguf", name=name, file_type=26, nextn=nextn, tensor_type="IQ3_XXS")
    BackburnerManager().validate_model(str(path))


from core.companion.backburner_models import METAL_WEIGHT_TYPES, PROFILE, tail_layer


@pytest.mark.parametrize("tensor_type", sorted(METAL_WEIGHT_TYPES))
@pytest.mark.parametrize("file_type", [None, 26])
def test_all_pinned_metal_weight_formats_ignore_filename_and_summary(tmp_path, tensor_type, file_type):
    path = write_qwen(tmp_path / "arbitrary-name.gguf", file_type=file_type, tensor_type=tensor_type)
    BackburnerManager().validate_model(str(path))


@pytest.mark.parametrize("tensor_type", ["NVFP4", "TQ1_0", "Q8_1", "Q8_K", "I8", "F64"])
def test_unsupported_weight_formats_fail_before_transfer(tmp_path, tensor_type):
    path = write_qwen(tmp_path / "bad.gguf", tensor_type=tensor_type)
    with pytest.raises(ValueError, match="cannot use these weight formats"):
        BackburnerManager().validate_model(str(path))


@pytest.mark.parametrize("key", list(PROFILE) + ["block_count", "nextn_predict_layers"])
def test_incompatible_geometry_is_rejected_even_with_qwen_name(tmp_path, key):
    path = write_qwen(tmp_path / "bad-geometry.gguf", changes={f"qwen35.{key}": 1})
    with pytest.raises(ValueError, match="geometry|trunk layers"):
        BackburnerManager().validate_model(str(path))


@pytest.mark.parametrize("device,per_layer,expected", [("iPhone18,2", 200*1024**2, 40),
    ("iPhone17,2", 200*1024**2, 52), ("iPhone18,2", 500*1024**2, 52), ("iPhone17,2", 500*1024**2, 60)])
def test_tail_budget_keeps_original_splits_or_shortens_heavy_quants(device, per_layer, expected):
    reader = SimpleNamespace(tensors=[SimpleNamespace(name=f"blk.{n}.ffn_up.weight", n_bytes=per_layer) for n in range(65)] +
        [SimpleNamespace(name="output.weight", n_bytes=100*1024**3)])
    assert tail_layer(reader, device) == expected


def test_impossible_tail_is_rejected():
    reader = SimpleNamespace(tensors=[SimpleNamespace(name="blk.63.ffn_up.weight", n_bytes=7*1024**3)])
    with pytest.raises(ValueError, match="smallest iPhone tail"):
        tail_layer(reader, "iPhone18,2")


def test_sharded_model_requires_merge_before_tail_preparation(tmp_path):
    path = write_qwen(tmp_path / "shard.gguf", changes={"split.count": 2})
    with pytest.raises(ValueError, match="Merge the GGUF shards"):
        BackburnerManager().validate_model(str(path))


def test_pinned_reader_does_not_replace_or_use_installed_gguf(tmp_path):
    import gguf
    import sys
    path = write_qwen(tmp_path / "new-format.gguf", tensor_type="Q2_0")
    reader = BackburnerManager()._model_reader(str(path))
    assert reader.tensors[0].tensor_type.name == "Q2_0"
    assert sys.modules["gguf"] is gguf
    assert type(reader).__module__ == "_unsloth_backburner_gguf.gguf_reader"


@pytest.mark.parametrize("name", ["token_embd.weight", "output.weight"])
def test_derivative_with_resized_draft_vocabulary_is_rejected(tmp_path, name):
    from core.companion.backburner_models import validate_profile
    path = write_qwen(tmp_path / "resized.gguf", name="Qwen3.8-27B Uncensored")
    reader = BackburnerManager()._model_reader(str(path))
    reader.tensors.append(SimpleNamespace(name=name, shape=[5120, 248321], tensor_type=SimpleNamespace(name="F16")))
    with pytest.raises(ValueError, match="changed the vocabulary"):
        validate_profile(reader)


@pytest.mark.parametrize("wrong_phone_sha", [None, "b" * 64])
def test_preflight_rejects_stale_tail_for_other_quant_or_finetune(tmp_path, monkeypatch, wrong_phone_sha):
    path = write_qwen(tmp_path / "abliterated-IQ3.gguf", nextn=1, tensor_type="IQ3_XXS")
    manager = BackburnerManager()
    manager.phone = {"memory": {"source_sha256": "a" * 64}}
    manager.root.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    (manager.root / "source.json").write_text(json.dumps({
        "file": [str(path.resolve()), stat.st_size, stat.st_mtime_ns], "sha256": "a" * 64}))
    draft = tmp_path / "dflash2-v2-q4km-self16.gguf"; draft.write_bytes(b"draft")
    manager.draft_path = str(draft)
    monkeypatch.setattr(manager, "status", lambda: {"ready": True, "prepared": True})
    monkeypatch.setattr(manager, "validate_draft", lambda *args: None)
    monkeypatch.setattr(bb, "_run", lambda args: str(24 * 1024**3).encode() if args[-1] == "hw.memsize" else b"20480")
    intent = SimpleNamespace(gguf_path=str(path), dflash_draft_path=None)
    assert manager.preflight(intent)["sourceSHA256"] == "a" * 64
    manager.phone["memory"]["source_sha256"] = wrong_phone_sha
    with pytest.raises(ValueError, match="different tail"):
        manager.preflight(intent)
    manager.phone["memory"]["source_sha256"] = "a" * 64
    path.touch()
    with pytest.raises(ValueError, match="Mac model changed"):
        manager.preflight(intent)


@pytest.mark.parametrize("path,variant,accepted", [("org/model", "IQ4_XS", True),
                                                  ("/model.gguf", None, True),
                                                  ("org/other", None, False),
                                                  ("org/model", "Q4_K_M", False)])
def test_speed_blocks_other_model_loads_before_teardown(path, variant, accepted):
    manager = BackburnerManager(); manager.mode = "speed"
    intent = SimpleNamespace(model_identifier="org/model", hf_repo="org/model", gguf_path="/model.gguf", hf_variant="IQ4_XS")
    if accepted:
        manager.require_selected_model(path, variant, intent)
    else:
        with pytest.raises(ValueError, match="Switch iPhone to Agent"):
            manager.require_selected_model(path, variant, intent)


@run_async
async def test_standard_runtime_exception_still_restores_agent(monkeypatch):
    manager = BackburnerManager(); companion = Companion(); backend = Backend(manager, fail=True)
    def load(intent):
        if manager.mode == "agent":
            raise RuntimeError("standard engine unavailable")
        return False
    backend.load_model = load
    monkeypatch.setattr(manager, "preflight", lambda intent: {})
    with pytest.raises(ValueError, match="failed to start"):
        await manager.select_mode("speed", backend, companion)
    assert manager.mode == "agent" and companion.settings.enabled
    assert "standard engine unavailable" in manager.error


@run_async
async def test_shutdown_terminates_owned_preparation_child(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    manager = BackburnerManager()
    manager._prepare_task = asyncio.create_task(asyncio.to_thread(
        manager._preparation_command, [sys.executable, "-c", "import time; time.sleep(60)"]))
    for _ in range(100):
        if manager._prepare_process is not None:
            break
        await asyncio.sleep(0.01)
    child = manager._prepare_process
    assert child is not None
    with pytest.raises(ValueError, match="cancelled at shutdown"):
        await manager.shutdown()
    assert child.poll() is not None and manager._prepare_process is None


@pytest.mark.parametrize("profile", [None, {"name": "memory-saving", "context": 8192,
    "totalContext": 50000, "kv": "q4_0", "bytesPerToken": 18432, "batch": 128,
    "ubatch": 64, "checkpoints": 1, "loadMode": "mmap", "draftMax": 3, "draftGpuLayers": 0}])
def test_original_launch_profile_and_isolation(tmp_path, monkeypatch, profile):
    from core.companion import backburner_runtime as runtime
    from core.inference.llama_cpp import GgufLoadIntent
    manager = BackburnerManager()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("LLAMA_SPLIT_TAIL", "192.168.1.2:50060")
    monkeypatch.setenv("SPEC_TYPE", "draft-mtp")
    phone = {"address": "169.254.1.2", "memory": {"sys_wired_mb": 4000, "avail_mb": 5000}}
    draft = tmp_path / "draft.gguf"; draft.write_bytes(b"draft")
    source_sha = "a" * 64
    monkeypatch.setattr(manager, "preflight", lambda intent: {"phone": phone, "draft": str(draft), "sourceSHA256": source_sha, **({"profile": profile} if profile else {})})
    monkeypatch.setattr(manager, "runtime_directory", lambda: tmp_path / "isolated-runtime")
    monkeypatch.setattr(runtime, "backburner_manager", manager)
    launches = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda command, **kwargs: launches.append((command, kwargs)) or SimpleNamespace(pid=999999))
    backend = SimpleNamespace(_lock=threading.Lock(), _cancel_event=threading.Event(), is_loaded=False,
                              _gguf_load_source_identity=lambda path: path, unload_model=lambda: None,
                              _find_free_port=lambda: 32150, _read_gguf_metadata=lambda path: None,
                              _record_server_pid=lambda pid: None, _drain_stdout=lambda: None,
                              _wait_for_health=lambda **kwargs: True, _binary_revision=lambda path: (path,))
    intent = GgufLoadIntent("model", gguf_path="/model.gguf", n_ctx=2048, n_parallel=8)
    assert runtime.load_backburner(backend, intent)
    _, launch = launches[0]; env = launch["env"]
    assert env["BIN"] == str(tmp_path / "isolated-runtime/bin")
    assert env["LLAMA_SPLIT_TAIL"] == "169.254.1.2:50060" and env["PHONE_KV"] == "169.254.1.2:50062"
    local = 8192 if profile else 65536
    kv = "q4_0" if profile else "q8_0"
    assert env["CTX"] == str(local) and env["KV"] == kv and env["PROXY"] == "1"
    if profile:
        assert env["LOAD_MODE"] == "mmap" and env["SPLIT_UB"] == "64"
        assert "-ngld 0" in env["SERVER_ARGS"] and "--spec-draft-n-max 3" in env["SERVER_ARGS"]
    assert "SPEC_TYPE" not in env and launch["start_new_session"] is True
    assert Path(env["CACHE_DIR"]).parent.name == backend._backburner_cache_identity
    assert Path(env["CACHE_DIR"]).name == kv
    expected = 50000 if profile else min(262144,65536+min((9400-4000)*1048576//34816//4096*4096,(5000-512)*1048576//34816//4096*4096))
    assert int(env["CTX_TOTAL"]) == expected == backend._context_length
    assert backend._effective_parallel_slots == 1 and backend._cache_type_kv == kv
    assert backend._last_load_intent.n_ctx == 2048 and backend._last_load_intent.n_parallel == 8
    backend._llama_log_fh.close()
    backend.is_loaded = True
    assert runtime.load_backburner(backend, intent) and len(launches) == 1
    source_sha = "b" * 64
    assert runtime.load_backburner(backend, intent) and len(launches) == 2
    assert launches[0][1]["env"]["CACHE_DIR"] != launches[1][1]["env"]["CACHE_DIR"]
    backend._llama_log_fh.close()
    draft.write_bytes(b"changed draft")
    assert runtime.load_backburner(backend, intent) and len(launches) == 3
    assert launches[1][1]["env"]["CACHE_DIR"] != launches[2][1]["env"]["CACHE_DIR"]
    backend._llama_log_fh.close()


@run_async
async def test_temporary_agent_suspension_survives_a_mac_restart(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from core.companion.manager import CompanionManager
    companion = CompanionManager()
    companion.settings_path = tmp_path / "settings.json"
    original = CompanionSettings(enabled=True)
    companion._write_model(companion.settings_path, original)
    saved = companion.settings_path.read_bytes()
    monkeypatch.setattr(companion, "_stop_listener", AsyncMock())
    companion.settings = original
    await companion.update_settings(original.model_copy(update={"enabled": False}), persist=False)
    assert not companion.settings.enabled and companion.settings_path.read_bytes() == saved
    restarted = companion._read_model(companion.settings_path, CompanionSettings, CompanionSettings())
    assert restarted.enabled is True
