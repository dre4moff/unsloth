# SPDX-License-Identifier: AGPL-3.0-only
"""Adapt the original serve.sh profile to Studio's existing inference transport."""
from __future__ import annotations

import os
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from core.companion.backburner import VENDOR, backburner_manager, phone_command


def load_backburner(backend, intent, cancel_event=None) -> bool:
    config = backburner_manager.preflight(intent)  # rejects BEFORE replacing a model
    identity = backend._gguf_load_source_identity(intent.gguf_path)
    draft = Path(config["draft"]).expanduser().resolve()
    draft_stat = draft.stat()
    profile = config.get("profile") or {"name": "original", "context": 65536, "kv": "q8_0",
        "bytesPerToken": 34816, "batch": 2048, "ubatch": 256, "checkpoints": 3, "loadMode": "none"}
    local, kv = profile["context"], profile["kv"]
    cache_identity = hashlib.sha256(json.dumps([
        config["sourceSHA256"], str(draft), draft_stat.st_size, draft_stat.st_mtime_ns,
        profile,
    ]).encode()).hexdigest()
    if (getattr(backend, "_backburner_active", False) and backend.is_loaded
        and backend._gguf_load_identity == identity
        and getattr(backend, "_backburner_cache_identity", None) == cache_identity):
        return True
    phone = config["phone"]
    root = backburner_manager.root
    runtime = backburner_manager.runtime_directory()
    root.mkdir(parents=True, exist_ok=True)
    # Scripts remain byte-identical upstream. Their writable cache and Python
    # interpreter shims live outside the installed wheel/app and standard engine.
    scripts = root / "scripts"
    if scripts.exists():
        shutil.rmtree(scripts)
    shutil.copytree(VENDOR / "scripts", scripts)
    shim = root / "bin"
    shim.mkdir(exist_ok=True)
    interpreter = shim / "python3"
    if interpreter.is_symlink():
        interpreter.unlink()
    interpreter.symlink_to(sys.executable)
    # Use upstream's phone reserve and bytes/token for the selected cache type.
    mem = phone["memory"]
    wired, available = int(mem.get("sys_wired_mb", 0)), int(mem.get("avail_mb", 0))
    if not 0 < wired < 9400 or available <= 512:
        raise ValueError("iPhone has insufficient available memory for the original Backburner profile.")
    bpt = profile["bytesPerToken"]
    share = min((9400-wired)*1048576//bpt//4096*4096,
                (available-512)*1048576//bpt//4096*4096)
    cap = min(262144, local + share)
    total = profile.get("totalContext", cap)
    if total > cap or (total > local and share <= 0):
        raise ValueError("iPhone has insufficient memory for remote KV pages.")
    backend.unload_model()
    with backend._lock:
        backend._cancel_event.clear()
        if cancel_event is not None and cancel_event.is_set():
            return False
        for attempt in range(32):
            candidate = backend._find_free_port()
            if candidate > 65435:
                continue
            try:
                with socket.socket() as upstream_port:
                    upstream_port.bind(("127.0.0.1", candidate + 100))
                backend._port = candidate
                break
            except OSError:
                continue
        else:
            raise ValueError("No free local port pair for the original Backburner proxy.")
        env = os.environ.copy()
        # An external environment must not silently rewrite the original profile.
        for key in list(env):
            if key.startswith(("LLAMA_", "GGML_", "SPEC_")) or key in {
                "SERVER_ARGS", "PHONE_DRAFT", "CTX_TOTAL", "KV", "SME", "MM_SME", "SPLIT_UB",
                "LOAD_MODE", "CACHE_RAM", "CTX_CHECKPOINTS", "PROXY", "PORT", "BIN", "DRAFT", "MODEL",
            }:
                env.pop(key)
        env.update(BIN=str(runtime / "bin"), MODEL=intent.gguf_path,
                   DRAFT=config["draft"], CTX=str(local), CTX_TOTAL=str(total), KV=kv,
                   PORT=str(backend._port), PHONE="0", PHONE_IP=phone["address"],
                   LLAMA_SPLIT_TAIL=f'{phone["address"]}:50060',
                   PHONE_KV=f'{phone["address"]}:50062',
                   CACHE_DIR=str(root / "cache" / cache_identity / kv), PROXY="1",
                   PATH=str(shim)+os.pathsep+env.get("PATH", "/usr/bin:/bin"))
        if profile["name"] == "memory-saving":
            env.update(LOAD_MODE="mmap", CACHE_RAM="0", CTX_CHECKPOINTS="1", SPLIT_UB="64",
                       SERVER_ARGS=f'-b 128 -ub 64 --spec-draft-n-max 3 -ngld {profile["draftGpuLayers"]}')
        backend._read_gguf_metadata(intent.gguf_path)
        backend._api_key = None
        backend._stdout_lines = []
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        backend._llama_log_path = logs / "server.log"
        backend._llama_log_fh = backend._llama_log_path.open("w", buffering=1)
        backend._process = subprocess.Popen(
            ["/bin/bash", str(scripts / "serve.sh")], cwd=root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", start_new_session=True,
        )
        backend._record_server_pid(backend._process.pid)
        backend._stdout_thread = threading.Thread(target=backend._drain_stdout, daemon=True)
        backend._stdout_thread.start()
    if not backend._wait_for_health(timeout=600):
        backend.unload_model()
        return False
    with backend._lock:
        backend._model_identifier = intent.model_identifier or intent.gguf_path
        backend._gguf_path = intent.gguf_path
        backend._hf_repo = intent.hf_repo
        backend._hf_variant = intent.hf_variant
        backend._last_load_intent = replace(intent, verified_gguf=None)
        backend._healthy = True
        backend._context_length = total
        backend._effective_context_length = total
        backend._max_context_length = total
        backend._kv_cache_context_total = total
        backend._effective_parallel_slots = 1
        backend._requested_n_parallel = 1
        backend._n_batch = profile["batch"]
        backend._n_ubatch = profile["ubatch"]
        backend._cache_type_kv = kv
        backend._effective_cache_types = (kv, kv)
        backend._gpu_offload_active = True
        backend._is_vision = False
        backend._is_audio = False
        backend._tensor_parallel = False
        backend._speculative_type = "draft-dflash"
        backend._spec_drafter_kind = "dflash"
        backend._requested_n_ctx = local
        backend._launch_binary_revision = backend._binary_revision(str(runtime / "bin/llama-server"))
        backend._gguf_load_identity = identity
        backend._backburner_active = True
        backend._backburner_draft = config["draft"]
        backend._backburner_cache_identity = cache_identity
    return True
