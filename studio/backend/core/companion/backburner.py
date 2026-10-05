# SPDX-License-Identifier: AGPL-3.0-only
"""Wired Backburner integration. The engine and protocols belong to StayLameBro.

Never uses Unsloth's updateable llama.cpp directory. Unknown USB speed, a Wi-Fi
route, and a protocol mismatch all fail closed. Mode selection is process-wide.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import platform
import plistlib
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "backburner"
UPSTREAM = json.loads((VENDOR / "UPSTREAM.json").read_text())
PATN = 0x4E544150
PATN_VERSION = 4


def _run(args: list[str], timeout: float = 8) -> bytes:
    return subprocess.check_output(args, timeout=timeout, stderr=subprocess.DEVNULL)


def _walk(value):
    if isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, dict):
        yield value
        yield from _walk(value.get("IORegistryEntryChildren", []))


def usb_phones(registry: Any) -> list[dict]:
    """USBSpeed is the negotiated tIOUSBHostConnectionSpeed, never bcdUSB.

    Apple's IOUSBHostFamilyDefinitions.h: 5=10 Gb/s, 6=20 Gb/s.
    The BSD interface must be a descendant of this exact USB device.
    """
    phones = []
    for node in _walk(registry):
        name = str(node.get("USB Product Name", node.get("kUSBProductString", "")))
        serial = node.get("USB Serial Number", node.get("kUSBSerialNumberString", ""))
        if node.get("idVendor") != 0x05AC or "iphone" not in name.lower():
            continue
        speed = {5: 10, 6: 20}.get(node.get("USBSpeed"))
        if not speed or not isinstance(serial, str) or not re.fullmatch(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{16}", serial):
            continue
        interfaces = sorted({str(n["BSD Name"]) for n in _walk(node) if "BSD Name" in n})
        for interface in interfaces:
            if re.fullmatch(r"en\d+", interface):
                phones.append({"deviceID": serial, "name": name, "speedGbps": speed, "interface": interface})
    return phones


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        part = sock.recv(count - len(result))
        if not part:
            raise ConnectionError("iPhone closed the Backburner connection")
        result.extend(part)
    return bytes(result)


def phone_hello(address: str, source: str) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        sock.bind((source, 0))
        sock.connect((address, 50062))
        sock.sendall(struct.pack("<IIQ", PATN, 1, 0))
        magic, kind, length = struct.unpack("<IIQ", _recv_exact(sock, 16))
        if (magic, kind, length) != (PATN, 2, 72):
            return False
        body = _recv_exact(sock, length)
        sock.sendall(struct.pack("<IIQ", PATN, 11, 0))
        return struct.unpack_from("<I", body)[0] == PATN_VERSION


def phone_command(phone: dict, command: str, timeout: float = 4) -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.bind((phone["sourceAddress"], 0))
        sock.connect((phone["address"], 50061))
        sock.sendall((command + "\n").encode())
        line = sock.makefile("rb").readline(65537)
        if len(line) > 65536:
            raise ValueError("Backburner reply exceeds its limit")
        return json.loads(line)


def detect_wired_phone(existing: dict | None = None) -> dict | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    if int(platform.mac_ver()[0].split(".")[0] or 0) < 14:
        return None
    # Include properties on every descendant: without -l ioreg prints BSD Name
    # only on matched nodes, hiding the NCM interface below the USB device.
    registry = plistlib.loads(_run(["/usr/sbin/ioreg", "-a", "-l", "-r", "-c", "IOUSBHostDevice"]))
    compatible = None
    for phone in usb_phones(registry):
        interface = phone["interface"]
        text = _run(["/sbin/ifconfig", interface]).decode()
        match = re.search(r"\binet (169\.254\.\d+\.\d+)\b", text)
        if not match:
            continue
        source = match[1]
        phone.update(sourceAddress=source, address=None, ready=False)
        compatible = compatible or phone
        if existing and existing["deviceID"] == phone["deviceID"] and existing.get("address"):
            # phone-attn has a single persistent client while generating. Its
            # separate command service remains available for status updates.
            address = existing["address"]
            route = _run(["/sbin/route", "-n", "get", address]).decode()
            if re.search(rf"interface:\s*{re.escape(interface)}\s", route):
                phone.update(address=address, ready=True)
                phone["memory"] = phone_command(phone, "mem")
                return phone
        # Only this USB interface is probed. Wi-Fi Bonjour and CoreDevice's
        # wireless connections are deliberately not used as evidence of a cable.
        try:
            pong = _run(["/sbin/ping", "-b", interface, "-c", "2", "-t", "2", "169.254.255.255"], 4).decode()
        except subprocess.CalledProcessError as exc:
            pong = (exc.output or b"").decode()
        for address in sorted(set(re.findall(r"bytes from (169\.254\.\d+\.\d+)", pong))):
            if address == source or not ipaddress.ip_address(address).is_link_local:
                continue
            route = _run(["/sbin/route", "-n", "get", address]).decode()
            if not re.search(rf"interface:\s*{re.escape(interface)}\s", route):
                continue
            try:
                if phone_hello(address, source):
                    phone.update(address=address, ready=True)
                    phone["memory"] = phone_command(phone, "mem")
                    return phone
            except (OSError, ValueError, ConnectionError):
                continue
    return compatible  # compatible cable present; Companion's speed page still closed


class BackburnerManager:
    def __init__(self):
        self.mode = "agent"
        self.phone: dict | None = None
        self.error: str | None = None
        self.preparing = False
        self.progress = ""
        self.draft_path: str | None = None
        self._checked = 0.0
        self._probe_lock = threading.Lock()
        self._mode_lock = asyncio.Lock()
        self._agent_settings = None
        self._prepare_task: asyncio.Task | None = None
        self._prepare_cancel = threading.Event()
        self._prepare_process = None
        self._process_lock = threading.Lock()

    @property
    def root(self) -> Path:
        from utils.paths.storage_roots import studio_root
        return studio_root() / "backburner" / UPSTREAM["engine_commit"]

    def _saved_preparation_file(self, name: str) -> Path:
        """Keep compatible tail identity and draft selection across engine updates.

        Runtime binaries and KV caches always use the new engine's own root.
        The existing preflight still checks the target file and phone content SHA.
        """
        current = self.root / name
        previous = UPSTREAM.get("compatible_preparation_engine")
        if name in {"source.json", "configuration.json"} and not current.exists() and previous:
            old = self.root.parent / previous / name
            if old.is_file():
                return old
        return current

    def status(self, refresh: bool = True) -> dict:
        if self.draft_path is None:
            try:
                self.draft_path = json.loads(self._saved_preparation_file("configuration.json").read_text()).get("draftPath")
            except (OSError, ValueError):
                pass
        if refresh and time.monotonic() - self._checked > 3:
            with self._probe_lock:
                if time.monotonic() - self._checked > 3:
                    try:
                        self.phone = detect_wired_phone(self.phone if self.mode == "speed" else None)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        self.phone = None
                    self._checked = time.monotonic()
        mem = (self.phone or {}).get("memory") or {}
        if self.mode == "speed" and self.phone is None:
            self.error = "USB connection lost. Switch to Agent to reload the Mac runtime."
        elif self.error and self.error.startswith("USB connection lost"):
            self.error = None
        return {"mode": self.mode, "available": self.phone is not None,
                "ready": bool((self.phone or {}).get("ready")), "phone": self.phone,
                "prepared": mem.get("tail_state") in {"ready", "connected"} and "pa_ane" in mem
                            and mem.get("engine_commit") == UPSTREAM["engine_commit"],
                "preparing": self.preparing, "progress": self.progress, "error": self.error,
                "draftPath": self.draft_path, "author": UPSTREAM["author"],
                "sourceURL": UPSTREAM["repository"], "engineCommit": UPSTREAM["engine_commit"]}

    def _model_reader(self, path: str):
        from core.companion.backburner_gguf import pinned_gguf
        return pinned_gguf().GGUFReader(str(Path(path).expanduser()))

    def validate_model(self, path: str) -> None:
        from core.companion.backburner_models import validate_profile
        validate_profile(self._model_reader(path))

    def validate_draft(self, draft_path: str, model_path: str) -> None:
        from core.companion.backburner_drafts import validate_draft
        validate_draft(draft_path, self._model_reader(model_path))

    def runtime_directory(self) -> Path:
        """Materialize the attested runtime in its own versioned install root.

        The normal llama.cpp updater never selects or writes this directory.
        Source-tree edits cannot silently replace a released engine either.
        """
        manifest = json.loads((VENDOR / "runtime/MANIFEST.json").read_text())
        destination = self.root / "runtime"
        for rel, expected in manifest["files"].items():
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("Invalid Backburner runtime manifest")
            source, target = VENDOR / "runtime" / rel, destination / rel
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Backburner runtime integrity check failed: {rel}")
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".installing")
            import shutil
            shutil.copyfile(source, temporary)
            temporary.chmod(0o755 if rel.startswith("bin/") else 0o644)
            temporary.replace(target)
        return destination

    def preflight(self, intent) -> dict:
        state = self.status()
        if not state["ready"] or not state["prepared"]:
            raise ValueError("Open Increase speed on the wired iPhone and prepare its model first.")
        if not (VENDOR / "runtime" / "bin" / "llama-server").is_file():
            raise ValueError("This build is missing the pinned Backburner engine.")
        self.validate_model(intent.gguf_path)
        stat = Path(intent.gguf_path).stat()
        marker = self._saved_preparation_file("source.json")
        if not marker.is_file():
            raise ValueError("Prepare this model's iPhone tail before increasing speed.")
        source = json.loads(marker.read_text())
        if source.get("file") != [str(Path(intent.gguf_path).resolve()), stat.st_size, stat.st_mtime_ns]:
            raise ValueError("The Mac model changed. Prepare its iPhone tail again.")
        if not re.fullmatch(r"[a-f0-9]{64}", str(source.get("sha256", ""))) or self.phone["memory"].get("source_sha256") != source.get("sha256"):
            raise ValueError("iPhone loaded a different tail. Finish preparation and restart its Speed page.")
        draft = self.draft_path or intent.dflash_draft_path or str(Path.home() / "Models/dflash2-v2-q4km-self16.gguf")
        from core.companion.backburner_profile import launch_profile, metal_budget
        self.validate_draft(draft, intent.gguf_path)
        wired = int(_run(["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"]).strip())
        physical = int(_run(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip())
        profile = launch_profile(intent.gguf_path, draft, getattr(intent, "n_ctx", 65536), physical, wired,
                                 metal_budget(physical, wired) if physical < 24 * 1024**3 else None)
        return {"phone": self.phone.copy(), "draft": draft, "sourceSHA256": source["sha256"], "profile": profile}

    def require_selected_model(self, model_path: str, variant: str | None, intent) -> None:
        if self.mode != "speed":
            return
        names = {intent.model_identifier, intent.gguf_path, intent.hf_repo} if intent else set()
        if model_path not in names or (variant and variant != intent.hf_variant):
            raise ValueError("Switch iPhone to Agent before loading another model or GGUF variant.")

    async def select_mode(self, mode: str, backend, companion) -> dict:
        from core.inference.llama_keepwarm import inference_lifecycle_gate
        from state import active_generations
        async with self._mode_lock, inference_lifecycle_gate():
            from core.inference.llama_keepwarm import other_inference_request_count
            if active_generations.count() or other_inference_request_count(current_request_counted=False, include_pending=False) or companion.has_pending_work() or self.preparing:
                raise ValueError("Finish running Mac/iPhone tasks before switching modes.")
            if mode == self.mode:
                return await asyncio.to_thread(self.status)
            intent = backend.last_load_intent
            if mode == "speed":
                if intent is None or not backend.is_loaded:
                    raise ValueError("Load a compatible Qwen3.8-27B GGUF on the Mac before increasing speed.")
                await asyncio.to_thread(self.preflight, intent)
                if active_generations.count() or companion.has_pending_work():
                    raise ValueError("A task started while checking the iPhone. Finish it before switching.")
                self._agent_settings = companion.settings.model_copy(deep=True)
                self.mode = "speed"
                try:
                    await companion.update_settings(companion.settings.model_copy(update={"enabled": False}), persist=False)
                    if not await asyncio.to_thread(backend.load_model, intent):
                        raise ValueError("Backburner failed to start. See the engine log.")
                except Exception:
                    self.mode = "agent"
                    try:
                        await asyncio.to_thread(backend.unload_model)
                        if not await asyncio.to_thread(backend.load_model, intent):
                            self.error = "Backburner failed and the standard runtime could not restore the model."
                    except Exception as restore_error:
                        self.error = f"Standard runtime restoration failed: {restore_error}"
                    finally:
                        await companion.update_settings(self._agent_settings, persist=False)
                        self._agent_settings = None
                    raise
            else:
                self.mode = "agent"
                try:
                    if intent is not None:
                        await asyncio.to_thread(backend.unload_model)
                        if not await asyncio.to_thread(backend.load_model, intent):
                            self.error = "The standard Mac runtime could not reload the model."
                finally:
                    if self._agent_settings is not None:
                        await companion.update_settings(self._agent_settings, persist=False)
                        self._agent_settings = None
            return await asyncio.to_thread(self.status)

    async def prepare(self, model_path: str, draft_path: str, companion) -> dict:
        from state import active_generations
        async with self._mode_lock:
            if self.mode == "speed" or self.preparing or active_generations.count() or companion.has_pending_work():
                raise ValueError("Finish tasks and switch to Agent before preparing the iPhone.")
            state = await asyncio.to_thread(self.status)
            if not state["ready"]:
                raise ValueError("Open Increase speed on an iPhone connected by a 10 Gb/s USB cable.")
            await asyncio.to_thread(self.validate_model, model_path)
            await asyncio.to_thread(self.validate_draft, draft_path, model_path)
            template = VENDOR / "runtime/anekv/tmpl16k.mlmodelc"
            if not template.is_dir():
                raise ValueError("This build is missing the original Neural Engine template.")
            self.draft_path = str(Path(draft_path).expanduser().resolve())
            self.root.mkdir(parents=True, exist_ok=True)
            config = self.root / "configuration.json.installing"
            config.write_text(json.dumps({"draftPath": self.draft_path}))
            config.replace(self.root / "configuration.json")
            self._agent_settings = companion.settings.model_copy(deep=True)
            if companion.has_pending_work():
                raise ValueError("An iPhone task started during preparation checks. Finish it first.")
            self.preparing = True
            try:
                await companion.update_settings(companion.settings.model_copy(update={"enabled": False}), persist=False)
            except Exception:
                self.preparing = False
                self._agent_settings = None
                raise
            self._prepare_cancel.clear()
            self.error = None
            phone = self.phone.copy()
            async def transfer():
                try:
                    await asyncio.to_thread(self._prepare_files, str(Path(model_path).expanduser().resolve()), template, phone)
                    self.progress = "Files copied. On iPhone turn Increase speed off and on to load its model and Neural Engine template."
                except Exception as exc:
                    self.error = str(exc)
                finally:
                    self.preparing = False
                    await companion.update_settings(self._agent_settings, persist=False)
                    self._agent_settings = None
                    self._checked = 0
            self._prepare_task = asyncio.create_task(transfer())
            return self.status(refresh=False)

    async def shutdown(self) -> None:
        self._prepare_cancel.set()
        with self._process_lock:
            process = self._prepare_process
            if process is not None and process.poll() is None:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
        if self._prepare_task is not None:
            await self._prepare_task
            self._prepare_task = None

    def _check_preparation_cancelled(self) -> None:
        if self._prepare_cancel.is_set():
            raise ValueError("Backburner preparation cancelled at shutdown.")

    def _preparation_command(self, args: list[str]) -> None:
        import signal
        with self._process_lock:
            self._check_preparation_cancelled()
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=True)
            from utils.process_lifetime import adopt_pid
            adopt_pid(process.pid)
            self._prepare_process = process
        try:
            while True:
                try:
                    _, error = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if self._prepare_cancel.is_set():
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                        self._check_preparation_cancelled()
            self._check_preparation_cancelled()
            if process.returncode:
                raise ValueError(error[-2000:] or "Backburner preparation failed")
        finally:
            from utils.process_lifetime import forget_pid
            forget_pid(process.pid)
            with self._process_lock:
                self._prepare_process = None

    def _prepare_files(self, model: str, template: Path, phone: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        mem = phone_command(phone, "mem")
        from core.companion.backburner_models import tail_layer, validate_profile
        device = str(mem.get("device_model", ""))
        reader = self._model_reader(model)
        validate_profile(reader)
        layer = tail_layer(reader, device)
        del reader
        signature = [model, Path(model).stat().st_size, Path(model).stat().st_mtime_ns, layer]
        identity = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
        tail = self.root / f"tail-{identity}-L{layer}-nohead.gguf"
        manifest = tail.with_suffix(".json")
        if not tail.is_file() or not manifest.is_file() or json.loads(manifest.read_text()) != signature:
            self.progress = f"Preparing the original iPhone tail (L{layer})…"
            partial = tail.with_suffix(".partial.gguf")
            self._preparation_command([sys.executable, str(VENDOR / "scripts/split-gguf.py"), model,
                                       str(partial), "-L", str(layer), "--no-head"])
            partial.replace(tail)
            manifest.write_text(json.dumps(signature))
        if signature[1:3] != [Path(model).stat().st_size, Path(model).stat().st_mtime_ns]:
            raise ValueError("Model changed while preparing its tail; no files were copied.")
        self.progress = "Verifying the Mac model and iPhone tail…"
        digest = hashlib.sha256()
        with open(model, "rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                self._check_preparation_cancelled()
                digest.update(block)
        if signature[1:3] != [Path(model).stat().st_size, Path(model).stat().st_mtime_ns]:
            raise ValueError("Model changed during verification; prepare its tail again.")
        source = self.root / "source.json"
        source.write_text(json.dumps({"file": signature[:3], "sha256": digest.hexdigest()}))
        for local, remote in [(tail, "tail.gguf"), (template, "anekv/tmpl16k.mlmodelc"), (source, "source.json")]:
            self.progress = f"Copying {remote} over USB…"
            self._preparation_command([sys.executable, str(VENDOR / "scripts/phone-push.py"),
                                       phone["address"], str(local), remote])


backburner_manager = BackburnerManager()
