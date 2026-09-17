# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lifecycle bridge for the bundled ``kaggle-tpu-lab`` launcher.

This module deliberately does not provision a TPU, install vLLM, download a
model, or implement a tunnel.  It starts the vendored upstream ``launch.py``
CLI, consumes its human-readable progress, and persists only the resulting
OpenAI-compatible endpoint in Unsloth. The Kaggle API token and generated
endpoint API key stay in the installation credential store and are never
returned by the status API.
"""

from __future__ import annotations

import asyncio
import json
import time
import signal

import httpx
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from storage import credential_secrets, providers_db
from utils.paths import ensure_dir, studio_root

logger = structlog.get_logger(__name__)

KAGGLE_TPU_STATES = frozenset(
    {
        "STOPPED",
        "STARTING",
        "PROVISIONING",
        "LOADING",
        "READY",
        "DISCONNECTED",
        "ERROR",
        "STOPPING",
    }
)

_ENDPOINT_RE = re.compile(r"https://[^\s]+\.trycloudflare\.com(?:/v1)?", re.I)
_API_KEY_RE = re.compile(r"API\s+key\s*:\s*(sk-[A-Za-z0-9._-]+)", re.I)
_MODEL_RE = re.compile(r"model\s*:\s*([^\s]+).*?context:\s*(\d+)", re.I)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bundled_launcher_root() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "kaggle_tpu_lab"


def _launcher_path(managed_config: dict[str, Any]) -> Path:
    configured = str(managed_config.get("lab_path") or "").strip()
    configured = configured or os.environ.get("UNSLOTH_KAGGLE_TPU_LAB", "").strip()
    root = Path(configured).expanduser().resolve() if configured else _bundled_launcher_root()
    launcher = root / "launch.py"
    if not root.is_dir() or not launcher.is_file():
        if configured:
            raise ValueError(f"No launch.py was found in {root}.")
        raise ValueError("The bundled Kaggle TPU launcher is missing from this Studio build.")
    return launcher


def _launcher_state_file(provider_id: str) -> Path:
    return studio_root() / "kaggle-tpu" / f"{provider_id}.json"


def _kaggle_api_token(provider_id: str) -> str:
    token = credential_secrets.get_kaggle_api_token(provider_id) or os.environ.get(
        "KAGGLE_API_TOKEN", ""
    ).strip()
    if not token:
        raise ValueError("Add your Kaggle API token to this connection before starting the TPU.")
    return token


def _launcher_environment(provider_id: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["KAGGLE_API_TOKEN"] = _kaggle_api_token(provider_id)
    state_file = _launcher_state_file(provider_id)
    ensure_dir(state_file.parent)
    env["UNSLOTH_KAGGLE_TPU_STATE_FILE"] = str(state_file)
    return env


def _launcher_command(managed_config: dict[str, Any], action: str) -> list[str]:
    launcher = _launcher_path(managed_config)
    command = [sys.executable, "-u", str(launcher), action]
    if launcher.parent == _bundled_launcher_root():
        command.append("--json")
    if action != "serve":
        return command
    command.extend(
        [
            "--max-model-len",
            str(int(managed_config.get("context_length") or 262144)),
            "--max-num-seqs",
            str(int(managed_config.get("max_num_seqs") or 4)),
            "--mtp",
            str(
                int(
                    managed_config.get("mtp_tokens")
                    if managed_config.get("mtp_tokens") is not None
                    else 3
                )
            ),
            "--reasoning-effort",
            str(managed_config.get("reasoning_effort_default") or "xhigh"),
            "--keepalive-min",
            str(int(managed_config.get("keepalive_minutes") or 480)),
        ]
    )
    if managed_config.get("text_only"):
        command.append("--text-only")
    if managed_config.get("fast_start"):
        command.append("--fast-start")
    return command


def _parse_launcher_line(line: str) -> dict[str, Any]:
    """Parse one launcher line without ever returning it verbatim to a log."""
    update: dict[str, Any] = {}
    try:
        event = json.loads(line)
    except (ValueError, TypeError):
        event = None
    if isinstance(event, dict) and event.get("status") in KAGGLE_TPU_STATES | {"MONITOR_RETRY"}:
        if event["status"] != "MONITOR_RETRY":
            update["state"] = event["status"]
        message = str(event.get("message") or "")
        message = re.sub(r"(?:KGAT_|sk-)[A-Za-z0-9._-]+|https?://\S+", "[redacted]", message)
        update["message"] = message[:600]
        update["session_ended"] = event.get("session_ended") is True
        url = event.get("base_url")
        if isinstance(url, str) and re.fullmatch(r"https://[A-Za-z0-9-]+\.trycloudflare\.com/v1", url):
            update["base_url"] = url
        for key in ("api_key", "model"):
            if isinstance(event.get(key), str) and event[key]:
                update[key] = event[key]
        length = event.get("context_length")
        if isinstance(length, int) and 1024 <= length <= 262144:
            update["context_length"] = length
        return update
    lower = line.lower()
    endpoint = _ENDPOINT_RE.search(line)
    if endpoint:
        url = endpoint.group(0).rstrip(".,)")
        update["base_url"] = url if url.endswith("/v1") else f"{url}/v1"
    key = _API_KEY_RE.search(line)
    if key:
        update["api_key"] = key.group(1)
    model = _MODEL_RE.search(line)
    if model:
        update["model"] = model.group(1)
        update["context_length"] = int(model.group(2))

    if "your endpoint is live" in lower:
        update.update(state="READY", message="Kaggle TPU endpoint is ready.")
    elif "failed at step" in lower or "push failed" in lower:
        update.update(state="ERROR", message="The Kaggle TPU launcher reported an error.")
    elif "queued" in lower or "provisioning" in lower or "pushing kernel" in lower:
        update.update(state="PROVISIONING", message="Kaggle is provisioning a TPU.")
    elif any(
        marker in lower
        for marker in (
            "building the python runtime",
            "runtime ready",
            "weights found",
            "starting vllm",
            "loading / compiling",
            "server is healthy",
            "endpoint url reserved",
        )
    ):
        update.update(state="LOADING", message="The model server is loading.")
    elif "auto-shutdown" in lower or "kernel finished with status complete" in lower or "cancelacknowledged" in lower:
        update.update(state="STOPPED", message="The Kaggle TPU session ended.")
    elif "kernel finished with status error" in lower:
        update.update(state="ERROR", message="The Kaggle kernel stopped with an error.")
    return update


@dataclass
class _Runtime:
    provider_id: str
    state: str = "STOPPED"
    message: str = ""
    base_url: Optional[str] = None
    model: Optional[str] = None
    context_length: Optional[int] = None
    updated_at: str = field(default_factory=_utc_now)
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    last_health_check: float = field(default=0.0, repr=False)
    session_ended: bool = field(default=False, repr=False)
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "state": self.state,
            "message": self.message,
            "base_url": self.base_url,
            "model": self.model,
            "context_length": self.context_length,
            "updated_at": self.updated_at,
        }


class KaggleTPULifecycleManager:
    def __init__(self) -> None:
        self._runtime: dict[str, _Runtime] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, provider_id: str) -> asyncio.Lock:
        return self._locks.setdefault(provider_id, asyncio.Lock())

    def _row(self, provider_id: str) -> dict[str, Any]:
        row = providers_db.get_provider(provider_id)
        if row is None:
            raise ValueError("Provider connection not found.")
        if row.get("provider_type") != "kaggle_tpu":
            raise ValueError("This connection is not a Kaggle TPU provider.")
        return row

    def _get_runtime(self, provider_id: str, row: Optional[dict] = None) -> _Runtime:
        runtime = self._runtime.get(provider_id)
        if runtime is not None:
            return runtime
        row = row or self._row(provider_id)
        has_endpoint = bool(row.get("base_url"))
        runtime = _Runtime(
            provider_id=provider_id,
            state="DISCONNECTED" if has_endpoint else "STOPPED",
            message=(
                "A saved endpoint exists; reconnect to verify it."
                if has_endpoint
                else "The Kaggle TPU session is stopped."
            ),
            base_url=row.get("base_url") or None,
            model=(row.get("models") or [None])[0],
            context_length=(row.get("capabilities") or {}).get("context_length"),
        )
        self._runtime[provider_id] = runtime
        return runtime

    def _set(self, runtime: _Runtime, **changes: Any) -> None:
        for key, value in changes.items():
            if key != "api_key" and hasattr(runtime, key):
                setattr(runtime, key, value)
        runtime.updated_at = _utc_now()

    @staticmethod
    def _protect_launcher_state_file(provider_id: str) -> None:
        state = _launcher_state_file(provider_id)
        try:
            if state.is_file():
                state.chmod(0o600)
        except OSError:
            logger.warning("kaggle_tpu.state_file_permissions_failed")

    @staticmethod
    def _persist_ready(
        provider_id: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        context_length: int,
        text_only: bool,
    ) -> None:
        row = providers_db.get_provider(provider_id)
        if row is None or row.get("provider_type") != "kaggle_tpu":
            raise ValueError("The Kaggle TPU connection was removed while starting.")
        capabilities = dict(row.get("capabilities") or {})
        capabilities.update(
            {
                "supports_streaming": True,
                "supports_tool_calling": True,
                "supports_reasoning": True,
                "supports_vision": not text_only,
                "supports_images": not text_only,
                "context_length": context_length,
                "api_mode": "chat_completions",
            }
        )
        credential_secrets.get_or_create_credential_encryption_key()
        credential_secrets.save_provider_api_key(provider_id, api_key)
        providers_db.update_provider(
            id=provider_id,
            base_url=base_url,
            models=[model],
            available_models=[model],
            capabilities=capabilities,
        )
        KaggleTPULifecycleManager._protect_launcher_state_file(provider_id)

    async def _persist_ready_guarded(self, provider_id: str, **ready: Any) -> None:
        # Share the same lock as provider edit/delete so an endpoint cannot
        # publish its credential into a connection that was concurrently removed.
        from routes.provider_credentials import provider_config_guard

        async with provider_config_guard(provider_id):
            await asyncio.to_thread(self._persist_ready, provider_id, **ready)

    @staticmethod
    async def _terminate(process) -> None:
        if process is None or process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
        except ProcessLookupError:
            pass
        except asyncio.TimeoutError:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            await process.wait()

    async def _spawn(self, provider_id, managed, action, *, follow=False):
        command = await asyncio.to_thread(_launcher_command, managed, action)
        if follow:
            command.append("--follow")
        if action == "serve":
            command.extend(["--slug", "unsloth-tpu-" + provider_id])
        env = await asyncio.to_thread(_launcher_environment, provider_id)
        return await asyncio.create_subprocess_exec(
            *command, cwd=str(Path(command[2]).parent),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=env, start_new_session=(os.name == "posix"), limit=1024 * 1024,
        )

    async def _healthy(self, runtime, api_key=None):
        if not runtime.base_url:
            return False
        key = api_key or await asyncio.to_thread(
            credential_secrets.get_provider_api_key, runtime.provider_id
        )
        if not key:
            return False
        # Do not follow redirects with a saved secret. Only the validated tunnel
        # discovered by this connection's launcher receives its inference key.
        if not re.fullmatch(r"https://[A-Za-z0-9-]+\.trycloudflare\.com/v1", runtime.base_url):
            return False
        runtime.last_health_check = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=False) as client:
                response = await client.get(runtime.base_url + "/models", headers={"Authorization": "Bearer " + key})
                response.raise_for_status()
                return any(m.get("id") == runtime.model for m in response.json().get("data", []))
        except Exception:
            return False

    async def _accept_ready(self, runtime, key, managed, *, attempts=1):
        if not (runtime.base_url and key and runtime.model and runtime.context_length):
            return False
        healthy = False
        for attempt in range(attempts):
            if await self._healthy(runtime, key):
                healthy = True
                break
            if attempt + 1 < attempts:
                self._set(runtime, state="LOADING", message="The model is loaded; waiting for its authenticated tunnel to become reachable.")
                await asyncio.sleep(5)
        if not healthy:
            self._set(runtime, state="DISCONNECTED", message="The model loaded but its tunnel is not reachable. Use Reconnect; the Kaggle session may still be running.")
            return False
        await self._persist_ready_guarded(
            runtime.provider_id, base_url=runtime.base_url, api_key=key,
            model=runtime.model, context_length=runtime.context_length,
            text_only=bool(managed.get("text_only")),
        )
        self._set(runtime, state="READY", message="Kaggle TPU connected. The model is ready for chat and Studio tools.")
        return True

    async def start(self, provider_id: str) -> dict[str, Any]:
        async with self._lock(provider_id):
            row = await asyncio.to_thread(self._row, provider_id)
            if row.get("is_enabled") is False:
                raise ValueError("Enable this Kaggle connection before starting it.")
            runtime = self._get_runtime(provider_id, row)
            if runtime.task is not None and not runtime.task.done():
                return runtime.public()
            managed = dict(row.get("managed_config") or {})
            managed["context_length"] = (row.get("capabilities") or {}).get("context_length")
            await asyncio.to_thread(_launcher_path, managed)
            await asyncio.to_thread(_kaggle_api_token, provider_id)
            if _launcher_state_file(provider_id).exists():
                await self._refresh_locked(runtime, managed)
                if runtime.state != "STOPPED" and not (runtime.state == "ERROR" and runtime.session_ended):
                    # Reattach instead of consuming another session after a Mac restart.
                    if runtime.state != "ERROR":
                        self._watch(runtime, managed, action="status")
                    return runtime.public()
            self._set(runtime, state="STARTING", message="Starting the Kaggle TPU session.",
                      base_url=None, session_ended=False, model=(row.get("models") or ["qwen3.8-27b"])[0],
                      context_length=managed["context_length"] or 262144)
            self._watch(runtime, managed)
            return runtime.public()

    def _watch(self, runtime, managed, *, action="serve"):
        runtime.task = asyncio.create_task(self._serve(runtime, managed, action=action),
                                           name=f"kaggle-tpu-{runtime.provider_id}")

    async def _serve(self, runtime, managed, *, action="serve"):
        api_key = None
        ready_seen = False
        try:
            runtime.process = await self._spawn(runtime.provider_id, managed, action, follow=action == "status")
            while True:
                raw = await runtime.process.stdout.readline()
                if not raw:
                    break
                update = _parse_launcher_line(raw.decode("utf-8", errors="replace"))
                api_key = update.pop("api_key", api_key)
                if update.get("state") == "READY":
                    ready_seen = True
                    update["state"] = "LOADING"
                if update:
                    self._set(runtime, **update)
                self._protect_launcher_state_file(runtime.provider_id)
                if ready_seen and api_key and runtime.base_url:
                    if await self._accept_ready(runtime, api_key, managed, attempts=6):
                        api_key = None
                        ready_seen = False
            return_code = await runtime.process.wait()
            if runtime.state not in ("STOPPED", "ERROR", "STOPPING", "READY", "DISCONNECTED"):
                self._set(runtime, state="DISCONNECTED" if return_code == 0 else "ERROR",
                          message="The Kaggle launcher detached. Use Reconnect to check the session.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set(runtime, state="ERROR", message="The Kaggle launcher failed. Check the connection and retry Start.")
            logger.warning("kaggle_tpu.launch_failed", provider_id=runtime.provider_id, error_type=type(exc).__name__)
        finally:
            api_key = None
            await self._terminate(runtime.process)
            runtime.process = None

    async def _refresh_locked(self, runtime, managed):
        process = None
        try:
            process = await self._spawn(runtime.provider_id, managed, "status")
            output, _ = await asyncio.wait_for(process.communicate(), timeout=75)
            api_key = None
            ready_seen = False
            self._set(runtime, state="DISCONNECTED", message="Checking the saved Kaggle session.")
            for raw in output.decode("utf-8", errors="replace").splitlines():
                update = _parse_launcher_line(raw)
                api_key = update.pop("api_key", api_key)
                if update.get("state") == "READY":
                    ready_seen = True
                    update["state"] = "LOADING"
                if update:
                    self._set(runtime, **update)
            if process.returncode and runtime.state != "ERROR":
                self._set(runtime, state="ERROR", message="Could not read Kaggle session status. Check the API token and retry.")
            elif ready_seen and runtime.state not in ("ERROR", "STOPPED"):
                await self._accept_ready(runtime, api_key, managed)
        except asyncio.TimeoutError:
            self._set(runtime, state="DISCONNECTED", message="Kaggle status timed out. Retry Reconnect; no new session was started.")
        finally:
            await self._terminate(process)
            self._protect_launcher_state_file(runtime.provider_id)
        return runtime.public()

    async def refresh(self, provider_id: str) -> dict[str, Any]:
        async with self._lock(provider_id):
            row = await asyncio.to_thread(self._row, provider_id)
            runtime = self._get_runtime(provider_id, row)
            managed = dict(row.get("managed_config") or {})
            # A running watcher is detached locally, without stopping the TPU.
            # Then status --follow restores progress and discovers a rotated URL.
            if runtime.task is not None and not runtime.task.done():
                runtime.task.cancel()
                await asyncio.gather(runtime.task, return_exceptions=True)
            await self._refresh_locked(runtime, managed)
            if runtime.state not in ("STOPPED", "ERROR"):
                self._watch(runtime, managed, action="status")
            return runtime.public()

    async def status(self, provider_id: str) -> dict[str, Any]:
        async with self._lock(provider_id):
            row = await asyncio.to_thread(self._row, provider_id)
            runtime = self._get_runtime(provider_id, row)
            if runtime.state == "READY" and time.monotonic() - runtime.last_health_check > 30:
                if not await self._healthy(runtime):
                    self._set(runtime, state="DISCONNECTED", message="The Kaggle tunnel is unavailable. Use Reconnect; the session may still be running.")
            return runtime.public()

    async def stop(self, provider_id: str) -> dict[str, Any]:
        async with self._lock(provider_id):
            row = await asyncio.to_thread(self._row, provider_id)
            runtime = self._get_runtime(provider_id, row)
            if runtime.task is not None and not runtime.task.done():
                runtime.task.cancel()
                await asyncio.gather(runtime.task, return_exceptions=True)
            self._set(runtime, state="STOPPING", message="Stopping the Kaggle TPU session.")
            process = None
            try:
                process = await self._spawn(provider_id, dict(row.get("managed_config") or {}), "stop")
                output, _ = await asyncio.wait_for(process.communicate(), timeout=55)
                state = "STOPPED" if process.returncode == 0 else "ERROR"
                message = "The Kaggle TPU session is stopped." if state == "STOPPED" else "Kaggle could not stop the session. Retry Stop or stop it on Kaggle."
                for line in output.decode("utf-8", errors="replace").splitlines():
                    update = _parse_launcher_line(line)
                    if update.get("state") == "ERROR":
                        state, message = "ERROR", update["message"]
                self._set(runtime, state=state, message=message)
            except asyncio.TimeoutError:
                self._set(runtime, state="ERROR", message="Stopping Kaggle timed out. Check the notebook before assuming its TPU has stopped.")
            finally:
                await self._terminate(process)
            return runtime.public()

    async def detach(self, provider_id: str) -> None:
        async with self._lock(provider_id):
            runtime = self._runtime.pop(provider_id, None)
            if runtime and runtime.task is not None and not runtime.task.done():
                runtime.task.cancel()
                await asyncio.gather(runtime.task, return_exceptions=True)
            if runtime:
                await self._terminate(runtime.process)

    async def restore_auto_start(self) -> None:
        for row in await asyncio.to_thread(providers_db.list_providers):
            if row.get("provider_type") != "kaggle_tpu" or not row.get("is_enabled"):
                continue
            try:
                if (row.get("managed_config") or {}).get("auto_start"):
                    await self.start(row["id"])
                elif _launcher_state_file(row["id"]).exists():
                    await self.refresh(row["id"])
            except Exception:
                runtime = self._get_runtime(row["id"], row)
                self._set(runtime, state="ERROR", message="Could not reconnect to Kaggle. Check the saved token and retry.")

    async def shutdown(self) -> None:
        rows = {row["id"]: row for row in await asyncio.to_thread(providers_db.list_providers)
                if row.get("provider_type") == "kaggle_tpu"}
        for provider_id in list(self._runtime):
            if (rows.get(provider_id, {}).get("managed_config") or {}).get("auto_stop"):
                try:
                    await self.stop(provider_id)
                except Exception:
                    logger.warning("kaggle_tpu.auto_stop_failed", provider_id=provider_id)
            await self.detach(provider_id)


kaggle_tpu_manager = KaggleTPULifecycleManager()
