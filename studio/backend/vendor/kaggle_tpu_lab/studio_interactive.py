"""Studio-only Kaggle interactive-session transport.

The upstream project launches a committed kernel version with ``kernels push``.
Kaggle's web editor instead creates an interactive kernel session and talks to
its Jupyter server.  Studio uses that same public API path here, while keeping
the upstream serving kernel unchanged.

Only non-secret ownership data (operation name / kernel session id) is returned
to the launcher for persistence.  Tokenized Jupyter URLs and Jupyter tokens are
kept in memory for the lifetime of this helper call.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx
from websockets.sync.client import connect


# Retained for recovery of mlx.26-28 sessions. New managed launches submit the
# complete server as a batch version so execution survives local disconnection.
INTERACTIVE_TPU_SHAPE = "TpuV5E8"


class InteractiveSessionClosed(RuntimeError):
    """The allocation operation ended, rather than merely remaining pending."""


@dataclass
class InteractiveSession:
    operation_name: str
    kernel_session_id: int
    jupyter_url: str
    tokenized_jupyter_url: str


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _request_types():
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiCancelKernelSessionRequest,
        ApiCreateKernelSessionRequest,
    )

    return ApiCreateKernelSessionRequest, ApiCancelKernelSessionRequest


def _status_request_type():
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    return ApiGetKernelSessionStatusRequest


def _quota_request_type():
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    return ApiGetAcceleratorQuotaStatisticsRequest


def _status_name(value) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(value or "UNKNOWN").rsplit(".", 1)[-1].upper()


def _session_status_with_client(client, kernel: str) -> tuple[str, str]:
    user, slug = kernel.split("/", 1)
    request = _status_request_type()()
    request.user_name = user
    request.kernel_slug = slug
    response = client.kernels.kernels_api_client.get_kernel_session_status(request)
    return _status_name(getattr(response, "status", None)), str(getattr(response, "failure_message", "") or "")


def session_status(kernel: str) -> tuple[str, str]:
    """Return Kaggle's public worker state for the managed interactive kernel."""
    api = _api()
    with api.build_kaggle_client() as client:
        return _session_status_with_client(client, kernel)


def _tpu_quota_with_client(client) -> dict[str, float]:
    request = _quota_request_type()()
    response = client.kernels.kernels_api_client.get_accelerator_quota_statistics(request)
    quota = getattr(response, "tpu_quota", None)
    if quota is None:
        return {}

    def seconds(value) -> float:
        total = getattr(value, "total_seconds", None)
        return float(total()) if callable(total) else 0.0

    return {
        "used_s": seconds(getattr(quota, "time_used", None)),
        "reserved_s": seconds(getattr(quota, "time_reserved", None)),
        "allowed_s": seconds(getattr(quota, "total_time_allowed", None)),
    }


def bootstrap_fingerprint(weights_dataset: str, env_dataset: str) -> str:
    payload = f"studio-interactive-v1\n{weights_dataset}\n{env_dataset}\n"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def ensure_bootstrap(
    kaggle_cli: Callable,
    kernel: str,
    weights_dataset: str,
    env_dataset: str,
    *,
    emit: Callable[[str, str], None],
    force: bool = False,
    timeout_s: int = 300,
) -> str:
    """Ensure a private notebook exists without requesting an accelerator.

    ``create-session`` requires a saved kernel slug.  A tiny CPU-only commit is
    therefore created once per connection.  Later Starts reuse that notebook;
    Stop cancels only the interactive session and leaves this bootstrap in
    place.  The bootstrap carries the two datasets so the interactive session
    inherits the fast-start mounts.
    """

    fingerprint = bootstrap_fingerprint(weights_dataset, env_dataset)
    status = kaggle_cli("kernels", "status", kernel)
    if status.returncode == 0 and not force:
        return fingerprint

    user, slug = kernel.split("/", 1)
    emit("PROVISIONING", "Creating the private Kaggle notebook used for interactive TPU sessions.")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                }
            },
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": "print('Unsloth Studio interactive bootstrap ready')",
                }
            ],
        }
        (root / "bootstrap.ipynb").write_text(json.dumps(notebook))
        (root / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel,
                    "title": slug,
                    "code_file": "bootstrap.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": "true",
                    "enable_gpu": "false",
                    "enable_tpu": "false",
                    "enable_internet": "false",
                    "dataset_sources": [weights_dataset, env_dataset],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                },
                indent=1,
            )
        )
        pushed = kaggle_cli("kernels", "push", "-p", str(root))
        if pushed.returncode:
            raise RuntimeError("Kaggle could not create the private interactive bootstrap notebook.")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = kaggle_cli("kernels", "status", kernel)
        raw = (status.stdout or "") + (status.stderr or "")
        if "KernelWorkerStatus.COMPLETE" in raw:
            return fingerprint
        if "KernelWorkerStatus.ERROR" in raw or status.returncode:
            raise RuntimeError("Kaggle could not finish the private interactive bootstrap notebook.")
        time.sleep(2)
    raise RuntimeError("Kaggle timed out while preparing the private interactive bootstrap notebook.")


def create_session(
    kernel: str,
    *,
    on_allocated: Callable[[str, int], None],
    emit: Callable[[str, str], None],
    operation_timeout_s: Optional[int] = None,
) -> InteractiveSession:
    """Create the same class of interactive session used by Kaggle notebooks."""

    ApiCreateKernelSessionRequest, _ = _request_types()
    api = _api()
    request = ApiCreateKernelSessionRequest()
    request.slug = kernel
    request.language = "python"
    request.kernel_type = "notebook"
    request.machine_shape = INTERACTIVE_TPU_SHAPE
    request.enable_internet = True

    emit("PROVISIONING", "Requesting an interactive Kaggle TPU session.")
    with api.build_kaggle_client() as client:
        operation = client.kernels.kernels_api_client.create_kernel_session(request)
        data = json.loads(str(operation))
        name = str(data.get("name") or "")
        session_id = int((data.get("metadata") or {}).get("kernelSessionId") or 0)
        if not name:
            raise RuntimeError("Kaggle did not return an interactive-session operation.")
        if session_id:
            on_allocated(name, session_id)

        deadline = (
            time.monotonic() + operation_timeout_s
            if operation_timeout_s is not None
            else None
        )
        allocation_started = time.monotonic()
        last_notice = 0.0
        quota = {}
        try:
            quota = _tpu_quota_with_client(client)
        except Exception:
            quota = {}
        while deadline is None or time.monotonic() < deadline:
            current = json.loads(str(client.common.operations_client.get_operation(name=name)))
            metadata = current.get("metadata") or {}
            current_id = int(metadata.get("kernelSessionId") or 0)
            if current_id and current_id != session_id:
                session_id = current_id
                on_allocated(name, session_id)
            if current.get("done"):
                error = current.get("error")
                if error:
                    raise RuntimeError("Kaggle rejected the interactive TPU session.")
                response = current.get("response") or {}
                session_id = int(response.get("kernelSessionId") or session_id or 0)
                jupyter_url = str(response.get("jupyterUrl") or "").rstrip("/")
                tokenized_url = str(response.get("tokenizedJupyterUrl") or "")
                if not (session_id and jupyter_url and tokenized_url):
                    raise RuntimeError("Kaggle returned an incomplete interactive-session response.")
                on_allocated(name, session_id)
                return InteractiveSession(name, session_id, jupyter_url, tokenized_url)
            now = time.monotonic()
            if now - last_notice >= 15:
                minutes = max(0, int((now - allocation_started) // 60))
                # kernels/status addresses the saved batch version, NOT this
                # interactive operation. Its COMPLETE/RUNNING cannot describe
                # whether this allocation has hardware or is waiting in a queue.
                emit("PROVISIONING", f"Kaggle has not completed the interactive allocation: {minutes} minutes elapsed.")
                last_notice = now
            time.sleep(1)
    raise RuntimeError("Kaggle timed out while allocating the interactive TPU session.")


def restore_session(operation_name: str) -> Optional[InteractiveSession]:
    """Recover a Jupyter URL in memory from a persisted non-secret operation id."""

    if not operation_name:
        return None
    api = _api()
    with api.build_kaggle_client() as client:
        current = json.loads(str(client.common.operations_client.get_operation(name=operation_name)))
    if current.get("error"):
        raise InteractiveSessionClosed("Kaggle closed the interactive allocation.")
    if not current.get("done"):
        return None
    response = current.get("response") or {}
    session_id = int(response.get("kernelSessionId") or (current.get("metadata") or {}).get("kernelSessionId") or 0)
    jupyter_url = str(response.get("jupyterUrl") or "").rstrip("/")
    tokenized_url = str(response.get("tokenizedJupyterUrl") or "")
    if not (session_id and jupyter_url and tokenized_url):
        return None
    return InteractiveSession(operation_name, session_id, jupyter_url, tokenized_url)


def _jupyter_token(session: InteractiveSession) -> str:
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(session.tokenized_jupyter_url).query)
    token = (query.get("token") or [""])[0]
    if not token:
        raise RuntimeError("Kaggle did not return a Jupyter access token.")
    return token


def session_alive(session: InteractiveSession) -> bool:
    token = _jupyter_token(session)
    try:
        with httpx.Client(
            timeout=12.0,
            follow_redirects=False,
            headers={"Authorization": "token " + token},
        ) as client:
            response = client.get(session.jupyter_url + "/api/kernels")
            return response.status_code == 200
    except Exception:
        return False


def _phase_event(line: str) -> Optional[dict]:
    import re

    match = re.search(r"PHASE ([\w-]+)\s*(.*)$", line)
    if not match:
        return None
    extra = {}
    if match[2].strip():
        try:
            parsed = json.loads(match[2])
            if isinstance(parsed, dict):
                extra = parsed
        except ValueError:
            pass
    return {**extra, "phase": match[1]}


def execute_kernel(
    session: InteractiveSession,
    source: str,
    *,
    on_event: Callable[[dict, str], None],
) -> Optional[dict]:
    """Run the prepared upstream source in the interactive Jupyter kernel."""

    token = _jupyter_token(session)
    headers = {"Authorization": "token " + token}
    with httpx.Client(timeout=30.0, follow_redirects=False, headers=headers) as client:
        response = client.get(session.jupyter_url + "/api/kernels")
        response.raise_for_status()
        kernels = response.json()
        if kernels:
            kernel_id = kernels[0]["id"]
        else:
            response = client.post(session.jupyter_url + "/api/kernels", json={"name": "python3"})
            response.raise_for_status()
            kernel_id = response.json()["id"]

    message_id = uuid.uuid4().hex
    ws_url = (
        session.jupyter_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        + f"/api/kernels/{kernel_id}/channels?session_id={uuid.uuid4().hex}"
    )
    request = {
        "header": {
            "msg_id": message_id,
            "username": "unsloth-studio",
            "session": message_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "channel": "shell",
        "content": {
            "code": "exec(compile(" + repr(source) + ", 'serve_qwen38.py', 'exec'))",
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
    }

    failure = None
    log_tail = ""
    pending = ""
    with connect(ws_url, additional_headers=headers, open_timeout=30, close_timeout=5) as socket:
        socket.send(json.dumps(request))
        while True:
            try:
                raw = socket.recv(timeout=90)
            except TimeoutError:
                # TPU graph compilation can be quiet for longer than a single
                # websocket read window. Keep the monitor attached while the
                # authenticated Jupyter service itself is still alive.
                if session_alive(session):
                    continue
                raise RuntimeError("The interactive Kaggle Jupyter session stopped responding.") from None
            if not isinstance(raw, str):
                continue
            message = json.loads(raw)
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind = message.get("msg_type") or message.get("header", {}).get("msg_type")
            content = message.get("content") or {}
            if kind == "stream":
                text = str(content.get("text") or "")
                log_tail = (log_tail + text)[-20000:]
                pending += text
                parts = pending.split("\n")
                pending = parts.pop()
                for line in parts:
                    event = _phase_event(line)
                    if event:
                        on_event(event, log_tail)
                        if event.get("phase") == "failed":
                            failure = event
            elif kind == "error":
                failure = {
                    "phase": "failed",
                    "step": "server",
                    "tail": str(content.get("evalue") or content.get("ename") or "Jupyter execution failed")[-1000:],
                }
                on_event(failure, log_tail)
            elif kind == "status" and content.get("execution_state") == "idle":
                return failure


def cancel_session(kernel_session_id: int) -> None:
    if not kernel_session_id:
        return
    _, ApiCancelKernelSessionRequest = _request_types()
    api = _api()
    request = ApiCancelKernelSessionRequest()
    request.kernel_session_id = int(kernel_session_id)
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.cancel_kernel_session(request)
    if getattr(response, "error_message", ""):
        raise RuntimeError("Kaggle could not cancel the interactive TPU session.")
