from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import secrets
import shutil
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from websockets.asyncio.server import ServerConnection, serve
from zeroconf import IPVersion, ServiceInfo, Zeroconf

from auth import storage as auth_storage
from core.companion.models import (
    PROTOCOL_VERSION,
    ActiveCompanionTask,
    CapabilityReport,
    CompanionResult,
    CompanionSettings,
    CompanionStatus,
    CompanionTask,
    DeviceStatus,
    Envelope,
    PairedDevice,
    PendingPairing,
    SERVICE_TYPE,
)
from core.companion.security import DesktopIdentity
from utils.native_tls import create_server_ssl_context


class CompanionUnavailable(RuntimeError):
    pass


class CompanionUserCancelled(CompanionUnavailable):
    pass


@dataclass
class DeviceSession:
    device_id: UUID
    websocket: ServerConnection
    paired: PairedDevice
    status: DeviceStatus
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_results: dict[UUID, asyncio.Future[CompanionResult]] = field(default_factory=dict)
    pending_tasks: dict[UUID, ActiveCompanionTask] = field(default_factory=dict)
    last_heartbeat_sent: float = 0
    last_heartbeat_received: float = field(default_factory=time.monotonic)


@dataclass
class PairingFlow:
    pending: PendingPairing
    nonce: bytes
    websocket: ServerConnection


@dataclass
class PairingOfferFlow:
    device_id: UUID
    device_public_key: str
    nonce: bytes
    websocket: ServerConnection
    created_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class BackgroundCompanionJob:
    job_id: UUID
    thread_id: str | None
    task_ids: list[UUID]
    item_ids: list[str]
    kind: str
    idempotency_key: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: str = "queued"
    completed_at: datetime | None = None
    collected_at: datetime | None = None
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    runner: asyncio.Task[None] | None = None
    cleanup_paths: list[Path] = field(default_factory=list)


class CompanionManager:
    def __init__(self) -> None:
        self.root = Path(auth_storage.DB_PATH).parent / "companion"
        self.settings_path = self.root / "settings.json"
        self.devices_path = self.root / "paired-devices.json"
        self.identity: DesktopIdentity | None = None
        self.settings = CompanionSettings()
        self.paired: dict[UUID, PairedDevice] = {}
        self.sessions: dict[UUID, DeviceSession] = {}
        self.pending_pairings: dict[UUID, PairingFlow] = {}
        self.pairing_offers: dict[int, PairingOfferFlow] = {}
        self._idempotent_results: dict[str, asyncio.Future[CompanionResult]] = {}
        self._background_jobs: dict[UUID, BackgroundCompanionJob] = {}
        self._background_job_keys: dict[str, UUID] = {}
        self.server: Any = None
        self.listener_port: int | None = None
        self.zeroconf: Zeroconf | None = None
        self.service_info: ServiceInfo | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None
        self._seen_messages: dict[UUID, float] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False

    async def start(self) -> None:
        async with self._lifecycle_lock:
            self._event_loop = asyncio.get_running_loop()
            self.root.mkdir(parents=True, exist_ok=True)
            self.identity = DesktopIdentity(self.root)
            self.settings = self._read_model(self.settings_path, CompanionSettings, CompanionSettings())
            devices = self._read_json(self.devices_path, [])
            self.paired = {value.deviceID: value for value in (PairedDevice.model_validate(item) for item in devices)}
            if self.settings.enabled:
                await self._start_listener()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            await self._stop_listener()

    async def update_settings(self, value: CompanionSettings) -> CompanionStatus:
        async with self._lifecycle_lock:
            was_enabled = self.settings.enabled
            self.settings = value
            self._write_model(self.settings_path, value)
            if value.enabled and not was_enabled:
                await self._start_listener()
            elif not value.enabled and was_enabled:
                await self._stop_listener()
        return self.status()

    def status(self) -> CompanionStatus:
        devices: list[DeviceStatus] = []
        for device_id, paired in self.paired.items():
            if session := self.sessions.get(device_id):
                devices.append(self._scored(session.status))
            else:
                devices.append(DeviceStatus(deviceID=device_id, name=paired.name, enabled=paired.enabled))
        return CompanionStatus(
            settings=self.settings,
            devices=sorted(devices, key=lambda item: item.score, reverse=True),
            pendingPairings=[flow.pending for flow in self.pending_pairings.values()],
            activeTasks=sorted(
                (task for session in self.sessions.values() for task in session.pending_tasks.values()),
                key=lambda item: item.startedAt,
            ),
            listenerPort=self.listener_port,
            certificateSHA256=self.identity.certificate_sha256 if self.identity else None,
        )

    async def confirm_pairing(self, pairing_id: UUID) -> None:
        flow = self.pending_pairings.get(pairing_id)
        if not flow:
            raise KeyError("Pairing request not found")
        flow.pending.desktopConfirmed = True
        await self._finish_pairing_if_ready(flow)

    async def reject_pairing(self, pairing_id: UUID) -> None:
        flow = self.pending_pairings.pop(pairing_id, None)
        if not flow:
            raise KeyError("Pairing request not found")
        await self._send(flow.websocket, Envelope(type="pairing_result", payload={"accepted": False}))
        await flow.websocket.close(code=1008, reason="Pairing rejected")

    async def revoke(self, device_id: UUID) -> None:
        self.paired.pop(device_id, None)
        self._save_devices()
        if session := self.sessions.pop(device_id, None):
            await self._cancel_all(session, "desktop_request")
            await session.websocket.close(code=1008, reason="Pairing revoked")

    async def rename(self, device_id: UUID, name: str) -> None:
        device = self.paired.get(device_id)
        if not device:
            raise KeyError("Device not found")
        device.name = name.strip()[:80]
        self._save_devices()
        if session := self.sessions.get(device_id):
            session.status.name = device.name

    async def set_device_enabled(self, device_id: UUID, enabled: bool) -> None:
        device = self.paired.get(device_id)
        if not device:
            raise KeyError("Device not found")
        device.enabled = enabled
        self._save_devices()
        if session := self.sessions.get(device_id):
            session.status.enabled = enabled
            if not enabled:
                await self._cancel_all(session, "companion_disabled")

    async def dispatch(self, task: CompanionTask) -> CompanionResult:
        if existing := self._idempotent_results.get(task.idempotencyKey):
            try:
                return await asyncio.wait_for(asyncio.shield(existing), timeout=task.timeoutSeconds)
            except asyncio.TimeoutError as exc:
                raise CompanionUnavailable("Duplicate iPhone task is still running") from exc
        shared: asyncio.Future[CompanionResult] = asyncio.get_running_loop().create_future()
        self._idempotent_results[task.idempotencyKey] = shared
        excluded: set[UUID] = set()
        try:
            while True:
                session = self._select_session(task.kind, excluded)
                if not session:
                    raise CompanionUnavailable("No eligible iPhone Companion is ready")
                try:
                    result = await self._dispatch_once(session, task)
                    if not shared.done():
                        shared.set_result(result)
                    asyncio.get_running_loop().call_later(120, self._expire_idempotency, task.idempotencyKey, shared)
                    return result
                except CompanionUserCancelled:
                    raise
                except CompanionUnavailable:
                    excluded.add(session.device_id)
                    if not task.shardCount or task.shardCount <= 1 or self._select_session(task.kind, excluded) is None:
                        raise
        except asyncio.CancelledError:
            self._idempotent_results.pop(task.idempotencyKey, None)
            if not shared.done(): shared.cancel()
            await self.cancel(task.taskID, explicit_user=False)
            raise
        except Exception:
            self._idempotent_results.pop(task.idempotencyKey, None)
            if not shared.done(): shared.cancel()
            raise

    async def dispatch_many(self, tasks: list[CompanionTask]) -> list[CompanionResult]:
        if not tasks:
            return []
        # Each iPhone runtime accepts one active task. Run one worker per eligible
        # selected device so Multi-iPhone truly executes in parallel without ever
        # overfilling a phone; remaining independent items wait for the next slot.
        slots = max(1, len(self._eligible_sessions()))
        semaphore = asyncio.Semaphore(slots)

        async def run(task: CompanionTask) -> CompanionResult:
            async with semaphore:
                return await self.dispatch(task)

        outcomes = await asyncio.gather(*(run(task) for task in tasks), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return [outcome for outcome in outcomes if isinstance(outcome, CompanionResult)]

    async def submit_background(
        self,
        tasks: list[CompanionTask],
        *,
        item_ids: list[str],
        thread_id: str | None,
        cleanup_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Queue Companion work and return before any iPhone generation completes.

        The caller transfers ownership of ``cleanup_paths`` only after this method
        accepts the job. They are retained for media transfer and removed as soon as
        the background job reaches a terminal state.
        """
        self._expire_background_jobs()
        owned_paths = list(cleanup_paths or [])
        if not tasks or len(tasks) != len(item_ids):
            await asyncio.to_thread(self._remove_paths, owned_paths)
            raise CompanionUnavailable("A background iPhone job needs matching tasks and item IDs")

        canonical_key = hashlib.sha256(
            ((thread_id or "") + "\n" + "\n".join(task.idempotencyKey for task in tasks)).encode()
        ).hexdigest()
        if existing_id := self._background_job_keys.get(canonical_key):
            existing = self._background_jobs.get(existing_id)
            if existing is not None:
                await asyncio.to_thread(self._remove_paths, owned_paths)
                snapshot = self._background_snapshot(existing, include_outcomes=True)
                snapshot["duplicate"] = True
                return snapshot

        missing_kind = next(
            (task.kind for task in tasks if self._select_session(task.kind) is None),
            None,
        )
        if missing_kind is not None:
            await asyncio.to_thread(self._remove_paths, owned_paths)
            raise CompanionUnavailable(
                f"No eligible iPhone Companion is ready for {missing_kind}"
            )

        parent_ids = {task.parentTaskID for task in tasks if task.parentTaskID is not None}
        job_id = next(iter(parent_ids)) if len(parent_ids) == 1 else tasks[0].taskID
        if job_id in self._background_jobs:
            job_id = uuid4()
        job = BackgroundCompanionJob(
            job_id=job_id,
            thread_id=thread_id,
            task_ids=[task.taskID for task in tasks],
            item_ids=list(item_ids),
            kind=tasks[0].kind,
            idempotency_key=canonical_key,
            cleanup_paths=owned_paths,
        )
        self._background_jobs[job_id] = job
        self._background_job_keys[canonical_key] = job_id
        job.runner = asyncio.create_task(
            self._run_background_job(job, tasks),
            name=f"iphone-companion-{job_id}",
        )
        # Let the runner claim its first available phone, but never wait for its
        # result. This is the async boundary observed by the Mac tool loop.
        await asyncio.sleep(0)
        return self._background_snapshot(job)

    async def background_status(
        self,
        *,
        thread_id: str | None,
        job_id: UUID | None = None,
    ) -> dict[str, Any]:
        self._expire_background_jobs()
        if job_id is not None:
            job = self._background_job(job_id, thread_id)
            return {"source": "iPhone Companion", "jobs": [self._background_snapshot(job)]}
        jobs = sorted(
            self._background_jobs_for_thread(thread_id),
            key=lambda value: value.created_at,
        )
        return {
            "source": "iPhone Companion",
            "jobs": [self._background_snapshot(job) for job in jobs],
            "counts": self.background_counts(thread_id),
        }

    async def collect_background(
        self,
        *,
        thread_id: str | None,
        job_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Take completed work without waiting; running jobs return status only."""
        self._expire_background_jobs()
        if job_id is not None:
            jobs = [self._background_job(job_id, thread_id)]
        else:
            jobs = [
                job
                for job in self._background_jobs_for_thread(thread_id)
                if self._job_is_terminal(job) and job.collected_at is None
            ]
        now = datetime.now(timezone.utc)
        snapshots: list[dict[str, Any]] = []
        for job in jobs:
            terminal = self._job_is_terminal(job)
            if terminal:
                job.collected_at = now
            snapshots.append(self._background_snapshot(job, include_outcomes=terminal))
        return {
            "source": "iPhone Companion",
            "jobs": snapshots,
            "counts": self.background_counts(thread_id),
            "message": (
                "Completed iPhone work collected. Critically evaluate it before using it."
                if any(self._job_is_terminal(job) for job in jobs)
                else "No completed iPhone work is ready. Continue on the Mac; do not busy-poll."
            ),
        }

    def background_counts(self, thread_id: str | None) -> dict[str, int]:
        self._expire_background_jobs()
        jobs = self._background_jobs_for_thread(thread_id)
        return {
            "pending": sum(not self._job_is_terminal(job) for job in jobs),
            "ready": sum(
                self._job_is_terminal(job) and job.collected_at is None for job in jobs
            ),
            "collected": sum(job.collected_at is not None for job in jobs),
        }

    async def _run_background_job(
        self,
        job: BackgroundCompanionJob,
        tasks: list[CompanionTask],
    ) -> None:
        job.state = "running"
        slots = (
            max(1, len(self._eligible_sessions()))
            if self.settings.mode.value == "multiple"
            else 1
        )
        semaphore = asyncio.Semaphore(slots)

        async def run_one(index: int, task: CompanionTask) -> dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.dispatch(task)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - retained for later Mac fallback
                    return {
                        "id": job.item_ids[index],
                        "taskID": str(task.taskID),
                        "error": str(exc),
                    }
                return {
                    "id": job.item_ids[index],
                    "taskID": str(result.taskID),
                    "durationMS": result.durationMS,
                    "tokensGenerated": result.tokensGenerated,
                    "result": result.result,
                }

        try:
            job.outcomes = list(
                await asyncio.gather(
                    *(run_one(index, task) for index, task in enumerate(tasks))
                )
            )
            succeeded = sum("result" in outcome for outcome in job.outcomes)
            if succeeded == len(job.outcomes):
                job.state = "completed"
            elif succeeded:
                job.state = "partial"
                job.error = "Some iPhone subtasks failed; completed results remain available"
            else:
                job.state = "failed"
                job.error = "; ".join(
                    dict.fromkeys(
                        str(outcome.get("error") or "iPhone task failed")
                        for outcome in job.outcomes
                    )
                )
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.error = "Background iPhone job cancelled"
            raise
        finally:
            job.completed_at = datetime.now(timezone.utc)
            job.runner = None
            paths, job.cleanup_paths = job.cleanup_paths, []
            await asyncio.to_thread(self._remove_paths, paths)

    def submit_background_sync(
        self,
        tasks: list[CompanionTask],
        *,
        item_ids: list[str],
        thread_id: str | None,
        cleanup_paths: list[Path] | None = None,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            self._remove_paths(list(cleanup_paths or []))
            raise CompanionUnavailable("iPhone Companion submission was cancelled")
        return self._run_sync(
            self.submit_background(
                tasks,
                item_ids=item_ids,
                thread_id=thread_id,
                cleanup_paths=cleanup_paths,
            )
        )

    def background_status_sync(
        self,
        *,
        thread_id: str | None,
        job_id: UUID | None = None,
    ) -> dict[str, Any]:
        return self._run_sync(self.background_status(thread_id=thread_id, job_id=job_id))

    def collect_background_sync(
        self,
        *,
        thread_id: str | None,
        job_id: UUID | None = None,
    ) -> dict[str, Any]:
        return self._run_sync(self.collect_background(thread_id=thread_id, job_id=job_id))

    def _run_sync(self, coroutine: Any) -> Any:
        loop = self._event_loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise CompanionUnavailable("iPhone Companion event loop is unavailable")
        pending = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            # Submission and mailbox reads do not wait for phone inference. A
            # bounded bridge timeout only protects against a stopped backend loop.
            return pending.result(timeout=10)
        except concurrent.futures.TimeoutError as exc:
            pending.cancel()
            raise CompanionUnavailable("iPhone Companion mailbox is unavailable") from exc

    def _background_job(self, job_id: UUID, thread_id: str | None) -> BackgroundCompanionJob:
        job = self._background_jobs.get(job_id)
        if job is None or job.thread_id != thread_id:
            raise CompanionUnavailable("iPhone Companion job not found in this chat")
        return job

    def _background_jobs_for_thread(
        self, thread_id: str | None
    ) -> list[BackgroundCompanionJob]:
        return [job for job in self._background_jobs.values() if job.thread_id == thread_id]

    @staticmethod
    def _job_is_terminal(job: BackgroundCompanionJob) -> bool:
        return job.state in {"completed", "partial", "failed", "cancelled"}

    def _background_snapshot(
        self,
        job: BackgroundCompanionJob,
        *,
        include_outcomes: bool = False,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "jobID": str(job.job_id),
            "state": job.state,
            "kind": job.kind,
            "taskIDs": [str(task_id) for task_id in job.task_ids],
            "createdAt": job.created_at.isoformat(),
            "completedAt": job.completed_at.isoformat() if job.completed_at else None,
            "collected": job.collected_at is not None,
        }
        if job.error:
            value["error"] = job.error
        if include_outcomes and self._job_is_terminal(job):
            if len(job.outcomes) == 1:
                value["result"] = job.outcomes[0]
            else:
                value["parallelResults"] = job.outcomes
        return value

    def _expire_background_jobs(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - 24 * 60 * 60
        for job_id, job in list(self._background_jobs.items()):
            if not self._job_is_terminal(job) or job.completed_at is None:
                continue
            if job.completed_at.timestamp() >= cutoff:
                continue
            self._background_jobs.pop(job_id, None)
            if self._background_job_keys.get(job.idempotency_key) == job_id:
                self._background_job_keys.pop(job.idempotency_key, None)

    @staticmethod
    def _remove_paths(paths: list[Path]) -> None:
        for path in paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    async def _dispatch_once(self, session: DeviceSession, task: CompanionTask) -> CompanionResult:
        future: asyncio.Future[CompanionResult] = asyncio.get_running_loop().create_future()
        session.pending_results[task.taskID] = future
        session.pending_tasks[task.taskID] = ActiveCompanionTask(
            taskID=task.taskID,
            deviceID=session.device_id,
            deviceName=session.paired.name,
            kind=task.kind,
            startedAt=datetime.now(timezone.utc),
        )
        session.status.queueDepth += 1
        try:
            payload = task.model_dump(mode="json")
            payload["input"]["maximumTokens"] = self._safe_maximum_tokens(session, task)
            media_paths = [Path(value) for value in payload["input"].pop("mediaPaths", [])]
            if media_paths and task.mediaPolicy == "semantic_only":
                raise CompanionUnavailable("Semantic-only tasks cannot transfer media files")
            remote_names = [f"{index:04d}-{path.name}" for index, path in enumerate(media_paths)]
            payload["input"]["mediaFiles"] = remote_names
            payload["input"]["expectedMediaBytes"] = sum(path.stat().st_size for path in media_paths)
            await self._send(session.websocket, Envelope(type="task_submit", payload=payload), session.send_lock)
            for path, remote_name in zip(media_paths, remote_names, strict=True):
                await self._send_blob(session, task.taskID, path, remote_name)
            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout=task.timeoutSeconds)
            except asyncio.TimeoutError as exc:
                await self.cancel(task.taskID, explicit_user=False)
                raise CompanionUnavailable("iPhone task timed out; Mac fallback required") from exc
        finally:
            session.pending_results.pop(task.taskID, None)
            session.pending_tasks.pop(task.taskID, None)
            session.status.queueDepth = max(0, session.status.queueDepth - 1)

    def _safe_maximum_tokens(self, session: DeviceSession, task: CompanionTask) -> int:
        """Honor the Mac model's budget while reserving room in the selected iPhone context."""
        try:
            requested = int(task.input.get("maximumTokens") or 4_096)
        except (TypeError, ValueError):
            requested = 4_096
        requested = max(256, min(16_384, requested))

        capabilities = session.status.capabilities
        context_size = int(capabilities.contextSize) if capabilities else 0
        if context_size <= 0:
            return min(requested, 4_096)

        text = str(task.input.get("text") or "")
        instruction = str(task.input.get("instruction") or "")
        # Gemma tokenization varies by language; one token per three characters is a
        # deliberately conservative Desktop-side estimate. The iPhone still performs
        # exact tokenization and rejects any generation that would finish by truncation.
        estimated_prompt_tokens = max(64, (len(text) + len(instruction) + 2) // 3)
        safe_available = context_size - estimated_prompt_tokens - 256
        if task.mediaPolicy != "semantic_only":
            safe_available = min(safe_available, context_size // 2)
        if safe_available < 256:
            raise CompanionUnavailable("iPhone context is too small for this delegated prompt; Mac fallback required")
        return max(256, min(requested, safe_available))

    def dispatch_sync(self, task: CompanionTask, cancel_event: Any = None) -> CompanionResult:
        loop = self._event_loop
        if loop is None or not loop.is_running():
            raise CompanionUnavailable("iPhone Companion event loop is unavailable")
        pending = asyncio.run_coroutine_threadsafe(self.dispatch(task), loop)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                asyncio.run_coroutine_threadsafe(self.cancel(task.taskID, explicit_user=True), loop)
                pending.cancel()
                raise CompanionUnavailable("iPhone Companion task was cancelled")
            try:
                return pending.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                continue

    def dispatch_many_sync(self, tasks: list[CompanionTask], cancel_event: Any = None) -> list[CompanionResult]:
        loop = self._event_loop
        if loop is None or not loop.is_running():
            raise CompanionUnavailable("iPhone Companion event loop is unavailable")
        pending = asyncio.run_coroutine_threadsafe(self.dispatch_many(tasks), loop)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                for task in tasks:
                    asyncio.run_coroutine_threadsafe(
                        self.cancel(task.taskID, explicit_user=True), loop
                    )
                pending.cancel()
                raise CompanionUnavailable("iPhone Companion tasks were cancelled")
            try:
                return pending.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                continue

    def compress_context_sync(self, messages: list[dict[str, Any]], thread_id: str | None, cancel_event: Any = None) -> str:
        canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(((thread_id or "") + "\n" + canonical).encode()).hexdigest()
        task = CompanionTask(
            taskID=uuid4(),
            idempotencyKey=f"context-{digest}",
            kind="context_compression",
            priority=80,
            timeoutSeconds=120,
            input={"text": canonical, "instruction": "Preserve message roles and every binding instruction.", "maximumTokens": 1536},
            resultSchema={"type": "object", "required": ["compressedContext", "preservedFacts"]},
        )
        result = self.dispatch_sync(task, cancel_event=cancel_event)
        compressed = result.result.get("compressedContext")
        if not isinstance(compressed, str) or not compressed.strip():
            raise CompanionUnavailable("iPhone returned no compressed context")
        return compressed.strip()

    async def cancel(self, task_id: UUID, explicit_user: bool) -> None:
        for session in self.sessions.values():
            if task_id in session.pending_results:
                await self._send(session.websocket, Envelope(type="task_cancel", payload={"taskID": str(task_id), "explicitUser": explicit_user}), session.send_lock)
                if future := session.pending_results.get(task_id):
                    if not future.done():
                        future.set_exception(CompanionUserCancelled("Task cancelled by the user") if explicit_user else CompanionUnavailable("Task cancelled by the Desktop"))
                return

    def _select_session(self, kind: str, excluded: set[UUID] | None = None) -> DeviceSession | None:
        candidates = [
            session
            for session in self._eligible_sessions(excluded)
            if session.status.capabilities
            and kind in session.status.capabilities.taskKinds
        ]
        return max(candidates, key=lambda value: self._scored(value.status).score, default=None)

    def available_task_kinds(self) -> set[str]:
        """Capabilities currently dispatchable under the user's device policy."""
        kinds: set[str] = set()
        for session in self._eligible_sessions():
            if session.status.capabilities:
                kinds.update(session.status.capabilities.taskKinds)
        return kinds

    def _eligible_sessions(self, excluded: set[UUID] | None = None) -> list[DeviceSession]:
        if not self.settings.enabled or self._stopping:
            return []
        excluded = excluded or set()
        candidates = [
            session for session in self.sessions.values()
            if session.device_id not in excluded and session.paired.enabled and session.status.connected and session.status.authenticated
            and session.status.capabilities
            and session.status.state in {"ready", "leased", "running"}
            and session.status.queueDepth == 0
            and not session.status.lowPowerMode and session.status.battery >= 0.10 and session.status.thermalState < 2
        ]
        if self.settings.mode.value == "multiple":
            candidates = [value for value in candidates if value.device_id in self.settings.selectedDeviceIDs]
        return candidates

    async def _start_listener(self) -> None:
        if self.server:
            return
        assert self.identity is not None
        context = create_server_ssl_context()
        context.load_cert_chain(self.identity.cert_path, self.identity.key_path)
        self.server = await serve(self._handler, "0.0.0.0", 0, ssl=context, max_size=4 * 1024 * 1024, compression=None, ping_interval=None)
        socket_value = next(iter(self.server.sockets))
        self.listener_port = int(socket_value.getsockname()[1])
        local_ipv4 = self._local_ipv4()
        address = socket.inet_aton(local_ipv4)
        host_label = self._bonjour_host_label()
        tls_hostname = f"{host_label}.local"
        self.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        self.service_info = ServiceInfo(
            SERVICE_TYPE,
            f"Unsloth Desktop {host_label}.{SERVICE_TYPE}",
            addresses=[address],
            port=self.listener_port,
            properties={
                "v": "1",
                "cert": self.identity.certificate_sha256,
                "name": host_label,
                "host": tls_hostname,
                "address": local_ipv4,
                "port": str(self.listener_port),
            },
            server=f"{host_label}.local.",
        )
        await asyncio.to_thread(self.zeroconf.register_service, self.service_info)
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _stop_listener(self) -> None:
        self._stopping = True
        try:
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                await asyncio.gather(self.heartbeat_task, return_exceptions=True)
                self.heartbeat_task = None
            for session in list(self.sessions.values()):
                await self._cancel_all(session, "companion_disabled")
                await session.websocket.close(code=1001, reason="Companion stopped")
            self.sessions.clear()
            self.pairing_offers.clear()
            self.pending_pairings.clear()
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None
            if self.zeroconf and self.service_info:
                await asyncio.to_thread(self.zeroconf.unregister_service, self.service_info)
            if self.zeroconf:
                await asyncio.to_thread(self.zeroconf.close)
            self.zeroconf = None
            self.service_info = None
            self.listener_port = None
        finally:
            self._stopping = False

    async def _handler(self, websocket: ServerConnection) -> None:
        session: DeviceSession | None = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=15)
            envelope = self._parse_envelope(raw)
            if envelope.type != "hello":
                raise ValueError("hello required")
            device_id = UUID(str(envelope.payload["deviceID"]))
            if paired := self.paired.get(device_id):
                session = await self._authenticate(websocket, paired)
            else:
                await self._begin_pairing(websocket, device_id, envelope.payload)
            async for message in websocket:
                if isinstance(message, bytes):
                    raise ValueError("unexpected binary frame")
                incoming = self._parse_envelope(message)
                if session is None:
                    await self._handle_pairing_message(websocket, incoming)
                    for candidate in self.sessions.values():
                        if candidate.websocket is websocket:
                            session = candidate
                            break
                else:
                    await self._handle_session_message(session, incoming)
        except Exception:
            await websocket.close(code=1008, reason="Protocol or authentication failure")
        finally:
            if session and self.sessions.get(session.device_id) is session:
                self.sessions.pop(session.device_id, None)
                for future in session.pending_results.values():
                    if not future.done():
                        future.set_exception(CompanionUnavailable("iPhone disconnected; Mac fallback required"))
            for pairing_id, flow in list(self.pending_pairings.items()):
                if flow.websocket is websocket:
                    self.pending_pairings.pop(pairing_id, None)
            self.pairing_offers.pop(id(websocket), None)

    async def _authenticate(self, websocket: ServerConnection, paired: PairedDevice) -> DeviceSession:
        assert self.identity is not None
        nonce = secrets.token_bytes(32)
        await self._send(websocket, Envelope(type="challenge", payload={"desktopID": str(self._desktop_id()), "nonce": base64.b64encode(nonce).decode(), "signature": self.identity.sign_base64(nonce)}))
        response = self._parse_envelope(await asyncio.wait_for(websocket.recv(), timeout=15))
        if response.type != "challenge_response" or UUID(str(response.payload["deviceID"])) != paired.deviceID:
            raise ValueError("challenge response mismatch")
        self.identity.verify_base64(paired.signingPublicKey, str(response.payload["signature"]), nonce)
        status = DeviceStatus(deviceID=paired.deviceID, name=paired.name, enabled=paired.enabled, connected=True, authenticated=True, lastSeen=datetime.now(timezone.utc))
        session = DeviceSession(device_id=paired.deviceID, websocket=websocket, paired=paired, status=status)
        self.sessions[paired.deviceID] = session
        return session

    async def _begin_pairing(self, websocket: ServerConnection, device_id: UUID, hello: dict[str, Any]) -> None:
        assert self.identity is not None
        nonce = secrets.token_bytes(32)
        device_public_key = str(hello["devicePublicKey"])
        public_key_bytes = base64.b64decode(device_public_key, validate=True)
        if len(public_key_bytes) != 65 or public_key_bytes[0] != 4:
            raise ValueError("invalid P-256 device public key")
        self.pairing_offers[id(websocket)] = PairingOfferFlow(
            device_id=device_id,
            device_public_key=device_public_key,
            nonce=nonce,
            websocket=websocket,
        )
        payload = {
            "desktopID": str(self._desktop_id()), "desktopName": socket.gethostname(),
            "signingPublicKey": self.identity.public_key_base64, "nonce": base64.b64encode(nonce).decode(),
            "certificateSHA256": self.identity.certificate_sha256,
        }
        await self._send(websocket, Envelope(type="pairing_offer", payload=payload))

    async def _handle_pairing_message(self, websocket: ServerConnection, envelope: Envelope) -> None:
        if envelope.type != "pairing_confirm":
            raise ValueError("pairing confirmation required")
        payload = envelope.payload
        if payload.get("preview"):
            assert self.identity is not None
            offer = self.pairing_offers.get(id(websocket))
            if not offer or offer.websocket is not websocket:
                raise ValueError("pairing offer expired")
            device_id = UUID(str(payload["deviceID"]))
            public_key = str(payload["devicePublicKey"])
            nonce = base64.b64decode(str(payload["nonce"]), validate=True)
            if device_id != offer.device_id or public_key != offer.device_public_key or not secrets.compare_digest(nonce, offer.nonce):
                raise ValueError("pairing preview does not match offer")
            self.identity.verify_base64(public_key, str(payload["signature"]), nonce)
            self.pairing_offers.pop(id(websocket), None)
            pairing_id = uuid4()
            code = self.identity.pairing_code(self.identity.public_key_base64, public_key, nonce, self.identity.certificate_sha256)
            pending = PendingPairing(pairingID=pairing_id, deviceID=device_id, deviceName=str(payload["deviceName"])[:80], signingPublicKey=public_key, code=code, createdAt=datetime.now(timezone.utc))
            self.pending_pairings[pairing_id] = PairingFlow(pending=pending, nonce=nonce, websocket=websocket)
            return
        flow = next((value for value in self.pending_pairings.values() if value.websocket is websocket), None)
        if not flow or not payload.get("confirmed"):
            raise ValueError("pairing rejected or missing preview")
        if UUID(str(payload["desktopID"])) != self._desktop_id() or UUID(str(payload["deviceID"])) != flow.pending.deviceID:
            raise ValueError("pairing identities changed")
        if str(payload["devicePublicKey"]) != flow.pending.signingPublicKey:
            raise ValueError("pairing key changed")
        assert self.identity is not None
        self.identity.verify_base64(flow.pending.signingPublicKey, str(payload["signature"]), flow.nonce)
        flow.pending.phoneConfirmed = True
        await self._finish_pairing_if_ready(flow)

    async def _finish_pairing_if_ready(self, flow: PairingFlow) -> None:
        if not (flow.pending.phoneConfirmed and flow.pending.desktopConfirmed):
            return
        paired = PairedDevice(deviceID=flow.pending.deviceID, name=flow.pending.deviceName, signingPublicKey=flow.pending.signingPublicKey, pairedAt=datetime.now(timezone.utc))
        self.paired[paired.deviceID] = paired
        self._save_devices()
        self.pending_pairings.pop(flow.pending.pairingID, None)
        await self._send(flow.websocket, Envelope(type="pairing_result", payload={"accepted": True}))
        status = DeviceStatus(deviceID=paired.deviceID, name=paired.name, connected=True, authenticated=True, enabled=True, lastSeen=datetime.now(timezone.utc))
        self.sessions[paired.deviceID] = DeviceSession(device_id=paired.deviceID, websocket=flow.websocket, paired=paired, status=status)

    async def _handle_session_message(self, session: DeviceSession, envelope: Envelope) -> None:
        session.status.lastSeen = datetime.now(timezone.utc)
        if envelope.type == "capabilities":
            session.status.capabilities = CapabilityReport.model_validate(envelope.payload)
        elif envelope.type == "heartbeat":
            session.last_heartbeat_received = time.monotonic()
            sent = float(envelope.payload.get("echoMonotonic", session.last_heartbeat_sent))
            session.status.latencyMS = max(0, (time.monotonic() - sent) * 1000)
            session.status.battery = float(envelope.payload.get("battery", 0))
            session.status.lowPowerMode = bool(envelope.payload.get("lowPowerMode", False))
            session.status.thermalState = int(envelope.payload.get("thermalState", 0))
            session.status.state = str(envelope.payload.get("state", "ready"))
            session.status.storageFreeBytes = int(envelope.payload.get("storageFreeBytes", 0))
        elif envelope.type == "task_completed":
            result = CompanionResult.model_validate(envelope.payload)
            expected = hashlib.sha256(json.dumps(result.result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if expected != result.sha256:
                raise ValueError("result checksum mismatch")
            if future := session.pending_results.get(result.taskID):
                if not future.done(): future.set_result(result)
        elif envelope.type in {"task_failed", "task_cancelled"}:
            task_id = UUID(str(envelope.payload["taskID"]))
            if future := session.pending_results.get(task_id):
                if not future.done():
                    message = str(envelope.payload.get("message") or envelope.payload.get("reason"))
                    error = CompanionUserCancelled(message) if envelope.type == "task_cancelled" and envelope.payload.get("reason") == "user" else CompanionUnavailable(message)
                    future.set_exception(error)
        elif envelope.type == "client_draining":
            session.status.state = "draining"

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            for key, offer in list(self.pairing_offers.items()):
                if now - offer.created_monotonic > 60:
                    self.pairing_offers.pop(key, None)
                    await offer.websocket.close(code=1008, reason="Pairing offer expired")
            current_time = datetime.now(timezone.utc)
            for pairing_id, flow in list(self.pending_pairings.items()):
                if (current_time - flow.pending.createdAt).total_seconds() > 300:
                    self.pending_pairings.pop(pairing_id, None)
                    await flow.websocket.close(code=1008, reason="Pairing confirmation expired")
            for session in list(self.sessions.values()):
                if now - session.last_heartbeat_received > 15:
                    session.status.connected = False
                    session.status.authenticated = False
                    session.status.state = "suspended"
                    await session.websocket.close(code=1011, reason="Heartbeat timeout")
                    continue
                session.last_heartbeat_sent = now
                try:
                    await self._send(session.websocket, Envelope(type="heartbeat", payload={"echoMonotonic": now}), session.send_lock)
                    for task_id in list(session.pending_results):
                        await self._send(session.websocket, Envelope(type="lease_renew", payload={"taskID": str(task_id)}), session.send_lock)
                except Exception:
                    await session.websocket.close()

    async def _send_blob(self, session: DeviceSession, task_id: UUID, path: Path, remote_name: str) -> None:
        if not path.is_file():
            raise CompanionUnavailable(f"Media file is unavailable: {path.name}")
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024): digest.update(chunk)
        blob_id = uuid4()
        descriptor = {"blobID": str(blob_id), "taskID": str(task_id), "name": remote_name, "mimeType": mimetypes.guess_type(path.name)[0] or "application/octet-stream", "size": size, "sha256": digest.hexdigest()}
        await self._send(session.websocket, Envelope(type="blob_begin", payload=descriptor), session.send_lock)
        async with session.send_lock:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024): await session.websocket.send(chunk)
        await self._send(session.websocket, Envelope(type="blob_end", payload={"blobID": str(blob_id)}), session.send_lock)

    async def _cancel_all(self, session: DeviceSession, reason: str) -> None:
        for task_id, future in list(session.pending_results.items()):
            try: await self._send(session.websocket, Envelope(type="task_cancel", payload={"taskID": str(task_id), "reason": reason}), session.send_lock)
            except Exception: pass
            if not future.done(): future.set_exception(CompanionUnavailable("Task transferred to Mac fallback"))

    async def _send(self, websocket: ServerConnection, envelope: Envelope, lock: asyncio.Lock | None = None) -> None:
        payload = envelope.model_dump_json(exclude_none=True)
        if lock:
            async with lock: await websocket.send(payload)
        else:
            await websocket.send(payload)

    def _parse_envelope(self, raw: str | bytes) -> Envelope:
        if isinstance(raw, bytes): raise ValueError("text envelope required")
        envelope = Envelope.model_validate_json(raw)
        if envelope.protocolVersion != PROTOCOL_VERSION: raise ValueError("protocol version mismatch")
        now = datetime.now(timezone.utc)
        if abs((now - envelope.sentAt).total_seconds()) > 60: raise ValueError("stale envelope")
        cutoff = time.monotonic() - 120
        self._seen_messages = {key: seen for key, seen in self._seen_messages.items() if seen >= cutoff}
        if envelope.messageID in self._seen_messages: raise ValueError("replayed envelope")
        self._seen_messages[envelope.messageID] = time.monotonic()
        return envelope

    def _scored(self, status: DeviceStatus) -> DeviceStatus:
        score = 100 - status.queueDepth * 20 - status.latencyMS / 20
        score += max(0, min(20, status.battery * 20))
        score -= status.thermalState * 25
        if status.lowPowerMode or not status.connected: score -= 100
        status.score = round(score, 2)
        return status

    def _expire_idempotency(self, key: str, expected: asyncio.Future[CompanionResult]) -> None:
        if self._idempotent_results.get(key) is expected:
            self._idempotent_results.pop(key, None)

    def _desktop_id(self) -> UUID:
        path = self.root / "desktop-id"
        if path.exists(): return UUID(path.read_text().strip())
        value = uuid4()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(value))
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        return value

    def _save_devices(self) -> None:
        self._write_json(self.devices_path, [value.model_dump(mode="json") for value in self.paired.values()])

    @staticmethod
    def _bonjour_host_label() -> str:
        hostname = socket.gethostname().strip().rstrip(".")
        if hostname.casefold().endswith(".local"):
            hostname = hostname[:-6].rstrip(".")
        hostname = hostname.replace(".", "-").strip("-")
        return hostname or "unsloth-desktop"

    @staticmethod
    def _local_ipv4() -> str:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try: probe.connect(("192.0.2.1", 9)); return str(probe.getsockname()[0])
        except OSError: return "127.0.0.1"
        finally: probe.close()

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try: return json.loads(path.read_text())
        except (OSError, ValueError): return default

    def _read_model(self, path: Path, model: Any, default: Any) -> Any:
        try: return model.model_validate(self._read_json(path, {}))
        except Exception: return default

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def _write_model(self, path: Path, value: Any) -> None:
        self._write_json(path, value.model_dump(mode="json"))


companion_manager = CompanionManager()
