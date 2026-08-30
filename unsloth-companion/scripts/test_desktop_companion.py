from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import ssl
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.exceptions import InvalidSignature
from jsonschema import Draft202012Validator, FormatChecker
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
BACKEND_ROOT = WORKSPACE_ROOT / "studio" / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

auth_package = types.ModuleType("auth")
auth_package.__path__ = []
auth_storage = types.ModuleType("auth.storage")
auth_storage.DB_PATH = Path(tempfile.gettempdir()) / "unsloth-companion-manager-tests" / "auth.db"
auth_package.storage = auth_storage
sys.modules["auth"] = auth_package
sys.modules["auth.storage"] = auth_storage

from core.companion import manager as manager_module
from core.companion.manager import CompanionManager, CompanionUnavailable, CompanionUserCancelled, DeviceSession
from core.companion.models import (
    CapabilityReport,
    CompanionResult,
    CompanionSettings,
    CompanionTask,
    ConnectionMode,
    DeviceStatus,
    Envelope,
    PairedDevice,
)
from core.companion.security import DesktopIdentity


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.closed: list[tuple[int, str]] = []

    async def send(self, value: str | bytes) -> None:
        self.sent.append(value)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))


def make_manager(root: Path) -> CompanionManager:
    value = CompanionManager()
    value.root = root
    value.settings_path = root / "settings.json"
    value.devices_path = root / "paired-devices.json"
    value.identity = DesktopIdentity(root)
    value.settings = CompanionSettings()
    return value


def make_session(name: str, battery: float, latency: float, *, state: str = "ready") -> DeviceSession:
    device_id = uuid4()
    device_key = ec.generate_private_key(ec.SECP256R1())
    signing_public_key = base64.b64encode(device_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)).decode()
    paired = PairedDevice(
        deviceID=device_id,
        name=name,
        signingPublicKey=signing_public_key,
        pairedAt=datetime.now(timezone.utc),
    )
    capabilities = CapabilityReport(
        protocolVersion=1,
        deviceID=device_id,
        deviceName=name,
        runtimeVersion="llama.cpp-3173a564",
        taskKinds={"summary", "context_compression"},
        selectedModelID="fast-e2b",
        supportsVision=True,
        supportsAudio=True,
        contextSize=8192,
        maxRawMediaBytes=2_147_483_648,
    )
    status = DeviceStatus(
        deviceID=device_id,
        name=name,
        connected=True,
        authenticated=True,
        battery=battery,
        latencyMS=latency,
        state=state,
        storageFreeBytes=8_000_000_000,
        capabilities=capabilities,
    )
    return DeviceSession(device_id=device_id, websocket=FakeWebSocket(), paired=paired, status=status)


def make_task(*, shard_count: int | None = None, key: str | None = None) -> CompanionTask:
    return CompanionTask(
        taskID=uuid4(),
        parentTaskID=uuid4() if shard_count else None,
        shardIndex=0 if shard_count else None,
        shardCount=shard_count,
        idempotencyKey=key or f"summary-{uuid4().hex}",
        kind="summary",
        priority=50,
        timeoutSeconds=2,
        leaseSeconds=30,
        input={"text": "A local test input."},
        resultSchema={"type": "object"},
    )


def make_result(task: CompanionTask, text: str = "done") -> CompanionResult:
    payload = {"summary": text}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return CompanionResult(taskID=task.taskID, result=payload, sha256=digest, durationMS=10, tokensGenerated=2)


class ProtocolContractTests(unittest.TestCase):
    def test_schema_fixture_and_swift_message_types_match(self) -> None:
        schema = json.loads((PACKAGE_ROOT / "protocol" / "schema-v1.json").read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        fixture = json.loads((PACKAGE_ROOT / "protocol" / "fixtures" / "task-summary.json").read_text())
        validator.validate(fixture)
        envelope = Envelope.model_validate(fixture)
        task = CompanionTask.model_validate(envelope.payload)
        self.assertEqual(task.kind, "summary")
        self.assertEqual(task.leaseSeconds, 30)

        invalid = json.loads(json.dumps(fixture))
        invalid["payload"]["unexpected"] = True
        self.assertTrue(list(validator.iter_errors(invalid)))

        swift = (PACKAGE_ROOT / "Unsloth Companion" / "Unsloth Companion" / "Core" / "ProtocolModels.swift").read_text()
        body = re.search(r"enum CompanionMessageType:.*?\{(.*?)\n\}", swift, re.S)
        self.assertIsNotNone(body)
        swift_types: set[str] = set()
        for case_line in re.findall(r"case ([^\n]+)", body.group(1)):
            for item in case_line.split(","):
                match = re.fullmatch(r'\s*([A-Za-z0-9_]+)(?:\s*=\s*"([^"]+)")?\s*', item)
                self.assertIsNotNone(match)
                swift_types.add(match.group(2) or match.group(1))
        self.assertEqual(swift_types, set(schema["properties"]["type"]["enum"]))

    def test_p256_signatures_sas_and_private_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity = DesktopIdentity(Path(directory))
            device_key = ec.generate_private_key(ec.SECP256R1())
            device_public = base64.b64encode(device_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)).decode()
            nonce = bytes(range(32))
            signature = base64.b64encode(device_key.sign(nonce, ec.ECDSA(hashes.SHA256()))).decode()
            DesktopIdentity.verify_base64(device_public, signature, nonce)
            with self.assertRaises(InvalidSignature):
                DesktopIdentity.verify_base64(device_public, signature, nonce + b"changed")
            code = identity.pairing_code(identity.public_key_base64, device_public, nonce, identity.certificate_sha256)
            self.assertRegex(code, r"^[0-9]{6}$")
            self.assertEqual(code, identity.pairing_code(identity.public_key_base64, device_public, nonce, identity.certificate_sha256))
            self.assertEqual(identity.key_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(identity.cert_path.stat().st_mode & 0o777, 0o600)
            r, s = decode_dss_signature(base64.b64decode(signature))
            self.assertGreater(r, 0)
            self.assertGreater(s, 0)


class CompanionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.manager = make_manager(Path(self.temporary.name))

    async def asyncTearDown(self) -> None:
        for session in self.manager.sessions.values():
            for future in session.pending_results.values():
                if not future.done():
                    future.cancel()
        self.manager.sessions.clear()
        await self.manager.stop()
        self.temporary.cleanup()

    def test_bonjour_host_label_is_a_single_local_dns_label(self) -> None:
        with patch.object(manager_module.socket, "gethostname", return_value="MacBook-Pro.local"):
            self.assertEqual(self.manager._bonjour_host_label(), "MacBook-Pro")
        with patch.object(manager_module.socket, "gethostname", return_value="studio.example.com."):
            self.assertEqual(self.manager._bonjour_host_label(), "studio-example-com")
        with patch.object(manager_module.socket, "gethostname", return_value=".local"):
            self.assertEqual(self.manager._bonjour_host_label(), "unsloth-desktop")

    async def test_full_pairing_requires_matching_signed_offer_and_both_confirmations(self) -> None:
        websocket = FakeWebSocket()
        device_id = uuid4()
        device_key = ec.generate_private_key(ec.SECP256R1())
        public_key = base64.b64encode(device_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)).decode()

        await self.manager._begin_pairing(websocket, device_id, {"devicePublicKey": public_key})
        offer = json.loads(websocket.sent[-1])["payload"]
        nonce = base64.b64decode(offer["nonce"])
        signature = base64.b64encode(device_key.sign(nonce, ec.ECDSA(hashes.SHA256()))).decode()
        preview = Envelope(type="pairing_confirm", payload={
            "preview": True, "confirmed": False, "deviceID": str(device_id), "deviceName": "Test iPhone",
            "devicePublicKey": public_key, "nonce": offer["nonce"], "signature": signature,
        })
        await self.manager._handle_pairing_message(websocket, preview)
        flow = next(iter(self.manager.pending_pairings.values()))
        self.assertRegex(flow.pending.code, r"^[0-9]{6}$")
        await self.manager.confirm_pairing(flow.pending.pairingID)
        self.assertNotIn(device_id, self.manager.paired)

        final = Envelope(type="pairing_confirm", payload={
            "desktopID": str(self.manager._desktop_id()), "deviceID": str(device_id), "deviceName": "Test iPhone",
            "devicePublicKey": public_key, "signature": signature, "confirmed": True,
        })
        await self.manager._handle_pairing_message(websocket, final)
        self.assertIn(device_id, self.manager.paired)
        self.assertTrue(self.manager.sessions[device_id].status.authenticated)
        self.assertTrue(json.loads(websocket.sent[-1])["payload"]["accepted"])

    async def test_real_wss_pairing_capabilities_and_task_result(self) -> None:
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(self.manager.identity.cert_path, self.manager.identity.key_path)
        self.manager.server = await serve(
            self.manager._handler,
            "127.0.0.1",
            0,
            ssl=server_context,
            max_size=4 * 1024 * 1024,
            compression=None,
            ping_interval=None,
        )
        listener_socket = next(iter(self.manager.server.sockets))
        self.manager.listener_port = int(listener_socket.getsockname()[1])

        client_context = ssl.create_default_context()
        client_context.check_hostname = False
        client_context.verify_mode = ssl.CERT_NONE
        device_id = uuid4()
        device_key = ec.generate_private_key(ec.SECP256R1())
        device_public_key = base64.b64encode(
            device_key.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint,
            )
        ).decode()

        async with connect(
            f"wss://127.0.0.1:{self.manager.listener_port}",
            ssl=client_context,
            max_size=4 * 1024 * 1024,
            compression=None,
            ping_interval=None,
        ) as websocket:
            ssl_object = websocket.transport.get_extra_info("ssl_object")
            certificate = ssl_object.getpeercert(binary_form=True)
            self.assertEqual(hashlib.sha256(certificate).hexdigest(), self.manager.identity.certificate_sha256)

            await websocket.send(Envelope(type="hello", payload={
                "deviceID": str(device_id),
                "deviceName": "Loopback iPhone",
                "devicePublicKey": device_public_key,
            }).model_dump_json())
            offer = Envelope.model_validate_json(await websocket.recv())
            self.assertEqual(offer.type, "pairing_offer")
            self.assertEqual(offer.payload["certificateSHA256"], self.manager.identity.certificate_sha256)
            nonce = base64.b64decode(offer.payload["nonce"])
            signature = base64.b64encode(device_key.sign(nonce, ec.ECDSA(hashes.SHA256()))).decode()
            expected_code = DesktopIdentity.pairing_code(
                offer.payload["signingPublicKey"],
                device_public_key,
                nonce,
                offer.payload["certificateSHA256"],
            )
            await websocket.send(Envelope(type="pairing_confirm", payload={
                "preview": True,
                "confirmed": False,
                "deviceID": str(device_id),
                "deviceName": "Loopback iPhone",
                "devicePublicKey": device_public_key,
                "nonce": offer.payload["nonce"],
                "signature": signature,
            }).model_dump_json())
            for _ in range(100):
                if self.manager.pending_pairings:
                    break
                await asyncio.sleep(0.01)
            flow = next(iter(self.manager.pending_pairings.values()))
            self.assertEqual(flow.pending.code, expected_code)
            await self.manager.confirm_pairing(flow.pending.pairingID)
            await websocket.send(Envelope(type="pairing_confirm", payload={
                "desktopID": str(self.manager._desktop_id()),
                "deviceID": str(device_id),
                "deviceName": "Loopback iPhone",
                "devicePublicKey": device_public_key,
                "signature": signature,
                "confirmed": True,
            }).model_dump_json())
            pairing_result = Envelope.model_validate_json(await websocket.recv())
            self.assertEqual(pairing_result.type, "pairing_result")
            self.assertTrue(pairing_result.payload["accepted"])

            capabilities = CapabilityReport(
                protocolVersion=1,
                deviceID=device_id,
                deviceName="Loopback iPhone",
                runtimeVersion="llama.cpp-3173a564",
                taskKinds={"summary"},
                selectedModelID="fast-e2b",
                supportsVision=False,
                supportsAudio=False,
                contextSize=8192,
                maxRawMediaBytes=2_147_483_648,
            )
            await websocket.send(Envelope(type="capabilities", payload=capabilities.model_dump(mode="json")).model_dump_json())
            await websocket.send(Envelope(type="heartbeat", payload={
                "echoMonotonic": time.monotonic(),
                "battery": 0.85,
                "lowPowerMode": False,
                "thermalState": 0,
                "state": "ready",
                "storageFreeBytes": 8_000_000_000,
            }).model_dump_json())
            for _ in range(100):
                session = self.manager.sessions.get(device_id)
                if session and session.status.state == "ready" and session.status.capabilities:
                    break
                await asyncio.sleep(0.01)

            task = make_task()
            pending_dispatch = asyncio.create_task(self.manager.dispatch(task))
            submitted = Envelope.model_validate_json(await websocket.recv())
            self.assertEqual(submitted.type, "task_submit")
            self.assertEqual(UUID(submitted.payload["taskID"]), task.taskID)
            result_payload = {"summary": "WSS loopback result"}
            digest = hashlib.sha256(json.dumps(result_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            await websocket.send(Envelope(type="task_completed", payload={
                "taskID": str(task.taskID),
                "result": result_payload,
                "sha256": digest,
                "durationMS": 12,
                "tokensGenerated": 3,
            }).model_dump_json())
            result = await pending_dispatch
            self.assertEqual(result.result, result_payload)
            self.assertEqual(self.manager.status().activeTasks, [])

    async def test_automatic_best_multi_selection_and_draining_exclusion(self) -> None:
        fast = make_session("Fast", 0.9, 10)
        slow = make_session("Slow", 0.5, 150)
        draining = make_session("Draining", 1.0, 1, state="draining")
        self.manager.sessions = {value.device_id: value for value in (fast, slow, draining)}
        self.assertIs(self.manager._select_session("summary"), fast)

        self.manager.settings = CompanionSettings(mode=ConnectionMode.multiple, selectedDeviceIDs=[slow.device_id])
        self.assertIs(self.manager._select_session("summary"), slow)
        self.manager.settings = CompanionSettings(mode=ConnectionMode.multiple, selectedDeviceIDs=[])
        self.assertIsNone(self.manager._select_session("summary"))
        self.manager.settings = CompanionSettings()
        fast.status.lowPowerMode = True
        self.assertIs(self.manager._select_session("summary"), slow)
        slow.status.battery = 0.09
        self.assertIsNone(self.manager._select_session("summary"))

    async def test_global_switch_stops_without_deadlock(self) -> None:
        result = await asyncio.wait_for(
            self.manager.update_settings(CompanionSettings(enabled=False)),
            timeout=1,
        )
        self.assertFalse(result.settings.enabled)
        self.assertIsNone(self.manager.listener_port)

    async def test_duplicate_idempotency_key_shares_one_execution(self) -> None:
        session = make_session("Only", 0.8, 10)
        self.manager.sessions[session.device_id] = session
        release = asyncio.Event()
        calls = 0

        async def dispatch_once(selected: DeviceSession, task: CompanionTask) -> CompanionResult:
            nonlocal calls
            calls += 1
            await release.wait()
            return make_result(task)

        self.manager._dispatch_once = dispatch_once
        key = f"shared-{uuid4().hex}"
        first_task = make_task(key=key)
        second_task = first_task.model_copy(update={"taskID": uuid4()})
        first = asyncio.create_task(self.manager.dispatch(first_task))
        await asyncio.sleep(0)
        second = asyncio.create_task(self.manager.dispatch(second_task))
        await asyncio.sleep(0)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        self.assertEqual(calls, 1)
        self.assertEqual(first_result, second_result)

    async def test_background_job_returns_immediately_and_is_collected_later(self) -> None:
        session = make_session("Async", 0.8, 10)
        self.manager.sessions[session.device_id] = session
        started = asyncio.Event()
        release = asyncio.Event()

        async def dispatch_once(selected: DeviceSession, task: CompanionTask) -> CompanionResult:
            self.assertIs(selected, session)
            started.set()
            await release.wait()
            return make_result(task, "background result")

        self.manager._dispatch_once = dispatch_once
        task = make_task()
        media_root = Path(self.temporary.name) / "owned-media"
        media_root.mkdir()
        (media_root / "input.bin").write_bytes(b"test")

        self.manager._event_loop = asyncio.get_running_loop()
        submitted = await asyncio.wait_for(
            asyncio.to_thread(
                self.manager.submit_background_sync,
                [task],
                item_ids=["only"],
                thread_id="chat-a",
                cleanup_paths=[media_root],
            ),
            timeout=1,
        )
        self.assertIn(submitted["state"], {"queued", "running"})
        self.assertEqual(submitted["taskIDs"], [str(task.taskID)])
        await asyncio.wait_for(started.wait(), timeout=1)
        self.assertFalse(release.is_set(), "submission must not wait for iPhone inference")
        self.assertEqual(self.manager.background_counts("chat-a")["pending"], 1)

        with self.assertRaisesRegex(CompanionUnavailable, "this chat"):
            await self.manager.collect_background(
                thread_id="chat-b",
                job_id=UUID(submitted["jobID"]),
            )

        runner = self.manager._background_jobs[UUID(submitted["jobID"])].runner
        self.assertIsNotNone(runner)
        release.set()
        await asyncio.wait_for(runner, timeout=1)
        self.assertFalse(media_root.exists())
        self.assertEqual(self.manager.background_counts("chat-a")["ready"], 1)

        collected = await asyncio.to_thread(
            self.manager.collect_background_sync,
            thread_id="chat-a",
            job_id=UUID(submitted["jobID"]),
        )
        job = collected["jobs"][0]
        self.assertEqual(job["state"], "completed")
        self.assertEqual(job["result"]["result"], {"summary": "background result"})
        self.assertTrue(job["collected"])
        self.assertEqual(self.manager.background_counts("chat-a")["ready"], 0)

    async def test_background_failure_stays_available_for_mac_fallback(self) -> None:
        session = make_session("Failing", 0.8, 10)
        self.manager.sessions[session.device_id] = session

        async def dispatch_once(_: DeviceSession, __: CompanionTask) -> CompanionResult:
            raise CompanionUnavailable("phone went away")

        self.manager._dispatch_once = dispatch_once
        submitted = await self.manager.submit_background(
            [make_task()],
            item_ids=["failed"],
            thread_id="chat-failure",
        )
        runner = self.manager._background_jobs[UUID(submitted["jobID"])].runner
        if runner is not None:
            await asyncio.wait_for(runner, timeout=1)

        collected = await self.manager.collect_background(thread_id="chat-failure")
        self.assertEqual(collected["jobs"][0]["state"], "failed")
        self.assertIn("phone went away", collected["jobs"][0]["result"]["error"])

    async def test_only_divisible_shard_retries_another_iphone(self) -> None:
        first = make_session("First", 0.9, 1)
        second = make_session("Second", 0.8, 20)
        self.manager.sessions = {first.device_id: first, second.device_id: second}
        visited: list[UUID] = []

        async def dispatch_once(session: DeviceSession, task: CompanionTask) -> CompanionResult:
            visited.append(session.device_id)
            if len(visited) == 1:
                raise CompanionUnavailable("connection lost")
            return make_result(task)

        self.manager._dispatch_once = dispatch_once
        task = make_task(shard_count=2)
        self.assertEqual((await self.manager.dispatch(task)).taskID, task.taskID)
        self.assertEqual(visited, [first.device_id, second.device_id])

        visited.clear()
        with self.assertRaises(CompanionUnavailable):
            await self.manager.dispatch(make_task())
        self.assertEqual(visited, [first.device_id])

        visited.clear()
        async def user_cancelled(session: DeviceSession, task: CompanionTask) -> CompanionResult:
            visited.append(session.device_id)
            raise CompanionUserCancelled("user")
        self.manager._dispatch_once = user_cancelled
        with self.assertRaises(CompanionUserCancelled):
            await self.manager.dispatch(make_task(shard_count=2))
        self.assertEqual(visited, [first.device_id])

    async def test_explicit_cancel_and_late_result_are_not_reassigned(self) -> None:
        session = make_session("Only", 0.8, 10)
        task = make_task()
        future = asyncio.get_running_loop().create_future()
        session.pending_results[task.taskID] = future
        self.manager.sessions[session.device_id] = session
        await self.manager.cancel(task.taskID, explicit_user=True)
        with self.assertRaises(CompanionUserCancelled):
            await future
        cancel = json.loads(session.websocket.sent[-1])
        self.assertEqual(cancel["type"], "task_cancel")
        self.assertTrue(cancel["payload"]["explicitUser"])

        session.pending_results.pop(task.taskID)
        result = make_result(task, "late")
        await self.manager._handle_session_message(session, Envelope(type="task_completed", payload=result.model_dump(mode="json")))
        self.assertNotIn(task.taskID, session.pending_results)

    async def test_running_task_is_exposed_in_status_until_completion(self) -> None:
        session = make_session("Visible", 0.8, 10)
        self.manager.sessions[session.device_id] = session
        task = make_task()
        dispatch = asyncio.create_task(self.manager.dispatch(task))
        for _ in range(100):
            if session.pending_results.get(task.taskID):
                break
            await asyncio.sleep(0.01)
        active = self.manager.status().activeTasks
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].taskID, task.taskID)
        session.pending_results[task.taskID].set_result(make_result(task))
        await dispatch
        self.assertEqual(self.manager.status().activeTasks, [])

    async def test_heartbeat_renews_active_lease_and_expires_stale_session(self) -> None:
        session = make_session("Lease", 0.8, 10)
        task_id = uuid4()
        session.pending_results[task_id] = asyncio.get_running_loop().create_future()
        self.manager.sessions[session.device_id] = session
        sleep_calls = 0

        async def one_iteration(_: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError

        with patch.object(manager_module.asyncio, "sleep", new=one_iteration):
            with self.assertRaises(asyncio.CancelledError):
                await self.manager._heartbeat_loop()
        sent_types = [json.loads(value)["type"] for value in session.websocket.sent if isinstance(value, str)]
        self.assertEqual(sent_types, ["heartbeat", "lease_renew"])

        session.websocket.sent.clear()
        session.last_heartbeat_received = time.monotonic() - 16
        sleep_calls = 0
        with patch.object(manager_module.asyncio, "sleep", new=one_iteration):
            with self.assertRaises(asyncio.CancelledError):
                await self.manager._heartbeat_loop()
        self.assertEqual(session.websocket.closed[-1], (1011, "Heartbeat timeout"))

    async def test_replay_stale_envelope_and_bad_result_checksum_are_rejected(self) -> None:
        envelope = Envelope(type="heartbeat", payload={"echoMonotonic": 1})
        raw = envelope.model_dump_json()
        self.manager._parse_envelope(raw)
        with self.assertRaisesRegex(ValueError, "replayed"):
            self.manager._parse_envelope(raw)

        stale = Envelope(sentAt=datetime.now(timezone.utc) - timedelta(seconds=61), type="heartbeat", payload={"echoMonotonic": 1})
        with self.assertRaisesRegex(ValueError, "stale"):
            self.manager._parse_envelope(stale.model_dump_json())

        session = make_session("Only", 0.8, 10)
        task = make_task()
        session.pending_results[task.taskID] = asyncio.get_running_loop().create_future()
        invalid = make_result(task).model_copy(update={"sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "checksum"):
            await self.manager._handle_session_message(session, Envelope(type="task_completed", payload=invalid.model_dump(mode="json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
