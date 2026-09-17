# SPDX-License-Identifier: AGPL-3.0-only
"""Move complete HF model repositories without downloading their weights again.

Only a verified destination is published. Cancellation or copy failures retain
all source files; an interrupted cleanup leaves an extra copy, never lost weights.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException

_CHUNK = 8 * 1024 * 1024
_jobs: dict[str, dict] = {}
_lock = threading.RLock()
_tasks: set[asyncio.Task] = set()
_ACTIVE = {"queued", "checking", "copying", "verifying", "finishing"}


class MoveCancelled(Exception):
    pass


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise MoveCancelled()


def _manifest(root: Path) -> dict:
    """Reject special files and escaping links; never copy unrelated user data.

    Studio may deduplicate completed Hugging Face blobs into ``<hub>/blobs``
    and leave ``<repo>/blobs/<etag>`` as a symlink to that shared store. Those
    links are part of the model even though their bytes live just outside the
    repository directory, so record them for materialization during a move.
    """
    result = {}
    shared_blobs = root.parent / "blobs"
    try:
        shared_info = shared_blobs.lstat()
        shared_root = (
            shared_blobs.resolve(strict = True)
            if stat.S_ISDIR(shared_info.st_mode) and not stat.S_ISLNK(shared_info.st_mode)
            else None
        )
    except (FileNotFoundError, OSError, RuntimeError):
        shared_root = None
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        relative = path.relative_to(root)
        if stat.S_ISLNK(info.st_mode):
            link_value = os.readlink(path)
            declared = Path(link_value)
            if not declared.is_absolute():
                declared = path.parent / declared
            declared = Path(os.path.abspath(os.path.normpath(os.fspath(declared))))
            resolved = path.resolve(strict = True)
            if declared.is_relative_to(root) and resolved.is_file():
                result[str(relative)] = (
                    "link",
                    link_value,
                    str(declared.relative_to(root)),
                )
            elif (
                relative.parts
                and relative.parts[0] == "blobs"
                and shared_root is not None
                and resolved.is_relative_to(shared_root)
                and resolved.is_file()
            ):
                target_info = resolved.stat()
                result[str(relative)] = (
                    "shared_blob",
                    link_value,
                    str(resolved),
                    target_info.st_size,
                    target_info.st_mtime_ns,
                    target_info.st_ino,
                )
            else:
                raise ValueError("The model contains a link outside its repository or to a folder.")
        elif stat.S_ISREG(info.st_mode):
            if path.name.endswith(".incomplete"):
                raise ValueError("Complete or remove the partial model download before moving it.")
            result[str(relative)] = ("file", info.st_size, info.st_mtime_ns, info.st_ino)
        elif stat.S_ISDIR(info.st_mode):
            result[str(relative)] = ("dir",)
        else:
            raise ValueError("The model contains an unsupported special file.")
    if not result:
        raise ValueError("The model folder is empty.")
    return result


def _digest(
    path: Path,
    cancel: threading.Event,
    progress = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            _check_cancel(cancel)
            digest.update(chunk)
            if progress:
                progress(len(chunk))
    return digest.hexdigest()


def _same_filesystem(source: Path, parent: Path) -> bool:
    return source.stat().st_dev == parent.stat().st_dev


def move_repository(
    source: Path,
    destination: Path,
    *,
    cancel: threading.Event,
    update,
    publish,
    guard = lambda: None,
) -> None:
    """Filesystem transaction, also used directly by the fault-injection tests."""
    source = source.resolve(strict = True)
    destination = destination.resolve(strict = False)
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Choose a different folder outside the model's current location.")
    if destination.exists() or destination.is_symlink():
        raise ValueError("This destination already contains the model. Choose another folder.")
    manifest = _manifest(source)
    total = sum(
        value[1] if value[0] == "file" else value[3]
        for value in manifest.values()
        if value[0] in {"file", "shared_blob"}
    )
    update(phase = "checking", total_bytes = total, completed_bytes = 0)
    _check_cancel(cancel)
    guard()
    # Rename on one filesystem is instantaneous and preserves links and metadata.
    if _same_filesystem(source, destination.parent) and not any(
        value[0] == "shared_blob"
        or (value[0] == "link" and os.path.isabs(value[1]))
        for value in manifest.values()
    ):
        update(phase = "finishing")
        publish()
        source.rename(destination)
        update(completed_bytes = total)
        return
    # Test symlink support before calculating required space. exFAT does not
    # support links; materialize snapshots there (and account for those bytes).
    probe = destination.parent / (".unsloth-link-test-" + uuid.uuid4().hex)
    supports_links = True
    try:
        probe.symlink_to("unsloth-link-test-target")
    except OSError:
        supports_links = False
    finally:
        if probe.is_symlink():
            probe.unlink()
    extra = (
        sum((source / value[2]).stat().st_size for value in manifest.values() if value[0] == "link")
        if not supports_links
        else 0
    )
    required = total + extra
    if shutil.disk_usage(destination.parent).free < required + 16 * 1024 * 1024:
        raise ValueError("Not enough free space on the destination disk.")
    stage = destination.parent / (".unsloth-move-" + uuid.uuid4().hex)
    stage.mkdir()
    completed = 0

    def advance(size):
        nonlocal completed
        completed += size
        update(completed_bytes = completed)

    try:
        update(phase = "copying", total_bytes = required * 2)
        hashes = {}
        for name, value in manifest.items():
            _check_cancel(cancel)
            target = stage / name
            target.parent.mkdir(parents = True, exist_ok = True)
            if value[0] == "dir":
                target.mkdir(exist_ok = True)
            elif value[0] in {"file", "shared_blob"} or (
                value[0] == "link" and not supports_links
            ):
                digest = hashlib.sha256()
                with (source / name).open("rb") as reader, target.open("xb") as writer:
                    while chunk := reader.read(_CHUNK):
                        _check_cancel(cancel)
                        writer.write(chunk)
                        digest.update(chunk)
                        advance(len(chunk))
                    writer.flush()
                    os.fsync(writer.fileno())
                hashes[name] = digest.hexdigest()
                shutil.copystat(source / name, target)
            else:
                # Rebase absolute as well as relative HF links inside the new repo.
                target.symlink_to(os.path.relpath(stage / value[2], target.parent))
        # Preserve revision ordering for loaders that choose snapshots by mtime.
        for name, value in reversed(list(manifest.items())):
            if value[0] == "dir":
                shutil.copystat(source / name, stage / name)
        update(phase = "verifying")
        for name, expected in hashes.items():
            if _digest(stage / name, cancel, advance) != expected:
                raise OSError("Copy verification failed. The original model was kept.")
        if _manifest(source) != manifest:
            raise OSError("The source model changed during the move. The original was kept.")
        _check_cancel(cancel)
        guard()
        # Record the external cache before publishing it; a crash after this
        # point can leave two discoverable copies, but cannot orphan the model.
        update(phase = "finishing")
        publish()
        if destination.exists() or destination.is_symlink():
            raise ValueError("The destination appeared during the move; nothing was overwritten.")
        stage.rename(destination)
        try:
            shutil.rmtree(source)
        except OSError:
            update(
                warning = "The model was copied and verified, but the original could not be fully removed. "
                f"You can remove the remaining copy at {source} after checking the destination."
            )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def status(repo_id: str, subject: str) -> dict | None:
    with _lock:
        job = _jobs.get(repo_id.casefold())
        if not job or job["subject"] != subject:
            return None
        return {key: value for key, value in job.items() if key not in {"cancel", "subject"}}


def cancel_move(repo_id: str, subject: str) -> dict | None:
    with _lock:
        job = _jobs.get(repo_id.casefold())
        if job and job["subject"] == subject and job["phase"] != "finishing":
            job["cancel"].set()
    return status(repo_id, subject)


def _assert_unloaded(repo_id: str):
    from hub.services.models import deletion

    if (
        deletion._llama_cpp_blocks_delete(repo_id, None)
        or deletion._inference_backend_blocks_delete(repo_id)
        or deletion._diffusion_blocks_delete(repo_id)
        or deletion._video_blocks_delete(repo_id)
    ):
        raise ValueError("Unload this model before moving it to another disk.")
    # Training may retain a base model lazily. Avoid initializing a trainer just
    # to query it; if it is running, don't move weights from under it.
    import sys

    module = sys.modules.get("core.training.training")
    backend = getattr(module, "_training_backend", None)
    if backend is not None and backend.is_training_active():
        raise ValueError("Wait for training to finish before moving models.")
    service = getattr(sys.modules.get("core.training.diffusion_training_service"), "_service", None)
    if service is not None and service.is_active():
        raise ValueError("Wait for training to finish before moving models.")


async def start_move(repo_id: str, folder: str, subject: str) -> dict:
    from hub.utils.paths import is_valid_repo_id
    from hub.services.models import downloads

    if not is_valid_repo_id(repo_id):
        raise HTTPException(400, "Invalid model repository ID")
    key = repo_id.casefold()
    with _lock:
        if any(job["phase"] in _ACTIVE for job in _jobs.values()):
            raise HTTPException(409, "A model move is already in progress.")
        owner = object()
        claimed, _ = downloads.registry.claim_repository_owner(repo_id, owner)
        if not claimed:
            raise HTTPException(
                409, "Finish or cancel the model download or update before moving it."
            )
        # Retain only the latest completed move per repo; bounded across repos.
        if len(_jobs) >= 32:
            _jobs.pop(next(iter(_jobs)))
        _jobs[key] = dict(
            repo_id = repo_id,
            phase = "queued",
            completed_bytes = 0,
            total_bytes = 0,
            destination = "",
            error = None,
            warning = None,
            cancel = threading.Event(),
            subject = subject,
        )
    task = asyncio.create_task(_run(repo_id, folder, subject, owner))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return status(repo_id, subject)


async def _run(repo_id: str, folder: str, subject: str, owner):
    from core.inference.llama_keepwarm import inference_lifecycle_gate
    from hub.services.models import downloads
    from hub.utils.hf_cache_state import iter_repo_cache_dirs
    from hub.utils.inventory_scan import invalidate_hf_cache_scans
    from utils.hf_cache_settings import _validate_cache_home, remember_model_storage_home
    from utils.model_storage_activity import relocation_reservation

    key = repo_id.casefold()

    def update(**values):
        with _lock:
            if values.get("phase") == "finishing" and _jobs[key]["cancel"].is_set():
                raise MoveCancelled()
            _jobs[key].update(values)

    def work():
        update(phase = "checking")
        _assert_unloaded(repo_id)
        sources = list(
            dict.fromkeys(path.resolve() for path in iter_repo_cache_dirs("model", repo_id))
        )
        if len(sources) != 1:
            raise ValueError("The model must have exactly one available cached copy to move it.")
        source = sources[0]
        selected = Path(folder).expanduser()
        if not selected.is_absolute() or not selected.is_dir():
            raise ValueError("Connect the destination disk and select an existing folder.")
        # Never place cache metadata directly at a volume root.
        home = _validate_cache_home(str(selected / "Unsloth Models"))
        destination = home / "hub" / source.name
        update(destination = str(destination))
        move_repository(
            source,
            destination,
            cancel = _jobs[key]["cancel"],
            update = update,
            publish = lambda: remember_model_storage_home(home, repo_id, destination, source),
            guard = lambda: _assert_unloaded(repo_id),
        )

    try:
        # Same gate used by manual loads, auto-switch and idle reload: no new
        # chat model can open the source while it is copied and retired.
        async with inference_lifecycle_gate():
            with relocation_reservation():
                worker = asyncio.create_task(asyncio.to_thread(work))
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Shutdown must not release the gate/reservation while the file
                    # worker still owns source files. Signal then wait for cleanup.
                    _jobs[key]["cancel"].set()
                    try:
                        await worker
                    except MoveCancelled:
                        pass
                    raise
        update(phase = "completed")
    except MoveCancelled:
        update(phase = "cancelled")
    except Exception as exc:
        update(phase = "failed", error = str(exc))
    finally:
        downloads.registry.release_repository_owner(repo_id, owner)
        invalidate_hf_cache_scans()
