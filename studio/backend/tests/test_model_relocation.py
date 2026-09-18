# SPDX-License-Identifier: AGPL-3.0-only
"""Real filesystem transactions with injected disk/copy/persistence failures."""

import asyncio
import functools
import shutil
import threading
from pathlib import Path

import pytest

from hub.services.models import relocation as move


def _async_test(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapped


@pytest.fixture
def model(tmp_path):
    source = tmp_path / "source" / "models--org--model"
    (source / "blobs").mkdir(parents = True)
    (source / "blobs" / "weights").write_bytes(b"weights" * 4096)
    snapshot = source / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model.gguf").symlink_to("../../blobs/weights")
    (source / "refs").mkdir()
    (source / "refs" / "main").write_text("revision")
    parent = tmp_path / "destination" / "hub"
    parent.mkdir(parents = True)
    return source, parent / source.name


def run(
    model,
    monkeypatch,
    *,
    cross = True,
    publish = lambda: None,
    finalize = lambda: None,
    update = lambda **kw: None,
    cancel = None,
    guard = lambda: None,
):
    monkeypatch.setattr(move, "_same_filesystem", lambda *_: not cross)
    return move.move_repository(
        *model,
        cancel = cancel or threading.Event(),
        update = update,
        publish = publish,
        finalize = finalize,
        guard = guard,
    )


def test_cross_disk_verifies_preserves_links_and_removes_source(model, monkeypatch):
    source, dest = model
    updates = []
    published = []
    run(
        model,
        monkeypatch,
        update = lambda **kw: updates.append(kw),
        publish = lambda: published.append(True),
    )
    assert not source.exists()
    assert (dest / "snapshots/revision/model.gguf").read_bytes() == b"weights" * 4096
    assert (dest / "snapshots/revision/model.gguf").is_symlink()
    assert published == [True]
    assert "verifying" in [u.get("phase") for u in updates]
    assert not list(dest.parent.glob(".unsloth-*"))


def test_finalize_runs_after_verified_destination_is_published(model, monkeypatch):
    source, destination = model
    finalized = []

    def finalize():
        assert destination.is_dir()
        assert source.is_dir()
        finalized.append(True)

    run(model, monkeypatch, finalize = finalize)
    assert finalized == [True]
    assert destination.is_dir()
    assert not source.exists()


def test_failed_verification_never_finalizes_restore_metadata(model, monkeypatch):
    finalized = []
    monkeypatch.setattr(move, "_digest", lambda *args, **kwargs: "not-the-copy-digest")
    with pytest.raises(OSError, match = "verification failed"):
        run(model, monkeypatch, finalize = lambda: finalized.append(True))
    assert finalized == []
    assert model[0].is_dir()
    assert not model[1].exists()


def test_restore_accepts_verified_existing_original_copy(model, monkeypatch):
    source, destination = model
    shutil.copytree(source, destination, symlinks = True)
    finalized = []
    updates = []
    move.restore_repository(
        source,
        destination,
        cancel = threading.Event(),
        update = lambda **kw: updates.append(kw),
        finalize = lambda: finalized.append(True),
    )
    assert finalized == [True]
    assert destination.is_dir()
    assert not source.exists()
    assert "verifying" in [entry.get("phase") for entry in updates]


def test_restore_refuses_different_existing_original_copy(model, monkeypatch):
    source, destination = model
    shutil.copytree(source, destination, symlinks = True)
    (destination / "blobs" / "weights").write_bytes(b"different")
    finalized = []
    with pytest.raises(ValueError, match = "different or incomplete copy"):
        move.restore_repository(
            source,
            destination,
            cancel = threading.Event(),
            update = lambda **kw: None,
            finalize = lambda: finalized.append(True),
        )
    assert finalized == []
    assert source.is_dir()
    assert destination.is_dir()


def test_same_disk_rename_does_not_copy(model, monkeypatch):
    monkeypatch.setattr(
        move, "_digest", lambda *a: pytest.fail("same-disk move must not hash/copy")
    )
    run(model, monkeypatch, cross = False)
    assert not model[0].exists()
    assert (model[1] / "snapshots/revision/model.gguf").is_file()


def test_absolute_links_are_rebased(model, monkeypatch):
    source, dest = model
    link = source / "snapshots/revision/model.gguf"
    link.unlink()
    link.symlink_to(source / "blobs/weights")
    run(model, monkeypatch, cross = False)
    assert (dest / "snapshots/revision/model.gguf").read_bytes() == b"weights" * 4096


def test_cancel_keeps_original_and_cleans_stage(model, monkeypatch):
    cancelled = threading.Event()

    def update(**values):
        if values.get("phase") == "copying":
            cancelled.set()

    with pytest.raises(move.MoveCancelled):
        run(model, monkeypatch, cancel = cancelled, update = update)
    assert model[0].is_dir() and not model[1].exists()
    assert not list(model[1].parent.glob(".unsloth-*"))


def test_verification_failure_keeps_original(model, monkeypatch):
    monkeypatch.setattr(move, "_digest", lambda *a: "corrupted")
    with pytest.raises(OSError, match = "verification failed"):
        run(model, monkeypatch)
    assert model[0].is_dir() and not model[1].exists()


@pytest.mark.parametrize("cross", [False, True])
def test_persistence_failure_keeps_original(model, monkeypatch, cross):
    def fail():
        raise OSError("database unavailable")

    with pytest.raises(OSError, match = "database"):
        run(model, monkeypatch, cross = cross, publish = fail)
    assert model[0].is_dir() and not model[1].exists()


def test_existing_destination_is_never_overwritten(model, monkeypatch):
    model[1].mkdir()
    (model[1] / "owned").write_text("keep")
    with pytest.raises(ValueError, match = "already contains"):
        run(model, monkeypatch)
    assert (model[1] / "owned").read_text() == "keep"
    assert model[0].is_dir()


def test_no_space_keeps_source(model, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(move.shutil, "disk_usage", lambda _: usage(1, 1, 0))
    with pytest.raises(ValueError, match = "free space"):
        run(model, monkeypatch)
    assert model[0].is_dir()


def test_escaping_link_is_rejected(model, monkeypatch, tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("private")
    (model[0] / "blobs" / "escape").symlink_to(secret)
    with pytest.raises(ValueError, match = "outside"):
        run(model, monkeypatch)
    assert secret.read_text() == "private"


@pytest.mark.parametrize("cross", [False, True])
def test_shared_hub_blob_is_materialized_on_move(model, monkeypatch, cross):
    source, dest = model
    xet_hash = "a" * 64
    shared_root = source.parent / "blobs"
    (shared_root / ".huggingface-shared-blobs").parent.mkdir(parents = True, exist_ok = True)
    (shared_root / ".huggingface-shared-blobs").write_text("1\n")
    shared = shared_root / xet_hash[:2] / xet_hash
    shared.parent.mkdir(parents = True)
    shared.write_bytes(b"shared" * 4096)
    manifest = shared.with_name(f"{xet_hash}.refs")
    manifest.write_text(f"{source.name}/blobs/weights\n")
    repo_blob = source / "blobs" / "weights"
    repo_blob.unlink()
    repo_blob.symlink_to(shared)

    run(model, monkeypatch, cross = cross)

    moved_blob = dest / "blobs" / "weights"
    assert moved_blob.read_bytes() == b"shared" * 4096
    assert moved_blob.is_file() and not moved_blob.is_symlink()
    assert (dest / "snapshots/revision/model.gguf").is_symlink()
    assert (dest / "snapshots/revision/model.gguf").read_bytes() == b"shared" * 4096
    assert not shared.exists()
    assert not manifest.exists()


def test_shared_blob_gc_runs_only_after_source_removal(model, monkeypatch):
    source, _dest = model
    shared = source.parent / "blobs" / "aa" / "shared-weight"
    shared.parent.mkdir(parents = True)
    shared.write_bytes(b"shared" * 4096)
    repo_blob = source / "blobs" / "weights"
    repo_blob.unlink()
    repo_blob.symlink_to(shared)
    swept = []

    monkeypatch.setattr(
        move,
        "_sweep_materialized_shared_blobs",
        lambda manifest, cache_dir: swept.append((manifest, cache_dir)) or (0, False),
    )
    run(model, monkeypatch)

    assert not source.exists()
    assert len(swept) == 1
    assert swept[0][1] == source.parent


def test_shared_blob_gc_retains_blob_used_by_another_repo(model, monkeypatch):
    source, dest = model
    xet_hash = "b" * 64
    shared_root = source.parent / "blobs"
    shared_root.mkdir(parents = True, exist_ok = True)
    (shared_root / ".huggingface-shared-blobs").write_text("1\n")
    shared = shared_root / xet_hash[:2] / xet_hash
    shared.parent.mkdir(parents = True)
    shared.write_bytes(b"shared" * 4096)

    repo_blob = source / "blobs" / "weights"
    repo_blob.unlink()
    repo_blob.symlink_to(shared)

    other_blob = source.parent / "models--org--other" / "blobs" / "weights"
    other_blob.parent.mkdir(parents = True)
    other_blob.symlink_to(shared)
    manifest = shared.with_name(f"{xet_hash}.refs")
    manifest.write_text(
        f"{source.name}/blobs/weights\n"
        "models--org--other/blobs/weights\n"
    )

    run(model, monkeypatch)

    assert not source.exists()
    assert (dest / "blobs/weights").is_file()
    assert shared.is_file()
    assert other_blob.read_bytes() == b"shared" * 4096
    assert manifest.read_text() == "models--org--other/blobs/weights\n"


def test_shared_blob_gc_is_skipped_when_source_cleanup_fails(model, monkeypatch):
    source, _dest = model
    shared = source.parent / "blobs" / "aa" / "shared-weight"
    shared.parent.mkdir(parents = True)
    shared.write_bytes(b"shared" * 4096)
    repo_blob = source / "blobs" / "weights"
    repo_blob.unlink()
    repo_blob.symlink_to(shared)
    original = move.shutil.rmtree

    def fail_source(path, *args, **kwargs):
        if path == source:
            raise PermissionError("read only")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(move.shutil, "rmtree", fail_source)
    monkeypatch.setattr(
        move,
        "_sweep_materialized_shared_blobs",
        lambda *_: pytest.fail("shared blob GC must wait for source removal"),
    )
    run(model, monkeypatch)


def test_exfat_materializes_links(model, monkeypatch):
    original = Path.symlink_to

    def unsupported(self, target, **kwargs):
        if self.name.startswith(".unsloth-link-test"):
            raise OSError("links unsupported")
        return original(self, target, **kwargs)

    monkeypatch.setattr(Path, "symlink_to", unsupported)
    run(model, monkeypatch)
    assert (model[1] / "snapshots/revision/model.gguf").read_bytes() == b"weights" * 4096
    assert not (model[1] / "snapshots/revision/model.gguf").is_symlink()


def test_changed_source_is_not_removed(model, monkeypatch):
    def update(**values):
        if values.get("phase") == "verifying":
            (model[0] / "new-file").write_text("new")

    with pytest.raises(OSError, match = "source model changed"):
        run(model, monkeypatch, update = update)
    assert (model[0] / "new-file").read_text() == "new"


def test_cleanup_failure_keeps_verified_destination(model, monkeypatch):
    original = move.shutil.rmtree

    def fail_source(path, *args, **kwargs):
        if path == model[0]:
            raise PermissionError("read only")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(move.shutil, "rmtree", fail_source)
    updates = []
    run(model, monkeypatch, update = lambda **kw: updates.append(kw))
    assert model[0].is_dir() and model[1].is_dir()
    assert any("warning" in value for value in updates)


def test_loaded_guard_prevents_any_copy(model, monkeypatch):
    def guard():
        raise ValueError("Unload the model")

    with pytest.raises(ValueError, match = "Unload"):
        run(model, monkeypatch, guard = guard)
    assert model[0].is_dir() and not model[1].exists()


def test_move_blocks_new_media_loads_and_training_starts():
    from fastapi import HTTPException
    from utils.model_storage_activity import model_file_operation, relocation_reservation

    calls = []

    @model_file_operation
    def load():
        calls.append(True)

    with relocation_reservation():
        with pytest.raises(HTTPException, match = "move to finish"):
            load()
    load()
    assert calls == [True]


def test_loading_model_blocks_move_and_exception_releases_reservation():
    from fastapi import HTTPException
    from utils.model_storage_activity import model_file_operation, relocation_reservation

    @model_file_operation
    def load():
        with pytest.raises(HTTPException, match = "current model load"):
            with relocation_reservation():
                pytest.fail("move started during load")
        raise ValueError("load failed")

    with pytest.raises(ValueError):
        load()
    with relocation_reservation():
        pass


def test_job_status_and_cancel_are_subject_scoped(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(
        move, "_jobs", {"org/model": {"subject": "owner", "phase": "copying", "cancel": event}}
    )
    assert move.status("org/model", "other") is None
    assert move.cancel_move("org/model", "other") is None
    assert not event.is_set()
    assert move.cancel_move("ORG/MODEL", "owner") == {"phase": "copying"}
    assert event.is_set()


@_async_test
async def test_background_job_publishes_and_releases_repo(model, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from hub.services.models import downloads
    from hub.utils import hf_cache_state
    from utils import hf_cache_settings

    reservations, releases, published = [], [], []
    owner_ref = []

    def claim(repo, owner):
        reservations.append(repo)
        owner_ref.append(owner)
        return True, "owned"

    monkeypatch.setattr(
        downloads,
        "registry",
        SimpleNamespace(
            claim_repository_owner = claim,
            release_repository_owner = lambda repo, owner: releases.append((repo, owner)),
        ),
    )
    monkeypatch.setattr(move, "_jobs", {})
    monkeypatch.setattr(move, "_assert_unloaded", lambda _: None)
    monkeypatch.setattr(hf_cache_state, "iter_repo_cache_dirs", lambda *args: iter([model[0]]))
    monkeypatch.setattr(
        hf_cache_settings, "remember_model_storage_home", lambda *args: published.append(args)
    )
    result = await move.start_move("org/model", str(model[1].parent), "owner")
    assert result["phase"] == "queued"
    await asyncio.gather(*list(move._tasks))
    result = move.status("org/model", "owner")
    assert result["phase"] == "completed", result
    assert Path(result["destination"]).is_dir()
    assert not model[0].exists()
    assert published and reservations == ["org/model"]
    assert releases == [("org/model", owner_ref[0])]


@_async_test
async def test_restore_with_disconnected_disk_fails_before_changing_metadata(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from hub.services.models import downloads
    from hub.utils import hf_cache_state
    from utils import hf_cache_settings

    missing = tmp_path / "external" / "Unsloth Models" / "hub" / "models--org--model"

    monkeypatch.setattr(
        downloads,
        "registry",
        SimpleNamespace(
            claim_repository_owner = lambda *args: (True, "owned"),
            release_repository_owner = lambda *args: None,
        ),
    )
    monkeypatch.setattr(move, "_jobs", {})
    monkeypatch.setattr(move, "_assert_unloaded", lambda _: None)
    monkeypatch.setattr(hf_cache_state, "iter_repo_cache_dirs", lambda *args: iter([]))
    monkeypatch.setattr(hf_cache_settings, "relocated_model_repo_path", lambda _: missing)
    monkeypatch.setattr(
        hf_cache_settings,
        "finish_model_storage_restore",
        lambda *args: pytest.fail("restore metadata changed while the source disk was missing"),
    )

    result = await move.start_restore("org/model", "owner")
    assert result["phase"] == "queued"
    await asyncio.gather(*list(move._tasks))
    result = move.status("org/model", "owner", "restore")
    assert result["phase"] == "failed"
    assert "Reconnect the disk" in result["error"]


@_async_test
async def test_restore_uses_registered_source_even_when_another_cached_copy_exists(model, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from hub.services.models import downloads
    from hub.utils import hf_cache_state
    from utils import hf_cache_settings

    source, destination = model
    shutil.copytree(source, destination, symlinks = True)
    finalized = []

    monkeypatch.setattr(
        downloads,
        "registry",
        SimpleNamespace(
            claim_repository_owner = lambda *args: (True, "owned"),
            release_repository_owner = lambda *args: None,
        ),
    )
    monkeypatch.setattr(move, "_jobs", {})
    monkeypatch.setattr(move, "_assert_unloaded", lambda _: None)
    monkeypatch.setattr(
        hf_cache_state,
        "iter_repo_cache_dirs",
        lambda *args: pytest.fail("restore must not scan all cached copies"),
    )
    monkeypatch.setattr(hf_cache_settings, "relocated_model_repo_path", lambda _: source)
    monkeypatch.setattr(
        hf_cache_settings,
        "model_storage_restore_destination",
        lambda *args: destination,
    )
    monkeypatch.setattr(
        hf_cache_settings,
        "finish_model_storage_restore",
        lambda *args: finalized.append(args),
    )

    result = await move.start_restore("org/model", "owner")
    assert result["phase"] == "queued"
    await asyncio.gather(*list(move._tasks))
    result = move.status("org/model", "owner", "restore")
    assert result["phase"] == "completed", result
    assert finalized
    assert destination.is_dir()
    assert not source.exists()


@_async_test
async def test_busy_download_refuses_move(model, monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from hub.services.models import downloads

    monkeypatch.setattr(move, "_jobs", {})
    monkeypatch.setattr(
        downloads,
        "registry",
        SimpleNamespace(claim_repository_owner = lambda *args: (False, "running")),
    )
    with pytest.raises(HTTPException, match = "download or update"):
        await move.start_move("org/model", str(model[1].parent), "owner")
    assert model[0].is_dir()
