# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.companion import backburner_drafts as drafts
from core.companion.backburner_profile import GIB, launch_profile


def test_repository_id_is_not_a_local_file():
    with pytest.raises(ValueError, match="repository ID"):
        drafts.validate_draft("HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF")


def test_discovery_deduplicates_hf_snapshots_and_ignores_incomplete_downloads(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    blob = tmp_path / "weights"; blob.write_bytes(b"draft")
    repo = hub / "models--user--DFlash2"
    first = repo / "snapshots/one/renamed.gguf"
    second = repo / "snapshots/two/Q6_K/draft.gguf"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True); path.symlink_to(blob)
    (repo / "blobs").mkdir(); (repo / "blobs/download.incomplete").write_bytes(b"partial")
    invalid = first.with_name("mtp.gguf"); invalid.write_bytes(b"not a dflash")
    calls = []
    def validate(path):
        calls.append(path)
        if Path(path).resolve() != blob.resolve():
            raise ValueError("wrong architecture")
        return {"path": str(Path(path).resolve()), "name": Path(path).name, "sizeBytes": 5}
    monkeypatch.setattr(drafts, "validate_draft", validate)
    found = drafts.discover_drafts([hub, hub], str(second))
    assert len(found) == 1 and found[0]["path"] == str(blob)
    assert found[0]["repository"] == "user/DFlash2"
    assert len(calls) == 2 and all("incomplete" not in path for path in calls)


@pytest.mark.parametrize("size", [561241824, 1143006720])
def test_16gb_profile_admits_user_iq3_with_both_drafts_at_50k(tmp_path, size):
    main = tmp_path / "main.gguf"; draft = tmp_path / "draft.gguf"
    with main.open("wb") as stream: stream.truncate(10442827296)
    with draft.open("wb") as stream: stream.truncate(size)
    profile = launch_profile(str(main), str(draft), 50000, 16*GIB, 0)
    assert profile["context"] == 8192 and profile["totalContext"] == 50000
    assert profile["kv"] == "q4_0" and profile["loadMode"] == "mmap"
    assert profile["estimatedBytes"] < 13*GIB
    assert profile["checkpoints"] == 1
    assert profile["draftGpuLayers"] == (999 if size < 1024**3 else 0)


def test_16gb_small_context_does_not_allocate_remote_pages(tmp_path):
    model = tmp_path / "model.gguf"; model.write_bytes(b"model")
    draft = tmp_path / "draft.gguf"; draft.write_bytes(b"draft")
    profile = launch_profile(str(model), str(draft), 4096, 16*GIB, 0)
    assert profile["context"] == profile["totalContext"] == 4096


@pytest.mark.parametrize("architecture", ["qwen35", "dflash"])
def test_original_filename_does_not_bypass_metadata_validation(tmp_path, architecture):
    from core.companion.backburner_gguf import pinned_gguf
    path = tmp_path / "dflash2-v2-q4km-self16.gguf"
    writer = pinned_gguf().GGUFWriter(str(path), architecture)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with pytest.raises(ValueError, match="Incompatible"):
        drafts.validate_draft(str(path))


def test_16gb_profile_rejects_weights_over_budget_without_reducing_user_context(tmp_path):
    main = tmp_path / "big.gguf"; draft = tmp_path / "draft.gguf"
    with main.open("wb") as stream: stream.truncate(13*GIB)
    draft.write_bytes(b"draft")
    with pytest.raises(ValueError, match="reduce context"):
        launch_profile(str(main), str(draft), 50000, 16*GIB, 0)


def test_original_profile_keeps_original_memory_requirement():
    with pytest.raises(ValueError, match="wired_limit_mb"):
        launch_profile("unused", "unused", 50000, 24*GIB, 0)
    assert launch_profile("unused", "unused", 50000, 24*GIB, 20480)["kv"] == "q8_0"


@pytest.mark.parametrize("physical", [0, 8*GIB])
def test_unknown_or_insufficient_ram_fails_closed(physical):
    with pytest.raises(ValueError):
        launch_profile("unused", "unused", 50000, physical, 20480)
