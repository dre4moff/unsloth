"""Installer cache isolation and real uv initialization before venv replacement."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="POSIX shell installer test",
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "install.sh").read_text()
HELPER = SOURCE.split("_configure_studio_uv_cache() {", 1)[1].split("\n}\n", 1)[0]
HELPER = "_configure_studio_uv_cache() {" + HELPER + "\n}\n"


def run_helper(tmp_path, *, cache=None, os_name="macos", real_uv=False):
    env = dict(os.environ, STUDIO_HOME=str(tmp_path / "Studio with spaces"), OS=os_name)
    env.pop("UV_CACHE_DIR", None)
    if cache is not None:
        env["UV_CACHE_DIR"] = str(cache)
    env["UV_TEST_PYTHON"] = shutil.which("python3")
    env["UV_TEST_VENV"] = str(tmp_path / "probe venv")
    code = "set -eu\ntauri_log() { printf '%s %s\\n' \"$1\" \"$2\" >&2; }\n" + HELPER
    code += "_configure_studio_uv_cache\nprintf 'cache=%s\\n' \"${UV_CACHE_DIR:-unset}\"\n"
    if real_uv:
        code += 'uv venv --python "$UV_TEST_PYTHON" --python-preference only-system "$UV_TEST_VENV"\n'
    return subprocess.run(["bash", "-c", code], env=env, text=True, capture_output=True)


@pytest.mark.parametrize("value", [None, "", " \t "])
def test_missing_or_blank_cache_uses_existing_backend_default(tmp_path, value):
    result = run_helper(tmp_path, cache=value)
    assert result.returncode == 0, result.stderr
    selected = tmp_path / "Studio with spaces/cache/uv"
    assert f"cache={selected}" in result.stdout and selected.is_dir()
    assert list(selected.iterdir()) == []


def test_explicit_cache_with_spaces_is_preserved(tmp_path):
    selected = tmp_path / "Custom cache"
    result = run_helper(tmp_path, cache=selected)
    assert result.returncode == 0, result.stderr
    assert f"cache={selected}" in result.stdout
    assert not (tmp_path / "Studio with spaces/cache/uv").exists()


def test_uncreatable_cache_fails_before_venv_is_touched(tmp_path):
    blocker = tmp_path / "file"; blocker.write_text("preserve me")
    result = run_helper(tmp_path, cache=blocker / "cache")
    assert result.returncode != 0 and "Could not create" in result.stderr
    assert blocker.read_text() == "preserve me"


def test_other_platforms_keep_their_existing_cache_behavior(tmp_path):
    result = run_helper(tmp_path, os_name="linux")
    assert result.returncode == 0 and "cache=unset" in result.stdout
    assert not (tmp_path / "Studio with spaces").exists()


def test_cache_is_selected_before_environment_replacement():
    invocation = SOURCE.index("\n_configure_studio_uv_cache || exit 1\n")
    assert invocation < SOURCE.index('\nif [ -x "$VENV_DIR/bin/python" ]; then', invocation)
    assert invocation < SOURCE.index('_uv_venv_arm64 "create venv"')


@pytest.mark.skipif(shutil.which("uv") is None or getattr(os, "geteuid", lambda: 0)() == 0, reason="real uv and non-root POSIX user required")
def test_real_uv_ignores_unwritable_shared_bucket(tmp_path, monkeypatch):
    home = tmp_path / "home"; bucket = home / ".cache/uv/sdists-v9"
    bucket.mkdir(parents=True); marker = bucket / ".git"; marker.touch(); marker.chmod(0o444)
    bucket.chmod(0o555)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    broken = subprocess.run(["uv", "venv", "--python", shutil.which("python3"), str(tmp_path / "failed venv")],
                            env={k:v for k,v in os.environ.items() if k != "UV_CACHE_DIR"}, capture_output=True, text=True)
    assert broken.returncode != 0 and "Permission denied" in broken.stderr
    result = run_helper(tmp_path, real_uv=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "probe venv/bin/python").exists()
    assert marker.stat().st_mode & 0o777 == 0o444
