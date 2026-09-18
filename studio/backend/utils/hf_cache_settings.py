# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live, persisted Hugging Face cache routing for Unsloth Studio.

Hugging Face reads cache environment variables at import time.  Studio therefore
owns an explicit cache snapshot for each operation instead of trying to refresh
``huggingface_hub.constants`` in the long-running API process.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping, Optional


CACHE_HOME_SETTING_KEY = "hugging_face_cache_home"
CACHE_HISTORY_SETTING_KEY = "hugging_face_cache_history"
MAX_CACHE_HISTORY = 16
MODEL_STORAGE_HOMES_KEY = "model_storage_homes"
MODEL_STORAGE_LOCATIONS_KEY = "model_storage_locations"
MODEL_STORAGE_REDIRECTS_KEY = "model_storage_redirects"

CacheSource = Literal["default", "studio", "environment"]

_CACHE_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
)
# Imported by storage_roots._setup_cache_env before Studio seeds defaults.
_EXPLICIT_CACHE_ENV = {
    key: value.strip()
    for key in _CACHE_ENV_KEYS
    if (value := os.environ.get(key)) is not None and value.strip()
}
_settings_lock = threading.RLock()
_spawn_env_lock = threading.RLock()


@dataclass(frozen = True)
class HuggingFaceCachePaths:
    cache_home: Path
    hub_cache: Path
    xet_cache: Path
    source: CacheSource
    environment_variable: Optional[str] = None

    @property
    def editable(self) -> bool:
        return self.source != "environment"

    @property
    def is_custom(self) -> bool:
        return self.source == "studio"

    def child_env(self, base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        # Scrub either way: an explicit base is usually the caller's own os.environ
        # copy, so it carries any scoped offline flags an open guard has set.
        from utils.utils import hf_environment_for_spawn, hf_environment_scrubbed

        env = hf_environment_for_spawn() if base is None else hf_environment_scrubbed(base)
        # Do not rewrite HF_HOME. It also owns HF's token path, and credentials
        # must not be moved onto a removable cache volume.
        env["HF_HUB_CACHE"] = str(self.hub_cache)
        env["HF_XET_CACHE"] = str(self.xet_cache)
        env.pop("HUGGINGFACE_HUB_CACHE", None)
        return env


def _default_cache_home() -> Path:
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface"


def _canonical(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict = False)


def _environment_paths() -> Optional[HuggingFaceCachePaths]:
    explicit_home = _EXPLICIT_CACHE_ENV.get("HF_HOME")
    explicit_hub = _EXPLICIT_CACHE_ENV.get("HF_HUB_CACHE") or _EXPLICIT_CACHE_ENV.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if not explicit_home and not explicit_hub:
        return None
    explicit_xet = _EXPLICIT_CACHE_ENV.get("HF_XET_CACHE")
    default_home = _default_cache_home()
    hf_home = _canonical(explicit_home) if explicit_home else default_home
    hub = _canonical(explicit_hub) if explicit_hub else hf_home / "hub"
    xet = _canonical(explicit_xet) if explicit_xet else hf_home / "xet"
    controlling = next(
        key
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")
        if key in _EXPLICIT_CACHE_ENV
    )
    # Settings describes model downloads, so an explicit hub path is the
    # displayed/opened location even when HF_HOME points somewhere else for
    # credentials or XET data.
    display_home = (
        (hub.parent if explicit_hub and hub.name.lower() == "hub" else hub)
        if explicit_hub
        else hf_home
    )
    return HuggingFaceCachePaths(display_home, hub, xet, "environment", controlling)


def _stored_cache_home() -> Optional[Path]:
    try:
        from storage.studio_db import get_app_setting
        value = get_app_setting(CACHE_HOME_SETTING_KEY, None)
    except Exception:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _canonical(value.strip())
    except (OSError, RuntimeError, ValueError):
        return None


def configured_cache_key() -> str:
    """The configured cache location, for keying caches and in-flight work.

    Deliberately unresolved: resolve() can block on the very volume a caller is
    trying to move off. Only equality matters here, not the real path.
    """
    explicit = (
        _EXPLICIT_CACHE_ENV.get("HF_HUB_CACHE")
        or _EXPLICIT_CACHE_ENV.get("HUGGINGFACE_HUB_CACHE")
        or _EXPLICIT_CACHE_ENV.get("HF_HOME")
    )
    if explicit:
        return "env:" + explicit
    try:
        from storage.studio_db import get_app_setting
        value = get_app_setting(CACHE_HOME_SETTING_KEY, None)
    except Exception:
        return "default"
    if isinstance(value, str) and value.strip():
        return "studio:" + value.strip()
    return "default"


def get_hf_cache_paths() -> HuggingFaceCachePaths:
    env_paths = _environment_paths()
    if env_paths is not None:
        return env_paths
    stored = _stored_cache_home()
    if stored is not None:
        xet = _EXPLICIT_CACHE_ENV.get("HF_XET_CACHE")
        return HuggingFaceCachePaths(
            stored,
            stored / "hub",
            _canonical(xet) if xet else stored / "xet",
            "studio",
        )
    home = _default_cache_home()
    xet = _EXPLICIT_CACHE_ENV.get("HF_XET_CACHE")
    return HuggingFaceCachePaths(
        home,
        home / "hub",
        _canonical(xet) if xet else home / "xet",
        "default",
    )


def active_hf_hub_cache() -> str:
    """Return the current hub cache as a string for library call kwargs."""

    return str(get_hf_cache_paths().hub_cache)


@contextmanager
def _xet_loader_barrier() -> Iterator[None]:
    """Block while a Xet shim loader holds its process-wide env override. Never fails a spawn."""
    try:
        from utils.hf_xet_fallback import env_override_barrier
        barrier = env_override_barrier()
    except Exception:  # noqa: BLE001 - the shim is optional; a spawn must never depend on it
        yield
        return
    with barrier:
        yield


@contextmanager
def child_environment_for_spawn(environment: Mapping[str, str]) -> Iterator[None]:
    """Apply captured env before spawn imports the child entrypoint.

    Applying variables only inside the multiprocessing target can be too late
    for libraries that snapshot environment variables at import. The lock keeps
    this short parent-process override atomic through ``Process.start()``.
    """

    from utils.utils import hf_environment_restored_for_spawn

    # Also exclude the Xet shim's GPU-init override window: a child spawned inside it inherits the
    # flag for life, whereupon unsloth_zoo hands it STUB triton and bitsandbytes and the run
    # silently produces nothing. Filtering a child env dict cannot help here, since spawn copies the
    # live environment and takes no env argument.
    with _spawn_env_lock, _xet_loader_barrier(), hf_environment_restored_for_spawn():
        missing = object()
        saved_environment: dict[str, str | object] = {}
        for key, value in environment.items():
            saved_environment[key] = os.environ.get(key, missing)
            os.environ[key] = value
        try:
            yield
        finally:
            for key, previous in saved_environment.items():
                if previous is missing:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(previous)


def initialize_hf_cache_environment() -> HuggingFaceCachePaths:
    """Seed import-time HF variables once during backend startup."""

    paths = get_hf_cache_paths()
    # Preserve an explicit HF_HOME, otherwise keep credentials at the platform
    # default while routing cache bytes through the selected home.
    if not os.environ.get("HF_HOME", "").strip():
        os.environ["HF_HOME"] = str(_default_cache_home())
    os.environ["HF_HUB_CACHE"] = str(paths.hub_cache)
    os.environ["HF_XET_CACHE"] = str(paths.xet_cache)
    if "HUGGINGFACE_HUB_CACHE" not in _EXPLICIT_CACHE_ENV:
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    for directory in (paths.hub_cache, paths.xet_cache):
        try:
            directory.mkdir(parents = True, exist_ok = True)
        except OSError:
            pass
    return paths


def _validate_cache_home(raw_path: str) -> Path:
    value = raw_path.strip()
    if not value:
        raise ValueError("Choose a cache folder.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("The Hugging Face cache folder must be an absolute path.")
    try:
        resolved = candidate.resolve(strict = False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The Hugging Face cache folder is invalid.") from exc

    if resolved.parent == resolved:
        raise ValueError("Choose a folder inside the filesystem or drive root.")
    try:
        from hub.storage.scan_folders import (
            contains_sensitive_path_component,
            is_denied_system_path,
        )
    except ImportError:
        contains_sensitive_path_component = is_denied_system_path = None
    if is_denied_system_path is not None and is_denied_system_path(str(resolved)):
        raise ValueError("System folders cannot be used for model downloads.")
    if contains_sensitive_path_component is not None and contains_sensitive_path_component(
        str(resolved)
    ):
        raise ValueError("Credential or config folders cannot be used for model downloads.")

    parent = resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("The parent folder does not exist.")
    try:
        resolved.mkdir(exist_ok = True)
        if not resolved.is_dir():
            raise ValueError("The selected cache location is not a folder.")
        for child in (resolved / "hub", resolved / "xet"):
            child.mkdir(exist_ok = True)
            with tempfile.NamedTemporaryFile(prefix = ".unsloth-write-test-", dir = child):
                pass
    except PermissionError as exc:
        raise ValueError("Studio does not have permission to write to this folder.") from exc
    except OSError as exc:
        raise ValueError(f"Studio cannot use this cache folder: {exc}") from exc
    return resolved


def _stored_history() -> list[Path]:
    try:
        from storage.studio_db import get_app_setting
        raw = get_app_setting(CACHE_HISTORY_SETTING_KEY, [])
    except Exception:
        raw = []
    if not isinstance(raw, list):
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            path = _canonical(value)
        except (OSError, RuntimeError, ValueError):
            continue
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out[:MAX_CACHE_HISTORY]


def set_hf_cache_home(cache_home: Optional[str]) -> HuggingFaceCachePaths:
    if _environment_paths() is not None:
        raise RuntimeError("The Hugging Face cache location is managed by an environment variable.")
    with _settings_lock:
        previous = _stored_cache_home()
        next_home = _validate_cache_home(cache_home) if cache_home is not None else None
        history = _stored_history()
        if previous is not None and previous != next_home:
            history.insert(0, previous)
        deduped: list[str] = []
        seen: set[str] = set()
        for path in history:
            key = os.path.normcase(str(path))
            if key in seen or path == next_home:
                continue
            seen.add(key)
            deduped.append(str(path))
            if len(deduped) >= MAX_CACHE_HISTORY:
                break
        from storage.studio_db import upsert_app_settings

        upsert_app_settings(
            {
                CACHE_HOME_SETTING_KEY: str(next_home) if next_home is not None else None,
                CACHE_HISTORY_SETTING_KEY: deduped,
            }
        )
    # Inventory scans are cached independently from settings. Invalidate after
    # persistence so the next request sees both the new active root and history.
    from hub.utils.inventory_scan import invalidate_hf_cache_scans

    invalidate_hf_cache_scans()
    return get_hf_cache_paths()


def remember_model_storage_home(
    home: Path, repo_id: Optional[str] = None, repo_path: Optional[Path] = None,
    previous_path: Optional[Path] = None
) -> None:
    """Keep relocated models discoverable without changing new-download settings.

    Unlike recent cache history, these roots must never be evicted by later
    folder changes. They contain the user's only copy of a moved model.
    """
    from storage.studio_db import get_app_setting, upsert_app_settings
    with _settings_lock:
        homes = get_app_setting(MODEL_STORAGE_HOMES_KEY, [])
        homes = [value for value in homes if isinstance(value, str)] if isinstance(homes, list) else []
        value = str(home.resolve())
        if value not in homes:
            homes.append(value)
        updates = {MODEL_STORAGE_HOMES_KEY: homes}
        if repo_id:
            locations = get_app_setting(MODEL_STORAGE_LOCATIONS_KEY, {})
            locations = dict(locations) if isinstance(locations, dict) else {}
            locations[repo_id.casefold()] = str(repo_path or home / "hub" / ("models--" + repo_id.replace("/", "--")))
            updates[MODEL_STORAGE_LOCATIONS_KEY] = locations
        if previous_path is not None and repo_path is not None:
            redirects = get_app_setting(MODEL_STORAGE_REDIRECTS_KEY, {})
            redirects = dict(redirects) if isinstance(redirects, dict) else {}
            old, new = str(previous_path), str(repo_path)
            # Collapse prior moves so old chats still resolve after moving again.
            redirects = {key: new if target == old else target for key, target in redirects.items()}
            redirects[old] = new
            updates[MODEL_STORAGE_REDIRECTS_KEY] = redirects
        upsert_app_settings(updates)


def relocated_model_repo_path(model_id: str) -> Optional[Path]:
    """Return the registered repository root for a model moved out of the default cache."""
    from storage.studio_db import get_app_setting

    locations = get_app_setting(MODEL_STORAGE_LOCATIONS_KEY, {})
    raw = locations.get(model_id.casefold()) if isinstance(locations, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return _canonical(raw)
    except (OSError, RuntimeError, ValueError):
        return None


def is_model_relocated(model_id: str) -> bool:
    """Whether a moved-model registration currently points to an available repository."""
    repo = relocated_model_repo_path(model_id)
    return repo is not None and repo.is_dir()


def model_storage_restore_destination(model_id: str, current_path: Path) -> Path:
    """Recover the pre-move repository path, falling back to the active local Hub cache."""
    from storage.studio_db import get_app_setting

    current = _canonical(current_path)
    redirects = get_app_setting(MODEL_STORAGE_REDIRECTS_KEY, {})
    homes = get_app_setting(MODEL_STORAGE_HOMES_KEY, [])
    external_homes: list[Path] = []
    if isinstance(homes, list):
        for raw in homes:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                external_homes.append(_canonical(raw))
            except (OSError, RuntimeError, ValueError):
                continue

    prior: list[Path] = []
    if isinstance(redirects, dict):
        for source, target in redirects.items():
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            try:
                if _canonical(target) != current:
                    continue
                candidate = _canonical(source)
            except (OSError, RuntimeError, ValueError):
                continue
            if (
                candidate == current
                or candidate.name != current.name
                or candidate.parent.name != "hub"
            ):
                continue
            prior.append(candidate)
            candidate_home = candidate.parent.parent
            if candidate_home not in external_homes:
                return candidate

    fallback = _canonical(get_hf_cache_paths().hub_cache / current.name)
    if fallback != current:
        return fallback
    if prior:
        return prior[0]
    raise ValueError("The model's original cache location could not be determined.")


def finish_model_storage_restore(
    model_id: str,
    restored_path: Path,
    previous_path: Path,
) -> None:
    """Publish a verified restore and retire its external-location registration."""
    from storage.studio_db import get_app_setting, upsert_app_settings

    restored = _canonical(restored_path)
    previous = _canonical(previous_path)
    with _settings_lock:
        locations = get_app_setting(MODEL_STORAGE_LOCATIONS_KEY, {})
        locations = dict(locations) if isinstance(locations, dict) else {}
        locations.pop(model_id.casefold(), None)

        redirects = get_app_setting(MODEL_STORAGE_REDIRECTS_KEY, {})
        redirects = dict(redirects) if isinstance(redirects, dict) else {}
        old, new = str(previous), str(restored)
        redirects = {key: new if target == old else target for key, target in redirects.items()}
        redirects.pop(new, None)  # The original path must not redirect to itself.
        if old != new:
            redirects[old] = new

        homes = get_app_setting(MODEL_STORAGE_HOMES_KEY, [])
        homes = [value for value in homes if isinstance(value, str)] if isinstance(homes, list) else []
        old_home = previous.parent.parent if previous.parent.name == "hub" else None
        if old_home is not None:
            remaining = []
            for raw in locations.values():
                if not isinstance(raw, str):
                    continue
                try:
                    remaining.append(_canonical(raw))
                except (OSError, RuntimeError, ValueError):
                    continue
            kept_homes = []
            for raw in homes:
                try:
                    home = _canonical(raw)
                except (OSError, RuntimeError, ValueError):
                    kept_homes.append(raw)
                    continue
                if home != old_home:
                    kept_homes.append(raw)
                    continue
                hub = home / "hub"
                if any(path == hub or path.is_relative_to(hub) for path in remaining):
                    kept_homes.append(raw)
            homes = kept_homes

        upsert_app_settings(
            {
                MODEL_STORAGE_HOMES_KEY: homes,
                MODEL_STORAGE_LOCATIONS_KEY: locations,
                MODEL_STORAGE_REDIRECTS_KEY: redirects,
            }
        )


def relocated_model_path(model_id: str) -> Optional[Path]:
    """Resolve moved repo IDs locally; an unplugged disk must not redownload GBs."""
    from storage.studio_db import get_app_setting
    repo = relocated_model_repo_path(model_id)
    if repo is None:
        redirects = get_app_setting(MODEL_STORAGE_REDIRECTS_KEY, {})
        if not isinstance(redirects, dict) or not Path(model_id).is_absolute():
            return None
        requested = Path(model_id)
        for source, destination in redirects.items():
            if not isinstance(source, str) or not isinstance(destination, str):
                continue
            try:
                suffix = requested.relative_to(source)
            except ValueError:
                continue
            resolved = Path(destination) / suffix
            if resolved.exists():
                return resolved
            if requested.exists():
                return None  # original retained after an interrupted commit
            raise ValueError("The disk containing this model is unavailable. Reconnect it before loading the model.")
        return None
    if not repo.is_dir():
        # The registration is persisted before commit. A failed commit may
        # still have the original, which remains a valid local fallback.
        from hub.utils.hf_cache_state import iter_repo_cache_dirs
        if any(path.is_dir() for path in iter_repo_cache_dirs("model", model_id)):
            return None
        raise ValueError("The disk containing this model is unavailable. Reconnect it before loading the model.")
    from hub.utils.hf_cache_state import ref_snapshot_dir, latest_snapshot_dir
    snapshot = ref_snapshot_dir(repo) or latest_snapshot_dir(repo)
    if snapshot is None:
        raise ValueError("The moved model has no usable snapshot. Reconnect its disk or check its files.")
    return snapshot


def known_hf_cache_homes() -> list[Path]:
    paths = get_hf_cache_paths()
    stored = _stored_cache_home()
    candidates: list[Path] = []
    if paths.source != "environment":
        candidates.append(paths.cache_home)
    elif explicit_home := _EXPLICIT_CACHE_ENV.get("HF_HOME"):
        candidates.append(_canonical(explicit_home))
    if stored is not None:
        candidates.append(stored)
    candidates.extend([*_stored_history(), _default_cache_home()])
    try:
        from storage.studio_db import get_app_setting
        relocated = get_app_setting(MODEL_STORAGE_HOMES_KEY, [])
    except Exception:
        relocated = []
    if isinstance(relocated, list):
        candidates.extend(Path(value) for value in relocated if isinstance(value, str) and value.strip())
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            canonical = _canonical(candidate)
        except (OSError, RuntimeError, ValueError):
            continue
        key = os.path.normcase(str(canonical))
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def known_hf_hub_caches() -> list[Path]:
    active = get_hf_cache_paths()
    out = [active.hub_cache]
    seen = {os.path.normcase(str(_canonical(active.hub_cache)))}
    for home in known_hf_cache_homes():
        hub = _canonical(home / "hub")
        key = os.path.normcase(str(hub))
        if key not in seen:
            seen.add(key)
            out.append(hub)
    return out


def cache_status(paths: Optional[HuggingFaceCachePaths] = None) -> dict:
    paths = paths or get_hf_cache_paths()
    available = paths.cache_home.is_dir()
    writable = available and os.access(paths.cache_home, os.W_OK | os.X_OK)
    free_bytes: Optional[int] = None
    if available:
        try:
            free_bytes = int(shutil.disk_usage(paths.cache_home).free)
        except OSError:
            pass
    return {
        "cache_home": str(paths.cache_home),
        "hub_cache": str(paths.hub_cache),
        "xet_cache": str(paths.xet_cache),
        "source": paths.source,
        "editable": paths.editable,
        "is_custom": paths.is_custom,
        "available": available,
        "writable": writable,
        "free_bytes": free_bytes,
        "environment_variable": paths.environment_variable,
    }
