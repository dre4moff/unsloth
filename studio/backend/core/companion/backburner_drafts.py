# SPDX-License-Identifier: AGPL-3.0-only
"""Local DFlash2 discovery and validation for the pinned Backburner engine."""
from __future__ import annotations

import threading
from pathlib import Path

from core.companion.backburner_gguf import pinned_gguf
from core.companion.backburner_models import METAL_WEIGHT_TYPES

_lock = threading.Lock()
_validated: dict[tuple, dict] = {}
_rejected: set[tuple] = set()


def validate_draft(path: str, target=None) -> dict:
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError("Select a downloaded DFlash2 GGUF file, rather than a Hugging Face repository ID.")
    stat = file.stat()
    key = (str(file.resolve()), stat.st_size, stat.st_mtime_ns)
    with _lock:
        cached = _validated.get(key) if target is None else None
        if cached is not None:
            return cached.copy()
        reader = pinned_gguf().GGUFReader(str(file))
        def field(name, default=None):
            value = reader.get_field(name)
            return value.contents() if value is not None else default
        expected = {
            "general.architecture": "dflash", "dflash.block_count": 5,
            "dflash.embedding_length": 5120, "dflash.feed_forward_length": 17408,
            "dflash.attention.head_count": 32, "dflash.attention.head_count_kv": 8,
            "dflash.attention.key_length": 128, "dflash.attention.value_length": 128,
            "dflash.block_size": 8, "dflash.conv_kernel_size": 2,
            "dflash.conv_group_size": 16, "dflash.selector_rank": 256,
            "dflash.selector_top_k": 16, "dflash.target_layers": [6, 20, 34, 48, 62],
        }
        for name, value in expected.items():
            if field(name) != value:
                raise ValueError(f"Incompatible Qwen3.8-27B DFlash2 draft: {name} must be {value}.")
        if field("split.count", 1) != 1:
            raise ValueError("Merge the DFlash2 GGUF shards before selecting the draft.")
        shapes = {
            "fc.weight": [25600, 5120], "enc.output_norm.weight": [5120],
            "output_norm.weight": [5120], "selector_hidden.weight": [5120, 256],
            "selector_predecessor.weight": [256, 248320],
            "selector_successor.weight": [256, 248320],
        }
        block = {
            "attn_norm.weight": [5120], "ffn_norm.weight": [5120],
            "attn_q.weight": [5120, 4096], "attn_k.weight": [5120, 1024],
            "attn_v.weight": [5120, 1024], "attn_output.weight": [4096, 5120],
            "attn_q_norm.weight": [128], "attn_k_norm.weight": [128],
            "ffn_gate.weight": [5120, 17408], "ffn_up.weight": [5120, 17408],
            "ffn_down.weight": [17408, 5120],
            "attn_conv_base": [5120, 2, 2], "ffn_conv_base": [5120, 2, 2],
            "attn_conv_proj.weight": [5120, 1280], "ffn_conv_proj.weight": [5120, 1280],
        }
        for i in range(5):
            shapes.update({f"blk.{i}.{name}": shape for name, shape in block.items()})
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        for name, shape in shapes.items():
            if name not in tensors or list(tensors[name].shape) != shape:
                raise ValueError(f"Incompatible or incomplete DFlash2 tensor: {name}.")
            if (len(shape) == 1 or name.endswith("conv_base")) and tensors[name].tensor_type.name != "F32":
                raise ValueError(f"The pinned DFlash2 graph requires F32 normalization/convolution bases: {name}.")
        unsupported = sorted({t.tensor_type.name for t in reader.tensors} - METAL_WEIGHT_TYPES)
        if unsupported:
            raise ValueError(f"Unsupported Backburner draft weight formats: {', '.join(unsupported)}.")
        tokens = field("tokenizer.ggml.tokens", [])
        if len(tokens) != 248320:
            raise ValueError("DFlash2 must contain the Qwen3.8-27B tokenizer (248,320 tokens).")
        if target is not None:
            for name in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.model",
                         "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
                value = target.get_field(name)
                if field(name) != (value.contents() if value is not None else None):
                    raise ValueError(f"DFlash2 and the selected target have different tokenizers: {name}.")
        result = {"path": str(file.resolve()), "name": file.name, "sizeBytes": stat.st_size,
                  "weightTypes": sorted({t.tensor_type.name for t in reader.tensors})}
        after = file.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("DFlash2 changed during validation. Wait for its download to finish.")
        if len(_validated) >= 128:
            _validated.clear()
        _validated[key] = result
        return result.copy()


def discover_drafts(roots: list[Path], selected: str | None = None) -> list[dict]:
    """Read finished GGUFs only; never initiate downloads or scan the user's home."""
    files: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        patterns = ("models--*/snapshots/*/**/*.gguf", "*.gguf", "*/*.gguf", "*/*/*.gguf", "*/*/*/*.gguf")
        for pattern in patterns:
            for path in root.glob(pattern):
                try:
                    if path.is_file():
                        files.setdefault(str(path.resolve()), path)
                except OSError:
                    continue
    if selected:
        path = Path(selected).expanduser()
        if path.is_file():
            files.setdefault(str(path.resolve()), path)
    results = []
    for path in files.values():
        key = None
        try:
            # Include renamed drafts too. Large target files cannot be the small
            # DFlash2 profile; explicitly named/selected files are always checked.
            stat = path.stat()
            key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            with _lock:
                if key in _rejected:
                    continue
            if stat.st_size > 4 * 1024**3 and "dflash" not in path.name.lower() and str(path.resolve()) != selected:
                continue
            result = validate_draft(str(path))
            # A persisted selection uses the blob's canonical path. Keep the
            # human-readable snapshot filename even after validating that blob.
            result["name"] = path.name
            repo = next((p[len("models--"):].replace("--", "/") for p in path.parts if p.startswith("models--")), None)
            result["repository"] = repo
            results.append(result)
        except (OSError, ValueError, KeyError, IndexError, TypeError, OverflowError):
            if key is not None:
                with _lock:
                    if len(_rejected) >= 512:
                        _rejected.clear()
                    _rejected.add(key)
            continue
    return sorted(results, key=lambda item: (item["sizeBytes"], item["name"]))


def local_drafts(selected: str | None = None) -> list[dict]:
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir, lmstudio_model_dirs
    from utils.hf_cache_settings import known_hf_hub_caches
    from storage.studio_db import list_scan_folders
    roots = [Path("./models").resolve(), Path.home() / "Models", *known_hf_hub_caches(),
             legacy_hf_cache_dir(), hf_default_cache_dir(), *lmstudio_model_dirs()]
    roots.extend(Path(folder["path"]) for folder in list_scan_folders())
    return discover_drafts(list(dict.fromkeys(Path(root) for root in roots if root is not None)), selected)
