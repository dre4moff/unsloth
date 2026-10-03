# SPDX-License-Identifier: AGPL-3.0-only
"""Isolate the pinned GGUF parser from Studio's updateable gguf package."""
from __future__ import annotations

import importlib
import sys
import threading
import types

_lock = threading.Lock()
_package = None


def pinned_gguf():
    global _package
    with _lock:
        if _package is not None:
            return _package
        from core.companion.backburner import VENDOR
        directory = VENDOR / "llama.cpp/gguf-py/gguf"
        name = "_unsloth_backburner_gguf"
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
        try:
            constants = importlib.import_module(f"{name}.constants")
            # The upstream reader has one absolute import. Adapt only that
            # namespace, leaving the vendored source and parsing algorithm intact.
            path = directory / "gguf_reader.py"
            source = path.read_text()
            absolute = "from gguf.constants import ("
            if source.count(absolute) != 1:
                raise ValueError("Unexpected pinned Backburner GGUF reader import")
            reader = types.ModuleType(f"{name}.gguf_reader")
            reader.__package__ = name
            reader.__file__ = str(path)
            sys.modules[reader.__name__] = reader
            exec(compile(source.replace(absolute, "from .constants import ("), str(path), "exec"), reader.__dict__)
            package.GGUFReader = reader.GGUFReader
            for key in ("GGMLQuantizationType", "GGML_QUANT_SIZES", "LlamaFileType"):
                setattr(package, key, getattr(constants, key))
            package.GGUFWriter = importlib.import_module(f"{name}.gguf_writer").GGUFWriter
            _package = package
            return package
        except Exception:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)
            raise
