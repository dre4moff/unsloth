# SPDX-License-Identifier: AGPL-3.0-only
"""Memory profiles using options already supported by pinned upstream serve.sh."""
from __future__ import annotations

from pathlib import Path

GIB = 1024**3


def metal_budget(physical_bytes: int, wired_mb: int) -> int:
    try:
        import mlx.core as mx
        recommended = int(mx.device_info()["max_recommended_working_set_size"])
    except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        recommended = physical_bytes * 7 // 10
    # A positive explicit cap is a further restriction. Zero is the automatic cap.
    return min(recommended, wired_mb * 1024**2) if wired_mb > 0 else recommended


def launch_profile(model: str, draft: str, requested_context: int, physical_bytes: int, wired_mb: int,
                   metal_bytes: int | None = None) -> dict:
    if physical_bytes <= 0:
        raise ValueError("Cannot determine Mac RAM for Backburner memory checks.")
    if physical_bytes >= 24 * GIB:
        if wired_mb < 20000:
            raise ValueError("Backburner's original profile requires: sudo sysctl iogpu.wired_limit_mb=20480 (after each reboot).")
        return {"name": "original", "context": 65536, "kv": "q8_0", "bytesPerToken": 34816,
                "batch": 2048, "ubatch": 256, "checkpoints": 3, "loadMode": "none"}
    if physical_bytes < 16 * GIB:
        raise ValueError("Backburner requires at least 16 GiB Mac RAM and sufficiently small target/draft weights.")
    requested_context = requested_context or 65536
    if requested_context < 512 or requested_context > 262144:
        raise ValueError("Choose a Backburner context between 512 and 262,144 tokens.")
    # Preserve the selected logical limit while retaining only 8k local cells.
    # Original phone-attn can hold older pages once this smaller ring is full.
    context = ((min(requested_context, 8192) + 255) // 256) * 256
    # Conservative: count the entire files, even clean/lazy embedding and MTP
    # pages. Reserve 1.25 GiB for recurrent/draft state and graph buffers, plus
    # 3 GiB for macOS/Studio. This is an admission estimate, not a speed promise.
    estimate = Path(model).stat().st_size + Path(draft).stat().st_size + context * 18432 + 5 * GIB // 4
    budget = physical_bytes - 3 * GIB
    if estimate > budget:
        raise ValueError(f"Backburner memory estimate {estimate/GIB:.1f} GiB exceeds this Mac's {budget/GIB:.1f} GiB budget. "
                         "Select a smaller DFlash2/target quantization or reduce context.")
    gpu_budget = metal_bytes if metal_bytes is not None else physical_bytes * 74 // 100
    draft_gpu_layers = 999
    if estimate > gpu_budget:
        # CPU and Metal share memory on Apple Silicon. Upstream supports a CPU
        # drafter borrowing the target head; this reduces GPU residency without
        # changing draft weights or the original verification algorithm.
        draft_gpu_layers = 0
        if estimate - Path(draft).stat().st_size > gpu_budget:
            raise ValueError("The target and Backburner buffers exceed this Mac's GPU budget even with a CPU draft. "
                             "Select a smaller target quantization or reduce context.")
    # mmap is upstream's supported pageable mode; never increase a kernel
    # wired limit past physical RAM, and do not interpret its automatic 0 as 0 RAM.
    return {"name": "memory-saving", "context": context, "totalContext": max(context, requested_context),
            "kv": "q4_0", "bytesPerToken": 18432, "batch": 128, "ubatch": 64,
            "draftMax": 3, "draftGpuLayers": draft_gpu_layers, "checkpoints": 1,
            "loadMode": "mmap", "estimatedBytes": estimate}
