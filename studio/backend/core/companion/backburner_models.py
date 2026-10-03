# SPDX-License-Identifier: AGPL-3.0-only
"""Model compatibility for the pinned Backburner engine and ANE template."""
from __future__ import annotations

import re

# Weight kernels in f8c7683 ggml-metal. NVFP4 and TQ1_0 explicitly lack Metal
# matmul/get_rows; Q8_1/Q8_K and integer types are intermediate, not weight formats.
METAL_WEIGHT_TYPES = frozenset({
    "F32", "F16", "BF16", "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS",
    "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "MXFP4", "TQ2_0",
})

# Remote q8 KV sizing and the compiled ANE page template depend on these values.
PROFILE = {
    "embedding_length": 5120, "feed_forward_length": 17408,
    "attention.head_count": 24, "attention.head_count_kv": 4,
    "attention.key_length": 256, "attention.value_length": 256,
    "full_attention_interval": 4, "ssm.conv_kernel": 4, "ssm.state_size": 128,
    "ssm.group_count": 16, "ssm.time_step_rank": 48, "ssm.inner_size": 6144,
}


def validate_profile(reader) -> None:
    def field(name, default=None):
        item = reader.get_field(name)
        return item.contents() if item is not None else default

    identity = " ".join(str(field(key, "")) for key in (
        "general.name", "general.basename", "general.base_model.0.name",
        "general.base_model.0.repo_url",
    ))
    if (field("general.architecture") != "qwen35"
            or not re.search(r"qwen[\s_-]*3[.]8\b", identity, re.IGNORECASE)):
        raise ValueError("Backburner requires Qwen3.8-27B or a compatible derivative (including abliterated/uncensored).")
    blocks, nextn = field("qwen35.block_count"), field("qwen35.nextn_predict_layers", 0)
    if not isinstance(blocks, int) or not isinstance(nextn, int) or nextn < 0 or blocks - nextn != 64:
        raise ValueError("Backburner requires 64 Qwen3.8-27B trunk layers; optional MTP layers are excluded from the iPhone tail.")
    for key, expected in PROFILE.items():
        # The original loader defaults full_attention_interval to 4.
        actual = field(f"qwen35.{key}", 4 if key == "full_attention_interval" else None)
        if actual != expected:
            raise ValueError(f"Incompatible Qwen3.8-27B geometry: {key} must be {expected} (found {actual}).")
    if field("split.count", 1) != 1:
        raise ValueError("Merge the GGUF shards into one file before preparing the Backburner iPhone tail.")
    if not reader.tensors:
        raise ValueError("The Qwen GGUF contains no model tensors.")
    for tensor in reader.tensors:
        if tensor.name in {"token_embd.weight", "output.weight"} and list(tensor.shape) != [5120, 248320]:
            raise ValueError("This derivative changed the vocabulary/embedding shape required by the original DFlash2 draft.")
    unsupported = sorted({tensor.tensor_type.name for tensor in reader.tensors} - METAL_WEIGHT_TYPES)
    if unsupported:
        raise ValueError(f"The pinned Backburner Metal engine cannot use these weight formats: {', '.join(unsupported)}.")
    # general.file_type and filename are summaries, not a per-tensor quantization
    # contract. Dynamic/mixed quantizations (e.g. UD/GSQ) keep their exact weights.


def tail_layer(reader, device: str) -> int:
    """Keep upstream L40/L52; shorten heavier tails to a conservative weight budget.

    These 6/3 GiB ceilings leave room for the original tail context and remote KV.
    Runtime preflight still checks the phone's real wired/app memory after loading.
    """
    start, budget = (40, 6 * 1024**3) if device.startswith("iPhone18,") else (52, 3 * 1024**3)
    for layer in range(start, 64, 4):
        total = 0
        for tensor in reader.tensors:
            match = re.match(r"blk\.(\d+)\.", tensor.name)
            if match:
                if not layer <= int(match[1]) < 64:
                    continue
            elif tensor.name in {"token_embd.weight", "output.weight"}:
                continue
            total += tensor.n_bytes
        if total <= budget:
            return layer
    raise ValueError("Even the smallest iPhone tail exceeds the safe weight budget for this quantization/device.")
