#!/usr/bin/env python3
"""Convert a DFlash2 draft checkpoint (HF safetensors) to GGUF.

llama.cpp has full runtime support for the `dflash` architecture and gguf-py already knows
the tensor names, but there is no converter upstream - so these drafters cannot be used.
This fills that gap.

DFlash2 is a block-diffusion drafter: it predicts a whole block of draft tokens in ONE
forward pass (instead of K sequential ones) and a low-rank candidate selector traces a
coherent path through the per-position candidates. Decoding stays lossless.

    python3 scripts/convert-dflash2.py <draft_dir> -o <out.gguf> \
        --target-gguf <target.gguf>

The draft has no embeddings or lm_head; it shares the target's via llama.cpp's ctx_other.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

# HF name -> (gguf name template, is_per_layer)
DIRECT = {
    "fc.weight":                                      "fc.weight",
    "hidden_norm.weight":                             "enc.output_norm.weight",
    "norm.weight":                                    "output_norm.weight",
    "candidate_selector.hidden_projection.weight":    "selector_hidden.weight",
    "candidate_selector.predecessor_codebook":        "selector_predecessor.weight",
    "candidate_selector.successor_codebook":          "selector_successor.weight",
}

PER_LAYER = {
    "input_layernorm.weight":               "blk.{i}.attn_norm.weight",
    "post_attention_layernorm.weight":      "blk.{i}.ffn_norm.weight",
    "self_attn.q_proj.weight":              "blk.{i}.attn_q.weight",
    "self_attn.k_proj.weight":              "blk.{i}.attn_k.weight",
    "self_attn.v_proj.weight":              "blk.{i}.attn_v.weight",
    "self_attn.o_proj.weight":              "blk.{i}.attn_output.weight",
    "self_attn.q_norm.weight":              "blk.{i}.attn_q_norm.weight",
    "self_attn.k_norm.weight":              "blk.{i}.attn_k_norm.weight",
    "mlp.gate_proj.weight":                 "blk.{i}.ffn_gate.weight",
    "mlp.up_proj.weight":                   "blk.{i}.ffn_up.weight",
    "mlp.down_proj.weight":                 "blk.{i}.ffn_down.weight",
    # The base kernels are raw rank-3 tensors. Unlike projections, their GGUF
    # names intentionally have no `.weight` suffix (see dflash.cpp).
    "attention_conv.base_kernel":           "blk.{i}.attn_conv_base",
    "attention_conv.kernel_projection.weight": "blk.{i}.attn_conv_proj.weight",
    "mlp_conv.base_kernel":                 "blk.{i}.ffn_conv_base",
    "mlp_conv.kernel_projection.weight":    "blk.{i}.ffn_conv_proj.weight",
}

DT = {"BF16": 2, "F16": 2, "F32": 4}

# The dynamic convolution coefficients are F32 graph outputs. ggml's CPU
# elementwise add requires the base operand to have the same type.
F32_TENSORS = {
    "attention_conv.base_kernel",
    "mlp_conv.base_kernel",
}


def read_safetensors(path: Path):
    """Yield (name, numpy array). bf16 is widened to f32 (it is just the top 16 bits)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype, shape = meta["dtype"], meta["shape"]
            if dtype not in DT:
                raise RuntimeError(f"{name}: unsupported dtype {dtype}")
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2")
                arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
            yield name, arr


def copy_vocab_kvs(w: "gguf.GGUFWriter", target: Path) -> bool:
    """Copy tokenizer.* (and the chat template) from the target GGUF.

    A DFlash draft has no embeddings, no lm_head and no vocab - it borrows the target's at
    runtime - but llama.cpp still builds a llama_vocab when loading it, so the KVs have to
    be present and must match the target exactly.
    """
    r = gguf.GGUFReader(target, "r")
    n = 0
    for key, field in r.fields.items():
        if not (key.startswith("tokenizer.") or key == "general.chat_template"):
            continue
        if not field.types:
            continue
        t = field.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            sub = field.types[1]
            if sub == gguf.GGUFValueType.STRING:
                vals = [bytes(field.parts[i]).decode("utf-8", errors="replace") for i in field.data]
            else:
                vals = [field.parts[i].tolist()[0] for i in field.data]
            if not vals:
                continue
            w.add_array(key, vals)
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(key, bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace"))
        elif t == gguf.GGUFValueType.BOOL:
            w.add_bool(key, bool(field.parts[field.data[0]][0]))
        elif t in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            w.add_float32(key, float(field.parts[field.data[0]][0]))
        else:
            w.add_uint32(key, int(field.parts[field.data[0]][0]))
        n += 1
    print(f"copied {n} tokenizer KVs from {target.name}")
    return any(key.endswith(".rope.dimension_sections") for key in r.fields)


def copy_head_tensors(w: "gguf.GGUFWriter", target: Path) -> None:
    """Copy token_embd/output verbatim (still quantized) from the target GGUF.

    llama.cpp marks both TENSOR_NOT_REQUIRED on dflash so a draft can share the target's via
    ctx_other - but a shared head pins the draft's graph to the target's buffer, so a draft
    that must run on another device needs its own copy.
    """
    r = gguf.GGUFReader(target, "r")
    want = {"token_embd.weight", "output.weight"}
    for t in r.tensors:
        if t.name not in want:
            continue
        # keep the target's quantization: pass the raw bytes through untouched
        # GGUFReader hands back the BYTE shape in t.data.shape (e.g. Q6_K -> 210 bytes per
        # 256-element block); gguf_writer converts that back to the logical shape itself.
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        print(f"embedded {t.name} ({t.tensor_type.name}, {t.data.nbytes/2**20:.0f} MiB)")
        want.discard(t.name)
    if want:
        print(f"warning: target has no {sorted(want)} - draft will still need ctx_other",
              file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("-o", "--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=["f32", "f16", "bf16"], default="f16")
    ap.add_argument("--target-gguf", type=Path, required=True,
                    help="the TARGET model's GGUF; its tokenizer/vocab KVs are copied in, since "
                         "a DFlash draft ships no vocab of its own")
    ap.add_argument("--embed-head", action="store_true",
                    help="also copy token_embd/output from the target, making the draft "
                         "self-contained. Required to run the draft on a DIFFERENT device than "
                         "the target (e.g. --spec-draft-device RPC0): otherwise the draft "
                         "borrows the target's lm_head and ggml-backend aborts with "
                         "'pre-allocated tensor (output.weight) in a buffer that cannot run the "
                         "operation'. Costs ~1.4 GB (quantized) and makes every draft pass "
                         "read it, so only use it when the draft must live elsewhere.")
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    if "DFlash2DraftModel" not in cfg.get("architectures", []):
        print(f"warning: architectures={cfg.get('architectures')}, expected DFlash2DraftModel")

    d = cfg["dflash_config"]
    n_embd  = cfg["hidden_size"]
    n_layer = cfg["num_hidden_layers"]
    # llama.cpp extracts layer inputs, while HF hidden_states[i + 1] is
    # layer i's output. Store the already-adjusted hook indices in GGUF.
    tgt     = [int(i) + 1 for i in d["target_layer_ids"]]

    w = gguf.GGUFWriter(args.outfile, "dflash")
    w.add_name(args.model_dir.name)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_rope_freq_base(cfg["rope_parameters"]["rope_theta"])
    w.add_vocab_size(cfg["vocab_size"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16"
                    else gguf.LlamaFileType.MOSTLY_BF16 if args.outtype == "bf16"
                    else gguf.LlamaFileType.ALL_F32)

    # which target layers' hidden states are concatenated into fc's input
    w.add_array(gguf.Keys.LLM.TARGET_LAYERS.format(arch="dflash"), tgt)

    # DFlash2 block-diffusion + selector geometry
    w.add_uint32(gguf.Keys.LLM.BLOCK_SIZE.format(arch="dflash"), d["block_size"])
    w.add_uint32("dflash.conv_kernel_size", d["conv_kernel_size"])
    w.add_uint32("dflash.conv_group_size",  d["conv_group_size"])
    w.add_uint32(gguf.Keys.LLM.SELECTOR_RANK.format(arch="dflash"), d["selector_rank"])
    w.add_uint32("dflash.selector_top_k",   d["selector_top_k"])

    if cfg.get("use_sliding_window") and cfg.get("sliding_window"):
        w.add_sliding_window(cfg["sliding_window"])
        w.add_array("dflash.attention.sliding_window_pattern", [True] * n_layer)

    target_uses_mrope = copy_vocab_kvs(w, args.target_gguf)
    if target_uses_mrope:
        # The draft uses temporal RoPE only, represented as degenerate M-RoPE.
        w.add_rope_dimension_sections([cfg["head_dim"] // 2, 0, 0, 0])
    # The target tokenizer does not declare a mask token, but DFlash uses its
    # trained mask ID to seed every non-anchor position in the noise block.
    w.add_mask_token_id(d["mask_token_id"])

    if args.embed_head:
        copy_head_tensors(w, args.target_gguf)

    shards = sorted(args.model_dir.glob("model*.safetensors"))
    if not shards:
        print("no safetensors found", file=sys.stderr)
        return 1

    np_dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32}[args.outtype]
    n = 0
    for shard in shards:
        for name, arr in read_safetensors(shard):
            if name.endswith((".scales", ".biases")):
                print(f"skip (quantized shard needs dequant): {name}", file=sys.stderr)
                return 1
            out = DIRECT.get(name)
            if out is None:
                if not name.startswith("layers."):
                    print(f"warning: unmapped tensor {name}", file=sys.stderr)
                    continue
                _, idx, rest = name.split(".", 2)
                tmpl = PER_LAYER.get(rest)
                if tmpl is None:
                    print(f"warning: unmapped tensor {name}", file=sys.stderr)
                    continue
                out = tmpl.format(i=int(idx))
            # Norms and convolution bases participate in F32 elementwise ops.
            data = arr.astype(np.float32) if arr.ndim == 1 or name.split(".", 2)[-1] in F32_TENSORS else arr.astype(np_dtype)
            w.add_tensor(out, data)
            n += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"\nwrote {n} tensors -> {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
