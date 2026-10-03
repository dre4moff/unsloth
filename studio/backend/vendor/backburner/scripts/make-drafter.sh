#!/bin/bash
# make-drafter.sh - the DFlash2 draft model the server uses: Hugging Face z-lab/Qwen3.8-27B-DFlash2 (safetensors) -> GGUF,
# Q4_K_M with the candidate selector kept in f16 (+7.5% accepted drafts vs an all-Q4_K_M file). CPU only, a few minutes.
#   huggingface-cli download z-lab/Qwen3.8-27B-DFlash2 --local-dir ~/Models/qwen38-27b-dflash2
#   scripts/make-drafter.sh ~/Models/qwen38-27b-dflash2 ~/Models/dflash2-v2-q4km-self16.gguf
# Needs: python3 with numpy and llama.cpp's gguf-py (pip install numpy; this script sets PYTHONPATH=llama.cpp/gguf-py),
# llama.cpp built (llama-quantize), and the target GGUF (the draft has no vocabulary of its own; it copies the target's).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$1; OUT=$2
TARGET=${TARGET:-$HOME/Models/Qwen3.8-27B-IQ4_XS.gguf}
TMP=${TMPDIR:-/tmp}/make-drafter.$$
mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT
PYTHONPATH=llama.cpp/gguf-py${PYTHONPATH:+:$PYTHONPATH} python3 scripts/convert-dflash2.py "$SRC" -o "$TMP/f16.gguf" --outtype f16 --target-gguf "$TARGET"
llama.cpp/build-metal/bin/llama-quantize --tensor-type selector=f16 "$TMP/f16.gguf" "$OUT" Q4_K_M 2>&1 | tail -2
ls -la "$OUT"
