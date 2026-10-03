#!/bin/bash
# phone-ane.sh IP [--off]: push the ANE page template (Documents/anekv/tmpl16k.mlmodelc; the app's phone-attn then serves the
# oldest 16k-key pages of each layer on the Neural Engine, docs/ANE.md) and relaunch the app. Builds the template
# first if missing (phone-attn/ane-kv/build.py, needs a python3 with coremltools + numpy). --off: push an empty anekv dir (engine off).
set -eu
cd "$(dirname "$0")/.."
IP=$1
TM=build/ane-kv/tmpl/kv_N16384_R48_fp16_pfix_cinput.mlmodelc
TMP=$(mktemp -d)/anekv; mkdir -p "$TMP"
if [ "${2:-}" != --off ]; then
  if [ ! -d "$TM" ]; then
    # the first python3 on the usual paths that has coremltools (PY=... to pick one)
    BPY=
    for p in ${PY:-} python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
      "$p" -c 'import coremltools, numpy' 2>/dev/null && { BPY=$p; break; }
    done
    [ -n "$BPY" ] || { echo "phone-ane: no python3 with coremltools and numpy: pip3 install coremltools numpy" >&2; exit 1; }
    "$BPY" phone-attn/ane-kv/build.py build/ane-kv/tmpl --keys 16384 --rows 48 --wq fp16 --pfix --center input
  fi
  cp -R "$TM" "$TMP/tmpl16k.mlmodelc"
fi
python3 scripts/phone-push.py "$IP" "$TMP" anekv | tail -1
exec scripts/phone-relaunch.sh "$IP"
