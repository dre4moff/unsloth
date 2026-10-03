#!/bin/bash
# phone-env.sh IP [KEY=VALUE ...]: write Documents/env.txt on the phone (the app applies it at launch) and relaunch the app.
# No KEY=VALUE = an empty env.txt (the app's built-in defaults).
set -eu
cd "$(dirname "$0")/.."
IP=$1; shift
TMP=$(mktemp -d)/env.txt
printf '# written by scripts/phone-env.sh %s\n' "$(date '+%F %T')" > "$TMP"
for kv in "$@"; do echo "$kv" >> "$TMP"; done
python3 scripts/phone-push.py "$IP" "$TMP" env.txt | tail -1
exec scripts/phone-relaunch.sh "$IP"
