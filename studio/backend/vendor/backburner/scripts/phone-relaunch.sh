#!/bin/bash
# phone-relaunch.sh IP: relaunch the Backburner app on the wired iPhone (env UDID to pick one) and wait for the prefill tail (:50060)
set -eu
IP=$1
UDID=${UDID:-$(xcrun devicectl list devices 2>/dev/null | grep -o '[0-9A-F]\{8\}-[0-9A-F]\{16\}' | while read u; do xcrun devicectl device info details --device "$u" 2>/dev/null | grep -q 'Transport Type: wired' && echo "$u"; done | head -1)}
BID=$("$(dirname "$0")/bundle-id.sh" "$UDID")
xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BID" 2>&1 | grep -E 'Launched|Locked|rror' | head -1
for i in $(seq 1 30); do nc -z -G1 "$IP" 50060 2>/dev/null && { echo "tail up after ~$((i*2)) s"; exit 0; }; sleep 2; done
echo "tail not up after 60 s" >&2; exit 1
