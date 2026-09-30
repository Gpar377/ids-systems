#!/usr/bin/env bash
# Run from the NOISE host (192.168.90.30), not the attacker. Generates traffic
# to the victim that the attacker neither sent nor can see. Test 2 runs this
# between two otherwise-identical attacker runs to show the victim's host-based
# window features (dst_host_*) shift with no attacker involvement.
set -euo pipefail
VICTIM="${VICTIM:-192.168.90.10}"
BURST="${BURST:-60}"
echo "[noise] $BURST third-party connections -> $VICTIM"
for i in $(seq 1 "$BURST"); do
  curl -s -m 2 "http://$VICTIM:80/" >/dev/null 2>&1 || true
done
echo "[noise] done"
