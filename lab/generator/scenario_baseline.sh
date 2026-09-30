#!/usr/bin/env bash
# Benign background: ordinary, well-spaced sessions to open services.
# These are the "normal" connections -- low volume, completed handshakes (SF).
set -euo pipefail
VICTIM="${VICTIM:-192.168.90.10}"
ROUNDS="${ROUNDS:-15}"
echo "[baseline] $ROUNDS benign rounds -> $VICTIM"
for i in $(seq 1 "$ROUNDS"); do
  curl -s -m 3 "http://$VICTIM:80/" >/dev/null 2>&1 || true
  printf 'USER anonymous\r\n' | nc -w 2 "$VICTIM" 21 >/dev/null 2>&1 || true
  printf 'a1 LOGIN x y\r\n'   | nc -w 2 "$VICTIM" 143 >/dev/null 2>&1 || true
  sleep 0.7      # spacing keeps the 2-second window count low, like real usage
done
echo "[baseline] done"
