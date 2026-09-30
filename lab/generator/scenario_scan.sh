#!/usr/bin/env bash
# Recon-style port sweep from the attacker host. This is the canonical example
# of Class B `count`/`srv_count` being RAISED (never lowered): the attacker
# generates connection volume, which is exactly what a scan is. Many target
# ports are closed -> REJ; that error pattern is what serror/rerror rates pick
# up, and -- per the taxonomy -- those rates are victim-determined, not set here.
set -euo pipefail
VICTIM="${VICTIM:-192.168.90.10}"
echo "[scan] TCP connect sweep -> $VICTIM"
nmap -sT -p 1-200 --max-retries 1 -n "$VICTIM" || true
echo "[scan] done"
