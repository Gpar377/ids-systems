#!/usr/bin/env bash
# End-to-end capture driver for the isolated lab. Brings the sandbox up, runs the
# scenarios, tears the capture down cleanly, and leaves PCAPs in lab/pcaps/.
#
#   ./run_capture.sh            # full run: baseline + scan + realize + test2
#   ./run_capture.sh test2      # just the third-party demo
#
# Requires Docker running. All traffic stays on the internal idsnet bridge.
set -euo pipefail
cd "$(dirname "$0")"
VIC=192.168.90.10
ATT=192.168.90.20

up() {
  echo "== building + starting isolated lab =="
  docker compose up -d --build
  sleep 4   # let victim listeners bind and tcpdump attach
}

down() { echo "== stopping lab =="; docker compose down -v; }

run() {
  echo "== [baseline] benign background =="
  docker compose exec -T generator bash scenario_baseline.sh
  echo "== [scan] recon sweep from attacker =="
  docker compose exec -T generator bash scenario_scan.sh
  echo "== [realize] attempt a target feature vector on the wire =="
  docker compose exec -T generator env VICTIM=$VIC python3 scenario_realize.py
}

test2() {
  echo "== [test2] identical attacker run, only third-party noise differs =="
  echo "-- run A: attacker only --"
  docker compose exec -T generator env VICTIM=$VIC COUNT=20 python3 scenario_realize.py
  docker compose exec -T generator bash -c "cp /pcaps/realize_manifest.json /pcaps/manifest_A.json" || true
  sleep 3
  echo "-- run B: same attacker traffic + noise host bursts --"
  docker compose exec -T noise bash scenario_noise.sh &
  docker compose exec -T generator env VICTIM=$VIC COUNT=20 python3 scenario_realize.py
  wait || true
}

case "${1:-full}" in
  up) up ;;
  down) down ;;
  test2) up; test2 ;;
  full) up; run; test2 ;;
  *) echo "usage: $0 [full|up|down|test2]"; exit 1 ;;
esac
echo "== done. PCAPs in lab/pcaps/ =="
