# Live-capture lab — turning the mask from assumption into measurement

The ML side of this project claims 31 of 41 NSL-KDD features are not
attacker-controllable over TCP/IP. This lab **proves it on real packets** rather
than asserting it from a table. It is the networking half of the case study
(Modules 1–3) and the empirical backing for `src/mask.py`.

Everything runs in an **isolated Docker bridge** (`internal: true`) — no route to
the internet or the LAN. Nothing here touches a network you don't own.

```
generator (192.168.90.20, attacker)  ─┐
noise     (192.168.90.30, 3rd party) ─┼─ idsnet ── victim (192.168.90.10)
                                       │              ▲
capture (tcpdump, in victim netns) ────┘──────────────┘  ← sensor vantage point
```

## Requirements

- Docker Desktop running.
- The Python venv (`../.venv`) with `dpkt` — already installed.

## Run it

From `lab/` in a shell with Docker on PATH:

```bash
bash run_capture.sh full      # baseline + scan + realize + third-party test
```

PCAPs land in `lab/pcaps/`. Then extract features (from the project venv):

```bash
# all connections
../.venv/Scripts/python.exe extract/pcap_to_nslkdd.py pcaps/capture.pcap -o pcaps/all.csv
# attacker-only rows, for the realizability comparison
../.venv/Scripts/python.exe extract/pcap_to_nslkdd.py pcaps/capture.pcap \
    --src 192.168.90.20 -o pcaps/attacker.csv
```

## The two experiments

### Test 1 — what can the attacker actually set?

`scenario_realize.py` tries to hit target values for the mutable features
(`src_bytes`, `duration`, `count`) against the real victim and writes
`pcaps/realize_manifest.json` listing what it *attempted* and what it had *no
knob for*. Compare attempted vs landed:

```bash
../.venv/Scripts/python.exe extract/realizability_report.py landed \
    pcaps/attacker.csv pcaps/realize_manifest.json
```

Expected: `src_bytes`/`duration`/`count` track their targets; the Class C
features take a spread of values the script never set.

### Test 2 — the killer demo

Two attacker runs with byte-identical attacker traffic, differing only in what
the **noise** host sends. Any feature that moves cannot be attacker-controlled.

```bash
bash run_capture.sh test2
# extract attacker rows from each run's pcap, then:
../.venv/Scripts/python.exe extract/realizability_report.py diff runA.csv runB.csv
```

Expected: `dst_host_count`, `dst_host_srv_count`, `dst_host_*_rate` move; the
attacker's own `src_bytes`/`duration` do not. That is the money screenshot: the
host-based window features shift with the attacker doing nothing differently.

## What the extractor does and does not recover

- **Recovered from the wire:** basic features (1–9) and all window features
  (23–41), with flag/window definitions matching the KDD Cup 1999 derivation
  NSL-KDD inherits (2-second time window; last-100-connection host window).
- **Not recoverable (emitted 0):** content features (10–22) — `logged_in`,
  `num_root`, etc. These are victim/host state that never appears on the wire,
  which is *itself* the taxonomy's argument for why they are Class C.

## Files

```
docker-compose.yml           isolated 4-service lab (internal network)
victim/                      multi-port TCP victim (SF on open, REJ on closed)
generator/                   attacker/noise toolkit + scenarios
  scenario_baseline.sh       benign background traffic
  scenario_scan.sh           recon sweep (raises count, Class B)
  scenario_realize.py        Test 1: attempt a target vector on the wire
  scenario_noise.sh          third-party traffic for Test 2
extract/
  pcap_to_nslkdd.py          PCAP -> 41-feature CSV
  realizability_report.py    attempted-vs-landed and third-party-diff reports
run_capture.sh               one-command driver
pcaps/                       output (git-ignored)
```
