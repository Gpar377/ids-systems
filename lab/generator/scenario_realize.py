"""Test 1: can the attacker actually PRODUCE a target feature vector on the wire?

mask.py says only src_bytes (A) and a handful of B features are attacker-set.
This script tries to hit specific target values for the mutable features against
the real victim, and writes a manifest of what it ATTEMPTED. The extractor then
reads the resulting PCAP and reports what actually LANDED. Comparing the two is
the empirical test of the mask: a Class C feature has no knob here at all, which
is the whole point.

Run from the attacker host:  python3 scenario_realize.py
"""
import json
import os
import socket
import time

VICTIM = os.environ.get("VICTIM", "192.168.90.10")
PORT = int(os.environ.get("PORT", "80"))          # -> service=http
TARGET_SRC_BYTES = int(os.environ.get("SRC_BYTES", "512"))  # Class A: free
TARGET_DURATION = float(os.environ.get("DURATION", "3"))    # Class B: hold time
TARGET_COUNT = int(os.environ.get("COUNT", "40"))           # Class B: raise-only

# What an attacker can even attempt to dial, per the taxonomy.
ATTACKER_CONTROLLABLE = {
    "src_bytes": TARGET_SRC_BYTES,   # A  -- bytes we choose to send
    "duration": TARGET_DURATION,     # B  -- how long we hold the connection
    "count": TARGET_COUNT,           # B  -- how many connections we open (up only)
}
# Features with NO knob in this script -- there is no line of code that could set
# them, because the wire does not expose them to the sender.
NOT_CONTROLLABLE = [
    "flag",                 # the victim decides how the connection resolves
    "dst_bytes",            # bytes the victim chooses to send back
    "logged_in",            # only 1 if auth actually succeeded
    "serror_rate", "rerror_rate",             # victim + window
    "same_srv_rate", "diff_srv_rate",         # window aggregates
    "dst_host_count", "dst_host_srv_count",   # last-100-conns to victim
    "dst_host_same_srv_rate", "dst_host_serror_rate",  # third-party window
]


def one_connection(hold: float, payload: bytes) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    try:
        s.connect((VICTIM, PORT))
        s.sendall(payload)          # <-- this is the only feature we truly set
        if hold > 0:
            time.sleep(hold)        # hold the connection open -> duration
        try:
            s.recv(256)
        except OSError:
            pass
    finally:
        s.close()


def main() -> None:
    payload = b"G" + b"x" * (TARGET_SRC_BYTES - 1)  # exactly TARGET_SRC_BYTES
    print(f"[realize] target -> {ATTACKER_CONTROLLABLE}")
    t0 = time.time()
    # Fire TARGET_COUNT connections in a tight burst to raise `count` within the
    # 2-second window. Only the first holds for `duration`, to show duration is
    # settable independently.
    for i in range(TARGET_COUNT):
        one_connection(TARGET_DURATION if i == 0 else 0.0, payload)
    elapsed = time.time() - t0
    manifest = {
        "victim": VICTIM, "port": PORT, "service": "http",
        "attempted": ATTACKER_CONTROLLABLE,
        "not_controllable": NOT_CONTROLLABLE,
        "wall_seconds": round(elapsed, 2),
        "note": "compare 'attempted' vs extractor output; "
                "'not_controllable' features have no knob in this script",
    }
    out = "/pcaps/realize_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[realize] {TARGET_COUNT} connections in {elapsed:.2f}s")
    print(f"[realize] wrote {out}")


if __name__ == "__main__":
    main()
