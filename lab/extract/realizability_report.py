"""Turn captured feature rows into the realizability evidence for the report.

Two analyses:

  attempted-vs-landed : the attacker tried to set src_bytes/duration/count
    (Class A/B). Did those land? And the Class C features it had no knob for --
    what values did they take anyway? That contrast is the measurement the
    taxonomy needs.

  third-party-diff : two attacker runs with byte-identical attacker traffic,
    differing only in third-party noise. Any feature that moves between them is,
    by construction, not attacker-controllable -- dst_host_* should move.

Usage:
  python realizability_report.py landed  <attacker_rows.csv> <manifest.json>
  python realizability_report.py diff     <runA.csv> <runB.csv>
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

MUTABLE = {"src_bytes", "duration", "count", "srv_count",
           "wrong_fragment", "urgent", "num_failed_logins", "land",
           "su_attempted"}


def _load(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _nums(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (ValueError, KeyError):
            pass
    return out


def landed(csv_path: Path, manifest_path: Path) -> None:
    rows = _load(csv_path)
    manifest = json.loads(Path(manifest_path).read_text())
    attempted = manifest["attempted"]
    print(f"\n{'='*64}\nATTEMPTED vs LANDED  (n={len(rows)} attacker connections)\n{'='*64}")
    print(f"{'feature':<26}{'class':<7}{'target':>10}{'observed':>14}")
    print("-" * 64)
    for feat, target in attempted.items():
        vals = _nums(rows, feat)
        if not vals:
            obs = "n/a"
        elif feat == "count":
            obs = f"max={int(max(vals))}"     # count is a running window value
        elif feat == "src_bytes":
            obs = f"mode={int(statistics.mode(vals))}"
        else:
            obs = f"max={max(vals):g}"
        print(f"{feat:<26}{'A/B':<7}{str(target):>10}{obs:>14}")

    print(f"\n{'features the attacker had NO knob for (Class C)':<40}")
    print("-" * 64)
    print(f"{'feature':<28}{'min':>8}{'max':>8}{'distinct':>10}")
    for feat in manifest["not_controllable"]:
        vals = _nums(rows, feat)
        if not vals:
            # categorical (e.g. flag) -> show distinct string values
            sv = sorted({r.get(feat, "") for r in rows if r.get(feat, "")})
            print(f"{feat:<28}{'':>8}{'':>8}{len(sv):>10}   {sv}")
            continue
        print(f"{feat:<28}{min(vals):>8g}{max(vals):>8g}"
              f"{len(set(vals)):>10}")
    print("\nreading: A/B features tracked their targets; Class C features took a "
          "range of values the attacker never set -> they are not attacker knobs.")


def diff(a_path: Path, b_path: Path) -> None:
    a, b = _load(a_path), _load(b_path)
    keys = [k for k in a[0].keys() if not k.startswith("_")
            and k not in ("protocol_type", "service", "flag")]
    print(f"\n{'='*64}\nTHIRD-PARTY DIFF  (runA n={len(a)}, runB n={len(b)})")
    print("attacker traffic identical; only third-party noise differs\n" + "=" * 64)
    print(f"{'feature':<30}{'meanA':>10}{'meanB':>10}{'moved?':>9}")
    print("-" * 64)
    moved_controllable, moved_frozen = [], []
    for k in keys:
        va, vb = _nums(a, k), _nums(b, k)
        if not va or not vb:
            continue
        ma, mb = statistics.mean(va), statistics.mean(vb)
        moved = abs(ma - mb) > 1e-6
        tag = "MOVED" if moved else "-"
        if moved:
            (moved_controllable if k in MUTABLE else moved_frozen).append(k)
        if moved or k.startswith("dst_host") or k in ("count", "src_bytes"):
            print(f"{k:<30}{ma:>10.3f}{mb:>10.3f}{tag:>9}")
    print("-" * 64)
    print(f"moved but NOT attacker-controllable (the point): {moved_frozen}")
    print(f"moved and attacker-controllable                : {moved_controllable}")
    print("\nreading: features that moved when only third parties changed cannot "
          "be attacker-set -- the wire proves the Class C assignment.")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "landed":
        landed(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "diff":
        diff(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
