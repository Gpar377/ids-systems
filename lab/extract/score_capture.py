"""Run real captured connections through the trained IDS.

Closes the loop: the same RandomForest from src/model.py, fed features extracted
from our own PCAP. Answers "does the model flag the real scan/realize traffic?"
and -- the point of the whole project -- lets us apply the realizability mask to
a REAL connection and see the model's verdict before vs after a realizable
perturbation.

The captured rows carry content features (10-22) as 0 and may use services unseen
in training; the fitted preprocessor's one-hot encoder handles unknowns as an
all-zero block, so this is a graceful, if imperfect, transfer to live data. That
gap is expected and worth stating in the report.

Usage:
  python score_capture.py <captured.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from model import load                       # noqa: E402
from preprocess import FEATURES, load_dataset  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cap = pd.read_csv(sys.argv[1])
    cap = cap[[c for c in cap.columns if not c.startswith("_")]]

    # Refit the preprocessor exactly as training did, then transform live rows.
    d = load_dataset()
    pre = d["preprocessor"]
    missing = [f for f in FEATURES if f not in cap.columns]
    if missing:
        print(f"WARNING: capture missing {len(missing)} features, filling 0: {missing}")
        for f in missing:
            cap[f] = 0
    X = pre.transform(cap[FEATURES])

    clf = load()
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]

    print(f"\nscored {len(cap)} real captured connections")
    print(f"  flagged as ATTACK : {int(pred.sum())} ({pred.mean():.1%})")
    print(f"  mean attack proba : {proba.mean():.3f}")
    if "service" in cap.columns:
        by_svc = pd.DataFrame({"service": cap["service"].values,
                               "flag": cap.get("flag", ""),
                               "pred": pred, "proba": proba.round(3)})
        print("\nby service:")
        print(by_svc.groupby("service").agg(
            n=("pred", "size"), flagged=("pred", "sum"),
            mean_proba=("proba", "mean")).to_string())
    print("\nnote: 0-valued content features + unseen services make this a "
          "conservative lower bound on detection, not the KDDTest+ number.")


if __name__ == "__main__":
    main()
