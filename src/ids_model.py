"""The improved IDS: soft-vote ensemble on log1p features + a tuned threshold.

Why it beats the paper's RF on KDDTest+ (~90% vs 77.9%): mostly the threshold.
At 0.5 the forest misses ~37% of attacks -- novel attack types score as
"suspicious" but not past 0.5. The threshold here is chosen on validation splits
made of WHOLE ATTACK TYPES held out of training (mimicking KDDTest+'s 17 unseen
types), never on the test set, and is the median over several such splits so no
single lucky split decides it. See src/benchmark.py for the evidence.

Trade-off to state in the report: precision falls ~0.97 -> ~0.92 (more false
alarms) while recall rises ~0.63 -> ~0.89 (far fewer missed attacks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import SoftVote, models, novel_attack_split, pick_threshold  # noqa: E402
from preprocess import (LOG_COLS, TRAIN_FILE, Preprocessor,  # noqa: E402
                        binarize_labels, load_raw)

RESULTS = Path(__file__).resolve().parent.parent / "results"
BUNDLE = RESULTS / "ids_improved.joblib"
THRESHOLD_SEEDS = (0, 1, 2, 3, 4)


def make_ensemble() -> SoftVote:
    m = models()
    return SoftVote([m["extra_trees"], m["hist_gb"], m["mlp"]])


class ImprovedIDS:
    """Ensemble + threshold, exposing predict / predict_proba like sklearn."""

    def __init__(self, ensemble: SoftVote, threshold: float):
        self.ensemble = ensemble
        self.threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def select_threshold(verbose: bool = True) -> tuple[float, list[float]]:
    """Median of per-split thresholds; each split holds out whole attack types."""
    tr = load_raw(TRAIN_FILE)
    per_split = []
    for seed in THRESHOLD_SEEDS:
        fit_df, val_df, held = novel_attack_split(tr, seed=seed)
        pv = Preprocessor(log_cols=list(LOG_COLS)).fit(fit_df)
        Xf, yf = pv.transform(fit_df), binarize_labels(fit_df["label"])
        Xv, yv = pv.transform(val_df), binarize_labels(val_df["label"])
        p = make_ensemble().fit(Xf, yf).predict_proba(Xv)[:, 1]
        t = pick_threshold(yv, p)
        per_split.append(t)
        if verbose:
            print(f"  split {seed}: held out {held} -> threshold {t:.2f}", flush=True)
    return float(np.median(per_split)), per_split


def build(X_train: np.ndarray, y_train: np.ndarray,
          verbose: bool = True) -> ImprovedIDS:
    thr, per_split = select_threshold(verbose)
    if verbose:
        print(f"  chosen threshold (median of {per_split}) = {thr:.2f}")
    ens = make_ensemble().fit(X_train, y_train)
    return ImprovedIDS(ens, thr)


def save(ids: ImprovedIDS, pre: Preprocessor, path: Path = BUNDLE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ids": ids, "preprocessor": pre}, path)
    print(f"saved improved IDS -> {path}")


def load(path: Path = BUNDLE) -> dict:
    return joblib.load(path)


if __name__ == "__main__":
    # Import ourselves by module name so the pickled class is
    # ids_model.ImprovedIDS, not __main__.ImprovedIDS (which nothing could load).
    import ids_model
    from model import print_report
    from preprocess import load_dataset

    d = load_dataset(log=True)
    ids = ids_model.build(d["X_train"], d["y_train"])
    print_report(ids, d["X_test"], d["y_test"], "TEST (KDDTest+) improved IDS")
    ids_model.save(ids, d["preprocessor"])
