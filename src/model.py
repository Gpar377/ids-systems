"""Random Forest baseline. Config matches the paper exactly (CLAUDE.md)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_PATH = MODEL_DIR / "rf_baseline.joblib"

# Paper config -- do not tune. The ~77% is the honest generalisation number.
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=RANDOM_STATE,
)


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


def train(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = build_model()
    clf.fit(X, y)
    return clf


def evaluate(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray,
             name: str = "eval") -> dict:
    pred = clf.predict(X)
    m = {
        "split": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "n": int(X.shape[0]),
    }
    return m


def print_report(clf, X, y, name: str) -> dict:
    m = evaluate(clf, X, y, name)
    pred = clf.predict(X)
    print(f"\n--- {name} (n={m['n']}) ---")
    print(f"accuracy : {m['accuracy']:.4f}")
    print(f"precision: {m['precision']:.4f}")
    print(f"recall   : {m['recall']:.4f}")
    print(f"f1       : {m['f1']:.4f}")
    print("confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, target_names=["normal", "attack"],
                                digits=4, zero_division=0))
    return m


def save(clf, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    print(f"saved model -> {path}")


def load(path: Path = MODEL_PATH) -> RandomForestClassifier:
    return joblib.load(path)


if __name__ == "__main__":
    from preprocess import load_dataset

    d = load_dataset()
    print(f"\nRF params: {RF_PARAMS}")
    clf = train(d["X_train"], d["y_train"])

    print_report(clf, d["X_train"], d["y_train"], "TRAIN (in-sample)")
    test_m = print_report(clf, d["X_test"], d["y_test"], "TEST (KDDTest+)")

    save(clf)
    print(f"\n>>> BASELINE CLEAN TEST ACCURACY: {test_m['accuracy']:.4f} "
          f"({test_m['accuracy']*100:.2f}%)")
