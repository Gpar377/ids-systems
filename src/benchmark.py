"""Model benchmark on the honest protocol: train KDDTrain+, test KDDTest+.

KDDTest+ holds 17 attack types absent from training, so this measures
generalisation to unseen attacks. The in-distribution protocol (random split of
KDDTrain+) is reported alongside for contrast, clearly labelled -- it is where
the "99%" numbers in the literature come from.

Two levers are tested besides the model itself:
  log1p   : bytes/counts span ~9 orders of magnitude; MinMax alone squashes
            almost every src_bytes value to ~0. log1p before scaling fixes that
            for distance/gradient models (trees don't care).
  threshold: chosen on a NOVEL-ATTACK validation split -- whole attack types held
            out of training -- so the choice mimics the test condition without
            ever touching the test set.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (NUMERIC, RATE_FEATURES, TEST_FILE, TRAIN_FILE,  # noqa: E402
                        Preprocessor, binarize_labels, load_raw)
from surrogate import MLPSurrogate  # noqa: E402

RS = 42
RESULTS = Path(__file__).resolve().parent.parent / "results"
LOG_COLS = [c for c in NUMERIC if c not in RATE_FEATURES]  # heavy-tailed ones


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LOG_COLS] = np.log1p(out[LOG_COLS].astype(float))
    return out


def models() -> dict:
    return {
        "rf_paper": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=RS, n_jobs=-1),
        "rf_full": lambda: RandomForestClassifier(
            n_estimators=300, random_state=RS, n_jobs=-1),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300, random_state=RS, n_jobs=-1),
        "hist_gb": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=RS),
        "logreg": lambda: LogisticRegression(max_iter=3000, random_state=RS),
        "mlp": lambda: MLPSurrogate(seed=RS, epochs=12, verbose=False),
    }


def proba(clf, X) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


def novel_attack_split(df: pd.DataFrame, frac_types: float = 0.3, seed: int = RS):
    """Hold out whole attack TYPES (plus a slice of normal) as validation, so the
    validation set contains attacks the model never trained on -- the same
    condition KDDTest+ imposes."""
    rng = np.random.default_rng(seed)
    types = sorted(t for t in df["label"].unique() if t != "normal")
    held = set(rng.choice(types, size=max(1, int(len(types) * frac_types)),
                          replace=False))
    normal_idx = df.index[df["label"] == "normal"]
    val_normal = set(rng.choice(normal_idx, size=len(normal_idx) // 5, replace=False))
    is_val = df["label"].isin(held) | df.index.isin(val_normal)
    return df[~is_val], df[is_val], sorted(held)


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Maximise balanced accuracy (prior-free: the test attack ratio differs)."""
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        pred = (p >= t).astype(int)
        tpr = pred[y == 1].mean() if (y == 1).any() else 0
        tnr = 1 - pred[y == 0].mean() if (y == 0).any() else 0
        if (tpr + tnr) / 2 > best:
            best, best_t = (tpr + tnr) / 2, t
    return float(best_t)


def score(y, p, t) -> dict:
    pred = (p >= t).astype(int)
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


class SoftVote:
    """Average predicted probabilities of several fitted-on-demand models."""

    def __init__(self, makers):
        self.makers = makers
        self.fitted = []

    def fit(self, X, y):
        self.fitted = [m().fit(X, y) for m in self.makers]
        return self

    def predict_proba(self, X):
        p = np.mean([proba(m, X) for m in self.fitted], axis=0)
        return np.column_stack([1 - p, p])

    def __getstate__(self):
        # makers are lambdas (unpicklable) and only needed to fit; drop on save.
        state = self.__dict__.copy()
        state["makers"] = None
        return state


def robustness(seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Is the threshold gain real, or one lucky choice of held-out attack types?
    Repeat the whole select-threshold-then-test procedure over several splits."""
    tr_raw, te_raw = load_raw(TRAIN_FILE), load_raw(TEST_FILE)
    tr, te = log_transform(tr_raw), log_transform(te_raw)
    pre = Preprocessor().fit(tr)
    Xtr, ytr = pre.transform(tr), binarize_labels(tr["label"])
    Xte, yte = pre.transform(te), binarize_labels(te["label"])
    m = models()
    cands = {
        "rf_paper": m["rf_paper"], "rf_full": m["rf_full"],
        "extra_trees": m["extra_trees"], "hist_gb": m["hist_gb"],
        "ensemble": lambda: SoftVote([m["extra_trees"], m["hist_gb"], m["mlp"]]),
    }
    # Fit each candidate on full train ONCE; only the threshold varies by seed.
    full_p = {k: proba(mk().fit(Xtr, ytr), Xte) for k, mk in cands.items()}
    rows = []
    for seed in seeds:
        fit_df, val_df, held = novel_attack_split(tr, seed=seed)
        pv = Preprocessor().fit(fit_df)
        Xf, yf = pv.transform(fit_df), binarize_labels(fit_df["label"])
        Xv, yv = pv.transform(val_df), binarize_labels(val_df["label"])
        for k, mk in cands.items():
            thr = pick_threshold(yv, proba(mk().fit(Xf, yf), Xv))
            rows.append({"seed": seed, "model": k, "thr": round(thr, 3),
                         "held_types": ",".join(held),
                         **score(yte, full_p[k], thr)})
            print(f"  seed {seed} {k:12s} thr={thr:.2f} "
                  f"acc={rows[-1]['accuracy']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "benchmark_robustness.csv", index=False)
    summ = df.groupby("model")["accuracy"].agg(["mean", "std", "min", "max"])
    summ["acc@0.5"] = [score(yte, full_p[k], 0.5)["accuracy"] for k in summ.index]
    print("\nacross threshold-selection splits (log1p features):")
    print(summ.sort_values("mean", ascending=False).round(4).to_string())
    return df


def main() -> None:
    if "--robust" in sys.argv:
        robustness()
        return
    tr_raw, te_raw = load_raw(TRAIN_FILE), load_raw(TEST_FILE)
    rows = []

    for feat_mode in ["minmax", "log1p+minmax"]:
        tf = log_transform if feat_mode.startswith("log") else (lambda d: d)
        tr, te = tf(tr_raw), tf(te_raw)

        # threshold-selection split, built once per feature mode
        fit_df, val_df, held = novel_attack_split(tr)
        pre_v = Preprocessor().fit(fit_df)
        Xf, yf = pre_v.transform(fit_df), binarize_labels(fit_df["label"])
        Xv, yv = pre_v.transform(val_df), binarize_labels(val_df["label"])

        pre = Preprocessor().fit(tr)
        Xtr, ytr = pre.transform(tr), binarize_labels(tr["label"])
        Xte, yte = pre.transform(te), binarize_labels(te["label"])

        for name, make in models().items():
            t0 = time.time()
            # 1) threshold from the novel-attack validation split
            thr = pick_threshold(yv, proba(make().fit(Xf, yf), Xv))
            # 2) refit on ALL of train, evaluate once on KDDTest+
            clf = make().fit(Xtr, ytr)
            p = proba(clf, Xte)
            for tname, t in [("0.5", 0.5), ("novel-val", thr)]:
                rows.append({"protocol": "KDDTrain+ -> KDDTest+ (honest)",
                             "features": feat_mode, "model": name,
                             "threshold": tname, "thr_value": round(t, 3),
                             **score(yte, p, t)})
            print(f"  {feat_mode:13s} {name:12s} "
                  f"acc@0.5={rows[-2]['accuracy']:.4f}  "
                  f"acc@thr({thr:.2f})={rows[-1]['accuracy']:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # In-distribution contrast: random 80/20 split of KDDTrain+ (leaky by design).
    pre = Preprocessor().fit(tr_raw)
    X, y = pre.transform(tr_raw), binarize_labels(tr_raw["label"])
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    clf = models()["rf_paper"]().fit(Xa, ya)
    rows.append({"protocol": "random split of KDDTrain+ (in-distribution)",
                 "features": "minmax", "model": "rf_paper", "threshold": "0.5",
                 "thr_value": 0.5, **score(yb, proba(clf, Xb), 0.5)})
    print(f"  in-distribution rf_paper acc={rows[-1]['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "benchmark.csv", index=False)
    honest = df[df.protocol.str.contains("honest")].sort_values("accuracy",
                                                                ascending=False)
    print("\nTop 8 on the honest protocol:")
    print(honest.head(8).to_string(index=False, formatters={
        k: "{:.4f}".format for k in ["accuracy", "precision", "recall", "f1"]}))
    print(f"\nheld-out attack types used for threshold selection: {held}")
    print(f"wrote {RESULTS / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
