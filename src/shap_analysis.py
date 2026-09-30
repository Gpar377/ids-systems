"""SHAP explanations + stability, cross-referenced against mutability class.

The interesting question is not "which features matter" but "can an attacker
touch the features that matter". If the model's importance mass sits in Class C,
the model is leaning on evidence the attacker cannot forge -- which is an
argument for the model, not against it, and it is invisible without the
taxonomy. This is where explainability and the realizability mask meet.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mask import CLASS_A, CLASS_B, CLASS_C, MUTABLE
from preprocess import FEATURES, Preprocessor

RESULTS = Path(__file__).resolve().parent.parent / "results"
SHAP_DIR = RESULTS / "shap"


def feature_class(name: str) -> str:
    if name in CLASS_A:
        return "A_free"
    if name in CLASS_B:
        return "B_constrained"
    return "C_frozen"


def _to_matrix(sv) -> np.ndarray:
    """shap returns (n,f) or (n,f,classes) or a list per class; normalise to (n,f)
    of the positive (attack) class."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    return sv


def encoded_to_raw(mean_abs: np.ndarray, pre: Preprocessor) -> pd.Series:
    """Sum |SHAP| over a categorical's one-hot columns to get its raw importance."""
    return pd.Series(
        {f: float(mean_abs[pre.feature_index_map[f]].sum()) for f in FEATURES}
    ).sort_values(ascending=False)


def explain(clf, X: np.ndarray, pre: Preprocessor, n: int = 2000,
            seed: int = 42) -> pd.DataFrame:
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    sv = _to_matrix(shap.TreeExplainer(clf).shap_values(X[idx], check_additivity=False))
    raw = encoded_to_raw(np.abs(sv).mean(axis=0), pre)
    total = raw.sum()
    df = raw.rename("mean_abs_shap").to_frame()
    df["share"] = df.mean_abs_shap / total
    df["mutability"] = [feature_class(f) for f in df.index]
    df["attacker_controllable"] = [f in MUTABLE for f in df.index]
    df.index.name = "feature"
    return df.reset_index()


def stability(clf, X: np.ndarray, pre: Preprocessor, n: int = 1500) -> dict:
    """Do two disjoint samples agree on the ranking? A SHAP story that flips
    between subsamples is not an explanation."""
    from scipy.stats import spearmanr
    a = explain(clf, X[: len(X) // 2], pre, n=n, seed=1).set_index("feature")
    b = explain(clf, X[len(X) // 2:], pre, n=n, seed=2).set_index("feature")
    common = a.index
    rho, p = spearmanr(a.loc[common, "mean_abs_shap"], b.loc[common, "mean_abs_shap"])
    top10 = len(set(a.nlargest(10, "mean_abs_shap").index)
                & set(b.nlargest(10, "mean_abs_shap").index))
    return {"spearman_rho": float(rho), "p_value": float(p), "top10_overlap": top10}


def main() -> None:
    from model import load
    from preprocess import load_dataset

    d = load_dataset()
    pre, clf = d["preprocessor"], load()
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    print("\ncomputing SHAP (TreeExplainer)...")
    df = explain(clf, d["X_test"], pre)
    df.to_csv(SHAP_DIR / "shap_importance.csv", index=False)

    print("\ntop 15 features by mean |SHAP|:")
    print(df.head(15).to_string(index=False,
          formatters={"mean_abs_shap": "{:.5f}".format, "share": "{:.3%}".format}))

    by_class = df.groupby("mutability")["share"].sum().sort_index()
    print("\nimportance mass by mutability class:")
    for k, v in by_class.items():
        print(f"  {k:15s} {v:7.2%}")
    ctrl = df.loc[df.attacker_controllable, "share"].sum()
    print(f"\n  attacker-controllable share of total importance: {ctrl:.2%}")
    print(f"  frozen (Class C) share                         : "
          f"{df.loc[df.mutability=='C_frozen','share'].sum():.2%}")

    print("\nstability across disjoint subsamples:")
    s = stability(clf, d["X_test"], pre)
    print(f"  spearman rho   : {s['spearman_rho']:.4f}  (p={s['p_value']:.2e})")
    print(f"  top-10 overlap : {s['top10_overlap']}/10")

    pd.Series({**s, "attacker_controllable_share": ctrl,
               **{f"share_{k}": v for k, v in by_class.items()}}
              ).to_csv(SHAP_DIR / "shap_summary.csv", header=["value"])
    print(f"\nwrote {SHAP_DIR/'shap_importance.csv'}, {SHAP_DIR/'shap_summary.csv'}")


if __name__ == "__main__":
    main()
