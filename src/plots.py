"""Report figures, generated from results/ -- never from hand-typed numbers.

    python src/plots.py      -> results/plots/*.png

Colours follow the demo app: our IDS = blue, the paper's IDS = orange, the
adversarially trained RF = aqua (slots 1-3 of the validated reference palette;
aqua is below 3:1 contrast on white, so every bar it colours carries a value
label and every figure has a matching table in the report).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"

OURS, PAPER, ADV = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e5e4e0", "#9a9893"
LABEL = {"baseline": "Paper's IDS (RF)", "adv_trained": "Adversarially trained RF",
         "improved_ids": "Our IDS (ensemble)"}
COLOR = {"baseline": PAPER, "adv_trained": ADV, "improved_ids": OURS}

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_metrics(m: pd.DataFrame) -> None:
    """Clean accuracy / precision / recall / F1 for the three IDSs."""
    clean = m[m.eval_set == "clean"].set_index("model")
    metrics = ["accuracy", "precision", "recall", "f1"]
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    models = ["baseline", "adv_trained", "improved_ids"]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for k, mod in enumerate(models):
        vals = clean.loc[mod, metrics].astype(float).to_numpy() * 100
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=COLOR[mod],
                      edgecolor="white", linewidth=1.2, label=LABEL[mod])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.6, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Score on KDDTest+ (%)")
    _grid(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              fontsize=9.5)
    fig.savefig(OUT / "fig_metrics.png")
    plt.close(fig)


def fig_evasion(ev: pd.DataFrame) -> None:
    """Evasion rate (PGD), unrestricted vs realistic, as two panels -- so colour
    keeps meaning 'which IDS' and the threat model is the panel, not a hue."""
    ev = ev[ev.attack == "pgd"]
    rows = [("baseline", "static"), ("adv_trained", "static"), ("improved_ids", "static")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, con, title in zip(axes, [False, True],
                              ["Unrestricted attacker (paper's threat model)",
                               "Realistic attacker (realizability mask)"]):
        vals = [float(ev[(ev.model == mod) & (ev.threat_model == t)
                         & (ev.constrained == con)].evasion_rate.iloc[0]) * 100
                for mod, t in rows]
        bars = ax.bar(range(3), vals, 0.62, color=[COLOR[r[0]] for r in rows],
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(range(3), ["Paper's\nIDS", "Adv.-trained\nRF", "Our\nIDS"])
        ax.set_title(title, fontsize=10)
        _grid(ax)
    axes[0].set_ylabel("Evasion rate (%)")
    axes[0].set_ylim(0, 36)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig_evasion.png")
    plt.close(fig)


def fig_confusion() -> None:
    """Confusion matrices on KDDTest+, recomputed from the saved models."""
    import joblib
    import ids_model
    from preprocess import load_dataset
    from sklearn.metrics import confusion_matrix

    mm, lg = load_dataset(log=False), load_dataset(log=True)
    rf = joblib.load(RESULTS / "rf_baseline.joblib")
    ours = ids_model.load()["ids"]
    cms = [("Paper's IDS (RF, threshold 0.5)",
            confusion_matrix(mm["y_test"], rf.predict(mm["X_test"]))),
           (f"Our IDS (ensemble, threshold {ours.threshold:.2f})",
            confusion_matrix(lg["y_test"], ours.predict(lg["X_test"])))]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f4f8fd", "#9cc3ee", OURS, "#174a8b"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, (title, cm) in zip(axes, cms):
        rowpct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                dark = rowpct[i, j] > 0.55
                ax.text(j, i, f"{cm[i, j]:,}\n({rowpct[i, j]*100:.1f}%)", ha="center",
                        va="center", fontsize=10, color="white" if dark else INK)
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "fig_confusion.png")
    plt.close(fig)


def fig_robustness(rb: pd.DataFrame) -> None:
    """Accuracy over the 5 held-out-attack-type threshold splits, per model."""
    order = ["rf_paper", "rf_full", "extra_trees", "hist_gb", "ensemble"]
    names = {"rf_paper": "Paper's RF (100 trees, depth 15)", "rf_full": "RF (300 trees)",
             "extra_trees": "Extra Trees", "hist_gb": "Hist. gradient boosting",
             "ensemble": "Ensemble (our IDS)"}
    col = {"rf_paper": PAPER, "ensemble": OURS}
    bm = pd.read_csv(RESULTS / "benchmark.csv")
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for y, mod in enumerate(order):
        acc = rb[rb.model == mod].accuracy.to_numpy() * 100
        c = col.get(mod, NEUTRAL)
        ax.plot([acc.min(), acc.max()], [y, y], color=c, linewidth=2, alpha=0.45)
        ax.scatter(acc, np.full(len(acc), y), s=34, color=c, zorder=3,
                   edgecolor="white", linewidth=1)
        ax.text(acc.max() + 0.35, y, f"{acc.mean():.1f} ± {acc.std(ddof=1):.1f}",
                va="center", fontsize=9, color=INK)
        # same model at the default 0.5 threshold, log features
        if mod != "ensemble":
            base = bm[(bm.model == mod) & (bm.features == "log1p+minmax")
                      & (bm.threshold == "0.5")].accuracy
            if len(base):
                ax.scatter([float(base.iloc[0]) * 100], [y], marker="|", s=160,
                           color=INK2, zorder=2, linewidth=1.6)
    ax.set_yticks(range(len(order)), [names[m] for m in order])
    ax.set_xlabel("Accuracy on KDDTest+ (%)")
    ax.set_xlim(76, 96)
    _grid(ax, "x")
    ax.scatter([], [], s=34, color=NEUTRAL, label="tuned threshold, one dot per split")
    ax.scatter([], [], marker="|", s=160, color=INK2, label="default threshold 0.5")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              fontsize=9)
    fig.savefig(OUT / "fig_robustness.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RESULTS / "metrics.csv")
    ev = pd.read_csv(RESULTS / "evasion.csv")
    rb = pd.read_csv(RESULTS / "benchmark_robustness.csv")
    fig_metrics(m)
    fig_evasion(ev)
    fig_robustness(rb)
    fig_confusion()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
