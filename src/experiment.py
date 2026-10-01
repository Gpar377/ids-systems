"""Orchestrator: constrained vs unconstrained, baseline vs defended.

Every number in the report comes from here. Run:  python src/experiment.py
Outputs results/metrics.csv and results/run_metadata.json. No hand-typed numbers.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attacks import (attack_labels, build_surrogate, fgsm, pgd,
                     perturbation_report, wrap)
from defense import adversarial_train, make_training_adversarials
from mask import summarize
from model import evaluate, train
from preprocess import load_dataset

RESULTS = Path(__file__).resolve().parent.parent / "results"

# eps=0.2 in MinMax [0,1] space = 20% of each feature's observed train range.
# alpha*iters >= eps so PGD can actually reach its budget -- the published
# alpha=0.01/iters=5 reaches only 0.05 and is strictly weaker than its own FGSM.
EPS, ALPHA, ITERS = 0.2, 0.02, 10


def lib_versions() -> dict:
    import importlib.metadata as md
    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "shap",
            "adversarial-robustness-toolbox", "matplotlib", "seaborn", "joblib"]
    v = {p: md.version(p) for p in pkgs}
    v["python"] = platform.python_version()
    v["platform"] = platform.platform()
    return v


def build_attack_sets(art, X, y, pre) -> dict[str, np.ndarray]:
    """Four evaluation sets: {FGSM,PGD} x {unconstrained, constrained}.

    `y` here must be the substitute's predicted labels, not ground truth."""
    out = {}
    for label, p in [("unconstrained", None), ("constrained", pre)]:
        t0 = time.time()
        out[f"fgsm_{label}"] = fgsm(art, X, y, eps=EPS, pre=p)
        out[f"pgd_{label}"] = pgd(art, X, y, eps=EPS, alpha=ALPHA,
                                  iters=ITERS, pre=p)
        print(f"  built {label} sets in {time.time()-t0:.1f}s")
    return out


def evasion_rows(name: str, victim, sub, pre, Xte, yte,
                 threat: str = "static") -> list[dict]:
    """Evasion rate: of the ATTACK connections the victim catches when clean,
    what fraction can the attacker disguise as normal?

    This is the IDS-relevant threat. Whole-test-set "accuracy under attack" (the
    paper's protocol) also perturbs NORMAL users' traffic -- which an attacker
    does not control -- and mostly measures false alarms, not evasion. Here the
    attacker perturbs only their own attack traffic, targeting "normal" (they
    know their traffic is an attack, so label 1 is legitimate knowledge)."""
    Xa = Xte[yte == 1]
    Xc = Xa[victim.predict(Xa) == 1]                    # attacks caught clean
    y1 = np.ones(len(Xc), dtype=int)
    out = []
    for label, p in [("unconstrained", None), ("constrained", pre)]:
        for aname in ["fgsm", "pgd"]:
            if aname == "fgsm":
                adv = fgsm(sub, Xc, y1, eps=EPS, pre=p)
            else:
                adv = pgd(sub, Xc, y1, eps=EPS, alpha=ALPHA, iters=ITERS, pre=p)
            out.append({"model": name, "threat_model": threat, "attack": aname,
                        "constrained": label == "constrained",
                        "n_detected_attacks": int(len(Xc)),
                        "evasion_rate": float((victim.predict(adv) == 0).mean())})
    return out


def improved_ids_rows() -> tuple[list[dict], list[dict]]:
    """Same transfer attacks + realizability mask, against the improved IDS
    (ensemble on log1p features, tuned threshold). The attacker's surrogate is
    trained in the same log feature space the victim uses, and the mask
    constrains in raw units via pre.to_raw/from_raw (see mask.py)."""
    import ids_model
    print("\n[improved IDS] ensemble + tuned threshold, log1p features")
    d = load_dataset(log=True)
    if ids_model.BUNDLE.exists():
        bundle = ids_model.load()
    else:
        bundle = {"ids": ids_model.build(d["X_train"], d["y_train"]),
                  "preprocessor": d["preprocessor"]}
        ids_model.save(bundle["ids"], bundle["preprocessor"])
    ids, pre = bundle["ids"], bundle["preprocessor"]
    assert pre.encoded_names == d["preprocessor"].encoded_names
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    sub = build_surrogate(Xtr, ytr, verbose=False)
    print(f"  surrogate (log space) clean test acc: "
          f"{evaluate(sub, Xte, yte)['accuracy']:.4f}; threshold {ids.threshold:.2f}")
    adv = build_attack_sets(wrap(sub), Xte, attack_labels(sub, Xte), pre)

    rows = []
    for sname, X in [("clean", Xte), *adv.items()]:
        m = evaluate(ids, X, yte, sname)
        m.update(model="improved_ids", eval_set=sname,
                 threat_model="clean" if sname == "clean" else "static",
                 constrained=("constrained" in sname and "unconstrained" not in sname),
                 attack=sname.split("_")[0] if sname != "clean" else "none")
        rows.append(m)
    ev = evasion_rows("improved_ids", ids, sub, pre, Xte, yte)
    return rows, ev


def main() -> None:
    print("=" * 66)
    print("IDS adversarial realizability experiment")
    print("=" * 66)
    d = load_dataset()
    pre = d["preprocessor"]
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    ms = summarize(pre)
    print(f"\nmask: {ms['continuously_mutable']}/41 raw features mutable; "
          f"{ms['n_mutable_encoded_cols']}/{ms['n_encoded_cols']} encoded dims "
          f"({ms['frac_encoded_mutable']*100:.1f}%)")

    print("\n[1/5] victim RandomForest")
    rf = train(Xtr, ytr)

    print("[2/5] attacker's substitute model (MLP, nonlinear)")
    sub = build_surrogate(Xtr, ytr)
    art = wrap(sub)
    sub_acc = evaluate(sub, Xte, yte)["accuracy"]
    print(f"  substitute clean test acc: {sub_acc:.4f}")

    print("[3/5] adversarial test sets")
    # Predicted, not true, labels -- see attacks.attack_labels.
    yte_hat = attack_labels(sub, Xte)
    adv_test = build_attack_sets(art, Xte, yte_hat, pre)

    print("[4/6] adversarial training (PGD-unconstrained examples from TRAIN)")
    t0 = time.time()
    ytr_hat = attack_labels(sub, Xtr)
    Xtr_adv = make_training_adversarials(art, Xtr, ytr_hat, EPS, ALPHA, ITERS, None)
    rf_def = adversarial_train(Xtr, ytr, Xtr_adv)
    print(f"  defended model trained in {time.time()-t0:.1f}s")

    # The static evaluation (adv_test) attacks a surrogate of the BASELINE, so it
    # does not test whether the defense holds against an attacker who adapts. An
    # adaptive attacker retrains a NEW surrogate to mimic the DEFENDED model
    # (query access -> the defended model's own predicted labels), then attacks
    # that. This is the honest defense evaluation.
    print("[5/6] adaptive attacker (surrogate retrained vs the DEFENDED model)")
    ytr_def = rf_def.predict(Xtr)
    sub_adapt = build_surrogate(Xtr, ytr_def, verbose=False)
    art_adapt = wrap(sub_adapt)
    agreement = float((sub_adapt.predict(Xte) == rf_def.predict(Xte)).mean())
    print(f"  adaptive surrogate agrees with defended model on {agreement:.3f} of test")
    yte_adapt = attack_labels(sub_adapt, Xte)
    adv_adapt = {f"adaptive_{k}": v
                 for k, v in build_attack_sets(art_adapt, Xte, yte_adapt, pre).items()}

    print("[6/6] evaluating")
    rows = []
    # Static threat model: baseline + defended on baseline-surrogate examples.
    for mname, clf in [("baseline", rf), ("adv_trained", rf_def)]:
        for sname, X in [("clean", Xte), *adv_test.items()]:
            m = evaluate(clf, X, yte, sname)
            m["model"] = mname
            m["eval_set"] = sname
            m["threat_model"] = "clean" if sname == "clean" else "static"
            m["constrained"] = ("constrained" in sname
                                and "unconstrained" not in sname)
            m["attack"] = sname.split("_")[0] if sname != "clean" else "none"
            rows.append(m)
    # Adaptive threat model: defended model vs a surrogate of itself.
    for sname, X in adv_adapt.items():
        m = evaluate(rf_def, X, yte, sname)
        m["model"] = "adv_trained"
        m["eval_set"] = sname
        m["threat_model"] = "adaptive"
        m["constrained"] = ("constrained" in sname and "unconstrained" not in sname)
        m["attack"] = sname.split("_")[1]   # adaptive_<attack>_<...>
        rows.append(m)

    imp_rows, imp_ev = improved_ids_rows()
    rows += imp_rows

    print("\n[evasion] attacker disguises only their OWN attack traffic")
    ev = (evasion_rows("baseline", rf, sub, pre, Xte, yte)
          + evasion_rows("adv_trained", rf_def, sub, pre, Xte, yte)
          + evasion_rows("adv_trained", rf_def, sub_adapt, pre, Xte, yte,
                         threat="adaptive")
          + imp_ev)
    ev_df = pd.DataFrame(ev)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(RESULTS / "evasion.csv", index=False)

    df = pd.DataFrame(rows)[["model", "eval_set", "threat_model", "attack",
                             "constrained", "accuracy", "precision", "recall",
                             "f1", "n"]]

    # Degradation relative to each model's own clean accuracy.
    clean = df[df.eval_set == "clean"].set_index("model")["accuracy"]
    df["clean_acc"] = df["model"].map(clean)
    df["degradation_pts"] = (df["clean_acc"] - df["accuracy"]) * 100

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "metrics.csv", index=False)

    perturb = {k: perturbation_report(v, Xte, pre) for k, v in adv_test.items()}
    for p in perturb.values():
        p.pop("cols_moved", None)  # keep the json small; names are in mask.py

    meta = {
        "libraries": lib_versions(),
        "attack_params": {"eps": EPS, "alpha": ALPHA, "iters": ITERS,
                          "note": "alpha*iters >= eps so PGD reaches its budget"},
        "random_state": 42,
        "data": {"n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                 "n_encoded_features": int(Xtr.shape[1]),
                 "novel_attack_types_in_test": len(d["novel_attack_types"])},
        "mask": ms,
        "perturbation": perturb,
        "surrogate": {"clean_test_accuracy": float(sub_acc),
                      "adaptive_agreement_with_defended": agreement},
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 66)
    print(df.to_string(index=False,
                       formatters={"accuracy": "{:.4f}".format,
                                   "precision": "{:.4f}".format,
                                   "recall": "{:.4f}".format,
                                   "f1": "{:.4f}".format,
                                   "clean_acc": "{:.4f}".format,
                                   "degradation_pts": "{:+.2f}".format}))
    print("=" * 66)

    for mname in ["baseline", "improved_ids"]:
        print(f"\nrealizability ({mname}): how much attack survives the mask")
        base = df[df.model == mname].set_index("eval_set")["degradation_pts"]
        for atk in ["fgsm", "pgd"]:
            u, c = base[f"{atk}_unconstrained"], base[f"{atk}_constrained"]
            surv = (c / u * 100) if u else float("nan")
            print(f"  {atk.upper():5s} unconstrained {u:+.2f} pts -> "
                  f"constrained {c:+.2f} pts ({surv:.1f}% survives)")

    # Static vs adaptive: does the defense still hold when the attacker retrains
    # against the defended model? Positive degradation = defense actually helps.
    print("\ndefense honesty (defended model): static vs adaptive attacker")
    dd = df[df.model == "adv_trained"].set_index("eval_set")["degradation_pts"]
    for atk in ["fgsm", "pgd"]:
        s = dd[f"{atk}_unconstrained"]
        a = dd[f"adaptive_{atk}_unconstrained"]
        print(f"  {atk.upper():5s} static {s:+.2f} pts -> adaptive {a:+.2f} pts "
              f"(unconstrained)")
        sc = dd[f"{atk}_constrained"]
        ac = dd[f"adaptive_{atk}_constrained"]
        print(f"        constrained: static {sc:+.2f} pts -> adaptive {ac:+.2f} pts")
    print("\n  (a defended model that only survives the STATIC attacker but "
          "collapses\n   under the ADAPTIVE one has not really been made robust.)")

    print("\nEVASION RATE -- of attacks caught clean, % disguised as normal:")
    piv = ev_df.assign(cond=np.where(ev_df.constrained, "realistic", "unrestricted"))
    piv = piv.pivot_table(index=["model", "threat_model", "attack"],
                          columns="cond", values="evasion_rate")
    print((piv * 100).round(2).to_string())
    print(f"\nwrote {RESULTS/'metrics.csv'} and {RESULTS/'run_metadata.json'}")


if __name__ == "__main__":
    main()
