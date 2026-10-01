# Status Report — IDS Case Study

**Project:** Improving Network Security through Intrusion Detection Systems
(SRM 21CSC302J, Computer Networks, Topic 10)
**Date:** 2026-09-30
**Repo:** `D:\SRM KTR\projects\IDS systems`

---

## 1. One-line summary

We built an IDS that is **more accurate (77.9% → 89.6% on unseen attacks)** and
**much harder to evade (31% → 5% evasion)** than the published one. We showed
that most of the published adversarial threat relies on values no packet can
carry, and we wrapped everything in an **offline demo app** for the viva.

---

## 2. What this project argues

ML intrusion detectors are usually attacked by nudging all 41 NSL-KDD features
freely. But a feature vector is not traffic: an adversarial example is only a
threat if a real attacker can *send* it. Over TCP/IP an attacker controls only
**9 of 41 features** (1 freely, 8 within limits); the other 32 are decided by the
victim, by third-party traffic, or cannot be moved by a gradient step at all. We
re-run the attacks under a **realizability mask** and measure what survives.
That makes it a **networking critique of an ML result**.

---

## 3. Progress

| Step | Description | Status |
|---|---|---|
| 0 | Scaffold, venv, pinned deps, data verified | ✅ |
| 1 | Paper's RandomForest baseline | ✅ 77.90% |
| 2 | FGSM / PGD, transfer threat model | ✅ |
| 3 | Adversarial training + adaptive-attacker check | ✅ defense holds |
| 4 | SHAP (now supporting evidence only) | ✅ |
| 5 | **Realizability mask** (the contribution) | ✅ |
| 6 | **Improved IDS** | ✅ **89.61%** |
| 7 | **Evasion-rate evaluation** | ✅ |
| 8 | **Demo app** (4 views, offline) | ✅ |
| 9 | Live capture on real packets | ⚙️ coded + tested; run it on the Docker machine |

46 tests pass. Python 3.12.13; everything on the D: drive.

---

## 4. Accuracy: 77.9% → 89.6%, measured honestly

Train on KDDTrain+, test on KDDTest+ (17 attack types never seen in training).

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Paper's RandomForest (threshold 0.5) | 77.90% | 0.969 | 0.632 |
| **Our IDS** (ensemble + tuned threshold) | **89.61%** | 0.924 | 0.891 |
| *Random split of KDDTrain+ (for contrast)* | *99.76%* | | |

**What was wrong:** mostly the decision threshold. At 0.5 the forest missed 37%
of attacks: novel attack types score as suspicious, but not past 0.5.

**Our IDS:** soft-vote of ExtraTrees + HistGradientBoosting + MLP on log1p
features (bytes and counts span ~9 orders of magnitude). The alarm threshold is
chosen on validation splits where **whole attack types are held out of training**,
which mimics KDDTest+ without ever touching the test set. It is the median over
5 such splits (0.04); across those splits accuracy is **89.6% ± 1.0%**
(`results/benchmark_robustness.csv`).

**Trade-off to state:** precision 0.97 → 0.92 (more false alarms) for recall
0.63 → 0.89 (missed attacks fall from 37% to 11%).

**Caveat:** log-scaling and the candidate models were picked after a first test
run, a mild optimistic bias. The main lever, the threshold, improves every model
under both feature settings, so it is not an artifact of that choice.

The 99.76% random-split number is the "99%" seen in the literature: it tests on
attack types the model already saw. We do not use it as our result.

---

## 5. The right metric: evasion rate

"Accuracy under attack" (the paper's protocol) perturbs **every** test connection,
including normal users' traffic. That mostly manufactures false alarms: under
attack our IDS's recall *rises* to 0.9995 while precision collapses, so the
accuracy drop measures normal traffic made to look malicious, which no attacker
controls. The IDS-relevant question is:

> Of the attacks the IDS catches, what fraction can the attacker disguise as normal?

| IDS | Unrestricted (FGSM / PGD) | Realistic, masked (FGSM / PGD) |
|---|---|---|
| Paper's RF | 26.7% / **31.1%** | 13.1% / 13.0% |
| Adversarially trained RF (static) | 0.0% / 0.0% | 7.1% / 7.1% |
| Adversarially trained RF (adaptive attacker) | 0.0% / 0.0% | 6.6% / 6.9% |
| **Our IDS** | 0.1% / **4.9%** | **1.8% / 3.0%** |

From `results/evasion.csv`. Three findings:

1. **Our IDS is both more accurate and far harder to evade.** About 1 in 3 attacks
   gets past the paper's RF; at most about 1 in 20 gets past ours.
2. **Realistic evasion is about half of unrestricted** on the paper's model, so the
   mask thesis holds on the right metric.
3. **Adversarial training on impossible perturbations misses the possible ones.**
   The defended RF blocks every unrestricted attack but lets through ~7% of
   realistic ones. Train against the threat you actually face.

---

## 6. Realizability on the paper's accuracy metric (for reference)

Whole-test-set degradation, kept so the reproduction is complete:

| Attack | Unrestricted | Realistic | Surviving |
|---|---|---|---|
| FGSM | −10.24 pts | −3.83 pts | 37.4% |
| PGD | −24.13 pts | −4.24 pts | 17.6% |

The masked attacker moves 9 of 122 encoded dimensions (7.4%). SHAP explains
why so little survives: **80.7% of the RF's importance mass sits in features no
attacker controls** (ρ = 0.9995 stability across subsamples).

---

## 7. Methodological findings worth stating in the viva

1. **The published PGD never reaches its budget:** α=0.01 × 5 steps = 0.05 < ε=0.2.
   We use α=0.02, 10 steps.
2. **A linear surrogate leaks the label:** its gradient is always ±w, so the
   perturbation encodes the class and adversarial training scores a fake 100%.
   We use a nonlinear MLP surrogate and predicted labels.
3. **Accuracy-under-attack conflates evasion with false-alarm injection** (§5).
4. **Adaptive attacker:** the defense was re-tested against an attacker whose
   surrogate mimics the *defended* model (93.4% agreement); no collapse.
5. **Functional realizability (open point):** even allowed changes can break the
   attack itself. In the demo, a `pod` (ping of death) attack's only realistic
   change is `src_bytes` 1480 → 0, and a ping of death with no payload is no longer
   an attack. The mask enforces what the wire allows, not what keeps the attack
   working, so our realistic evasion numbers are an *upper bound*.

---

## 8. Demo app

```bash
.venv/Scripts/python.exe demo/app.py     # or double-click demo/run_demo.bat
```

Offline, standard library only, opens at http://127.0.0.1:8765. Four views:

- **Overview:** headline numbers and the evasion chart, read live from `results/*.csv`.
- **Network simulation:** animated topology replaying real KDDTest+ connections
  through both IDSs, tallying misses and false alarms.
- **Attack lab:** pick an attack, choose unrestricted or realistic, and watch
  features move. Impossible values are flagged per feature, and the attacks are
  computed live.
- **Live capture:** drop a `.pcap` from the Docker lab and every connection is
  scored by both IDSs.

Verified end to end in a browser: all four views, no console errors, no
horizontal overflow at phone width. A suggested 3-minute viva flow is in
`demo/README.md`.

---

## 9. Live capture

The isolated Docker lab (`lab/`) and the PCAP → 41-feature extractor are built;
the extractor is validated by 6 unit tests on synthetic packets. It can't run on
this laptop because the campus network blocks UDP/53 DNS, so Docker can't pull
images. Run it on the machine that has working Docker:

1. Copy `lab/` over, run `bash run_capture.sh full`.
2. Bring back `lab/pcaps/capture.pcap` and drop it into the demo's Live capture tab.

A synthetic test capture confirmed the path: both IDSs flag all 40 port-sweep
connections.

---

## 10. Repository map

```
src/
  preprocess.py    encoding; optional log1p with exact raw<->model mapping
  model.py         paper's RandomForest
  ids_model.py     improved IDS (ensemble + held-out-attack-type threshold)
  benchmark.py     model comparison + threshold robustness
  surrogate.py     attacker's MLP surrogate
  attacks.py       FGSM / PGD, mask applied every step
  mask.py          realizability mask (works in raw units in both feature spaces)
  defense.py       adversarial training
  experiment.py    everything -> results/metrics.csv, evasion.csv
  shap_analysis.py supporting evidence
demo/              offline web app (app.py + static/)
lab/               Docker capture lab + PCAP extractor
tests/             46 tests
results/           all numbers in this report
```

---

## 11. Open items

- Run the capture lab on the Docker machine (§9).
- `CLAUDE.md` still says "do not improve the ~77%"; it should be updated to match
  the new direction.
- Plots for the written report (`results/plots/` is still empty).
- Write up Modules 1–3 (networking fundamentals) around the lab.
