# CLAUDE.md — IDS Case Study Build

If a `paper-code/` folder or any existing NSL-KDD / RF / ART scripts already exist in this repo, read them first and adapt rather than rebuild. This file assumes a from-scratch build; downgrade to a patch if prior code is present.

## What this is

A Computer Networks case study (SRM 21CSC302J, Topic 10: "Improving Network Security through Intrusion Detection Systems"). It reproduces and then extends a published IEEE paper (Gopa & Grace Shalini, ICADS 2026) on adversarial robustness + explainability for ML-based IDS.

The paper: NSL-KDD → Random Forest → SHAP → FGSM/PGD attacks → adversarial training defense.

The novel extension (this repo's whole reason to exist): the paper perturbs all 41 NSL-KDD features freely, but most features are not attacker-controllable over real TCP/IP. We re-run the attacks under a realizability mask and measure how much of the reported degradation survives. This is a networking critique of an ML result — that framing must survive into the code structure and comments.

## Goal (in order)

1. Reproduce the paper's baseline: RF on NSL-KDD, ~77% clean accuracy.
2. Reproduce the attacks: FGSM (ε=0.2), PGD (5 iter, α=0.01, ε=0.2); confirm degradation to ~70.6% / ~70.8%.
3. Reproduce the defense: adversarial training → ~80.6% / ~80.4%, clean stable ~76.75%.
4. Reproduce SHAP explanations + stability check.
5. NEW: apply the realizability mask, re-run constrained FGSM/PGD, tabulate constrained-vs-unconstrained degradation.
6. NEW: an improved IDS with better honest accuracy (`src/ids_model.py`, ~89.6% on KDDTest+), evaluated by **evasion rate**: of the attacks the IDS catches, the fraction an attacker can disguise as normal.

Steps 1–4 validate we match the paper. Step 5 is the contribution. Step 6 was added on 2026-09-30 at the user's request: exact reproduction of the paper's numbers is not required, and the project should not be framed as XAI.

## Stack

- Python 3.11 or 3.12 (not 3.13+ — check ART/SHAP wheels first)
- scikit-learn (RandomForest, metrics)
- pandas, numpy
- shap
- adversarial-robustness-toolbox (IBM ART) — for FGSM/PGD. Do NOT use CleverHans.
- matplotlib, seaborn (plots only)
- Use a venv. Pin exact versions in `requirements.txt` and record them in the report.

## Repo structure

```
data/            KDDTrain+.txt, KDDTest+.txt  (git-ignored, document how to obtain)
src/
  preprocess.py  load, encode protocol_type/service/flag, scale, FIXED column order
  model.py       RF train/eval (random_state fixed), save/load model
  attacks.py     FGSM/PGD via ART
  mask.py        THE NOVEL BIT — realizability mask (see spec below)
  shap_analysis.py
  experiment.py  orchestrates constrained vs unconstrained → results/metrics.csv
results/         metrics.csv, plots/, shap/
requirements.txt
README.md
```

## Model config (match the paper exactly)

- `RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)`
- Binary framing (normal vs attack) unless the paper's multiclass is being reproduced — confirm from data before assuming.

## NSL-KDD facts (do not hallucinate these)

- 41 features + 1 label (+ a difficulty column in some file variants — check and drop it).
- KDDTrain+ ≈ 125,973 rows; KDDTest+ ≈ 22,544 rows (exact count varies by file variant — print actual shape, don't hardcode).
- Categorical features: `protocol_type`, `service`, `flag` → one-hot encode.
- `num_outbound_cmds` is constant 0 — zero variance, expect it to carry no signal.
- Test set contains 17 attack types unseen in training → the paper RF's ~77% is honest generalization, not a bug. Never "fix" it by testing on a random split of KDDTrain+ (~99.8%, because it tests on attack types the model has already seen).

## THE CRITICAL GOTCHA — read before writing mask.py

The mask is defined in raw 41-feature space, but the model consumes scaled + one-hot-encoded vectors. If you apply integer-rounding or sign-clamping in the encoded/scaled space, the constraints silently do nothing and the whole experiment is invalid.

Required approach:

- Keep an explicit index map: raw feature name → its column(s) in the encoded matrix.
- Apply mask logic in raw space, OR invert the scaler → apply constraints → re-transform.
- After building the mask, write a unit test that perturbs a frozen (Class C) feature and asserts the model input is unchanged. If that test doesn't exist, the experiment can't be trusted.

## Realizability mask spec (mask.py)

Three feature classes from the taxonomy:

- **Class A — freely mutable (1):** `src_bytes`
- **Class B — constrained (9):** `duration`, `flag`, `land`, `wrong_fragment`, `urgent`, `num_failed_logins`, `su_attempted`, `count`, `srv_count`
- **Class C — frozen (31):** everything else → gradient zeroed, value pinned to original

Additional constraints:

- `INCREASE_ONLY = {count, srv_count, num_failed_logins}` → after perturbation, clamp `x_adv = max(x_adv, x_orig)` on these.
- `INTEGER = {src_bytes, duration, count, srv_count, wrong_fragment, urgent, num_failed_logins}` → round after each step.
- Clip all: non-negative; rate features (`*_rate`) to [0, 1].
- Categorical `flag` is Class B but not continuously perturbable — treat as frozen in the continuous attack for now and note the limitation in a comment. (`protocol_type`, `service` are Class C, frozen.)

```python
def apply_mask(x_adv, x_orig, feature_index_map):
    # 1. freeze Class C columns -> x_orig values
    # 2. increase-only clamp on INCREASE_ONLY
    # 3. integer-round INTEGER features
    # 4. clip: >=0 everywhere; rates in [0,1]
    # returns masked x_adv in the space the model expects
```

Call `apply_mask` inside the PGD loop after every gradient step, and once after FGSM.

## Reproducibility rules

- Fix `random_state=42` everywhere (RF, any split, ART where applicable).
- No silent hardcoded shapes — print and assert actual data dimensions.
- Every result that goes in the report comes from `results/metrics.csv`, regenerated by `python src/experiment.py`. No hand-typed numbers.
- Log library versions at runtime into the results file.

## Conventions

- Blunt, minimal comments — but every non-obvious networking assumption gets a one-line "why" (e.g. why a feature is frozen).
- Small functions, one job each. No notebook-style mega-scripts.
- Deterministic over clever.

## Do NOT

- Do not use Cisco Packet Tracer (irrelevant here, but it's a hard no on the project).
- Do not use CleverHans.
- Do not improve accuracy by changing the evaluation protocol. Improvements are allowed, and wanted, only on the honest protocol: train on KDDTrain+, test on KDDTest+. Choose thresholds and hyperparameters on training data (e.g. held-out attack types), never on KDDTest+. The paper's RF (77.9%) stays as the reproduction baseline.
- Do not apply mask constraints in encoded/scaled space (see gotcha).
- Do not fabricate NSL-KDD counts — print them.

## First session task

1. Scaffold the repo + venv + requirements.
2. `preprocess.py` + `model.py`: get the RF baseline printing clean accuracy. Confirm ~77%.
3. Stop and report the number before going further.
