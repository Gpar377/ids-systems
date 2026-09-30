"""FGSM / PGD under a transfer threat model.

A RandomForest has no gradients, so it cannot be attacked white-box. Rather than
pretend otherwise, we model what a real attacker actually faces: they do not
have the victim's forest. They train their own substitute on traffic they can
observe, attack that, and hope the examples transfer.

  victim   : RandomForest (the deployed IDS)
  attacker : MLP substitute (surrogate.py) -- nonlinear, so its input gradient
             varies per sample. A linear substitute gives only two perturbation
             directions, which leaks the label into the perturbation itself.

This is a stronger and more honest setup than white-boxing a model no attacker
could see, and it is where the realizability mask hooks in: the mask constrains
the attacker's *search*, which is exactly where a protocol constraint bites.

FGSM/PGD are implemented here rather than taken from ART because the mask must
run inside the PGD loop after every step (CLAUDE.md), and ART's PGD does not
accept an arbitrary projection operator.
"""

from __future__ import annotations

import numpy as np
from mask import apply_mask
from preprocess import Preprocessor
from surrogate import MLPSurrogate

RANDOM_STATE = 42


def build_surrogate(X: np.ndarray, y: np.ndarray, **kw) -> MLPSurrogate:
    """The attacker's substitute model. Trained on the same feature space, but
    it is NOT the victim -- adversarial examples must transfer to succeed."""
    return MLPSurrogate(seed=RANDOM_STATE, **kw).fit(X, y)


def wrap(model):
    """Kept for interface symmetry; MLPSurrogate already exposes loss_gradient.

    ART's SklearnClassifier is still used for the LINEAR baseline surrogate in
    the label-leaking demonstration (see notes in surrogate.py)."""
    return model


def attack_labels(surrogate, X: np.ndarray) -> np.ndarray:
    """Use the substitute's PREDICTED labels, not ground truth.

    Feeding true labels to FGSM leaks them into the perturbation direction, so
    a model retrained on those examples learns the offset instead of learning
    robustness (Kurakin et al., 2017). A real attacker does not have the labels
    anyway -- they have their own model's opinion."""
    return surrogate.predict(X)


def _onehot(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    return np.eye(n_classes)[y].astype(np.float32)


def _box(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valid scaled range is [0,1] (MinMax fit on train), widened so a clean
    test point outside that range is never itself clipped -- otherwise eps=0
    would already change the input and the comparison would be dishonest."""
    return np.minimum(0.0, X), np.maximum(1.0, X)


def _project(x_adv, x_orig, eps, lo, hi, pre=None):
    x_adv = np.clip(x_adv, x_orig - eps, x_orig + eps)  # L-inf budget
    x_adv = np.clip(x_adv, lo, hi)                      # valid feature range
    if pre is not None:
        x_adv = apply_mask(x_adv, x_orig, pre)          # realizability
    return x_adv


def fgsm(art_clf, X: np.ndarray, y: np.ndarray, eps: float = 0.2,
         pre: Preprocessor | None = None) -> np.ndarray:
    """Single signed-gradient step. Mask applied once, after the step."""
    X = np.asarray(X, dtype=np.float64)
    lo, hi = _box(X)
    grad = art_clf.loss_gradient(X.astype(np.float32), _onehot(y))
    x_adv = X + eps * np.sign(grad)
    return _project(x_adv, X, eps, lo, hi, pre)


def pgd(art_clf, X: np.ndarray, y: np.ndarray, eps: float = 0.2,
        alpha: float = 0.01, iters: int = 5,
        pre: Preprocessor | None = None) -> np.ndarray:
    """Iterative FGSM. Mask applied after EVERY step, not just at the end --
    a mask applied only once at the end would let the search wander through
    unrealizable space and land somewhere the projection cannot undo."""
    X = np.asarray(X, dtype=np.float64)
    lo, hi = _box(X)
    x_adv = X.copy()
    for _ in range(iters):
        grad = art_clf.loss_gradient(x_adv.astype(np.float32), _onehot(y))
        x_adv = x_adv + alpha * np.sign(grad)
        x_adv = _project(x_adv, X, eps, lo, hi, pre)
    return x_adv


def perturbation_report(x_adv: np.ndarray, x_orig: np.ndarray,
                        pre: Preprocessor) -> dict:
    """How big is the perturbation, and where did it actually land?"""
    d = x_adv - x_orig
    moved = np.abs(d).sum(axis=0) > 1e-12
    from preprocess import NUMERIC
    moved_names = [pre.encoded_names[i] for i in np.where(moved)[0]]
    return {
        "linf": float(np.abs(d).max()),
        "l2_mean": float(np.linalg.norm(d, axis=1).mean()),
        "n_cols_moved": int(moved.sum()),
        "n_cols_total": int(x_orig.shape[1]),
        "cols_moved": moved_names,
    }
