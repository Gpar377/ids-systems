"""Adversarial training.

Augment the training set with adversarial examples and retrain. The defended
model is evaluated against BOTH threat models, which is the point: a defense
tuned against an unconstrained attacker may be buying robustness against
perturbations no attacker could ever send.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from attacks import pgd
from model import RF_PARAMS


def adversarial_train(X: np.ndarray, y: np.ndarray, X_adv: np.ndarray
                      ) -> RandomForestClassifier:
    """Retrain on clean + adversarial. Labels are unchanged: an adversarial
    example is still the attack it always was -- that is the whole premise."""
    X_aug = np.vstack([X, X_adv])
    y_aug = np.concatenate([y, y])
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_aug, y_aug)
    return clf


def make_training_adversarials(art_clf, X, y, eps=0.2, alpha=0.02, iters=10,
                               pre=None) -> np.ndarray:
    """Adversarial examples drawn from TRAIN, never from test -- generating them
    from test would leak the evaluation set into the defense."""
    return pgd(art_clf, X, y, eps=eps, alpha=alpha, iters=iters, pre=pre)
