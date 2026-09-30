"""Realizability mask -- THE NOVEL BIT.

An adversarial example is only a threat if some real packet sequence, sent by a
real attacker over TCP/IP, actually produces that feature vector. The gradient
does not know that. This module enforces what the wire allows.

Feature classes come from nsl_kdd_mutability_taxonomy.md:
  A (1)  freely mutable      -- attacker sets it directly
  B (9)  constrained         -- influenced but bounded / coupled / asymmetric
  C (31) non-mutable         -- victim-controlled, third-party window aggregate,
                                categorical launch choice, or dead constant

CRITICAL: constraints are defined in RAW 41-feature units, but the model eats a
scaled + one-hot matrix. Rounding "to an integer" in MinMax-scaled space is
meaningless -- 0.5 scaled is not 1 byte. So we invert the scaler, constrain in
raw units, and re-transform. Doing this in encoded space silently does nothing
and invalidates the whole experiment. See CLAUDE.md.
"""

from __future__ import annotations

import numpy as np

from preprocess import CATEGORICAL, FEATURES, NUMERIC, RATE_FEATURES, Preprocessor

# --- taxonomy ---------------------------------------------------------------

CLASS_A = {"src_bytes"}  # the one genuinely free feature

CLASS_B = {
    "duration",           # attacker holds the connection; integer seconds
    "flag",               # victim-coupled: reflects how the victim responded
    "land",               # craftable via src=dst spoof, but self-incriminating
    "wrong_fragment",     # craftable malformed fragmentation, self-incriminating
    "urgent",             # URG-flag packets; craftable but unusual
    "num_failed_logins",  # attacker chooses tries; each try is real traffic
    "su_attempted",       # attacker action, binary and rare
    "count",              # raise-only: can't go below own footprint
    "srv_count",          # raise-only, per-service
}

CLASS_C = set(FEATURES) - CLASS_A - CLASS_B  # 31 features

# `flag` is Class B but categorical -- a connection-status change is not a small
# continuous step, so a gradient cannot walk it. Frozen in the continuous attack;
# a categorical-flip attack is future work. protocol_type/service are Class C
# (launch-time choices) and frozen for the same structural reason.
CATEGORICAL_FROZEN = set(CATEGORICAL)

# Continuously perturbable = A + B minus the categorical one.
MUTABLE = (CLASS_A | CLASS_B) - CATEGORICAL_FROZEN

# Asymmetric: an attacker generates connections, so these can only go up.
INCREASE_ONLY = {"count", "srv_count", "num_failed_logins"}

# Integer-valued on the wire -- you cannot send 0.2 of a byte.
INTEGER = {"src_bytes", "duration", "count", "srv_count",
           "wrong_fragment", "urgent", "num_failed_logins"}

# Binary flags -- round to {0,1}, not to arbitrary integers.
BINARY = {"land", "su_attempted"}

assert len(CLASS_A) == 1 and len(CLASS_B) == 9 and len(CLASS_C) == 31
assert CLASS_A | CLASS_B | CLASS_C == set(FEATURES)


def frozen_encoded_columns(pre: Preprocessor) -> np.ndarray:
    """Encoded column indices the attacker may not move (Class C + categoricals)."""
    frozen = CLASS_C | CATEGORICAL_FROZEN
    cols: list[int] = []
    for f in frozen:
        cols.extend(pre.feature_index_map[f])
    return np.array(sorted(cols), dtype=int)


def mutable_encoded_columns(pre: Preprocessor) -> np.ndarray:
    cols = [pre.feature_index_map[f][0] for f in sorted(MUTABLE)]
    return np.array(sorted(cols), dtype=int)


def gradient_mask(pre: Preprocessor) -> np.ndarray:
    """1.0 on columns the attack may move, 0.0 elsewhere. Multiply into the grad."""
    m = np.zeros(len(pre.encoded_names), dtype=np.float64)
    m[mutable_encoded_columns(pre)] = 1.0
    return m


def apply_mask(x_adv: np.ndarray, x_orig: np.ndarray,
               pre: Preprocessor) -> np.ndarray:
    """Project a perturbed batch back onto the realizable set.

    Order matters: freeze -> raw-space constraints -> re-transform -> re-freeze.
    Returns a matrix in the encoded space the model expects.
    """
    x_adv = np.atleast_2d(np.array(x_adv, dtype=np.float64, copy=True))
    x_orig = np.atleast_2d(np.asarray(x_orig, dtype=np.float64))
    assert x_adv.shape == x_orig.shape, f"{x_adv.shape} vs {x_orig.shape}"

    frozen = frozen_encoded_columns(pre)

    # 1. Class C + categoricals pinned to their original values.
    x_adv[:, frozen] = x_orig[:, frozen]

    # 2/3. Map back to raw units (inverse scaler, plus expm1 if log-scaled) so
    # constraints act on real bytes, seconds and connection counts rather than
    # on scaled or log-space surrogates.
    num = pre.numeric_slice
    raw_adv = pre.to_raw(x_adv[:, num])
    raw_orig = pre.to_raw(x_orig[:, num])

    for j, name in enumerate(NUMERIC):
        if name not in MUTABLE:
            raw_adv[:, j] = raw_orig[:, j]
            continue
        col = raw_adv[:, j]
        if name in BINARY:
            col = np.clip(np.round(col), 0.0, 1.0)
        elif name in INTEGER:
            col = np.round(col)
        if name in INCREASE_ONLY:
            col = np.maximum(col, raw_orig[:, j])  # can't drop below own footprint
        raw_adv[:, j] = col

    # 4. Universal validity: nothing negative; proportions stay proportions.
    raw_adv = np.maximum(raw_adv, 0.0)
    for j, name in enumerate(NUMERIC):
        if name in RATE_FEATURES:
            raw_adv[:, j] = np.clip(raw_adv[:, j], 0.0, 1.0)

    x_adv[:, num] = pre.from_raw(raw_adv)

    # Re-freeze: the inverse/transform round-trip introduces float drift, and a
    # frozen feature must be bit-stable or the test below is meaningless.
    x_adv[:, frozen] = x_orig[:, frozen]
    return x_adv


def summarize(pre: Preprocessor) -> dict:
    """Numbers for the report -- how much of the budget is actually usable."""
    n_enc = len(pre.encoded_names)
    return {
        "n_raw_features": len(FEATURES),
        "class_A": len(CLASS_A),
        "class_B": len(CLASS_B),
        "class_C": len(CLASS_C),
        "continuously_mutable": len(MUTABLE),
        "mutable_features": sorted(MUTABLE),
        "n_encoded_cols": n_enc,
        "n_mutable_encoded_cols": len(mutable_encoded_columns(pre)),
        "frac_encoded_mutable": len(mutable_encoded_columns(pre)) / n_enc,
    }
