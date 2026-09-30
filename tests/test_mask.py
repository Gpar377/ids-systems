"""The mask must actually bite.

CLAUDE.md: "write a unit test that perturbs a frozen (Class C) feature and
asserts the model input is unchanged. If that test doesn't exist, the experiment
can't be trusted." That is test_frozen_feature_cannot_move below; the rest guard
the constraints that make a perturbation physically realizable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (CATEGORICAL, FEATURES, NUMERIC, Preprocessor,
                        TRAIN_FILE, load_raw)
from mask import (BINARY, CLASS_A, CLASS_B, CLASS_C, INCREASE_ONLY, INTEGER,
                  MUTABLE, apply_mask, frozen_encoded_columns, gradient_mask,
                  mutable_encoded_columns)

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def _train():
    return load_raw(TRAIN_FILE)


# Every mask test runs in BOTH feature spaces: plain MinMax (paper reproduction)
# and log1p+MinMax (improved IDS). The log space is exactly where the CLAUDE.md
# gotcha would bite -- rounding in log space silently does nothing.
@pytest.fixture(scope="module", params=["minmax", "log1p"])
def ctx(request, _train):
    from preprocess import LOG_COLS
    log_cols = list(LOG_COLS) if request.param == "log1p" else []
    pre = Preprocessor(log_cols=log_cols).fit(_train)
    X = pre.transform(_train.head(300))
    return pre, X


# --- taxonomy bookkeeping ---------------------------------------------------

def test_class_counts_match_taxonomy():
    assert (len(CLASS_A), len(CLASS_B), len(CLASS_C)) == (1, 9, 31)
    assert len(CLASS_A | CLASS_B | CLASS_C) == 41


def test_only_nine_features_are_continuously_mutable():
    # flag is Class B but categorical -> frozen in the continuous attack.
    assert len(MUTABLE) == 9
    assert "flag" not in MUTABLE
    assert "src_bytes" in MUTABLE


def test_gradient_mask_zeroes_the_vast_majority(ctx):
    pre, _ = ctx
    g = gradient_mask(pre)
    assert g.sum() == 9
    assert g.shape[0] == len(pre.encoded_names)
    # The headline: the attacker moves a tiny fraction of the input space.
    assert g.mean() < 0.10


# --- THE test ---------------------------------------------------------------

def test_frozen_feature_cannot_move(ctx):
    """Perturb every Class C column hard; model input must be byte-identical."""
    pre, X = ctx
    frozen = frozen_encoded_columns(pre)
    x_adv = X.copy()
    x_adv[:, frozen] += RNG.normal(0, 5.0, size=(X.shape[0], len(frozen)))
    out = apply_mask(x_adv, X, pre)
    np.testing.assert_array_equal(out[:, frozen], X[:, frozen])


def test_frozen_covers_all_class_c_and_categoricals(ctx):
    pre, _ = ctx
    frozen = set(frozen_encoded_columns(pre).tolist())
    for f in CLASS_C | set(CATEGORICAL):
        assert set(pre.feature_index_map[f]).issubset(frozen), f
    for f in MUTABLE:
        assert not set(pre.feature_index_map[f]) & frozen, f


def test_mutable_and_frozen_partition_the_space(ctx):
    pre, _ = ctx
    fr = set(frozen_encoded_columns(pre).tolist())
    mu = set(mutable_encoded_columns(pre).tolist())
    assert not (fr & mu)
    assert fr | mu == set(range(len(pre.encoded_names)))


# --- realizability constraints ---------------------------------------------

def _raw(pre, X):
    return pre.to_raw(X[:, pre.numeric_slice])


def test_integer_features_are_integral_after_mask(ctx):
    pre, X = ctx
    x_adv = X + RNG.normal(0, 0.05, size=X.shape)
    out = apply_mask(x_adv, X, pre)
    raw = _raw(pre, out)
    for name in INTEGER:
        j = NUMERIC.index(name)
        col = raw[:, j]
        np.testing.assert_allclose(col, np.round(col), atol=1e-6,
                                   err_msg=f"{name} not integral")


def test_binary_features_stay_binary(ctx):
    pre, X = ctx
    x_adv = X + RNG.normal(0, 0.3, size=X.shape)
    out = apply_mask(x_adv, X, pre)
    raw = _raw(pre, out)
    for name in BINARY:
        j = NUMERIC.index(name)
        assert set(np.unique(np.round(raw[:, j], 6))).issubset({0.0, 1.0}), name


def test_increase_only_features_never_decrease(ctx):
    pre, X = ctx
    # Push hard downward -- the clamp must hold.
    x_adv = X - 0.5
    out = apply_mask(x_adv, X, pre)
    raw_out, raw_in = _raw(pre, out), _raw(pre, X)
    for name in INCREASE_ONLY:
        j = NUMERIC.index(name)
        assert np.all(raw_out[:, j] >= raw_in[:, j] - 1e-6), name


def test_nothing_goes_negative(ctx):
    pre, X = ctx
    out = apply_mask(X - 10.0, X, pre)
    assert _raw(pre, out).min() >= -1e-6


def test_rates_stay_in_unit_interval(ctx):
    pre, X = ctx
    from preprocess import RATE_FEATURES
    out = apply_mask(X + 10.0, X, pre)
    raw = _raw(pre, out)
    for name in RATE_FEATURES:
        j = NUMERIC.index(name)
        assert raw[:, j].min() >= -1e-6 and raw[:, j].max() <= 1 + 1e-6, name


def test_mask_is_idempotent(ctx):
    """Applied twice == applied once. Required: it runs every PGD step."""
    pre, X = ctx
    x_adv = X + RNG.normal(0, 0.1, size=X.shape)
    once = apply_mask(x_adv, X, pre)
    twice = apply_mask(once.copy(), X, pre)
    np.testing.assert_allclose(once, twice, atol=1e-9)


def test_unperturbed_input_is_a_fixed_point(ctx):
    """Masking a clean sample must not silently corrupt it."""
    pre, X = ctx
    out = apply_mask(X.copy(), X, pre)
    np.testing.assert_allclose(out, X, atol=1e-6)


def test_src_bytes_can_actually_move(ctx):
    """Sanity: the mask must not freeze everything."""
    pre, X = ctx
    j = NUMERIC.index("src_bytes")
    x_adv = X.copy()
    x_adv[:, j] += 0.2
    out = apply_mask(x_adv, X, pre)
    raw_out, raw_in = _raw(pre, out), _raw(pre, X)
    assert np.any(raw_out[:, j] != raw_in[:, j]), "src_bytes should be mutable"
