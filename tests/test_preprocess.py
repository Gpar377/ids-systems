"""Sanity checks on encoding + the raw->encoded index map.

The index map is what makes the realizability mask possible at all, so it gets
tested before any attack code is written. See CLAUDE.md "THE CRITICAL GOTCHA".
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (CATEGORICAL, FEATURES, NUMERIC, RATE_FEATURES,
                        Preprocessor, load_raw, TRAIN_FILE, TEST_FILE,
                        binarize_labels)


@pytest.fixture(scope="module")
def fitted():
    train = load_raw(TRAIN_FILE)
    pre = Preprocessor().fit(train)
    return pre, train


def test_feature_count():
    assert len(FEATURES) == 41
    assert len(CATEGORICAL) == 3
    assert len(NUMERIC) == 38


def test_index_map_covers_all_41(fitted):
    pre, _ = fitted
    assert set(pre.feature_index_map) == set(FEATURES)


def test_index_map_is_a_partition(fitted):
    """Every encoded column belongs to exactly one raw feature."""
    pre, _ = fitted
    all_idx = [i for idx in pre.feature_index_map.values() for i in idx]
    assert sorted(all_idx) == list(range(len(pre.encoded_names))), "overlap or gap"


def test_numeric_block_is_contiguous_and_first(fitted):
    """mask.py inverts the scaler on this block as one slice."""
    pre, _ = fitted
    for i, f in enumerate(NUMERIC):
        assert pre.feature_index_map[f] == [i]
    assert pre.numeric_slice == slice(0, 38)


def test_categoricals_are_multi_column(fitted):
    pre, _ = fitted
    for c in CATEGORICAL:
        assert len(pre.feature_index_map[c]) > 1


def test_transform_shape_and_range(fitted):
    pre, train = fitted
    X = pre.transform(train.head(500))
    assert X.shape[1] == len(pre.encoded_names)
    # MinMax on train -> [0,1]; chosen so eps=0.2 is a meaningful fraction of range.
    num = X[:, pre.numeric_slice]
    assert num.min() >= -1e-9 and num.max() <= 1 + 1e-9


def test_onehot_rows_sum_to_one(fitted):
    pre, train = fitted
    X = pre.transform(train.head(500))
    for c in CATEGORICAL:
        block = X[:, pre.feature_index_map[c]]
        assert np.allclose(block.sum(axis=1), 1.0)


def test_test_categoricals_are_subset_of_train(fitted):
    """KDDTest+ introduces 17 novel attack LABELS but no novel categorical values.

    Checked, not assumed: service is 70 values in train, 64 in test, test-only=0.
    So novelty lives in the labels, not the encoding -- the ~77% is genuine
    generalisation to unseen attack behaviour, not an artefact of unknown columns.
    """
    pre, train = fitted
    test = load_raw(TEST_FILE)
    for c in CATEGORICAL:
        assert not (set(test[c]) - set(train[c])), f"unexpected unseen {c}"
    novel = set(test["label"]) - set(train["label"])
    assert len(novel) == 17, f"expected 17 novel attack types, got {len(novel)}"


def test_unknown_category_would_encode_as_zero_block(fitted):
    """handle_unknown='ignore' is defensive: it never fires on this data, but if a
    future variant adds a service, it must yield an all-zero block, not a crash."""
    pre, train = fitted
    row = train.head(1).copy()
    row.loc[:, "service"] = "__not_a_real_service__"
    X = pre.transform(row)
    block = X[:, pre.feature_index_map["service"]]
    assert np.allclose(block.sum(axis=1), 0.0)


def test_num_outbound_cmds_is_dead(fitted):
    """Constant 0 across the dataset -> zero variance, carries no signal."""
    _, train = fitted
    assert train["num_outbound_cmds"].nunique() == 1
    assert train["num_outbound_cmds"].iloc[0] == 0


def test_rate_features_within_unit_interval(fitted):
    _, train = fitted
    assert len(RATE_FEATURES) > 0
    for f in RATE_FEATURES:
        assert train[f].min() >= 0.0 and train[f].max() <= 1.0


def test_binarize_labels():
    import pandas as pd
    s = pd.Series(["normal", "neptune", "normal", "smurf"])
    assert list(binarize_labels(s)) == [0, 1, 0, 1]


def test_log_mode_roundtrip_is_exact():
    """to_raw(from_raw(x)) == x in log mode, or the mask would drift features."""
    from preprocess import LOG_COLS
    train = load_raw(TRAIN_FILE)
    pre = Preprocessor(log_cols=list(LOG_COLS)).fit(train)
    raw = train[NUMERIC].head(500).to_numpy(dtype=float)
    back = pre.to_raw(pre.from_raw(raw))
    np.testing.assert_allclose(back, raw, rtol=1e-9, atol=1e-6)


def test_log_mode_spreads_src_bytes():
    """The reason log mode exists: MinMax alone crushes src_bytes toward 0."""
    from preprocess import LOG_COLS
    train = load_raw(TRAIN_FILE)
    j = NUMERIC.index("src_bytes")
    plain = Preprocessor().fit(train).transform(train)[:, j]
    logd = Preprocessor(log_cols=list(LOG_COLS)).fit(train).transform(train)[:, j]
    assert np.median(plain) < 1e-4          # crushed
    assert np.median(logd) > 0.1            # spread out
