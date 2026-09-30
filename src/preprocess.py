"""NSL-KDD loading and encoding.

Produces the FIXED-column-order encoded matrix the RF consumes, plus the
raw-name -> encoded-column index map that mask.py needs. The index map is not
optional bookkeeping: the realizability mask is defined over raw 41-feature
space, and without an explicit map there is no correct way to apply it to the
encoded matrix. See CLAUDE.md "THE CRITICAL GOTCHA".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# The 41 NSL-KDD features in file order, then label, then difficulty.
FEATURES = [
    # basic (1-9), from packet headers
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
    # content (10-22), payload / host state
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    # time-based traffic (23-31), 2-second window
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # host-based traffic (32-41), window of last 100 connections to the victim
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

# Rate features are proportions by construction -> valid range is [0, 1].
RATE_FEATURES = [f for f in NUMERIC if f.endswith("_rate")]

# Heavy-tailed magnitudes (bytes, counts, durations): candidates for log1p.
LOG_COLS = [f for f in NUMERIC if f not in RATE_FEATURES]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"


def load_raw(path: Path) -> pd.DataFrame:
    """Read one NSL-KDD file. Asserts shape rather than trusting the variant."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    n_cols = df.shape[1]
    # Some variants ship 42 cols (no difficulty). Fail loudly instead of
    # silently shifting every feature by one position.
    assert n_cols == 43, f"{path.name}: expected 43 columns, got {n_cols}"
    df = df.drop(columns=["difficulty"])  # difficulty is metadata, not a feature
    assert list(df.columns) == FEATURES + ["label"]
    print(f"  {path.name}: {df.shape[0]} rows, {len(FEATURES)} features")
    return df


def binarize_labels(labels: pd.Series) -> np.ndarray:
    """normal -> 0, anything else -> 1. Binary framing per CLAUDE.md."""
    return (labels != "normal").astype(int).to_numpy()


@dataclass
class Preprocessor:
    """Fit on train only; transform train and test identically.

    Encoded layout is fixed and deliberate:
        [ 38 scaled numeric ][ one-hot protocol_type ][ one-hot service ][ one-hot flag ]
    Numeric columns come first so mask.py can slice them as one contiguous
    block and invert the scaler on exactly that block.
    """

    # Numeric columns to log1p before scaling. Bytes and counts span ~9 orders of
    # magnitude; MinMax alone squashes nearly every src_bytes value to ~0. Empty
    # by default so the paper-reproduction pipeline is unchanged.
    log_cols: list[str] = field(default_factory=list)
    scaler: MinMaxScaler = field(default_factory=MinMaxScaler)
    encoder: OneHotEncoder = field(
        default_factory=lambda: OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float64
        )
    )
    feature_index_map: dict[str, list[int]] = field(default_factory=dict)
    encoded_names: list[str] = field(default_factory=list)

    @property
    def _log_idx(self) -> list[int]:
        return [NUMERIC.index(c) for c in self.log_cols]

    def _raw_to_model_space(self, raw: np.ndarray) -> np.ndarray:
        """Raw numeric units -> the pre-scaler space (log1p on log_cols)."""
        out = np.array(raw, dtype=np.float64, copy=True)
        if self.log_cols:
            out[:, self._log_idx] = np.log1p(np.maximum(out[:, self._log_idx], 0.0))
        return out

    def to_raw(self, scaled_num: np.ndarray) -> np.ndarray:
        """Encoded numeric block -> raw units (bytes, seconds, counts).
        mask.py constrains in THIS space, never in scaled/log space."""
        out = self.scaler.inverse_transform(scaled_num)
        if self.log_cols:
            out[:, self._log_idx] = np.expm1(out[:, self._log_idx])
        return out

    def from_raw(self, raw_num: np.ndarray) -> np.ndarray:
        """Raw units -> encoded numeric block. Inverse of to_raw."""
        return self.scaler.transform(self._raw_to_model_space(raw_num))

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        assert set(self.log_cols) <= set(NUMERIC), "log_cols must be numeric"
        self.scaler.fit(self._raw_to_model_space(df[NUMERIC].to_numpy(dtype=np.float64)))
        # handle_unknown='ignore': KDDTest+ contains `service` values absent
        # from KDDTrain+. Unknown -> all-zero block, which is the honest
        # encoding; dropping those rows would inflate accuracy.
        self.encoder.fit(df[CATEGORICAL])
        self._build_index_map()
        return self

    def _build_index_map(self) -> None:
        self.feature_index_map = {f: [i] for i, f in enumerate(NUMERIC)}
        self.encoded_names = list(NUMERIC)
        offset = len(NUMERIC)
        for cat, cats in zip(CATEGORICAL, self.encoder.categories_):
            idx = list(range(offset, offset + len(cats)))
            self.feature_index_map[cat] = idx
            self.encoded_names += [f"{cat}={c}" for c in cats]
            offset += len(cats)
        assert offset == len(self.encoded_names)
        assert set(self.feature_index_map) == set(FEATURES), "index map must cover all 41"

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self.from_raw(df[NUMERIC].to_numpy(dtype=np.float64))
        cat = self.encoder.transform(df[CATEGORICAL])
        X = np.hstack([num, cat]).astype(np.float64)
        assert X.shape[1] == len(self.encoded_names)
        return X

    @property
    def numeric_slice(self) -> slice:
        """The contiguous scaled-numeric block, for scaler inversion."""
        return slice(0, len(NUMERIC))


def load_dataset(log: bool = False) -> dict:
    """Load train+test, fit encoding on train, return everything downstream needs.

    log=True applies log1p to the heavy-tailed numeric columns (LOG_COLS) before
    scaling -- used by the improved IDS; the paper reproduction uses log=False."""
    print("Loading NSL-KDD:")
    train_df = load_raw(TRAIN_FILE)
    test_df = load_raw(TEST_FILE)

    y_train = binarize_labels(train_df["label"])
    y_test = binarize_labels(test_df["label"])

    pre = Preprocessor(log_cols=list(LOG_COLS) if log else []).fit(train_df)
    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)

    # Novel attack types in test only -> the ~77% is generalisation, not a bug.
    novel = set(test_df["label"].unique()) - set(train_df["label"].unique())

    print(f"  encoded dim: {X_train.shape[1]} "
          f"({len(NUMERIC)} numeric + {X_train.shape[1] - len(NUMERIC)} one-hot)")
    print(f"  train attack ratio: {y_train.mean():.4f} | test: {y_test.mean():.4f}")
    print(f"  novel attack types in test: {len(novel)}")
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[1] == X_train.shape[1]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "preprocessor": pre,
        "train_df": train_df, "test_df": test_df,
        "novel_attack_types": sorted(novel),
    }


if __name__ == "__main__":
    d = load_dataset()
    print(f"\nnovel types: {d['novel_attack_types']}")
