# NSL-KDD data

Two files are expected here:

- `KDDTrain+.txt` — 125,973 rows
- `KDDTest+.txt`  — 22,544 rows

Both are 43 comma-separated columns: **41 features + label + difficulty**.
The trailing difficulty column is NOT a feature; `preprocess.py` drops it.

## How to obtain

Canonical source is the Canadian Institute for Cybersecurity (UNB), which now
gates downloads behind a request form: https://www.unb.ca/cic/datasets/nsl.html

The copies used here were pulled from a public mirror:

```bash
curl -sSL -o "data/KDDTrain+.txt" https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
curl -sSL -o "data/KDDTest+.txt"  https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
```

## Verify what you downloaded

Do not trust a mirror blindly. Row counts and hashes of the files used in this repo:

| file | rows | sha256 |
|---|---|---|
| KDDTrain+.txt | 125973 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| KDDTest+.txt  | 22544  | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |

```bash
sha256sum data/KDDTrain+.txt data/KDDTest+.txt
wc -l data/KDDTrain+.txt data/KDDTest+.txt
```

`preprocess.py` re-asserts the column count and prints actual row counts at load
time, so a wrong file variant fails loudly rather than silently.
