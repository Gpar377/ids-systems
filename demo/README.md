# Demo app

An offline web app for presenting the project. It runs on the standard library
only: no internet, no CDN, nothing extra to install beyond the project venv.

```bash
.venv/Scripts/python.exe demo/app.py      # or double-click demo/run_demo.bat
```

Opens <http://127.0.0.1:8765>. Startup takes about 12 s (it loads both IDSs; the
attacker's surrogate models are cached in `results/` after the first run).

## The four views

| View | What it shows | Where the data comes from |
|---|---|---|
| **Overview** | Headline numbers, evasion chart, who controls each of the 41 features | `results/metrics.csv`, `results/evasion.csv` |
| **Network simulation** | Animated topology; real KDDTest+ connections replayed through both IDSs; misses and false alarms tallied | live model scores |
| **Attack lab** | Pick an attack, choose unrestricted vs realistic, watch features move and the verdict change; impossible values flagged per feature | attacks computed **live** per click |
| **Live capture** | Drop a `.pcap` from the Docker lab; every connection is scored by both IDSs | `lab/extract/pcap_to_nslkdd.py` |

## Suggested viva flow (about 3 minutes)

1. **Overview**: 77.9% becomes 89.6%; evasion drops from 31% to 5%; only 9 of 41 features are attacker-controllable.
2. **Attack lab**: click *"Pick an example that fools the paper's IDS"*. The IDS is evaded, but point at the red tiles: the "attack" needs values no packet can carry.
3. Switch to **Realistic** and launch again: 32 features lock and only real changes remain. Then switch the target to **Our IDS**.
4. **Network simulation**: run *Recon scan*. Novel scan types (e.g. `mscan`) get MISSED by the paper's IDS and caught by ours.
5. **Live capture** (if you ran the lab on the Docker machine): drop the pcap in and show real packets scored.

## Two IDSs compared

- **Paper's IDS**: RandomForest (100 trees, depth 15), MinMax features, threshold 0.5.
- **Our IDS**: soft-vote of ExtraTrees + HistGradientBoosting + MLP on log1p features,
  with the threshold chosen on held-out *attack types* (`src/ids_model.py`).
