"""Offline demo server for the IDS case study.

    python demo/app.py            -> http://127.0.0.1:8765 (opens the browser)
    python demo/app.py --port 9000   (if the port is taken it moves to the next free
                                      port on its own and prints which one)

Standard library only (http.server) so it runs on the viva laptop with no
internet and nothing extra to install. Loads both IDSs once, then computes
attacks LIVE per request -- nothing on screen is a canned animation of numbers.

Two victims:
  paper : the paper's RandomForest, MinMax features, threshold 0.5
  ours  : soft-vote ensemble on log1p features, tuned threshold (src/ids_model.py)
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = ROOT / "results"
UPLOADS = ROOT / "lab" / "pcaps"          # on D:, git-ignored
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lab" / "extract"))

import ids_model  # noqa: E402
from attacks import fgsm, pgd  # noqa: E402
from mask import (BINARY, CLASS_A, CLASS_B, INCREASE_ONLY, INTEGER,  # noqa: E402
                  MUTABLE)
from preprocess import (CATEGORICAL, FEATURES, NUMERIC, RATE_FEATURES,  # noqa: E402
                        load_dataset)
from surrogate import MLPSurrogate  # noqa: E402

# 8765, not 8000: 8000 is the default for many dev servers (and for another
# project on this machine). A shared address means Chrome shares cache and
# service workers between the two apps and can show the wrong one.
HOST, PORT = "127.0.0.1", 8765
ALPHA_DEFAULT, ITERS_DEFAULT = 0.02, 10

# NSL-KDD attack label -> family, for the scenario simulator.
FAMILY = {
    "recon": {"portsweep", "ipsweep", "nmap", "satan", "mscan", "saint"},
    "dos": {"neptune", "smurf", "back", "teardrop", "pod", "land", "apache2",
            "processtable", "udpstorm", "mailbomb"},
    "r2l": {"guess_passwd", "ftp_write", "imap", "phf", "multihop",
            "warezmaster", "warezclient", "spy", "xlock", "xsnoop", "snmpguess",
            "snmpgetattack", "httptunnel", "sendmail", "named", "worm"},
    "u2r": {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm"},
}


def family_of(label: str) -> str:
    if label == "normal":
        return "normal"
    for fam, members in FAMILY.items():
        if label in members:
            return fam
    return "other"


def feature_class(name: str) -> str:
    return "A" if name in CLASS_A else "B" if name in CLASS_B else "C"


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# --------------------------------------------------------------------------- #
# Model state, loaded once
# --------------------------------------------------------------------------- #

class Victim:
    def __init__(self, key, label, clf, pre, threshold, X, sub):
        self.key, self.label = key, label
        self.clf, self.pre, self.threshold = clf, pre, threshold
        self.X, self.sub = X, sub

    def proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(np.atleast_2d(X))[:, 1]

    def detects(self, X: np.ndarray) -> np.ndarray:
        return self.proba(X) >= self.threshold


def _surrogate(name: str, X, y) -> MLPSurrogate:
    """Attacker's model for this feature space, cached on D: after first build."""
    path = RESULTS / f"surrogate_{name}.joblib"
    if path.exists():
        return joblib.load(path)
    print(f"  training {name} surrogate (first run only)...", flush=True)
    sub = MLPSurrogate(seed=42, verbose=False).fit(X, y)
    joblib.dump(sub, path)
    return sub


class State:
    def __init__(self):
        t0 = time.time()
        print("loading data + models...", flush=True)
        mm = load_dataset(log=False)
        lg = load_dataset(log=True)
        self.test_df = mm["test_df"].reset_index(drop=True)
        self.y = mm["y_test"]
        self.labels = self.test_df["label"].to_numpy()
        self.families = np.array([family_of(l) for l in self.labels])

        rf_path = RESULTS / "rf_baseline.joblib"
        if rf_path.exists():
            rf = joblib.load(rf_path)
        else:
            from model import save, train
            rf = train(mm["X_train"], mm["y_train"])
            save(rf)
        if not ids_model.BUNDLE.exists():
            ids = ids_model.build(lg["X_train"], lg["y_train"])
            ids_model.save(ids, lg["preprocessor"])
        bundle = ids_model.load()

        self.victims = {
            "paper": Victim("paper", "Paper's IDS (Random Forest)", rf,
                            mm["preprocessor"], 0.5, mm["X_test"],
                            _surrogate("minmax", mm["X_train"], mm["y_train"])),
            "ours": Victim("ours", "Our IDS (ensemble + tuned threshold)",
                           bundle["ids"], bundle["preprocessor"],
                           float(bundle["ids"].threshold), lg["X_test"],
                           _surrogate("log", lg["X_train"], lg["y_train"])),
        }
        # Pre-score the whole test set once per victim (cheap, makes the
        # simulator and sample picker instant).
        self.p_all = {k: v.proba(v.X) for k, v in self.victims.items()}
        self.samples = self._curate_samples()
        print(f"ready in {time.time()-t0:.1f}s", flush=True)

    # -- sample picker -------------------------------------------------------
    def _curate_samples(self, per_type: int = 4) -> list[dict]:
        """Attack connections both IDSs catch when clean, spread over attack
        types, flagged with whether the paper's IDS can be evaded (unrestricted
        PGD) -- lets the demo reach a dramatic example on purpose."""
        v = self.victims["paper"]
        caught = ((self.y == 1)
                  & (self.p_all["paper"] >= v.threshold)
                  & (self.p_all["ours"] >= self.victims["ours"].threshold))
        idx = np.where(caught)[0]
        adv = pgd(v.sub, v.X[idx], np.ones(len(idx), dtype=int), eps=0.2,
                  alpha=ALPHA_DEFAULT, iters=ITERS_DEFAULT)
        evades = ~v.detects(adv)
        rng = np.random.default_rng(7)
        out = []
        for lab in sorted(set(self.labels[idx])):
            pos = np.where(self.labels[idx] == lab)[0]
            ev, st = pos[evades[pos]], pos[~evades[pos]]
            pick = list(rng.permutation(ev)[:2]) + list(rng.permutation(st)[:2])
            for p in pick[:per_type]:
                i = int(idx[p])
                r = self.test_df.iloc[i]
                out.append({"id": i, "label": lab, "family": family_of(lab),
                            "service": r.service, "protocol": r.protocol_type,
                            "flag": r.flag, "src_bytes": int(r.src_bytes),
                            "evades_paper": bool(evades[p])})
        return out

    # -- feature decoding ----------------------------------------------------
    def decode(self, v: Victim, x: np.ndarray) -> dict:
        """Encoded vector -> raw 41 features as a real packet summary would show."""
        raw = v.pre.to_raw(x[None, v.pre.numeric_slice])[0]
        out = {}
        for j, f in enumerate(NUMERIC):
            out[f] = float(raw[j])
        cats = v.pre.encoder.categories_
        for cat, choices in zip(CATEGORICAL, cats):
            block = x[v.pre.feature_index_map[cat]]
            top = int(np.argmax(block))
            valid = bool(np.isclose(block.max(), 1.0) and np.isclose(block.sum(), 1.0)
                         and np.all(np.isclose(block, 0) | np.isclose(block, 1)))
            out[cat] = {"value": str(choices[top]) if block.max() > 0 else "unknown",
                        "valid": valid,
                        "weight": float(block.max())}
        return out

    def attack(self, sid: int, victim: str, method: str, eps: float,
               realistic: bool) -> dict:
        v = self.victims[victim]
        x0 = v.X[sid:sid + 1]
        y1 = np.ones(1, dtype=int)                 # attacker targets "normal"
        pre = v.pre if realistic else None
        if method == "fgsm":
            xa = fgsm(v.sub, x0, y1, eps=eps, pre=pre)
        else:
            xa = pgd(v.sub, x0, y1, eps=eps, alpha=max(eps / 10, 0.005),
                     iters=ITERS_DEFAULT, pre=pre)
        before, after = self.decode(v, x0[0]), self.decode(v, xa[0])
        dims_moved = int((np.abs(xa[0] - x0[0]) > 1e-9).sum())

        feats, violations = [], 0
        for f in FEATURES:
            b, a = before[f], after[f]
            if f in CATEGORICAL:
                changed = (b["value"] != a["value"]) or (not a["valid"])
                why = []
                if changed and not a["valid"]:
                    why.append("not a valid category (one-hot smeared)")
                elif changed:
                    why.append("attacker cannot change this")
                bv, av = b["value"], a["value"]
            else:
                changed = abs(a - b) > 1e-6 * (1 + abs(b))
                why = []
                if changed:
                    if f not in MUTABLE:
                        why.append("attacker cannot control this")
                    if f in INTEGER and abs(a - round(a)) > 1e-6:
                        why.append("fractional value")
                    if f in BINARY and round(a, 6) not in (0.0, 1.0):
                        why.append("must be 0 or 1")
                    if f in INCREASE_ONLY and a < b - 1e-6:
                        why.append("can only increase")
                    if f in RATE_FEATURES and not (-1e-9 <= a <= 1 + 1e-9):
                        why.append("rate outside [0,1]")
                    if a < -1e-9:
                        why.append("negative")
                bv, av = b, a
            violations += int(bool(why))
            feats.append({"name": f, "cls": feature_class(f), "before": bv,
                          "after": av, "changed": bool(changed),
                          "violations": why})

        p0, p1 = float(v.proba(x0)[0]), float(v.proba(xa)[0])
        r = self.test_df.iloc[sid]
        return {"id": sid, "label": r.label, "victim": v.label,
                "threshold": v.threshold, "p_before": p0, "p_after": p1,
                "detected_before": p0 >= v.threshold,
                "detected_after": p1 >= v.threshold,
                "dims_moved": dims_moved, "dims_total": int(x0.shape[1]),
                "features_changed": sum(f["changed"] for f in feats),
                "impossible_values": violations, "features": feats,
                "mode": "realistic" if realistic else "unrestricted",
                "method": method, "eps": eps}

    # -- scenario simulator --------------------------------------------------
    def simulate(self, scenario: str, n: int, seed: int) -> dict:
        """Replay REAL KDDTest+ connections of a scenario through both IDSs."""
        rng = np.random.default_rng(seed)
        normal = np.where(self.families == "normal")[0]
        if scenario == "benign":
            idx = rng.choice(normal, n, replace=False)
        else:
            fam = np.where(self.families == scenario)[0]
            k = max(1, int(n * 0.45))
            burst = rng.choice(fam, min(k, len(fam)), replace=False)
            bg = rng.choice(normal, n - len(burst), replace=False)
            # benign background, then the attack arrives as a burst mid-stream
            cut = len(bg) // 3
            idx = np.concatenate([bg[:cut], burst, bg[cut:]])
        rows = []
        for i in idx:
            r = self.test_df.iloc[int(i)]
            rows.append({
                "id": int(i), "label": r.label, "family": self.families[i],
                "attack": bool(self.y[i]), "service": r.service,
                "protocol": r.protocol_type, "flag": r.flag,
                "src_bytes": int(r.src_bytes), "dst_bytes": int(r.dst_bytes),
                "count": int(r["count"]), "dst_host_count": int(r.dst_host_count),
                "p_paper": float(self.p_all["paper"][i]),
                "p_ours": float(self.p_all["ours"][i]),
            })
        return {"scenario": scenario, "rows": rows,
                "thr_paper": self.victims["paper"].threshold,
                "thr_ours": self.victims["ours"].threshold}

    # -- live capture ----------------------------------------------------------
    def score_pcap(self, data: bytes) -> dict:
        from pcap_to_nslkdd import compute_features, parse_connections
        UPLOADS.mkdir(parents=True, exist_ok=True)
        path = UPLOADS / f"upload_{int(time.time())}.pcap"
        path.write_bytes(data)
        rows = compute_features(parse_connections(path))
        if not rows:
            return {"file": path.name, "rows": [], "note": "no TCP connections found"}
        df = pd.DataFrame(rows)
        feats = df[FEATURES]
        out = []
        p = {k: v.proba(v.pre.transform(feats)) for k, v in self.victims.items()}
        for i, r in df.iterrows():
            out.append({"src": r["_src_ip"], "service": r["service"],
                        "flag": r["flag"], "src_bytes": int(r["src_bytes"]),
                        "duration": int(r["duration"]), "count": int(r["count"]),
                        "dst_host_count": int(r["dst_host_count"]),
                        "p_paper": float(p["paper"][i]),
                        "p_ours": float(p["ours"][i])})
        return {"file": path.name, "rows": out,
                "thr_paper": self.victims["paper"].threshold,
                "thr_ours": self.victims["ours"].threshold}


def read_meta() -> dict:
    """Headline numbers, read from the results files -- never typed in."""
    def load(name):
        p = RESULTS / name
        return list(csv.DictReader(open(p, newline=""))) if p.exists() else []
    metrics, evasion = load("metrics.csv"), load("evasion.csv")
    return {
        "metrics": metrics, "evasion": evasion,
        "classes": {"A": sorted(CLASS_A), "B": sorted(CLASS_B),
                    "C": sorted(set(FEATURES) - CLASS_A - CLASS_B)},
        "mutable": sorted(MUTABLE),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

STATE: State | None = None
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
         ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):      # quiet console
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(jsonable(obj)).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/meta":
                return self._json(read_meta())
            if u.path == "/api/samples":
                return self._json(STATE.samples)
            if u.path == "/api/simulate":
                return self._json(STATE.simulate(q.get("scenario", "recon"),
                                                  int(q.get("n", 36)),
                                                  int(q.get("seed", 1))))
            return self._static(u.path)
        except Exception as e:  # surface errors to the UI, not a blank page
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            if u.path == "/api/attack":
                r = json.loads(body)
                return self._json(STATE.attack(
                    int(r["id"]), r.get("victim", "paper"), r.get("method", "pgd"),
                    float(r.get("eps", 0.2)), bool(r.get("realistic", False))))
            if u.path == "/api/pcap":
                return self._json(STATE.score_pcap(body))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _static(self, path: str):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        return self._send(200, f.read_bytes(),
                          TYPES.get(f.suffix, "application/octet-stream"))


class ExclusiveServer(ThreadingHTTPServer):
    """Refuse to share a port.

    http.server sets SO_REUSEADDR, and on Windows that lets a second process bind
    a port another program is already listening on -- silently. The demo then
    reports "running" while the browser reaches the OTHER program. Exclusive
    binding turns that into an error we can react to."""
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):          # Windows only
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind_free_port(start: int, tries: int = 20) -> ExclusiveServer:
    for port in range(start, start + tries):
        try:
            return ExclusiveServer((HOST, port), Handler)
        except OSError:
            print(f"  port {port} is in use by another program, trying {port + 1}...",
                  flush=True)
    raise SystemExit(f"no free port in {start}-{start + tries - 1}; "
                     f"pass --port N to pick another range")


def main() -> None:
    global STATE
    port = PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    STATE = State()
    srv = bind_free_port(port)
    url = f"http://{HOST}:{srv.server_address[1]}/"
    print(f"demo running at {url}  (Ctrl+C to stop)", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
