"""Nonlinear substitute model for the attacker, with exact input gradients.

Why not LogisticRegression: a linear model's input gradient is always +/-w, so
FGSM produces exactly two perturbation directions -- one per class. The
perturbation then *encodes the label*, adversarial training learns that offset,
and the defended model scores a fake 100%. That is textbook label leaking
(Kurakin et al., 2017), and it was measured here, not assumed: every moved
dimension had a single sign across all samples of a class, and the two classes
pushed in opposite directions on 93/93 shared dimensions.

Why not PyTorch: a CPU wheel is a very large download, and a 2-hidden-layer MLP
with analytic backprop is ~60 lines of numpy. Deterministic, dependency-free,
and we control the loop -- which matters because the mask runs inside it.

Exposes the subset of the ART estimator interface attacks.py uses.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPSurrogate:
    """122 -> 64 -> 32 -> 2, ReLU, softmax + cross-entropy. Adam, fixed seed."""

    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=12, batch=256,
                 seed=42, verbose=True):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.batch, self.seed, self.verbose = batch, seed, verbose
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

    def _init(self, n_in: int, n_out: int) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [n_in, *self.hidden, n_out]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He init: keeps ReLU activations from collapsing.
            self.W.append(rng.normal(0, np.sqrt(2.0 / dims[i]),
                                     (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))

    def _forward(self, X: np.ndarray):
        acts, pre = [X], []
        h = X
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            pre.append(z)
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(h)
        return _softmax(h), acts, pre

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPSurrogate":
        X = np.asarray(X, dtype=np.float64)
        Y = np.eye(2)[np.asarray(y, dtype=int)]
        self._init(X.shape[1], 2)
        rng = np.random.default_rng(self.seed)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, epsA, t = 0.9, 0.999, 1e-8, 0

        for ep in range(self.epochs):
            idx = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), self.batch):
                bi = idx[s:s + self.batch]
                xb, yb = X[bi], Y[bi]
                p, acts, pre = self._forward(xb)
                tot += -np.sum(yb * np.log(p + 1e-12))
                t += 1
                g = (p - yb) / len(bi)
                for i in range(len(self.W) - 1, -1, -1):
                    gW = acts[i].T @ g
                    gb = g.sum(axis=0)
                    if i > 0:
                        g = (g @ self.W[i].T) * (pre[i - 1] > 0)
                    for prm, grd, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        m[i] = b1 * m[i] + (1 - b1) * grd
                        v[i] = b2 * v[i] + (1 - b2) * grd ** 2
                        prm[i] -= self.lr * (m[i] / (1 - b1 ** t)) / (
                            np.sqrt(v[i] / (1 - b2 ** t)) + epsA)
            if self.verbose and (ep + 1) % 4 == 0:
                acc = (self.predict(X) == y).mean()
                print(f"    epoch {ep+1:2d}/{self.epochs}  loss={tot/len(X):.4f}  acc={acc:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=np.float64))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dL/dX. Per-sample; scale is irrelevant to FGSM/PGD, which take sign()."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = np.eye(2)[Y.astype(int)]
        p, acts, pre = self._forward(X)
        g = p - Y
        for i in range(len(self.W) - 1, -1, -1):
            g = g @ self.W[i].T
            if i > 0:
                g = g * (pre[i - 1] > 0)
        return g
