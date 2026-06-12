"""Probability calibration for Venus outputs.

A raw sigmoid is not a probability. Agents (and the Signal Gate threshold)
reason over confidence as if it were P(TP before SL), so we fit isotonic
regression on out-of-fold validation predictions and apply it at inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> None:
        self._iso.fit(raw_probs, labels)
        self._fitted = True

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")
        return self._iso.predict(raw_probs)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self._iso))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        c = cls()
        c._iso = pickle.loads(path.read_bytes())
        c._fitted = True
        return c
