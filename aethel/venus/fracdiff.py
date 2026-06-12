"""Fixed-width fractional differentiation (López de Prado, AFML ch. 5).

Raw prices are non-stationary; plain returns are stationary but throw away
all memory of price levels. Fractional differentiation with d ≈ 0.3-0.6
makes the series stationary while PRESERVING long-range memory — typically
the single highest-value transform for financial ML inputs.

(Native implementation — mlfinlab's is no longer on PyPI.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fracdiff_weights(d: float, threshold: float = 1e-4, max_width: int = 200) -> np.ndarray:
    """Binomial expansion weights for (1-B)^d, truncated where |w| < threshold."""
    w = [1.0]
    for k in range(1, max_width):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w)


def fracdiff(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
    """Fixed-width-window fractionally differentiated series. The first
    len(weights)-1 values are NaN (insufficient history) — callers must
    treat them as warm-up, exactly like a long moving average."""
    w = fracdiff_weights(d, threshold)
    width = len(w)
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    # dot the reversed weight vector over each trailing window
    w_rev = w[::-1]
    for i in range(width - 1, len(values)):
        window = values[i - width + 1:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(w_rev @ window)
    return pd.Series(out, index=series.index)
