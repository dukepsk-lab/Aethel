"""Feature engineering for Venus. All features are computed causally
(no future bars) and normalized with rolling statistics — never with
full-dataset statistics, which would leak."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_20",
    "range_norm", "body_norm", "upper_wick", "lower_wick",
    "vol_z", "atr_norm", "rsi", "dist_ema20", "dist_ema50",
]


def compute_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    c, h, low, o = ohlc["close"], ohlc["high"], ohlc["low"], ohlc["open"]
    v = ohlc["tick_volume"].astype(float)
    atr = (h - low).rolling(14).mean()
    rng = (h - low).replace(0, np.nan)

    df = pd.DataFrame(index=ohlc.index)
    df["ret_1"] = c.pct_change()
    df["ret_5"] = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)
    df["range_norm"] = (h - low) / atr
    df["body_norm"] = (c - o) / rng
    df["upper_wick"] = (h - np.maximum(c, o)) / rng
    df["lower_wick"] = (np.minimum(c, o) - low) / rng
    df["vol_z"] = (v - v.rolling(100).mean()) / v.rolling(100).std()
    df["atr_norm"] = atr / c
    df["rsi"] = _rsi(c, 14)
    df["dist_ema20"] = (c - c.ewm(span=20).mean()) / atr
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / atr
    return df[FEATURE_COLUMNS]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)) / 100  # scaled to [0, 1]
