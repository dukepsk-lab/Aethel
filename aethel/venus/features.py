"""Feature engineering for Venus. All features are computed causally
(no future bars) and normalized with rolling statistics — never with
full-dataset statistics, which would leak.

Feature groups:
- returns/price:   ret_1/5/20, fracdiff_z (stationary but memory-preserving)
- candle anatomy:  range_norm, body_norm, wicks, streak
- trend:           dist_ema20/50, macd_hist_norm, adx
- mean-reversion:  rsi, stoch_k, bb_pctb, bb_width
- volatility:      atr_norm, atr_pctile (regime), vol_z
- volume/price:    vwap_dist, vp_poc_dist (volume profile point-of-control)
- clock:           session flags (Asia/London/NY), hour + day-of-week sin/cos
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aethel.venus.fracdiff import fracdiff

FEATURE_COLUMNS = [
    # returns / price
    "ret_1", "ret_5", "ret_20", "fracdiff_z",
    # candle anatomy
    "range_norm", "body_norm", "upper_wick", "lower_wick", "streak",
    # trend
    "dist_ema20", "dist_ema50", "macd_hist_norm", "adx",
    # mean reversion
    "rsi", "stoch_k", "bb_pctb", "bb_width",
    # volatility regime
    "atr_norm", "atr_pctile", "vol_z",
    # volume / price interaction
    "vwap_dist", "vp_poc_dist",
    # clock
    "sess_asia", "sess_london", "sess_ny",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# warm-up bars needed before features are valid (longest lookback below)
WARMUP_BARS = 500


def compute_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    c, h, low, o = ohlc["close"], ohlc["high"], ohlc["low"], ohlc["open"]
    v = ohlc["tick_volume"].astype(float)
    atr = (h - low).rolling(14).mean()
    rng = (h - low).replace(0, np.nan)

    df = pd.DataFrame(index=ohlc.index)

    # --- returns / price ---
    df["ret_1"] = c.pct_change()
    df["ret_5"] = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)
    fd = fracdiff(np.log(c), d=0.4)
    df["fracdiff_z"] = (fd - fd.rolling(200).mean()) / fd.rolling(200).std()

    # --- candle anatomy ---
    df["range_norm"] = (h - low) / atr
    df["body_norm"] = (c - o) / rng
    df["upper_wick"] = (h - np.maximum(c, o)) / rng
    df["lower_wick"] = (np.minimum(c, o) - low) / rng
    df["streak"] = _streak(np.sign(c.diff())) / 10.0  # consecutive same-direction bars

    # --- trend ---
    df["dist_ema20"] = (c - c.ewm(span=20).mean()) / atr
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / atr
    macd_hist = (c.ewm(span=12).mean() - c.ewm(span=26).mean()) - \
        (c.ewm(span=12).mean() - c.ewm(span=26).mean()).ewm(span=9).mean()
    df["macd_hist_norm"] = macd_hist / atr
    df["adx"] = _adx(h, low, c, 14) / 100.0

    # --- mean reversion ---
    df["rsi"] = _rsi(c, 14)
    lo_k, hi_k = low.rolling(14).min(), h.rolling(14).max()
    df["stoch_k"] = (c - lo_k) / (hi_k - lo_k).replace(0, np.nan)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pctb"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    df["bb_width"] = (4 * bb_std) / bb_mid

    # --- volatility regime ---
    df["atr_norm"] = atr / c
    df["atr_pctile"] = atr.rolling(480).rank(pct=True)  # where current vol sits vs ~5 days
    df["vol_z"] = (v - v.rolling(100).mean()) / v.rolling(100).std()

    # --- volume / price interaction ---
    typical = (h + low + c) / 3
    pv = (typical * v).rolling(288).sum()  # ~1 trading day of M5
    vwap = pv / v.rolling(288).sum()
    df["vwap_dist"] = (c - vwap) / atr
    df["vp_poc_dist"] = _volume_profile_poc_dist(typical, v, c, atr,
                                                 window=96, bins=20)

    # --- clock (UTC) ---
    hours = ohlc.index.hour + ohlc.index.minute / 60.0
    df["sess_asia"] = ((hours >= 0) & (hours < 7)).astype(float)
    df["sess_london"] = ((hours >= 7) & (hours < 16)).astype(float)
    df["sess_ny"] = ((hours >= 12) & (hours < 21)).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    dow = ohlc.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    return df[FEATURE_COLUMNS]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)) / 100  # scaled to [0, 1]


def _adx(h: pd.Series, low: pd.Series, c: pd.Series, period: int) -> pd.Series:
    up = h.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period).mean()


def _streak(direction: pd.Series) -> pd.Series:
    """Signed count of consecutive same-direction closes (capped by /10 upstream)."""
    vals = direction.fillna(0).to_numpy()
    out = np.zeros(len(vals))
    for i in range(1, len(vals)):
        d = vals[i]
        if d == 0:
            out[i] = 0
        elif np.sign(out[i - 1]) == d:
            out[i] = out[i - 1] + d
        else:
            out[i] = d
    return pd.Series(out, index=direction.index)


def _volume_profile_poc_dist(typical: pd.Series, volume: pd.Series, close: pd.Series,
                             atr: pd.Series, window: int, bins: int) -> pd.Series:
    """Distance (in ATRs) from the rolling volume-profile point of control —
    the price level where the most volume traded over the last `window` bars.
    Price tends to be attracted to / rejected from high-volume nodes."""
    tp = typical.to_numpy()
    vol = volume.to_numpy()
    cl = close.to_numpy()
    at = atr.to_numpy()
    n = len(tp)
    out = np.full(n, np.nan)
    for i in range(window, n):
        w_tp = tp[i - window:i + 1]
        w_v = vol[i - window:i + 1]
        lo, hi = w_tp.min(), w_tp.max()
        if hi <= lo or not np.isfinite(at[i]) or at[i] <= 0:
            continue
        hist, edges = np.histogram(w_tp, bins=bins, range=(lo, hi), weights=w_v)
        poc = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
        out[i] = (cl[i] - poc) / at[i]
    return pd.Series(out, index=typical.index)
