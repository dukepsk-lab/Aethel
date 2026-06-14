"""Selective triple-barrier labeling for Helios.

Only labels bars that pass a setup filter:
- Volume spike: tick_volume > 1.2x rolling 20-bar mean
- Trend strength: |dist_ema50| > 0.3x ATR (not at equilibrium)
- Volatility regime: ATR not in top 5% (avoid chaotic bars)

Unlabeled bars are dropped — Helios trains on fewer but cleaner samples.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from aethel.venus.labeling import triple_barrier_labels, ewm_volatility


def selective_labels(
    m15: pd.DataFrame,
    h1_features: pd.DataFrame,    # must contain 'dist_ema50', 'atr'
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 48,
) -> pd.DataFrame:
    """Return label DataFrame (same schema as triple_barrier_labels) but only
    for bars that pass all three setup filters."""
    close = m15["close"]
    vol = ewm_volatility(close, span=100)

    # resample h1 features to m15 index
    dist_ema50 = h1_features["dist_ema50"].reindex(m15.index, method="ffill")
    h1_atr = h1_features["atr"].reindex(m15.index, method="ffill")

    direction = np.sign(dist_ema50).fillna(0).astype(int)

    # Setup filters
    tick_vol = m15["tick_volume"]
    vol_ma = tick_vol.rolling(20, min_periods=5).mean()
    volume_spike = tick_vol > (1.2 * vol_ma)

    trend_strong = dist_ema50.abs() > (0.3 * h1_atr)

    atr_pct = vol.rank(pct=True)
    not_chaotic = atr_pct < 0.95

    setup_mask = volume_spike & trend_strong & not_chaotic & (direction != 0)

    # Only run triple_barrier_labels on filtered rows — but we need OHLC context
    # for the barrier walk. Label all bars first, then filter.
    all_labels = triple_barrier_labels(m15, direction, tp_mult=tp_mult,
                                       sl_mult=sl_mult, max_holding=max_holding)

    # apply setup filter — only keep labeled bars that passed the filter
    filtered = all_labels[setup_mask.reindex(all_labels.index, fill_value=False)]
    return filtered
