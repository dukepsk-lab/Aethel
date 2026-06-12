"""Triple-barrier labeling (López de Prado).

For each bar we set three barriers from the entry price:
- upper barrier:  entry + tp_mult * volatility
- lower barrier:  entry - sl_mult * volatility
- vertical barrier: max_holding bars

Label = 1 if the profit barrier is hit first (in the trade direction),
0 otherwise. Venus is trained on this label, so its calibrated output is
directly P(TP hit before SL) — exactly the confidence semantics that Ares
and Athena consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially weighted volatility of returns, in price terms."""
    returns = close.pct_change()
    return returns.ewm(span=span).std() * close


def triple_barrier_labels(
    ohlc: pd.DataFrame,
    direction: pd.Series,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 60,
    vol_span: int = 100,
) -> pd.DataFrame:
    """direction: +1 (long) / -1 (short) per bar; 0 rows are skipped.

    Returns DataFrame with columns: label (0/1), t_end (index of barrier touch).
    The t_end column is required by the purged walk-forward splitter to
    prevent leakage from overlapping label windows.
    """
    close, high, low = ohlc["close"], ohlc["high"], ohlc["low"]
    vol = ewm_volatility(close, span=vol_span)
    n = len(ohlc)
    labels = np.full(n, np.nan)
    t_end = np.full(n, -1, dtype=np.int64)

    for i in range(n - 1):
        d = direction.iloc[i]
        if d == 0 or np.isnan(vol.iloc[i]):
            continue
        entry = close.iloc[i]
        upper = entry + d * tp_mult * vol.iloc[i]
        lower = entry - d * sl_mult * vol.iloc[i]
        end = min(i + max_holding, n - 1)
        labels[i], t_end[i] = 0, end
        for j in range(i + 1, end + 1):
            hit_tp = high.iloc[j] >= upper if d > 0 else low.iloc[j] <= upper
            hit_sl = low.iloc[j] <= lower if d > 0 else high.iloc[j] >= lower
            if hit_sl:  # conservative: same-bar double touch counts as a loss
                labels[i], t_end[i] = 0, j
                break
            if hit_tp:
                labels[i], t_end[i] = 1, j
                break

    out = pd.DataFrame({"label": labels, "t_end": t_end}, index=ohlc.index)
    return out.dropna(subset=["label"])
