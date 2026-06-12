"""Rule-based market regime classification — deterministic Python, no ML.

Produces a coarse regime label from H1 candles the orchestrator already
fetches, injected as plain context into the Ares and Athena prompts. Venus
already learns regime implicitly through its ADX / ATR-percentile / BB-width
features; this module exists only so the LLM agents can reason about it in
words. Explainable thresholds beat a trained model here — the label needs
to be roughly right and auditable, not optimal.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd
from pydantic import BaseModel

from aethel.core.schemas import Candle
from aethel.venus.features import _adx

# Classic Wilder threshold: ADX above this means a directional trend.
ADX_TREND_THRESHOLD = 25.0
# Current ATR in the top decile of the lookback window = volatility event.
ATR_HIGH_VOL_PCTILE = 0.90
ATR_LOOKBACK = 240  # ~10 days of H1 bars


class RegimeLabel(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class RegimeContext(BaseModel):
    label: RegimeLabel
    adx: float            # 0-100
    atr_percentile: float  # 0-1, current ATR vs lookback window


def classify_regime(candles: list[Candle]) -> RegimeContext:
    """Classify the H1 regime. high_volatility wins over trend labels because
    an ATR spike changes risk geometry regardless of direction."""
    if len(candles) < 60:
        raise ValueError(f"need >= 60 H1 candles for regime, got {len(candles)}")

    h = pd.Series([c.high for c in candles])
    low = pd.Series([c.low for c in candles])
    c = pd.Series([c.close for c in candles])

    adx = float(_adx(h, low, c, 14).iloc[-1])
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pctile = float(atr.rolling(min(ATR_LOOKBACK, len(candles))).rank(pct=True).iloc[-1])

    if atr_pctile >= ATR_HIGH_VOL_PCTILE:
        label = RegimeLabel.HIGH_VOLATILITY
    elif adx >= ADX_TREND_THRESHOLD:
        ema20 = c.ewm(span=20).mean()
        rising = ema20.iloc[-1] > ema20.iloc[-5]
        label = RegimeLabel.TRENDING_UP if rising else RegimeLabel.TRENDING_DOWN
    else:
        label = RegimeLabel.RANGING

    return RegimeContext(label=label, adx=round(adx, 1),
                         atr_percentile=round(atr_pctile, 3))
