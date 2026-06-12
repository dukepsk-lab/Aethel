"""Regime classifier: synthetic series must map to the expected labels."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from aethel.core.regime import RegimeLabel, classify_regime
from aethel.core.schemas import Candle


def _candles(closes: list[float], spread: float = 0.0005) -> list[Candle]:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        out.append(Candle(
            time=t0 + timedelta(hours=i), open=o,
            high=max(o, c) + spread, low=min(o, c) - spread,
            close=c, tick_volume=100,
        ))
        prev = c
    return out


def test_uptrend_is_trending_up():
    closes = [1.10 + 0.0008 * i for i in range(300)]
    ctx = classify_regime(_candles(closes))
    assert ctx.label == RegimeLabel.TRENDING_UP
    assert ctx.adx >= 25


def test_downtrend_is_trending_down():
    closes = [1.30 - 0.0008 * i for i in range(300)]
    ctx = classify_regime(_candles(closes))
    assert ctx.label == RegimeLabel.TRENDING_DOWN


def test_flat_oscillation_is_ranging():
    closes = [1.10 + 0.0003 * math.sin(i / 3) for i in range(300)]
    ctx = classify_regime(_candles(closes))
    assert ctx.label == RegimeLabel.RANGING
    assert ctx.adx < 25


def test_volatility_spike_wins_over_trend():
    closes = [1.10 + 0.0003 * math.sin(i / 3) for i in range(280)]
    # sudden wide bars at the end -> ATR jumps into the top decile
    last = closes[-1]
    for i in range(20):
        last += 0.008 if i % 2 == 0 else -0.006
        closes.append(last)
    ctx = classify_regime(_candles(closes))
    assert ctx.label == RegimeLabel.HIGH_VOLATILITY
    assert ctx.atr_percentile >= 0.9


def test_too_few_candles_raises():
    with pytest.raises(ValueError):
        classify_regime(_candles([1.1] * 30))
