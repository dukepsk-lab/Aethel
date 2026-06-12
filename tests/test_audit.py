"""Weekly-audit math: the pure PnL-metric reduction Themis is graded on.

The DB aggregation in gather_weekly_stats is thin SQLAlchemy over pgvector
tables (not constructible on sqlite); the logic worth guarding is here.
"""

import pandas as pd

from aethel.observability.audit import _pnl_metrics


def _series(profits, start="2026-06-01"):
    p = pd.Series(profits, dtype=float)
    closed = pd.Series(pd.date_range(start, periods=len(profits), freq="6h"))
    return p, closed


def test_empty_week_returns_nones():
    p, closed = _series([])
    out = _pnl_metrics(p, closed, equity_base=10_000)
    assert out == {"sharpe": None, "profit_factor": None, "max_drawdown_pct": None}


def test_profit_factor():
    p, closed = _series([100, -50, 200, -100])
    out = _pnl_metrics(p, closed, equity_base=10_000)
    assert out["profit_factor"] == 2.0  # 300 gains / 150 losses


def test_all_wins_has_no_profit_factor():
    p, closed = _series([100, 50])
    out = _pnl_metrics(p, closed, equity_base=10_000)
    assert out["profit_factor"] is None
    assert out["max_drawdown_pct"] == 0.0


def test_max_drawdown():
    # equity 10000 -> 10500 -> 9450 : drawdown 10% from the peak
    p, closed = _series([500, -1050])
    out = _pnl_metrics(p, closed, equity_base=10_000)
    assert out["max_drawdown_pct"] == 10.0


def test_sharpe_computed_when_enough_days():
    p, closed = _series([100, -50, 200, -100, 150, 80, -30, 120])
    out = _pnl_metrics(p, closed, equity_base=10_000)
    assert out["sharpe"] is not None
