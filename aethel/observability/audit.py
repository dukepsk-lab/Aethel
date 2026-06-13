"""Weekly performance aggregation for Themis.

Pulls the last 7 days from the tables that already exist (Decision,
ClosedTrade, TradeMemory) plus the in-process metric counters, and reduces
them to a stats dict the auditor LLM can reason about. Ratio metrics
(Sharpe, profit factor, drawdown) come from quantstats when installed
(`pip install -e ".[audit]"`); otherwise a native fallback computes the
same numbers — the audit must never fail for lack of a plotting library.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aethel.core.schemas import utcnow
from aethel.db.models import ClosedTrade, Decision, TradeMemory
from aethel.observability.metrics import metrics


def _pnl_metrics(profits: pd.Series, closed_at: pd.Series, equity_base: float) -> dict:
    if profits.empty:
        return {"sharpe": None, "profit_factor": None, "max_drawdown_pct": None}
    daily = profits.groupby(closed_at.dt.date).sum()
    returns = daily / max(equity_base, 1.0)
    gains, losses = profits[profits > 0].sum(), -profits[profits < 0].sum()
    equity_curve = equity_base + profits.cumsum()
    drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    out = {
        "profit_factor": round(gains / losses, 2) if losses > 0 else None,
        "max_drawdown_pct": round(float(drawdown) * -100, 2),
        "sharpe": None,
    }
    try:
        import quantstats as qs
        ts = pd.Series(returns.values, index=pd.to_datetime(returns.index))
        out["sharpe"] = round(float(qs.stats.sharpe(ts)), 2)
    except Exception:
        if len(returns) >= 2 and returns.std() > 0:
            out["sharpe"] = round(float(returns.mean() / returns.std() * (252 ** 0.5)), 2)
    return out


async def gather_weekly_stats(session: AsyncSession, equity: float) -> dict:
    """Everything Themis needs to grade the week, as one JSON-able dict."""
    since = utcnow() - timedelta(days=7)

    trades = (await session.execute(
        select(ClosedTrade).where(ClosedTrade.closed_at >= since)
    )).scalars().all()
    profits = pd.Series([t.profit for t in trades], dtype=float)
    closed_at = pd.Series(pd.to_datetime([t.closed_at for t in trades]))

    per_symbol: dict[str, dict] = {}
    for t in trades:
        s = per_symbol.setdefault(t.symbol, {"trades": 0, "pnl": 0.0, "wins": 0})
        s["trades"] += 1
        s["pnl"] = round(s["pnl"] + t.profit, 2)
        s["wins"] += t.profit > 0

    decision_states = dict((await session.execute(
        select(Decision.state, func.count())
        .where(Decision.created_at >= since).group_by(Decision.state)
    )).all())

    lesson_quality = (await session.execute(
        select(func.count(TradeMemory.id)).where(TradeMemory.created_at >= since)
    )).scalar() or 0

    # Mnemosyne's hindsight quality scores, averaged over the week.
    quality_row = (await session.execute(
        select(func.avg(TradeMemory.ares_quality), func.avg(TradeMemory.athena_quality))
        .where(TradeMemory.created_at >= since)
    )).one_or_none()
    avg_ares_q = round(float(quality_row[0]), 2) if quality_row and quality_row[0] else None
    avg_athena_q = round(float(quality_row[1]), 2) if quality_row and quality_row[1] else None

    counters = dict(metrics.counters)
    return {
        "window_days": 7,
        "total_trades": len(trades),
        "win_rate": round(float((profits > 0).mean()), 3) if len(trades) else None,
        "net_pnl": round(float(profits.sum()), 2),
        **_pnl_metrics(profits, closed_at, equity),
        "per_symbol": per_symbol,
        "decision_states": decision_states,  # incl. VETOED / RISK_REJECTED counts
        "lessons_recorded": lesson_quality,
        "avg_ares_quality": avg_ares_q,
        "avg_athena_quality": avg_athena_q,
        "gate_blocked": counters.get("gate_blocked", 0),
        "risk_rejections": {k.removeprefix("risk_reject_"): v
                            for k, v in counters.items()
                            if k.startswith("risk_reject_")},
        "agent_failures": {
            "ares": counters.get("ares_failures", 0),
            "athena": counters.get("athena_failures", 0),
            "apollo": counters.get("apollo_failures", 0),
        },
    }
