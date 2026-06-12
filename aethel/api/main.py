"""NEXMIND backend — FastAPI app serving the Command Center.

Read-mostly by design: the frontend observes the system. Exactly one write
path exists — the manual kill switch — because a human must always be able
to stop trading instantly. There is no endpoint that places trades; only
the orchestrator pipeline can reach Hermes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from sqlalchemy import select

from aethel.config import SYMBOLS, get_settings
from aethel.db.models import ClosedTrade, Decision
from aethel.db.risk_state import RiskStateStore
from aethel.db.session import get_sessionmaker, init_db
from aethel.mt5 import get_mt5_client
from aethel.observability.alerts import send_alert
from aethel.observability.metrics import metrics
from aethel.orchestrator import Orchestrator

log = structlog.get_logger("api")

_orchestrator: Orchestrator | None = None
_loop_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _loop_task
    await init_db()
    _orchestrator = Orchestrator(get_mt5_client(), Path("models/artifacts"))
    _loop_task = asyncio.create_task(_orchestrator.run_forever())
    yield
    if _loop_task:
        _loop_task.cancel()


app = FastAPI(title="Aethel NEXMIND", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    s = get_settings()
    return {"status": "ok", "shadow_mode": s.shadow_mode, "symbols": SYMBOLS}


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.get("/decisions")
async def decisions(limit: int = 50, symbol: str | None = None):
    async with get_sessionmaker()() as session:
        stmt = select(Decision).order_by(Decision.created_at.desc()).limit(limit)
        if symbol:
            stmt = stmt.where(Decision.symbol == symbol)
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "decision_id": d.decision_id, "symbol": d.symbol, "state": d.state,
                "venus_signal": d.venus_signal, "ares_proposal": d.ares_proposal,
                "athena_decision": d.athena_decision, "risk_outcome": d.risk_outcome,
                "execution": d.execution, "created_at": d.created_at.isoformat(),
            }
            for d in rows
        ]


@app.get("/trades")
async def trades(limit: int = 100):
    async with get_sessionmaker()() as session:
        stmt = select(ClosedTrade).order_by(ClosedTrade.closed_at.desc()).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "ticket": t.ticket, "symbol": t.symbol, "action": t.action,
                "lots": t.lots, "entry": t.entry, "exit": t.exit_price,
                "profit": t.profit, "closed_at": t.closed_at.isoformat(),
            }
            for t in rows
        ]


@app.get("/risk/status")
async def risk_status():
    async with get_sessionmaker()() as session:
        store = RiskStateStore(session)
        account = await get_mt5_client().get_account()
        return {
            "equity": account.equity,
            "daily_pnl_pct": await store.daily_pnl_pct(account.equity),
            "trades_today": await store.trades_today(),
            "kill_switch_tripped": await store.is_tripped(),
            "open_positions": account.open_positions,
        }


@app.post("/risk/kill-switch")
async def manual_kill_switch():
    """The one human write path: stop all new trading immediately.
    Resets only at the next trading-day boundary — no un-trip endpoint
    by design."""
    async with get_sessionmaker()() as session:
        store = RiskStateStore(session)
        account = await get_mt5_client().get_account()
        await store.get_or_create(account.equity)
        await store.trip_kill_switch("manual trip from NEXMIND")
    await send_alert("🛑 Kill switch tripped MANUALLY from NEXMIND")
    return {"kill_switch_tripped": True}
