"""Aethel Command Center backend — FastAPI app serving the Command Center.

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
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from aethel.config import SYMBOL_SPECS, SYMBOLS, get_settings
from aethel.db.models import ClosedTrade, Decision
from aethel.db.risk_state import RiskStateStore
from aethel.db.session import get_sessionmaker, init_db
from aethel.mt5 import get_mt5_client
from aethel.news.calendar import JsonFeedCalendar, NewsService
from aethel.observability.alerts import send_alert
from aethel.observability.metrics import metrics
from aethel.observability.signal_log import signal_log
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


app = FastAPI(title="Aethel Command Center", version="0.1.0", lifespan=lifespan)

_cors_origins = get_settings().cors_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    s = get_settings()
    return {"status": "ok", "shadow_mode": s.shadow_mode, "symbols": SYMBOLS}


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.get("/retrain/status")
async def retrain_status():
    """Venus model registry — per-symbol last retrain time, action taken and
    the benchmark metrics that decided promotion."""
    from aethel.venus.retrain import get_registry

    return get_registry()


@app.get("/signals")
async def signals(limit: int = 100):
    """Every Venus signal evaluation — including gate-blocked ones with the
    block reason. In-memory ring buffer; cleared on restart."""
    return signal_log.snapshot(limit)


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


_news = NewsService(JsonFeedCalendar())


@app.get("/shadow-trades")
async def shadow_trades(limit: int = 50):
    """Orders that passed every gate in shadow mode but were never sent.
    Hypothetical floating P&L is marked-to-market against the current tick —
    this is what the system WOULD be holding if shadow mode were off."""
    mt5 = get_mt5_client()
    ticks: dict[str, float] = {}
    async with get_sessionmaker()() as session:
        stmt = (select(Decision)
                .where(Decision.state.in_(["EXECUTED", "MANAGED"]))
                .order_by(Decision.created_at.desc()).limit(limit))
        rows = (await session.execute(stmt)).scalars().all()
    out = []
    for d in rows:
        if not (d.execution or {}).get("shadow") or not d.risk_outcome:
            continue
        o = d.risk_outcome
        sym, spec = o["symbol"], SYMBOL_SPECS[o["symbol"]]
        if sym not in ticks:
            try:
                t = await mt5.get_tick(sym)
                ticks[sym] = (t.bid + t.ask) / 2
            except Exception:
                ticks[sym] = 0.0
        mid = ticks[sym]
        sign = 1 if o["action"] == "BUY" else -1
        pnl = ((mid - o["entry"]) / spec.pip_size * spec.pip_value_per_lot
               * o["lots"] * sign) if mid else None
        out.append({
            "decision_id": d.decision_id, "symbol": sym, "action": o["action"],
            "lots": o["lots"], "entry": o["entry"], "stop_loss": o["stop_loss"],
            "take_profit": o["take_profit"], "current_price": mid or None,
            "hypothetical_pnl": round(pnl, 2) if pnl is not None else None,
            "created_at": d.created_at.isoformat(),
        })
    return out


@app.get("/news")
async def news():
    """Upcoming calendar events (12h horizon) and recent events (last 6h) for
    traded symbols, plus the per-symbol blackout flag the Risk Gate is enforcing."""
    s = get_settings()
    events, seen = [], set()
    blackout = {}
    for sym in SYMBOLS:
        try:
            blocking = await _news.in_blackout(sym, s.news_blackout_minutes)
            blackout[sym] = blocking.title if blocking else None
            for e in await _news.upcoming_for_symbol(sym, hours=12):
                key = (e.time, e.currency, e.title)
                if key not in seen:
                    seen.add(key)
                    events.append({**e.model_dump(mode="json"), "upcoming": True})
            for e in await _news.recent_for_symbol(sym, hours_back=6):
                key = (e.time, e.currency, e.title)
                if key not in seen:
                    seen.add(key)
                    events.append({**e.model_dump(mode="json"), "upcoming": False})
        except Exception:
            blackout[sym] = None  # feed down — risk gate handles its own fetch
    events.sort(key=lambda e: e["time"], reverse=True)
    return {"events": events, "blackout": blackout,
            "blackout_minutes": s.news_blackout_minutes}


@app.get("/positions")
async def positions():
    """Currently open positions straight from MT5 (live floating P&L)."""
    pos = await get_mt5_client().get_positions()
    return [p.model_dump(mode="json") for p in pos]


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
            "shadow_mode": get_settings().shadow_mode,
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
        await store.trip_kill_switch("manual trip from Aethel Command Center")
    await send_alert("🛑 Kill switch tripped MANUALLY from Aethel Command Center")
    return {"kill_switch_tripped": True}


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("aethel.api.main:app", host=s.api_host, port=s.api_port,
                reload=False, log_level="info")
