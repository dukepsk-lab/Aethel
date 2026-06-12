"""Persisted daily risk state — the kill switch's source of truth.

The trading day boundary is daily_reset_hour_utc (default 21:00 UTC ≈ 5pm
New York broker rollover). A crash-restart reloads today's row; the loss
counter is never silently reset.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aethel.config import get_settings
from aethel.core.schemas import utcnow
from aethel.db.models import DailyRiskState


def current_trading_day() -> date:
    now = utcnow()
    reset_hour = get_settings().daily_reset_hour_utc
    # after the reset hour we are already in "tomorrow's" trading day
    return (now + timedelta(hours=24 - reset_hour)).date() if now.hour >= reset_hour \
        else now.date()


class RiskStateStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create(self, current_equity: float) -> DailyRiskState:
        day = current_trading_day()
        state = await self.session.get(DailyRiskState, day)
        if state is None:
            state = DailyRiskState(trading_day=day, start_equity=current_equity)
            self.session.add(state)
            await self.session.commit()
        return state

    async def daily_pnl_pct(self, current_equity: float) -> float:
        state = await self.get_or_create(current_equity)
        if state.start_equity <= 0:
            return 0.0
        return (current_equity - state.start_equity) / state.start_equity * 100.0

    async def record_trade(self) -> None:
        state = await self.session.get(DailyRiskState, current_trading_day())
        if state:
            state.trades_count += 1
            await self.session.commit()

    async def trip_kill_switch(self, reason: str) -> None:
        state = await self.session.get(DailyRiskState, current_trading_day())
        if state:
            state.kill_switch_tripped = True
            state.kill_switch_reason = reason
            await self.session.commit()

    async def is_tripped(self) -> bool:
        state = await self.session.get(DailyRiskState, current_trading_day())
        return bool(state and state.kill_switch_tripped)

    async def trades_today(self) -> int:
        state = await self.session.get(DailyRiskState, current_trading_day())
        return state.trades_count if state else 0
