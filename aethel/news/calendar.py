"""Economic calendar ingestion.

Cloud LLMs cannot browse — Athena's news awareness is only as good as what we
inject into her prompt, and the Risk Gate's news blackout needs the same data.
This module defines the interface plus a default JSON-feed implementation
(compatible with ForexFactory-style weekly calendar JSON); swap in a paid
provider by implementing CalendarSource.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel

from aethel.core.schemas import utcnow

CURRENCIES_BY_SYMBOL = {
    "EURUSD": {"EUR", "USD"},
    "GBPUSD": {"GBP", "USD"},
    "USDJPY": {"USD", "JPY"},
    "XAUUSD": {"USD"},
}


class CalendarEvent(BaseModel):
    time: datetime
    currency: str
    title: str
    impact: str  # "high" | "medium" | "low"


class CalendarSource(ABC):
    @abstractmethod
    async def get_events(self) -> list[CalendarEvent]: ...


class JsonFeedCalendar(CalendarSource):
    """Weekly calendar JSON feed (ForexFactory-compatible shape)."""

    def __init__(self, url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"):
        self.url = url
        self._cache: list[CalendarEvent] = []
        self._fetched_at: datetime | None = None

    async def get_events(self) -> list[CalendarEvent]:
        now = utcnow()
        if self._fetched_at and (now - self._fetched_at) < timedelta(hours=1):
            return self._cache
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.get(self.url)
            r.raise_for_status()
        events = []
        for item in r.json():
            try:
                events.append(CalendarEvent(
                    time=item["date"],
                    currency=item["country"],
                    title=item["title"],
                    impact=str(item.get("impact", "low")).lower(),
                ))
            except (KeyError, ValueError):
                continue
        self._cache, self._fetched_at = events, now
        return events


class NewsService:
    def __init__(self, source: CalendarSource) -> None:
        self.source = source

    async def upcoming_for_symbol(self, symbol: str, hours: int = 12) -> list[CalendarEvent]:
        now = utcnow()
        horizon = now + timedelta(hours=hours)
        currencies = CURRENCIES_BY_SYMBOL.get(symbol, set())
        return [
            e for e in await self.source.get_events()
            if e.currency in currencies and now <= e.time <= horizon
        ]

    async def in_blackout(self, symbol: str, blackout_minutes: int) -> CalendarEvent | None:
        """Return the blocking event if a high-impact release is within the
        blackout window (before or after), else None."""
        now = utcnow()
        window = timedelta(minutes=blackout_minutes)
        for e in await self.upcoming_for_symbol(symbol, hours=2):
            if e.impact == "high" and abs(e.time - now) <= window:
                return e
        return None
