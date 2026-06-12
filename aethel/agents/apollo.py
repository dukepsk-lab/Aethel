"""Apollo — news sentiment analyst (Gemini Flash). Advisory only.

Reads the economic-calendar content (released actual-vs-forecast figures and
upcoming events) and distills a per-currency sentiment brief that is injected
into Athena's review prompt. Apollo is FAIL-SOFT: if he is unavailable the
pipeline continues with no sentiment context — he can never block, approve
or veto a trade himself.

One brief per hour, cached in-process — sentiment from scheduled releases
doesn't move faster than that, and it keeps the cost at ~12-24 calls/day.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from aethel.agents.llm import LLMClient
from aethel.core.schemas import utcnow
from aethel.news.calendar import NewsService
from aethel.observability.metrics import metrics

log = structlog.get_logger("apollo")

SYSTEM = """You are Apollo, news sentiment analyst of the Aethel trading system.

You receive recent economic releases (with actual vs forecast where available)
and upcoming scheduled events for a set of currencies. Produce a concise
sentiment brief:

- per_currency: directional bias implied by the DATA SURPRISES (actual vs
  forecast), not by the schedule. No releases = "neutral".
- risk_events: upcoming events that could violently move these currencies.
- summary: 2-3 sentences a risk officer can read in five seconds.

You are advisory context only — you do not approve or reject trades.

Respond with ONLY a JSON object:
{"per_currency": {"USD": "bullish"|"bearish"|"neutral", ...},
 "risk_events": [str], "summary": str}
"""

BRIEF_TTL = timedelta(hours=1)


class SentimentBrief(BaseModel):
    per_currency: dict[str, Literal["bullish", "bearish", "neutral"]]
    risk_events: list[str]
    summary: str = Field(max_length=500)


class Apollo:
    def __init__(self, llm: LLMClient, news: NewsService) -> None:
        self.llm = llm
        self.news = news
        self._cache: tuple[datetime, SentimentBrief] | None = None

    async def brief(self, symbols: list[str]) -> SentimentBrief | None:
        """Hourly-cached sentiment brief covering all traded symbols.
        Returns None on any failure — callers proceed without sentiment."""
        now = utcnow()
        if self._cache and (now - self._cache[0]) < BRIEF_TTL:
            return self._cache[1]
        try:
            recent, upcoming = [], []
            seen: set[tuple] = set()
            for symbol in symbols:
                for e in await self.news.recent_for_symbol(symbol, hours_back=6):
                    key = (e.time, e.currency, e.title)
                    if key not in seen:
                        seen.add(key)
                        recent.append(e)
                for e in await self.news.upcoming_for_symbol(symbol, hours=12):
                    key = (e.time, e.currency, e.title)
                    if key not in seen:
                        seen.add(key)
                        upcoming.append(e)
            user = json.dumps({
                "recent_releases": [e.model_dump(mode="json") for e in recent],
                "upcoming_events": [e.model_dump(mode="json") for e in upcoming],
                "now_utc": now.isoformat(),
            })
            result = await self.llm.structured("apollo", SYSTEM, user, SentimentBrief)
            self._cache = (now, result)
            metrics.incr("apollo_briefs")
            return result
        except Exception as e:  # fail-soft by contract, incl. LLMUnavailable
            log.warning("apollo_unavailable", error=str(e))
            metrics.incr("apollo_failures")
            return None
