"""Forex market-hours helper.

Forex trades ~24/5: opens Sunday 22:00 UTC, closes Friday 22:00 UTC.
Shared by the orchestrator (to pause the trading loop on weekends) and the
API (to surface a "market closed" banner on the dashboard).
"""

from __future__ import annotations

from datetime import datetime, timezone


def market_is_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    wd = now.weekday()  # 0=Mon ... 5=Sat, 6=Sun
    if wd == 5:  # Saturday — closed all day
        return False
    if wd == 6 and now.hour < 22:  # Sunday before 22:00 UTC
        return False
    if wd == 4 and now.hour >= 22:  # Friday after 22:00 UTC
        return False
    return True


def market_status(now: datetime | None = None) -> dict:
    """Structured status for the API: open flag + human-readable note."""
    now = now or datetime.now(timezone.utc)
    is_open = market_is_open(now)
    return {
        "open": is_open,
        "note": (
            "Market open" if is_open
            else "Market closed — forex reopens Sunday 22:00 UTC"
        ),
    }
