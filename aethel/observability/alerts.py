"""Operational alerting — kill-switch trips, MT5 disconnects, execution
failures go to Telegram. Alert failures are logged and swallowed: alerting
must never take down the trading loop."""

from __future__ import annotations

import httpx
import structlog

from aethel.config import get_settings

log = structlog.get_logger("alerts")


async def send_alert(message: str) -> None:
    s = get_settings()
    if not s.telegram_bot_token or not s.telegram_chat_id:
        log.warning("alert_unconfigured", message=message)
        return
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            await http.post(
                f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage",
                json={"chat_id": s.telegram_chat_id, "text": f"🏛 Aethel\n{message}"},
            )
    except httpx.HTTPError as e:
        log.error("alert_failed", error=str(e), message=message)
