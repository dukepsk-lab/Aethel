"""Signal Gate — sits between Venus and the agent team.

Without it, M5-cadence signals × 4 symbols would mean hundreds of LLM calls
per day chasing noise. Agents are consulted only when:
- Venus confidence >= threshold,
- the per-symbol cooldown has elapsed,
- the global agent-calls-per-hour budget has headroom.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from aethel.config import get_settings
from aethel.core.schemas import VenusSignal, utcnow


class SignalGate:
    def __init__(self) -> None:
        self.s = get_settings()
        self._last_consult: dict[str, datetime] = {}
        self._call_times: deque[datetime] = deque()

    def should_consult_agents(self, signal: VenusSignal) -> tuple[bool, str]:
        now = utcnow()

        if signal.confidence < self.s.venus_confidence_threshold:
            return False, (f"confidence {signal.confidence:.2f} < "
                           f"threshold {self.s.venus_confidence_threshold}")

        last = self._last_consult.get(signal.symbol)
        if last and (now - last).total_seconds() < self.s.signal_cooldown_seconds:
            return False, f"cooldown active for {signal.symbol}"

        cutoff = now - timedelta(hours=1)
        while self._call_times and self._call_times[0] < cutoff:
            self._call_times.popleft()
        if len(self._call_times) >= self.s.max_agent_calls_per_hour:
            return False, "hourly agent call budget exhausted"

        return True, "ok"

    def record_consultation(self, symbol: str) -> None:
        now = utcnow()
        self._last_consult[symbol] = now
        self._call_times.append(now)
