"""In-memory ring buffer of recent Venus signal evaluations.

The Decision table only records signals that passed the Signal Gate and
reached the agents; gate-blocked signals would otherwise be invisible in
the Command Center. This log keeps the last N evaluations (passed AND
blocked, with the block reason) for the /signals endpoint. In-memory by
design — it's monitoring data, not audit data; restarts clearing it is fine.
"""

from __future__ import annotations

from collections import deque

from aethel.core.schemas import VenusSignal, utcnow

MAX_SIGNALS = 300


class SignalLog:
    def __init__(self, maxlen: int = MAX_SIGNALS) -> None:
        self._buf: deque[dict] = deque(maxlen=maxlen)

    def record(self, signal: VenusSignal, passed: bool, reason: str,
               helios_confidence: float | None = None,
               helios_agreed: bool | None = None) -> None:
        self._buf.append({
            "time": utcnow().isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "confidence": round(signal.confidence, 4),
            "passed_gate": passed,
            "reason": reason,
            "model_version": signal.model_version,
            "helios_confidence": round(helios_confidence, 4) if helios_confidence is not None else None,
            "helios_agreed": helios_agreed,
        })

    def snapshot(self, limit: int = 100) -> list[dict]:
        return list(self._buf)[-limit:][::-1]  # newest first


signal_log = SignalLog()
