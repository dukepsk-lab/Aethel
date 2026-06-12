"""Lightweight in-process metrics, exposed via the API for Aethel Command Center.

Tracks the numbers that tell you whether the system is healthy and whether
the agents are earning their latency: per-stage timings, veto rate,
gate-block reasons, execution outcomes.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from contextlib import contextmanager


class Metrics:
    def __init__(self, window: int = 500) -> None:
        self.counters: Counter[str] = Counter()
        self.stage_latency: dict[str, deque[float]] = {}
        self._window = window

    def incr(self, name: str) -> None:
        self.counters[name] += 1

    @contextmanager
    def time_stage(self, stage: str):
        start = time.monotonic()
        try:
            yield
        finally:
            bucket = self.stage_latency.setdefault(stage, deque(maxlen=self._window))
            bucket.append(time.monotonic() - start)

    def snapshot(self) -> dict:
        lat = {
            stage: {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values) * 1000, 1),
                "max_ms": round(max(values) * 1000, 1),
            }
            for stage, values in self.stage_latency.items() if values
        }
        total = self.counters["athena_approve"] + self.counters["athena_veto"]
        return {
            "counters": dict(self.counters),
            "stage_latency": lat,
            "veto_rate": round(self.counters["athena_veto"] / total, 3) if total else None,
        }


metrics = Metrics()
