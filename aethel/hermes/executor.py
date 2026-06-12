"""Hermes — execution engine. Pure Python, zero LLM calls in this path.

Properties:
- Idempotent: a decision_id can only ever be executed once (in-memory set
  plus DB unique constraint as the durable backstop).
- Shadow mode: when enabled, orders are fully validated and logged but never
  sent — the honest way to evaluate the system before risking capital,
  since LLM agents cannot be backtested historically.
- Broker-side SL/TP are part of the order itself (see MT5Client), so a dead
  bot process never means an unprotected position.
"""

from __future__ import annotations

import structlog

from aethel.config import get_settings
from aethel.core.schemas import ExecutionResult, ValidatedOrder
from aethel.mt5.base import MT5Client

log = structlog.get_logger("hermes")


class Hermes:
    def __init__(self, mt5: MT5Client) -> None:
        self.mt5 = mt5
        self.s = get_settings()
        self._executed: set[str] = set()

    async def execute(self, order: ValidatedOrder) -> ExecutionResult:
        if order.decision_id in self._executed:
            return ExecutionResult(
                decision_id=order.decision_id, success=False,
                detail="duplicate decision_id — already executed")
        self._executed.add(order.decision_id)

        if self.s.shadow_mode:
            log.info("shadow_execution", **order.model_dump())
            return ExecutionResult(decision_id=order.decision_id, success=True,
                                   shadow=True, detail="shadow mode — order logged, not sent")

        result = await self.mt5.place_limit_order(order)
        if result.success:
            log.info("order_placed", ticket=result.ticket, **order.model_dump())
        else:
            log.error("order_failed", retcode=result.retcode, detail=result.detail,
                      decision_id=order.decision_id)
        return result
