"""Rule-based trade management — pure Python, runs every management tick.

Agents participate at entry only. Exits are SL/TP plus these mechanical
rules, so there is no LLM latency or cost in the position-management path:
- move SL to breakeven at +1R,
- trail SL behind price by 1R once past +1.5R.
"""

from __future__ import annotations

import structlog

from aethel.config import get_settings
from aethel.core.schemas import Action, Position
from aethel.mt5.base import MT5Client

log = structlog.get_logger("hermes.management")


class TradeManager:
    def __init__(self, mt5: MT5Client) -> None:
        self.mt5 = mt5
        self.s = get_settings()
        # original risk per ticket, captured when we first see the position
        self._initial_risk: dict[int, float] = {}

    async def manage_all(self) -> None:
        positions = await self.mt5.get_positions()
        managed = [p for p in positions if p.magic >= self.s.magic_base]
        for pos in managed:
            try:
                await self._manage(pos)
            except Exception as e:  # one bad position must not stop the loop
                log.error("manage_failed", ticket=pos.ticket, error=str(e))

    async def _manage(self, pos: Position) -> None:
        risk = self._initial_risk.setdefault(pos.ticket, abs(pos.entry - pos.sl))
        if risk <= 0:
            return
        tick = await self.mt5.get_tick(pos.symbol)
        is_buy = pos.action == Action.BUY
        price = tick.bid if is_buy else tick.ask
        r_multiple = ((price - pos.entry) if is_buy else (pos.entry - price)) / risk

        new_sl = pos.sl
        if r_multiple >= self.s.trailing_start_rr:
            trail = price - risk if is_buy else price + risk
            new_sl = max(pos.sl, trail) if is_buy else min(pos.sl, trail)
        elif r_multiple >= self.s.breakeven_trigger_rr:
            be = pos.entry
            new_sl = max(pos.sl, be) if is_buy else min(pos.sl, be)

        if new_sl != pos.sl:
            ok = await self.mt5.modify_position(pos.ticket, new_sl, pos.tp)
            log.info("sl_moved", ticket=pos.ticket, old_sl=pos.sl, new_sl=new_sl,
                     r_multiple=round(r_multiple, 2), ok=ok)
