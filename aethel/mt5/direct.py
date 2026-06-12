"""Direct MetaTrader5 client — Windows only.

Imports the MetaTrader5 package lazily so the rest of Aethel can run on Linux.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from aethel.core.schemas import (
    AccountState,
    Action,
    Candle,
    ExecutionResult,
    Position,
    Tick,
    Timeframe,
    ValidatedOrder,
)
from aethel.config import SYMBOL_SPECS
from aethel.mt5.base import MT5Client

_TF_MAP = {"M1": 1, "M15": 15, "H1": 16385}  # MetaTrader5 timeframe constants


class DirectMT5Client(MT5Client):
    def __init__(self) -> None:
        import MetaTrader5 as mt5  # Windows only

        self._mt5 = mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    async def _run(self, fn, *args):
        # MetaTrader5 calls are blocking; keep the event loop responsive.
        return await asyncio.get_event_loop().run_in_executor(None, fn, *args)

    async def get_candles(self, symbol: str, timeframe: Timeframe, count: int) -> list[Candle]:
        rates = await self._run(
            self._mt5.copy_rates_from_pos, symbol, _TF_MAP[timeframe.value], 0, count
        )
        if rates is None:
            raise ConnectionError(f"copy_rates failed for {symbol}: {self._mt5.last_error()}")
        return [
            Candle(
                time=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]),
                tick_volume=int(r["tick_volume"]),
            )
            for r in rates
        ]

    async def get_tick(self, symbol: str) -> Tick:
        t = await self._run(self._mt5.symbol_info_tick, symbol)
        if t is None:
            raise ConnectionError(f"symbol_info_tick failed for {symbol}")
        return Tick(time=datetime.fromtimestamp(t.time, tz=timezone.utc), bid=t.bid, ask=t.ask)

    async def get_account(self) -> AccountState:
        info = await self._run(self._mt5.account_info)
        if info is None:
            raise ConnectionError("account_info failed")
        positions = await self.get_positions()
        return AccountState(
            balance=info.balance, equity=info.equity,
            margin_free=info.margin_free, open_positions=len(positions),
        )

    async def get_positions(self) -> list[Position]:
        raw = await self._run(self._mt5.positions_get) or []
        return [
            Position(
                ticket=p.ticket, symbol=p.symbol,
                action=Action.BUY if p.type == 0 else Action.SELL,
                lots=p.volume, entry=p.price_open, sl=p.sl, tp=p.tp,
                profit=p.profit, magic=p.magic,
            )
            for p in raw
        ]

    async def place_limit_order(self, order: ValidatedOrder) -> ExecutionResult:
        mt5 = self._mt5
        order_type = (
            mt5.ORDER_TYPE_BUY_LIMIT if order.action == Action.BUY
            else mt5.ORDER_TYPE_SELL_LIMIT
        )
        expiry = datetime.now(timezone.utc) + timedelta(seconds=order.order_expiry_seconds)
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": order.symbol,
            "volume": order.lots,
            "type": order_type,
            "price": order.entry,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "magic": order.magic,
            "comment": f"aethel:{order.decision_id[:8]}",
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": int(expiry.timestamp()),
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = await self._run(mt5.order_send, request)
        if result is None:
            return ExecutionResult(decision_id=order.decision_id, success=False,
                                   detail=str(mt5.last_error()))
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        return ExecutionResult(
            decision_id=order.decision_id, success=ok,
            ticket=result.order if ok else None,
            retcode=result.retcode, detail=result.comment,
        )

    async def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        mt5 = self._mt5
        positions = await self._run(mt5.positions_get)
        pos = next((p for p in positions or [] if p.ticket == ticket), None)
        if pos is None:
            return False
        result = await self._run(mt5.order_send, {
            "action": mt5.TRADE_ACTION_SLTP, "position": ticket,
            "symbol": pos.symbol, "sl": sl, "tp": tp,
        })
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    async def close_position(self, ticket: int) -> bool:
        mt5 = self._mt5
        positions = await self._run(mt5.positions_get)
        pos = next((p for p in positions or [] if p.ticket == ticket), None)
        if pos is None:
            return False
        tick = await self._run(mt5.symbol_info_tick, pos.symbol)
        is_buy = pos.type == 0
        result = await self._run(mt5.order_send, {
            "action": mt5.TRADE_ACTION_DEAL, "position": ticket,
            "symbol": pos.symbol, "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if is_buy else tick.ask,
            "magic": pos.magic, "comment": "aethel:close",
        })
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
