"""HTTP client for the Windows MT5 gateway — runs on the Linux brain."""

from __future__ import annotations

import httpx
from datetime import datetime
    AccountState,
    Candle,
    ExecutionResult,
    Position,
    Tick,
    Timeframe,
    ValidatedOrder,
)
from aethel.mt5.base import MT5Client


class GatewayMT5Client(MT5Client):
    def __init__(self, base_url: str, token: str, timeout: float = 10.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    async def _get(self, path: str, **params):
        r = await self._http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, payload: dict):
        r = await self._http.post(path, json=payload)
        r.raise_for_status()
        return r.json()

    async def get_candles(self, symbol: str, timeframe: Timeframe, count: int) -> list[Candle]:
        data = await self._get("/candles", symbol=symbol, timeframe=timeframe.value, count=count)
        return [Candle(**c) for c in data]

    async def get_tick(self, symbol: str) -> Tick:
        return Tick(**await self._get("/tick", symbol=symbol))

    async def get_account(self) -> AccountState:
        return AccountState(**await self._get("/account"))

    async def get_positions(self) -> list[Position]:
        return [Position(**p) for p in await self._get("/positions")]

    async def place_limit_order(self, order: ValidatedOrder) -> ExecutionResult:
        return ExecutionResult(**await self._post("/orders/limit", order.model_dump()))

    async def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        data = await self._post("/positions/modify", {"ticket": ticket, "sl": sl, "tp": tp})
        return bool(data.get("ok"))

    async def close_position(self, ticket: int) -> bool:
        data = await self._post("/positions/close", {"ticket": ticket})
        return bool(data.get("ok"))

    async def get_closed_deals(self, since: datetime) -> list[dict]:
        return await self._get("/deals/closed", since=since.isoformat())
