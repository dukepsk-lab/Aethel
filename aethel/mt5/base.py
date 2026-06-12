"""Broker connectivity interface.

The official MetaTrader5 Python package is Windows-only (it speaks IPC to a
running terminal). Aethel therefore abstracts the broker behind this
interface with two implementations:

- DirectMT5Client: runs on Windows next to the terminal.
- GatewayMT5Client: runs anywhere, talks HTTP to a thin Windows gateway
  (aethel/mt5/gateway_server.py) that wraps the direct client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aethel.core.schemas import (
    AccountState,
    Candle,
    ExecutionResult,
    Position,
    Tick,
    Timeframe,
    ValidatedOrder,
)


class MT5Client(ABC):
    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: Timeframe, count: int) -> list[Candle]: ...

    @abstractmethod
    async def get_tick(self, symbol: str) -> Tick: ...

    @abstractmethod
    async def get_account(self) -> AccountState: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def place_limit_order(self, order: ValidatedOrder) -> ExecutionResult:
        """Place a pending limit order with expiry. SL/TP are attached
        broker-side at order time — never managed only in software."""

    @abstractmethod
    async def modify_position(self, ticket: int, sl: float, tp: float) -> bool: ...

    @abstractmethod
    async def close_position(self, ticket: int) -> bool: ...
