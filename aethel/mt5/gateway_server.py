"""Thin FastAPI gateway that runs ON THE WINDOWS HOST next to the MT5 terminal.

Run with:  uvicorn aethel.mt5.gateway_server:app --host 0.0.0.0 --port 8001

It wraps DirectMT5Client and exposes exactly the operations the brain needs.
Protect it with AETHEL_MT5_GATEWAY_TOKEN and keep it on a private network.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aethel.core.schemas import Timeframe, ValidatedOrder
from aethel.mt5.direct import DirectMT5Client

app = FastAPI(title="Aethel MT5 Gateway")
_security = HTTPBearer()
_client: DirectMT5Client | None = None


def _auth(creds: HTTPAuthorizationCredentials = Depends(_security)) -> None:
    expected = os.environ.get("AETHEL_MT5_GATEWAY_TOKEN", "")
    if not expected or creds.credentials != expected:
        raise HTTPException(status_code=401, detail="invalid token")


def client() -> DirectMT5Client:
    global _client
    if _client is None:
        _client = DirectMT5Client()
    return _client


@app.get("/candles", dependencies=[Depends(_auth)])
async def candles(symbol: str, timeframe: Timeframe, count: int = 200):
    return [c.model_dump(mode="json") for c in await client().get_candles(symbol, timeframe, count)]


@app.get("/tick", dependencies=[Depends(_auth)])
async def tick(symbol: str):
    return (await client().get_tick(symbol)).model_dump(mode="json")


@app.get("/account", dependencies=[Depends(_auth)])
async def account():
    return (await client().get_account()).model_dump(mode="json")


@app.get("/positions", dependencies=[Depends(_auth)])
async def positions():
    return [p.model_dump(mode="json") for p in await client().get_positions()]


@app.post("/orders/limit", dependencies=[Depends(_auth)])
async def place_limit(order: ValidatedOrder):
    return (await client().place_limit_order(order)).model_dump(mode="json")


@app.post("/positions/modify", dependencies=[Depends(_auth)])
async def modify(payload: dict):
    ok = await client().modify_position(payload["ticket"], payload["sl"], payload["tp"])
    return {"ok": ok}


@app.post("/positions/close", dependencies=[Depends(_auth)])
async def close(payload: dict):
    return {"ok": await client().close_position(payload["ticket"])}
