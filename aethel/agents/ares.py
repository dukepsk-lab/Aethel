"""Ares — market strategist (DeepSeek). Proposes a trade setup from the
Venus signal and market structure. Proposes risk_pct, never lot size."""

from __future__ import annotations

import json

from aethel.agents.llm import LLMClient
from aethel.core.schemas import AresProposal, Candle, Tick, VenusSignal

SYSTEM = """You are Ares, the market strategist of the Aethel trading system.

You receive a calibrated ML signal (Venus) and recent market structure for one
symbol. Decide whether there is a tradeable setup in the signal's direction
and, if so, propose precise levels.

Rules:
- Entry must be a LIMIT price at or better than the current market (you may
  propose a small pullback entry). Stops go beyond recent structure, not at
  round numbers. Minimum risk:reward 1.5.
- risk_pct is the % of account equity risked if the stop is hit (max 1.0).
- BUY requires stop_loss < entry < take_profit. SELL requires the reverse.
- Respond with ONLY a JSON object:
{"symbol": str, "action": "BUY"|"SELL", "entry": float, "stop_loss": float,
 "take_profit": float, "risk_pct": float, "rationale": str}
"""


class Ares:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def propose(
        self,
        signal: VenusSignal,
        m15_candles: list[Candle],
        tick: Tick,
        memory_notes: list[str],
    ) -> AresProposal:
        recent = [
            {"t": c.time.isoformat(), "o": c.open, "h": c.high, "l": c.low, "c": c.close}
            for c in m15_candles[-40:]
        ]
        user = json.dumps({
            "venus_signal": signal.model_dump(mode="json"),
            "current_bid": tick.bid,
            "current_ask": tick.ask,
            "recent_m15": recent,
            "lessons_from_similar_past_trades": memory_notes,
        })
        proposal = await self.llm.structured("deepseek", SYSTEM, user, AresProposal)
        if proposal.symbol != signal.symbol:
            raise ValueError("Ares proposed a different symbol than the signal")
        return proposal
